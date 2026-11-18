import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from io import TextIOWrapper
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader

from scripts.check_data import read_labels
from scripts.image_data import HandImages, read_split
from scripts.train_resnet import make_model as make_resnet
from scripts.train_vit import make_model as make_vit


def read_holdout(archive):
    csvs = [name for name in archive.namelist() if Path(name).name == "Validation Dataset.csv"]
    if len(csvs) != 1:
        raise ValueError("Expected one Validation Dataset.csv in the holdout archive")

    labels = {}
    with archive.open(csvs[0]) as data, TextIOWrapper(data, encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            image_id = int(row["Image ID"])
            age = int(row["Bone Age (months)"])
            male = row["male"].strip().lower()
            if image_id in labels or image_id <= 0 or age < 0 or male not in ("true", "false"):
                raise ValueError(f"Invalid or duplicate holdout label for image {image_id}")
            labels[image_id] = (age, male == "true")
    if not labels:
        raise ValueError("The holdout has no labels")
    return labels


def unpack(archive, training, labels, folder):
    # Identical PNGs have the same size and CRC. Compare bytes only for matches.
    fingerprints = defaultdict(list)
    train_ids = set()
    for member in training.infolist():
        name = Path(member.filename)
        if name.suffix.lower() == ".png":
            image_id = int(name.stem)
            if image_id in train_ids:
                raise ValueError("Duplicate PNG ID in the training archive")
            train_ids.add(image_id)
            fingerprints[member.file_size, member.CRC].append(member)

    if train_ids & labels.keys():
        raise ValueError("The training and holdout archives share image IDs")

    found = set()
    for name in archive.namelist():
        if not name.lower().endswith(".zip"):
            continue
        nested = folder / "images.zip"
        with archive.open(name) as src, nested.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        with ZipFile(nested) as images:
            for member in images.infolist():
                path = Path(member.filename)
                if path.suffix.lower() != ".png":
                    continue
                image_id = int(path.stem)
                if image_id in found or image_id not in labels:
                    raise ValueError(f"Unexpected or duplicate holdout PNG ID: {image_id}")
                png = images.read(member)
                for candidate in fingerprints[member.file_size, member.CRC]:
                    if png == training.read(candidate):
                        raise ValueError("The training and holdout archives share a byte-identical PNG")
                (folder / f"{image_id}.png").write_bytes(png)
                found.add(image_id)
        nested.unlink()

    if found != labels.keys():
        raise ValueError("The holdout images do not match its labels")
    return train_ids


def summarize(predictions, rows):
    groups = defaultdict(list)
    for predicted, (age, male) in zip(predictions, rows, strict=True):
        error = abs(predicted - age)
        lower = min(age // 60 * 60, 180)
        band = f"{lower}-{lower + 59}" if lower < 180 else "180+"
        groups["overall"].append(error)
        groups[f"age_{band}"].append(error)
        groups["recorded_male" if male else "recorded_female"].append(error)
    return {
        name: {"count": len(errors), "mae_months": sum(errors) / len(errors)}
        for name, errors in groups.items()
    }


def load_head(run_folder, name, head, fingerprints):
    run = json.loads((run_folder / "run.json").read_text(encoding="utf-8"))
    if run["model"] != name or run["weights"] != "IMAGENET1K_V1":
        raise ValueError(f"Unexpected architecture or weights in {run_folder}")
    for key, expected in fingerprints.items():
        if run[key] != expected:
            raise ValueError(f"{key} does not match the saved run in {run_folder}")
    checkpoint = run_folder / "best-head.pt"
    head.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return run["best_epoch"], hashlib.sha256(checkpoint.read_bytes()).hexdigest()


@torch.inference_mode()
def predict(model, batches, device):
    model.to(device).eval()
    predictions = []
    for images, _ in batches:
        predictions.extend(model(images.to(device)).squeeze(1).tolist())
    return predictions


def evaluate(args):
    train_labels = read_labels(args.labels)
    train, _ = read_split(args.split, train_labels)
    baseline = median(train_labels[image_id][0] for image_id in train)
    fingerprints = {
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        with ZipFile(args.holdout) as archive, ZipFile(args.training_zip) as training:
            labels = read_holdout(archive)
            if labels.keys() & train_labels.keys():
                raise ValueError("The training and holdout annotations share image IDs")
            if unpack(archive, training, labels, folder) != train_labels.keys():
                raise ValueError("The training archive does not match the training labels")
        print(f"Checked {len(labels)} holdout PNGs for label matches and archive overlap", flush=True)
        ids = sorted(labels)
        rows = [labels[image_id] for image_id in ids]
        batches = DataLoader(HandImages(labels, ids, folder), batch_size=args.batch_size)
        report = {
            **fingerprints,
            "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "baseline_months": baseline,
            "results": {"median": summarize([baseline] * len(rows), rows)},
            "checkpoints": {},
        }
        for name, make_model, layer in (
            ("resnet18", make_resnet, "fc"), ("vit_b_16", make_vit, "heads.head")
        ):
            model = make_model(baseline)
            epoch, digest = load_head(args.runs / name, name, model.get_submodule(layer), fingerprints)
            report["checkpoints"][name] = {"epoch": epoch, "head_sha256": digest}
            report["results"][name] = summarize(predict(model, batches, device), rows)
            del model
            mae = report["results"][name]["overall"]["mae_months"]
            print(f"{name}: {mae:.2f} months MAE", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved heads on the RSNA holdout")
    parser.add_argument("labels", type=Path, help="original training labels CSV")
    parser.add_argument("training_zip", type=Path, help="official training image archive")
    parser.add_argument("holdout", type=Path, help="official validation archive")
    parser.add_argument("runs", type=Path, help="folder containing resnet18 and vit_b_16 run folders")
    parser.add_argument("--split", type=Path, default=Path("artifacts/median-baseline.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("artifacts/holdout.json"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.out.exists():
        parser.error("Use a positive batch size and a new output filename")

    report = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved aggregate results to {args.out}")


if __name__ == "__main__":
    main()
