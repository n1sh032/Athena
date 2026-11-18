import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from scripts.check_data import read_labels


to_input = Compose([
    ToTensor(),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def load_image(path):
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("Expected a PNG image")

            if image.mode == "I" or image.mode.startswith("I;16"):
                gray = image.convert("I").point(lambda value: value / 257).convert("L")
            else:
                gray = image.convert("L")

            padded = ImageOps.pad(gray, (224, 224), Image.Resampling.BILINEAR, color=0)
            return to_input(padded.convert("RGB"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load {path}: {error}") from error


class HandImages(Dataset):
    def __init__(self, labels, ids, image_dir):
        self.labels = labels
        self.ids = ids
        self.image_dir = image_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = load_image(self.image_dir / f"{image_id}.png")
        age = torch.tensor(self.labels[image_id][0], dtype=torch.float32)
        return image, age


def read_split(path, labels):
    split = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(split, dict):
        raise ValueError("Expected a baseline split JSON object")

    train = split.get("train_ids")
    dev = split.get("dev_ids")
    for ids in (train, dev):
        if not isinstance(ids, list) or not ids:
            raise ValueError("Split needs nonempty train_ids and dev_ids lists")
        if any(type(image_id) is not int for image_id in ids):
            raise ValueError("Split IDs must be integers")

    all_ids = train + dev
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("The saved split contains duplicate IDs")
    if set(all_ids) != labels.keys():
        raise ValueError("The saved split does not match the labels CSV")

    return train, dev


def make_datasets(labels_path, image_dir, split_path):
    labels = read_labels(labels_path)
    train, dev = read_split(split_path, labels)

    missing = [
        image_id for image_id in labels
        if not (image_dir / f"{image_id}.png").is_file()
    ]
    if missing:
        raise ValueError(f"Missing PNGs: {len(missing)}. First missing image: {missing[0]}.png")

    return HandImages(labels, train, image_dir), HandImages(labels, dev, image_dir)


def main():
    parser = argparse.ArgumentParser(description="Check the first image batch in each split")
    parser.add_argument("labels", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--split", type=Path, default=Path("artifacts/median-baseline.json"))
    args = parser.parse_args()

    try:
        train, dev = make_datasets(args.labels, args.images, args.split)
        for name, dataset in (("Train", train), ("Development", dev)):
            images, ages = next(iter(DataLoader(dataset, batch_size=4)))
            print(f"{name}: {len(dataset)} images, batch shape {tuple(images.shape)}")
            print(f"  Age labels in months: {ages.tolist()}")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Image check failed: {error}\n")


if __name__ == "__main__":
    main()
