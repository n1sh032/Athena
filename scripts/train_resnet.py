import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

from scripts.image_data import make_datasets


def make_model(start_age, weights=ResNet18_Weights.IMAGENET1K_V1):
    model = resnet18(weights=weights)
    model.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    nn.init.zeros_(model.fc.weight)
    nn.init.constant_(model.fc.bias, start_age)
    # Freezing parameters alone does not freeze the batch norm statistics.
    return model.eval()


def train_epoch(model, batches, optimizer, device):
    total = 0
    for images, ages in batches:
        images, ages = images.to(device), ages.to(device)
        optimizer.zero_grad()
        ages_out = model(images).squeeze(1)
        loss = nn.functional.l1_loss(ages_out, ages)
        loss.backward()
        optimizer.step()
        total += loss.item() * ages.numel()
    return total / len(batches.dataset)


@torch.inference_mode()
def score(model, batches, device):
    total = 0
    for images, ages in batches:
        ages_out = model(images.to(device)).squeeze(1)
        total += (ages_out - ages.to(device)).abs().sum().item()
    return total / len(batches.dataset)


def train(args):
    train_data, dev_data = make_datasets(args.labels, args.images, args.split)
    start_age = median(train_data.labels[i][0] for i in train_data.ids)
    baseline_mae = sum(abs(dev_data.labels[i][0] - start_age) for i in dev_data.ids)
    baseline_mae /= len(dev_data)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_batches = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_batches = DataLoader(dev_data, batch_size=args.batch_size)
    model = make_model(start_age).to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    args.out.mkdir(parents=True)
    run = {
        "model": "resnet18",
        "weights": "IMAGENET1K_V1",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "torch": str(torch.__version__),
        "torchvision": torchvision.__version__,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "baseline_mae": baseline_mae,
        "epochs": [],
    }
    best = float("inf")
    print(f"Device: {device}")
    print(f"Median baseline on this split: {baseline_mae:.2f} months", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_mae = train_epoch(model, train_batches, optimizer, device)
        dev_mae = score(model, dev_batches, device)
        run["epochs"].append({"epoch": epoch, "train_mae": train_mae, "dev_mae": dev_mae})
        if dev_mae < best:
            best = dev_mae
            run["best_epoch"] = epoch
            run["best_dev_mae"] = best
            torch.save(model.fc.state_dict(), args.out / "head.tmp")
            (args.out / "head.tmp").replace(args.out / "best-head.pt")

        (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        print(f"Epoch {epoch}: train MAE {train_mae:.2f}, dev MAE {dev_mae:.2f} months", flush=True)

    print(f"Best dev MAE: {best:.2f} months (epoch {run['best_epoch']})")
    print(f"Saved best head and run details in {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Train a ResNet-18 head to predict bone age")
    parser.add_argument("labels", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--split", type=Path, default=Path("artifacts/median-baseline.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/resnet18"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.lr < float("inf"):
        parser.error("epochs, batch size and learning rate must be positive and finite")
    if args.out.exists():
        parser.error("choose a new output folder for this run")

    try:
        train(args)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
