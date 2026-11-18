import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean, median

from check_data import read_labels


def split_ids(labels, seed):
    ids = sorted(labels)
    if len(ids) < 2:
        raise ValueError("Need at least two labeled images")

    random.Random(seed).shuffle(ids)
    dev_count = max(1, len(ids) // 5)
    return sorted(ids[dev_count:]), sorted(ids[:dev_count])


def baseline(labels, seed):
    train, dev = split_ids(labels, seed)
    prediction = median(labels[image_id][0] for image_id in train)
    errors = [abs(labels[image_id][0] - prediction) for image_id in dev]
    return {
        "seed": seed,
        "train_ids": train,
        "dev_ids": dev,
        "median_months": prediction,
        "dev_mae_months": mean(errors),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure the median bone age baseline")
    parser.add_argument("labels", type=Path, help="RSNA training labels CSV")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("artifacts/median-baseline.json"))
    args = parser.parse_args()

    try:
        if args.out.resolve() == args.labels.resolve():
            raise ValueError("Choose a different path for the output")

        result = baseline(read_labels(args.labels), args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Baseline failed: {error}\n")

    print(f"Training examples: {len(result['train_ids'])}")
    print(f"Development examples: {len(result['dev_ids'])}")
    print(f"Training median: {result['median_months']:g} months")
    print(f"Development MAE: {result['dev_mae_months']:.2f} months")
    print(f"Saved split and score to {args.out}")


if __name__ == "__main__":
    main()
