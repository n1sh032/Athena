import argparse
import csv
from pathlib import Path


def read_labels(path):
    labels = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        needed = {"id", "boneage", "male"}
        if not needed.issubset(rows.fieldnames or []):
            raise ValueError("CSV needs the columns id, boneage and male")

        for row in rows:
            try:
                image_id = int(row["id"])
                age = int(row["boneage"])
                male = row["male"].strip().lower()
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"Invalid label on CSV line {rows.line_num}") from error

            if image_id <= 0 or age < 0 or male not in ("true", "false"):
                raise ValueError(f"Invalid label on CSV line {rows.line_num}")
            if image_id in labels:
                raise ValueError(f"Duplicate image ID {image_id} on CSV line {rows.line_num}")
            labels[image_id] = (age, male == "true")

    if not labels:
        raise ValueError("CSV has no labels")
    return labels


def check(labels, images):
    expected = {f"{image_id}.png" for image_id in labels}
    found = {
        path.name for path in images.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    missing = sorted(expected - found)
    extra = sorted(found - expected)

    ages = [age for age, _ in labels.values()]
    male_count = sum(male for _, male in labels.values())
    print(f"Label rows: {len(labels)}")
    print(f"Age range: {min(ages)} to {max(ages)} months")
    print(f"Recorded male: {male_count}, recorded female: {len(labels) - male_count}")
    print(f"Matching PNGs: {len(expected & found)}")
    print(f"Missing PNGs: {len(missing)}")
    if missing:
        print("  " + ", ".join(missing[:5]))
    print(f"Extra PNGs: {len(extra)}")
    if extra:
        print("  " + ", ".join(extra[:5]))
    return 1 if missing else 0


def main():
    parser = argparse.ArgumentParser(description="Check the RSNA training dataset")
    parser.add_argument("labels", type=Path, help="training labels CSV")
    parser.add_argument("images", type=Path, help="folder containing the training PNGs")
    args = parser.parse_args()

    try:
        return check(read_labels(args.labels), args.images)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Data check failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
