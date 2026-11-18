# Athena

Athena is a computer vision study that estimates bone age in months from pediatric hand X-rays. The experiment asks whether a pretrained Vision Transformer improves on ResNet-18 when both see the same data. This project is for education and research.

## Project progress

- [x] Check the training labels and their matching image files.
- [x] Add the median age baseline and a saved development split.
- [x] Build the shared image loader and preprocessing.
- [ ] Train ResNet-18 and a Vision Transformer.
- [ ] Compare results and document where the models struggle.

## Inspect the data

Download the training images and training annotations from the [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017). The commands below use `data/raw/rsna-training-labels/train.csv` for the extracted annotations and `data/raw/boneage-training-dataset` for the folder containing the PNGs. The labels CSV has the columns `id`, `boneage`, and `male`. Each row points to an image named `<id>.png`.

Run from the repo root with Python 3. No additional packages are needed for this step.

```sh
python scripts/check_data.py data/raw/rsna-training-labels/train.csv data/raw/boneage-training-dataset
```

The script prints label counts, the age range, counts by the recorded sex field, and missing or extra PNG filenames. Invalid labels and missing images cause a nonzero exit. This is a filename check. Image decoding and tensor checks come with the image loader.

## Run the median baseline

Use only the training annotations for this step. The script predicts the same age for every development example, using the median age from the training portion alone.

```sh
python scripts/baseline.py data/raw/rsna-training-labels/train.csv
```

The command prints the development mean absolute error in months and saves the exact split and score to `artifacts/median-baseline.json`. Use that split for both image models.

With seed 42, the 10,089 training labels give a median of 132 months. Predicting 132 months for each of the 2,522 development examples gives a **development MAE of 33.33 months**. The final holdout is reserved for the model comparison.

## Check the image batches

After extracting the training images and running the baseline, install the image dependencies and run:

```sh
python -m pip install -r requirements.txt
python -m scripts.image_data data/raw/rsna-training-labels/train.csv data/raw/boneage-training-dataset
```

The loader checks that the saved split covers the CSV exactly and that every expected PNG exists. It decodes the first batch in each split and prints the image shape and the corresponding ages. Each batch has shape `(N, 3, 224, 224)`, where `N` is the number of images in the batch.

Run `python -m unittest discover -s tests -v` to check the loader with generated PNGs.

See the [benchmark plan](docs/benchmark-plan.md) for the comparison and the [data card](docs/data-card.md) for RSNA attribution and usage terms.
