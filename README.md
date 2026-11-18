# Athena

Athena is a computer vision study that estimates bone age in months from pediatric hand X-rays. The experiment asks whether a pretrained Vision Transformer improves on ResNet-18 when both see the same data. This project is for education and research.

## Project progress

- [x] Check the training labels and their matching image files.
- [x] Add the median age baseline and a saved development split.
- [x] Build the shared image loader and preprocessing.
- [x] Add the ResNet-18 training script and verify it with generated images.
- [x] Run the ResNet-18 development benchmark on the RSNA images.
- [x] Add the Vision Transformer training script.
- [x] Run the Vision Transformer development benchmark on the same RSNA split.
- [ ] Evaluate both models on the final holdout and document where they struggle.

On the development split, ViT-B/16 reached **17.15 months MAE**, ResNet-18 reached **19.16 months**, and the training-median baseline scored **33.33 months**. See the [development results](docs/development-results.md) for the settings and epoch timings.

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

## Train the ResNet-18 head

Run from the repository root after checking the images and creating the baseline split. Start with a cloud notebook with a GPU for the full dataset. The first run fetches the [pretrained ResNet-18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

```sh
python -m scripts.train_resnet data/raw/rsna-training-labels/train.csv data/raw/boneage-training-dataset
```

The head starts by predicting the training median. Training adjusts only the head, using absolute error in months. The script prints the development MAE after each epoch and saves the best head and `run.json` in `artifacts/resnet18/`. Use `--out artifacts/resnet18-run2` for a separate run. Keep the Torchvision ImageNet weights available alongside the head when loading it later.

## Train the Vision Transformer head

```sh
python -m scripts.train_vit data/raw/rsna-training-labels/train.csv data/raw/boneage-training-dataset
```

This uses [ViT-B/16 with ImageNet V1 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) and saves the output to `artifacts/vit_b_16/`. Both scripts use the same preprocessing, split, batch order and scoring code when run with the same settings. If the transformer runs out of GPU memory, lower `--batch-size` for both official comparison runs.

## Run the benchmark on a cloud GPU

[Open the benchmark in Colab](https://colab.research.google.com/github/n1sh032/Athena/blob/main/notebooks/rsna_benchmark.ipynb) and select a GPU runtime. The notebook downloads the official RSNA training data, checks the images, recreates the seed 42 split, and runs both models. Allow around 20 GB for the archive and extracted images, plus room for model weights and outputs.

The notebook starts with batch size 8 for both models. Save the two output ZIPs before ending the session. Each includes a regression head and a report with development MAE and timing. A free Colab GPU is subject to [availability and session limits](https://research.google.com/colaboratory/faq.html).

## Evaluate the final holdout

Download the validation archive from the [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017). Keep the original training image ZIP alongside it. From the repository root, run:

```sh
python -m scripts.evaluate_holdout data/raw/train.csv data/raw/images.zip data/raw/holdout.zip artifacts/rsna-dev-b8
```

Use the original training CSV and saved split from the two runs. The evaluator checks their hashes, verifies that the archives share neither image IDs nor identical PNG files, and reports MAE overall and by age band and recorded sex. It saves aggregate results to `artifacts/holdout.json`.

See the [benchmark plan](docs/benchmark-plan.md) for the comparison and the [data card](docs/data-card.md) for RSNA attribution and usage terms.
