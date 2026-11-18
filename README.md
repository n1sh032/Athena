# Athena

Athena is a computer vision study that estimates bone age in months from pediatric hand X-rays. The experiment asks whether a pretrained Vision Transformer improves on ResNet-18 when both see the same data. This project is for education and research.

## Project progress

- [x] Check the training labels and their matching image files.
- [ ] Measure the median age baseline.
- [ ] Build the shared image loader and preprocessing.
- [ ] Train ResNet-18 and a Vision Transformer.
- [ ] Compare results and document where the models struggle.

## Inspect the data

Download the training images and training annotations from the [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017). Extract the images under `data/raw/` or use any folder that directly contains the PNG files. The labels CSV should have the columns `id`, `boneage`, and `male`. Each row points to an image named `<id>.png`.

Run from the repo root with Python 3. No additional packages are needed for this step.

```sh
python scripts/check_data.py data/raw/boneage-training-dataset.csv data/raw/boneage-training-dataset
```

The script prints label counts, the age range, counts by the recorded sex field, and missing or extra PNG filenames. Invalid labels and missing images cause a nonzero exit. This is a filename check. Image decoding and tensor checks come with the image loader.

See the [benchmark plan](docs/benchmark-plan.md) for the comparison and the [data card](docs/data-card.md) for RSNA attribution and usage terms.
