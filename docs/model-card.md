# Athena model card

## Models and intended use

Athena compares two image regression models for an educational computer vision project. Each receives one hand X-ray and outputs an estimated skeletal age in months. The demo is designed for exploring RSNA example images.

ResNet-18 and ViT-B/16 use frozen Torchvision ImageNet V1 backbones. A single trainable linear layer converts the features to one age estimate. Freezing the backbones reduced training work and gave both models the same experiment setup. The shared loader preserves image proportions with padding and applies the same normalization to both models.

## Data and evaluation

The RSNA training download was split into 10,089 training and 2,522 development images with seed 42. We chose checkpoints on the development split and evaluated them on the separate 1,425-image RSNA validation download. Both epoch-5 checkpoints were scored on a Tesla T4. See the [data card](data-card.md) and [holdout report](holdout-results.md).

## Results and limits

ViT-B/16 reached 16.86 months mean absolute error versus 19.40 for ResNet-18 and 33.40 for the training median. ViT took about 11 minutes longer across five epochs. Its holdout error rose to 24.29 months for reference ages of 180 months and up. Group sizes vary, and aggregate group scores can reflect differences in age distribution.

The holdout had no shared image IDs or byte-identical PNGs with the training download. Patient-level separation could not be established from the available annotations. This single benchmark measures agreement with RSNA reference labels. Clinical decisions and use on images from other settings would require separate validation.
