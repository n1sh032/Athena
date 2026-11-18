# Benchmark plan

## What we are comparing

1. Predict the median bone age from the training split for every image.
2. Use pretrained ResNet-18 features with a small regression head.
3. Use pretrained ViT-B/16 features with a small regression head.

Start with frozen backbones and train only the new heads. Treat any later fine-tuning as a separate experiment. The two image models will use images alone. Use the same split, input resolution, and whole-hand framing for both, with any normalization differences from the pretrained weights recorded.

Both backbones use their ImageNet V1 weights in evaluation mode. Each linear head starts at the training median. The defaults are five epochs, batch size 32, Adam with learning rate 0.01, and seed 42. A separate seeded generator gives both models the same image order for matching run settings. Select the head with the lowest development MAE. The run report records the actual settings, time per epoch including development scoring, and hashes of the labels and split files.

## How we will measure it

Set aside development examples from the RSNA training download. Use those to choose settings and checkpoints. Plan to reserve RSNA's published validation download for Athena's final holdout, after checking its labels and overlap with the training download. If it cannot serve as the holdout, define and record a fixed internal test split before tuning. Use patient groups if identifiers are supplied. Report the split source, limitations, and seed.

The baseline sorts the training download's image IDs and shuffles them with seed 42. It sets aside 20% rounded down for development and uses the rest for training. The annotations identify images, so this is an image-level split. Keep the saved IDs in `artifacts/median-baseline.json` for reuse when training both image models.

The shared loader resizes each grayscale X-ray to fit inside 224 × 224 pixels and fills the remaining space with black padding. This preserves the image's proportions and keeps the edges in view. It copies grayscale into three channels and applies the ImageNet channel normalization used by the [ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and [ViT-B/16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html) ImageNet weights. This project uses padding in place of their default center crop.

Mean absolute error (MAE) in months is the main score. For instance, a prediction of 130 months against a reference of 124 has an absolute error of 6 months. Report each method's MAE on the same holdout, counts and MAE by age band and recorded sex, plus training time and hardware for the image models. Keep preprocessing settings, software versions, and checkpoints with the experiment records.

The result measures agreement with the reference labels on this dataset. Generalization to other hospitals and clinical use would require separate evaluation.
