# RSNA development results

ViT-B/16 reached **17.15 months MAE**, about **2.01 months lower** than ResNet-18. Its five epochs took about **11 minutes longer** on the same Tesla T4.

| Method | Best development MAE (months) | Best epoch | Five epochs, including development scoring |
|---|---:|---:|---:|
| Training median | 33.33 | — | — |
| ResNet-18 | 19.16 | 5 | 39.05 min |
| ViT-B/16 | 17.15 | 5 | 49.74 min |

| Epoch | ResNet-18 MAE (months) | ViT-B/16 MAE (months) |
|---:|---:|---:|
| 1 | 21.36 | 19.42 |
| 2 | 20.83 | 18.19 |
| 3 | 19.76 | 17.96 |
| 4 | 20.85 | 17.95 |
| 5 | 19.16 | 17.15 |

Both models used 10,089 training images and 2,522 development images. The run reports matched on the label and split file hashes. Each run used frozen ImageNet V1 weights, a linear head, Adam at 0.01, batch size 8, and seed 42. Both ran on a Colab Tesla T4 with PyTorch `2.14.1+cu130` and Torchvision `0.29.1+cu130`.

We used the development split to select each checkpoint. See the [final holdout report](holdout-results.md) for the separate RSNA validation archive results. Timing includes image loading and development scoring, so it describes this notebook run rather than model inference speed alone.
