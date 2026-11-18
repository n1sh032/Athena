# RSNA holdout results

ViT-B/16 reached **16.86 months MAE** on 1,425 holdout images. Its mean absolute error was **2.54 months lower** than ResNet-18, about a **13.1% reduction** in this run.

| Method | Development MAE (months) | Holdout MAE (months) |
|---|---:|---:|
| Training median | 33.33 | 33.40 |
| ResNet-18 | 19.16 | 19.40 |
| ViT-B/16 | 17.15 | 16.86 |

## Where errors were larger

All scores below are MAE in months. The age bands use the RSNA reference age.

| Group | Images | Median | ResNet-18 | ViT-B/16 |
|---|---:|---:|---:|---:|
| 0–59 months | 94 | 94.11 | 28.51 | 23.11 |
| 60–119 months | 394 | 42.92 | 21.17 | 19.19 |
| 120–179 months | 809 | 17.51 | 16.04 | 13.83 |
| 180+ months | 128 | 59.98 | 28.57 | 24.29 |
| Recorded female | 652 | 31.93 | 20.34 | 18.43 |
| Recorded male | 773 | 34.64 | 18.62 | 15.54 |

Both image models had their lowest error in the 120–179 month band, which contains 809 of the 1,425 images. ViT's highest error was 24.29 months in the 180+ band. The recorded-sex results describe this dataset's groups. Age mix and other differences could affect those averages.

## Evaluation setup

- The two saved epoch-5 heads were selected on the development split and evaluated without further training on the separate RSNA validation download.
- All 1,425 evaluation images matched their labels. The training and holdout archives had no shared image IDs or byte-identical PNGs. The available annotation IDs identify images and do not establish patient separation.
- The holdout report matched the training label and split hashes from both development runs. Both head hashes matched the downloaded checkpoint ZIPs. Inference ran on a Tesla T4.
- The 132-month baseline came from the 10,089 training labels. The same holdout images were scored for all three methods.

These results measure agreement with RSNA's reference labels in one experiment. Generalization to other hospitals and clinical use would require separate evaluation.
