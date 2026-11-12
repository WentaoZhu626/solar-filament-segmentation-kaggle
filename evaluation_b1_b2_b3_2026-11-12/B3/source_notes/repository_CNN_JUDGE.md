# CNN judge for the false-positive filter

Scripts: `stage2/cnn_filter.py` (stages `crops` / `train` / `eval`; env `S`, `TAG`, `ARCH`, `LR`, `BS`) and `stage2/cnn_test.py`
(test submission; env `MEMBERS="tag:arch:size,..."`, `W_CNN`, `THR`, `SUBNAME`). `stage2/vis_prep.py` rebuilds the out-of-fold
candidates (ens3 B >= 400 px + faint p > 0.3 blobs) and their 13 filter features.

## Setup
- 9795 out-of-fold candidates (4366 kept by the 13-feature GBM filter).
- Input: 7 channels at 128 px. Local crop (side = 1.5 x bbox + 32, min 64): image, ensemble probability, candidate mask.
  Context crop 512 px: image, probability, candidate mask, other candidates.
- Target y = share of annotator views that match the candidate (IoU > 0.5); soft BCE.
- timm backbone (ImageNet pre-trained, in_chans 7), 15 epochs, AdamW + OneCycle, flips / rot90 augmentation, 4-rotation TTA.
  Same month-grouped 5 folds as the U-Nets.
- Keep rule: 0.6 x mean(5 CNN judges) + 0.4 x GBM score >= 0.4.

## Out-of-fold results (official PQ bookkeeping)
| keep rule | PQ | TP / FP / FN |
|---|---|---|
| GBM filter (13 features), previous final | 0.4406 | 4968 / 2138 / 3231 |
| CNN ResNet34 alone | 0.4431 | 5046 / 2188 / 3153 |
| 0.5 CNN ResNet34 + 0.5 GBM | 0.4464 | 5039 / 2071 / 3160 |
| **0.6 mean of 5 CNNs + 0.4 GBM (final)** | **0.4481** | |

Judges (blend 0.5 with the GBM score): ResNet34 128 px 0.4464, ConvNeXt-Tiny 128 px 0.4469, EfficientNet-B3 128 px 0.4448,
ResNet50 128 px 0.4475, ResNet34 192 px 0.4475. Per fold (ResNet34 blend): 0.4482 / 0.4527 / 0.4270 / 0.4494 / 0.4557.
Simple averaging beats stacking a GBM on (score, CNN), which overfits the noisy target. Stronger backbones fit the training
folds better without better AUC; the remaining error is mostly annotator disagreement. Public leaderboard: 0.39.

Weights: `weights/cnn_judge/{r34_128,cnxt_128,effb3_128,r50_128,r34_192}/fold{0-4}.pth` (fp16 state dicts of timm
`resnet34`, `convnext_tiny`, `efficientnet_b3`, `resnet50`, `resnet34` with `in_chans=7, num_classes=1`).

## Ensemble-disagreement features (not adopted)
Seven features from the 3 member maps inside each candidate (per-pixel std across members, range, spread of member means,
votes): corr(disagreement, y) = -0.59, but GBM 0.4406 -> 0.4406 and blend 0.4481 -> 0.4477. Redundant with the mean
probability and the CNN judge. Script: `stage2/experiments/uncertainty_features.py`.
