# Boundary refiner (stage 2)

Scripts: `stage2/refine.py` (stages `crops` / `train` / `eval`) and `stage2/refine_test.py` (test submission: CNN judge
selection + refined masks).

## Method
- One crop per candidate: square of side 1.3 x max(bbox side) + 48 px centred on the candidate, resized to 256 px.
- Input (4 channels): pre-processed image, ensemble probability, the candidate's own mask, union of the other candidates.
- Target: mean of the annotators' masks that overlap the candidate with IoU > 0.2 (soft where annotators disagree);
  5189 of 9795 candidates have a target, the rest are not used for training.
- smp U-Net with ResNet34 encoder (ImageNet), BCE + soft Dice, D4 augmentation, 20 epochs, AdamW + OneCycle, batch 32,
  about 4.5 min per fold on one RTX 4080 SUPER. Same month-grouped 5 folds; D4 TTA at prediction time; test = mean of the
  5 fold models.
- Paste back: resize the crop probability to the crop size, threshold 0.5, keep the original mask if the result is empty;
  overlaps are resolved by original area (the larger instance keeps the pixels), so instances never overlap.

## Out-of-fold results (own rasterisation of the annotations, about 0.006 above the official-style number)
| selection | original masks | refined masks | per fold (refined) |
|---|---|---|---|
| GBM filter | 0.4468 | 0.4568 | 0.4618 / 0.4662 / 0.4335 / 0.4594 / 0.4644 (5/5 better) |
| **CNN judge (final)** | 0.4543 | **0.4655** | 0.4668 / 0.4704 / 0.4459 / 0.4716 / 0.4738 (5/5 better) |

The gain is mostly recognition quality: about 85 near-miss predictions cross IoU 0.5 (TP 5096 -> 5185, FP 2032 -> 1943);
segmentation quality rises only 0.683 -> 0.688. Refined masks are about 10 % larger: annotators draw filaments wider than
the U-Net. Thresholds 0.3 / 0.4 / 0.5 / 0.6 give 0.4529 / 0.4554 / 0.4568 / 0.4556 on the GBM selection.

Test set: 1074 filaments, 0 overlapping pixels, median IoU 0.91 with the unrefined masks. Public leaderboard 0.39 (the
public board uses about 90 images at two decimals, so +0.01 is below its resolution).

Also tried (out of fold, GBM selection, official bookkeeping, baseline 0.4406): geometric mean of the 3 members 0.4361;
D4 TTA for all 15 U-Nets 0.4408.

Weights: `weights/refine/fold{0-4}.pth` (fp16 state dicts of `smp.Unet('resnet34', in_channels=4, classes=1)`).
