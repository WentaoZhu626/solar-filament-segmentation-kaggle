# Boundary refiner (stage 2)

Scripts: `stage2/refine.py` (stages `crops` / `train` / `eval`) and `stage2/refine_test.py` (test submission: CNN-judge
selection + refined masks). All numbers are out-of-fold on 707 training observations with the **official** PQ bookkeeping
(pycocotools rasterisation of the annotations, IoU > 0.5, pooled over annotator views).

## Method
- One crop per candidate: square of side 1.3 x max(bbox side) + 48 px centred on the candidate, resized to 256 px.
- Input (4 channels): pre-processed image, ensemble probability, the candidate's own mask, union of the other candidates.
- Target: mean of the annotators' masks (pycocotools raster, as the official scorer) that overlap the candidate with
  IoU > 0.2; 5189 of 9795 candidates have a target.
- smp U-Net, ResNet34 encoder (ImageNet), BCE + soft Dice, D4 augmentation, 20 epochs, AdamW + OneCycle, batch 32
  (~2.3 min per fold on an RTX 4080 SUPER). Month-grouped 5 folds; D4 TTA; test = mean of the 5 fold models.
- Paste back: threshold 0.5, keep the original mask if empty; overlaps resolved by area (instances never overlap).

## Results
| selection | original masks | refined t 0.4 | **refined t 0.5** | refined t 0.6 |
|---|---|---|---|---|
| GBM filter | 0.4407 | 0.4443 | **0.4453** | 0.4446 |
| CNN judge (final) | 0.4482 | 0.4531 | **0.4542** | 0.4530 |

- CNN-judge selection per fold: 0.4501 / 0.4525 / 0.4296 / 0.4532 / 0.4562 -> 0.4541 / 0.4567 / 0.4358 / 0.4612 / 0.4638
  (5/5 better). TP / FP / FN 5063 / 2065 / 3136 -> 5105 / 2023 / 3094; SQ 0.678 -> 0.682.
- Paired bootstrap over images (2000 resamples): +0.0060, 95 % CI [+0.0040, +0.0081]; month-block bootstrap
  [+0.0042, +0.0076]. Per image: better 447, worse 236, equal 24.
- Public-leaderboard-like random 90-image subsets: PQ standard deviation 0.012; the paired gain (+0.006 +- 0.003) is often
  invisible at two decimals. Public leaderboard: 0.39.
- Test set: 1074 filaments, 0 overlapping pixels, median IoU 0.94 with the unrefined masks, median area ratio 1.00.

## Lesson: rasterise the annotations exactly like the scorer
A first version (Nov 9) rasterised the polygons with `cv2.fillPoly`. Those masks are about 11 % larger than the
pycocotools masks used by the official scorer (median area ratio 1.107, IoU 0.88 between the two rasters on 1045
polygons). The refiner learned the wider masks and looked +0.010 better when the evaluation used the same cv2 raster, but
with the official raster it was slightly worse (0.4481 -> 0.4464). Targets and evaluation now both use pycocotools.

Weights: `weights/refine/fold{0-4}.pth` (fp16 state dicts of `smp.Unet('resnet34', in_channels=4, classes=1)`).
