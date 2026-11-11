# Experiments that were tried and not adopted

Everything here was evaluated out of fold on the 707 training observations (1154 annotator views, month-grouped 5 folds)
with the official PQ bookkeeping unless stated otherwise. "Reference" is the pipeline that was current at the time.
The final pipeline and its numbers are in the README.

## 1. Segmentation models and training

| experiment | result | reference | why not adopted |
|---|---|---|---|
| Single encoders (same pipeline): ResNet34 / ResNet50 / ConvNeXt-Tiny / EfficientNet-B4 / RegNetY-032 | 0.4238 / 0.4265 / 0.4319 / 0.4287 / 0.4332 | – | the three best (ConvNeXt-T, EffNet-B4, RegNetY) form the final ensemble (0.4410) |
| 5-encoder ensemble (+ ResNet34, ResNet50) | 0.4375 | 3-encoder 0.4410 | the weaker ResNets dilute the equal-weight mean (worse in 4/5 folds) |
| ConvNeXt-Small, 60 epochs | 0.4258 single; as 4th member 0.4410 | ConvNeXt-T 0.4319 | larger backbone not better; adds nothing to the ensemble |
| Learned global ensemble weights (grid search) | 0.4353–0.4366 full pipeline | equal weights 0.4412 | small gain at the instance stage reverses after the false-positive filter |
| Stacking CNN on the member maps (dilated conv judge) | 0.4363 | 0.4412 | re-calibrates maps toward soft targets; thresholds tuned for averaged maps no longer fit |
| Two-resolution model (global 1024-px context channels) | 0.4241 single; ensemble 0.4381 | RegNetY 0.4332; ensemble 0.4412 | train/test mismatch of the context maps (out-of-fold at training, 5-fold mean at test) |
| Scale augmentation 0.8–1.25 + 60 epochs | 0.4303 single; ensemble 0.4380–0.4402 | 0.4332; 0.4412 | no gain |
| Loss weight x3 inside "unidentifiable-chirality" filaments | 0.4323 single; ensemble + CNN judge 0.4447 | 0.4332; 0.4464 | recall of the faint class rises slightly (0.49 -> 0.52) but paid with more false positives |

## 2. Extra training data

| experiment | result | reference | why not adopted |
|---|---|---|---|
| Neighbouring GONG frames (1–2 min apart) with copied labels, p = 0.5 | 0.4192 | 0.4233 (worse in 5/5 folds) | archive JPEGs differ from competition images (re-encoding, 4.2 grey levels); labels drawn on one frame do not fit faint parts of a neighbour (correlation 0.78); the labelled image is seen half as often |
| Pseudo labels on neighbour frames (teacher = other folds' ensemble) | 0.4307 | 0.4317 | no gain |
| Pseudo labels on 1180 new dates (one frame per day, never labelled) | 0.4268 single; ensemble 0.4371 | 0.4332; 0.4412 | confirmation bias: the student absorbs the teacher's uncertain faint guesses as positives (more and wider filaments) |
| Arbitrary-angle rotation + seeing blur augmentation | 0.4294 single; ensemble 0.4391 | 0.4317; 0.4412 | no gain |

## 3. Test-time augmentation and ensembling of maps

| experiment | result | reference | why not adopted |
|---|---|---|---|
| Multi-scale prediction (0.75 + 1.0) | 0.4206 | 0.4233 | down-scaling mainly removes false positives, which the false-positive filter already does |
| D4 TTA (8 flips/rotations instead of horizontal flip) for all 15 U-Nets | 0.4408 | 0.4406 | no gain |
| Geometric instead of arithmetic mean of the 3 encoders | 0.4361 | 0.4406 | suppresses faint filaments that only one member sees |

## 4. Instance post-processing

| experiment | result | reference | why not adopted |
|---|---|---|---|
| Hysteresis thresholding | 0.3826 | 0.3867 | – |
| Learned per-filament threshold | 0.3828 (oracle 0.4075) | 0.3867 | the predictor cannot find the right threshold per filament |
| Growth in all directions / along the spine corridor / spine-guided lengthening | ±0.001 | – | missing ends have probability < 0.3: very faint, mostly drawn by one annotator only |
| Thinner masks for small filaments, longer spine links | −0.001 to −0.008 | 0.4412 | predictions are short, not thin (width ratio 1.0, length ratio 0.5–0.8) |
| Learned merge model (atom pairs, 47 features, AUC 0.853 vs 0.58 for the spine rule) + path bridging | 0.4447 | 0.4406 | public leaderboard 0.38 vs 0.39: merging tuned to the training annotators does not transfer |
| Local hysteresis growth by size (and by chirality) | +0.0055 (+0.0004 more with true chirality), measured with a non-official raster | – | public leaderboard 0.38 vs 0.39; the gain was measured with a rasterisation that makes annotations ~11 % larger, so it was likely inflated |
| Per-filament expected-IoU-optimal mask (top-k pixels) after the refiner | −0.0005 vs fixed threshold 0.5 | – | refiner probabilities are already sharp |

## 5. Extra information for the false-positive filter / merge decisions

| experiment | result | why not adopted |
|---|---|---|
| Temporal context: GONG frames ±1/3/6 h, de-rotated and aligned, as features for merge decisions | merge-pair AUC 0.8534 vs 0.8530 without | the rules allow only the single test image at inference; alignment quality drops with time (0.86 / 0.77 / 0.66), residual misalignment 20–40 px vs filaments a few px wide; the single frame already holds the information |
| Time strips of false positives (−48 h … +48 h) | 3 of 6 inspected false positives are persistent structures visible days earlier | supports that many false positives are filaments an annotator did not draw, not model errors |
| Chirality classifier (ResNet34 on crops; left vs right 0.86, 0.97 on the confident half) as merge feature | pair AUC 0.8518 vs 0.8530 | the rule holds (opposite chirality merged 0.20 vs same 0.37) but only 12 % of fragments are confidently chiral; chirality decides how annotators draw (barbs included for left/right, minimal outline for "unidentifiable"), not whether something is a filament |
| Plage (bright region) context | no effect at fixed filter score | hypothesis "candidates in plage are not drawn" is false |
| Crowding (other candidates within 150 px) | 0.4406 = 0.4406 | strong signal but already captured by the filter's distance / count features |
| Frangi dark-ridge filter as extra detector | 0.1 % of new ridge candidates match an annotation | the model probability already separates annotated filaments from background (AUC ~1.0); the remaining blind filaments look like thousands of unannotated dark ridges |
| Ensemble-disagreement features (std / range across the 3 members inside a candidate) | 0.4477 vs 0.4481 (CNN-judge blend) | correlates with the target (−0.59) but redundant with the mean probability and the CNN judge |
| Stronger CNN judges (ConvNeXt-T, ResNet50, EfficientNet-B3, 192-px crops) | +0.001–0.002 each over one ResNet34 judge | kept as the 5-judge mean in the final pipeline; larger backbones fit the training folds more without better AUC |
| Stacking a GBM on (CNN score, filter score) | 0.4415 | simple averaging 0.4464–0.4481 is better; stacking overfits the noisy target |

## 6. Selection among alternative masks

See `docs/CONFLICT_GRAPH.md`: a conflict-graph selection over 56,596 alternative hypotheses (thresholds 0.3–0.7,
spine-merged instances, faint blobs, refined masks), exact maximum-weight independent set with τ = PQ/2. Oracle
upper bound 0.5606 vs 0.5175 for the plain keep rule, but learned scorers reach only 0.438–0.448; threshold-component
(nested) selection followed by the same refiner gives 0.4477 vs 0.4542 for the final pipeline. Choosing per filament
between original / refined / threshold versions with a CNN: 0.4500–0.4517 vs always refined 0.4540 (oracle 0.4811).

## 7. What these failures have in common

1. **Annotator disagreement dominates the remaining error.** Annotator-vs-annotator PQ is 0.340; 39 % of our false
   positives are another annotator's filament; in multi-annotator images most shape errors match what another annotator
   drew (57–69 %). Changes that fit the training annotators' style (merging, growth, wider masks) do not transfer to
   the test annotators (public leaderboard 0.39 -> 0.38 twice).
2. **The model already "sees" almost everything that is annotated consistently.** The probability map separates
   annotated filaments from background with AUC ~1.0; missed parts are very faint, thin and mostly drawn by one
   annotator only. Extra detectors (ridge filters), extra data (pseudo labels, neighbour frames) and loss weighting add
   positives that annotators did not draw.
3. **Information that is redundant with the probability map does not help** (disagreement, crowding, temporal context).
   What helped was looking at the image again with a different model (CNN judge, +0.0075) and re-drawing each kept
   filament's outline (boundary refiner, +0.006).
