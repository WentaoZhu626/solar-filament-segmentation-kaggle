# Error analysis of the CNN-judge pipeline (before the boundary refiner)

Out of fold, 707 observations. Pipeline analysed: ensemble candidates + keep rule 0.6 x mean(5 CNN judges) + 0.4 x GBM
filter >= 0.4. Counts are per annotator view. These tables were computed with a slightly different rasterisation of the
annotations than the official one; use them for relative comparisons (shares, ratios), not as absolute PQ.

Totals: TP 5095 / FP 2033 / FN 3104; segmentation quality 0.683, recognition quality 0.665.

## Missed annotated filaments (FN, 3104)
| category | n | share | median GT area (px) | drawn by one annotator only |
|---|---|---|---|---|
| 1 a removed candidate would have matched | 695 | 22 % | 646 | 50 % |
| 2 fragmented (two or more kept pieces cover >= 50 %) | 114 | 4 % | 5073 | 37 % |
| 3 oversized / merged (prediction covers >= 60 % of the filament but IoU < 0.5) | 445 | 14 % | 931 | 42 % |
| 4 near miss (IoU 0.3–0.5) | 482 | 16 % | 2330 | 30 % |
| 5 weak overlap with a kept prediction | 205 | 7 % | 3681 | 35 % |
| 6 only removed candidates overlap weakly | 758 | 24 % | 1325 | 42 % |
| 7–8 no candidate although probability >= 0.3 | 37 | 1 % | | |
| 9 model blind (probability < 0.3 everywhere) | 368 | 12 % | 548 | 50 % |

## Extra predictions (FP, 2033)
| category | n | share |
|---|---|---|
| a another annotator of the same image drew it (IoU > 0.5 in another view) | 801 | 39 % |
| b near miss in this view (IoU 0.3–0.5) | 483 | 24 % |
| c weak overlap in this view | 267 | 13 % |
| d only weak overlap in another view | 79 | 4 % |
| e spurious (no annotation in any view) | 403 | 20 % |

## Shape errors
- Oversized / merged: in 70 % the prediction also covers another filament of the same annotator — we draw one filament
  where the annotator drew several pieces (prediction 1.84x longer, 2.2x larger).
- Near miss: the prediction is short (length ratio 0.50, width ratio 1.0, 91 % of it inside the annotation); the
  annotator continues along faint extensions.
- In multi-annotator images the prediction behind a shape error matches another annotator in 69 % (merged), 62 % (near
  miss) and 57 % (fragmented) of cases: most shape errors are disagreements about splitting and extent.

## Who and where
- Recall by size: < 400 px 0.28, 400–1k 0.59, 1–3k 0.68, 3–8k 0.68, > 8k 0.57.
- Recall by chirality label: left 0.68, right 0.70, unidentifiable 0.51 (unidentifiable filaments = 48 % of all misses;
  annotators draw them about twice as wide: > 8k px median width 36 px vs 15.6 px for left/right).
- Distance from disk centre: r/R <= 0.8 recall 0.62–0.65; 0.8–0.9 0.50; > 0.9 0.09.
- Annotator agreement: filaments drawn by nobody else are recalled at 0.32, by some 0.63, by all 0.87.
- Annotator density: recall 0.71 when an annotator drew <= 3 filaments, 0.52 when > 15; per-annotator counts differ by
  >= 2x in 29 % of multi-annotator images.
- Observing site and year: flat (recall 0.61–0.64 per site).
- Precision of kept predictions by size: < 400 px 0.54, 400–1k 0.67, 1–3k 0.75, 3–8k 0.78, > 8k 0.92.

## Reading
1. About half of the remaining error is annotator inconsistency that no single prediction can satisfy.
2. The largest potentially fixable pool is faint / low-contrast / "unidentifiable" filaments, found only in part.
3. Boundary quality is the other lever (+0.05 IoU on all matches would be worth about +0.03 PQ) — this motivated the
   boundary refiner (`docs/BOUNDARY_REFINER.md`).
