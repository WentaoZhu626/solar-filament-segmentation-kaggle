# Error analysis — ens3 (ConvNeXt-T + EffNet-B4 + RegNetY-032, 15 models), OOF 0.4410, LB 0.39
Run: AutoDL `run_err.py --tag err3` (run_ens.py + analysis block). Saved: runs/err3/oof_maps.npy (707x2x2048x2048 uint8 ensemble OOF maps), runs/err3/err/{gt,fp,tp,img}.csv (per-GT / per-FP / per-TP / per-image tables), state.pkl (candidate tables + OOF filter scores).
Config: B(400) + faint blobs p>0.3 + GBM FP filter, keep score >= 0.35.

## Where PQ is lost
TP 5113 / FP 2403 / FN 3086 -> PQ 0.4410 = SQ 0.678 x RQ 0.651.
| what-if | PQ |
|---|---|
| no FP | 0.521 |
| no FN (IoU = SQ) | 0.591 |
| SQ = 1 | 0.651 |
| all 799 near-miss FN fixed (IoU 0.55, paired FP gone) | 0.497 |
| all 973 spurious FP removed | 0.470 |

FN types: untouched 1919 (62%), near miss (IoU 0.3–0.5) 799, fragmented 205 (164 of them >= 3000 px), weak overlap 163.
FP types: spurious 973, near miss 943, weak overlap 487. 871 / 2403 FP match ANOTHER annotator's filament (pure annotator disagreement on multi-annotator images).

## Recall by size x contrast quintile
| size | q1 faint | q2 | q3 | q4 | q5 dark |
|---|---|---|---|---|---|
| <400 | 0.04 | 0.13 | 0.16 | 0.39 | 0.53 |
| 400-1k | 0.21 | 0.49 | 0.67 | 0.76 | 0.77 |
| 1k-3k | 0.37 | 0.65 | 0.79 | 0.86 | 0.88 |
| >=3k | 0.30 | 0.70 | 0.81 | 0.88 | 0.97 |
Recall by annotator agreement (multi-annotator images): drawn by nobody else 0.348 (n 1855), by some 0.658 (868), by all others 0.874 (2270).

## TP quality
TP IoU quantiles 10/25/50/75/90: 0.56/0.61/0.68/0.74/0.79. Pred/GT area ratio median 1.02 (10–90%: 0.73–1.41); small (<400 px) TPs over-sized (ratio 1.36), large (>=3k) slightly under-sized (0.93).

## Other
- Per site PQ 0.433 (Lh) – 0.454 (Uh): no bad station.
- By #annotators: 1 view 0.443, 2 views 0.446, 3 views 0.436.
- Per-image PQ median 0.447 (10% 0.26, 90% 0.61); worst images are near-empty ones (1–5 GT filaments).
- Human ceiling (annotator vs other annotators, 296 images): 0.340; ours on the same images 0.440.

## Reading
1. Most "errors" are label ambiguity: filaments drawn by only one annotator are recalled at 0.35, 871 FP are another annotator's filament, and the model already beats annotator-vs-annotator agreement by 0.10.
2. Largest actionable lever = near misses (799 FN + 943 FP at IoU 0.3–0.5): fixing them is worth up to +0.056. These are extent/splitting disagreements (large filaments fragmented, small ones over-sized).
3. Untouched FN are dominated by faint and/or small filaments (faintest quintile recall 0.04–0.37) — mostly single-annotator.
