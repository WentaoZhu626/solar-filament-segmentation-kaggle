# Solar Filament Segmentation 2026 — mask + spine U-Net ensemble, CNN judge and boundary refiner

Instance segmentation of solar filaments on GONG H-alpha full-disk images
([Kaggle: Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)).

**Current pipeline:** 3 encoders x 5 folds (15 U-Nets) -> spine-guided instances + faint candidates -> keep rule
0.6 x mean of 5 CNN judges + 0.4 x GBM filter >= 0.4 -> stage-2 boundary refiner (U-Net on a crop around each kept
filament). Out-of-fold PQ **0.4541** (organizers' evaluation code, 707 observations), public leaderboard **0.40**.

## For the evaluation committee

| what | where |
|---|---|
| Report (4 pages, PDF) | [`Solar_Filament_Segmentation_2026_report.pdf`](Solar_Filament_Segmentation_2026_report.pdf) |
| Demo notebook (whole pipeline, already executed, outputs included) | [`inference/demo.ipynb`](inference/demo.ipynb) |
| Inference script used by the notebook | [`inference/predict_final.py`](inference/predict_final.py) |
| Trained weights (loaded automatically, no download needed) | [`weights/`](weights/): 15 U-Nets, `cnn_judge/` (25 CNN judges), `refine/` (5 boundary refiners), `stage2_filter_table.npz` |
| Our submitted predictions (public LB 0.40) | [`results/submission_final.csv`](results/submission_final.csv) |
| Out-of-fold predictions on all 707 training observations (PQ 0.4541) | [`results/oof_final_predictions.csv`](results/oof_final_predictions.csv) |
| Evaluation with the organizers' notebook (Dice, IoU, one-to-many / many-to-one, PQ), examples, sensitivity | [`evaluation_b1_b2_b3_2026-11-12/`](evaluation_b1_b2_b3_2026-11-12/) |
| Dependencies | [`requirements.txt`](requirements.txt) |

Run: `pip install -r requirements.txt`, then open `inference/demo.ipynb` and run all cells (the competition data is
found automatically under `/kaggle/input` or `./data`; otherwise set `DATA` in the first cell). Training from scratch is
optional (`RUN_TRAINING = True` in the notebook); the stored weights reproduce our submission without retraining
(98.5 % of the regenerated filaments match a submitted one with IoU > 0.9; 7.3 s per image on one RTX 4080 SUPER).


| stage | OOF PQ (official bookkeeping) |
|---|---|
| 15 U-Nets + B + faint candidates + GBM filter (steps 1–7 below) | 0.4406 |
| + CNN judge (`docs/CNN_JUDGE.md`) | 0.4481 |
| + boundary refiner (`docs/BOUNDARY_REFINER.md`) | **0.4541** (+0.0060, 95 % CI [+0.0040, +0.0081], 5/5 folds), LB 0.40 |

Out-of-fold predictions of the final pipeline: `results/oof_final_predictions.csv` (with `results/folds.csv`; loader,
scoring and overlay figures in `tools/oof_viewer.py`). Error analysis: `docs/ERROR_ANALYSIS_V2.md`. All experiments that
were tried and not adopted, with numbers and reasons: `docs/FAILED_EXPERIMENTS.md`.

Analyses: `docs/DECISION_THEORY.md` (keep threshold = PQ/2, derived and verified), `docs/CONFLICT_GRAPH.md` (conflict-graph /
component-tree selection, oracle analysis — not adopted). Tried without gain: D4 TTA (+0.0002), geometric-mean ensemble
(-0.0045), ensemble-disagreement features (-0.0004), component-tree selection (0.4477 vs 0.4541).

## Method in one page

1. **Pre-processing** (`preprocess` in `src/run.py`): solar disk = largest Otsu blob; limb darkening removed by dividing
   by a smooth background estimated inside the disk (computed at 1/8 resolution, Gaussian sigma 4); the ratio is mapped to
   8 bit as `(I/bg - 1) * 510 + 128` (background ~128, filaments dark). Full 2048x2048 resolution is kept — filaments are
   only a few pixels wide.
2. **Targets:** soft consensus of all annotators (mean of the rasterised polygons, pycocotools = official scorer) for the
   filament **mask**, and the same for the provided **spine** polylines (5 px wide). With 2 of 3 annotators drawing a
   filament the target is 0.67, so the model follows the majority.
3. **Model v2** (`MODEL_CODE` in `src/run.py`): `segmentation_models_pytorch` U-Net with 2 outputs (mask, spine).
   The network sees 3 views computed inside the model: raw image, Gaussian-denoised (sigma 1.5), local-contrast-normalised
   (sigma 12 window) — faint filaments are amplified. Loss: mask BCE + soft Dice + 0.2 clDice (centre-line Dice),
   spine BCE + soft Dice (weight 0.5). Training: random 768x768 crops at full resolution (70 % centred on a filament),
   rot90/flip/brightness augmentation, AdamW lr 3e-4 with one-cycle schedule, 40 epochs x 720 crops, AMP.
   Encoders: `tu-convnext_tiny`, `tu-efficientnet_b4`, `tu-regnety_032` (ImageNet pre-trained, via timm).
4. **Validation:** 5-fold GroupKFold grouped by month (filaments persist for days to weeks, so same-month images are
   near duplicates). Every number in this README is out-of-fold on all 707 observations, scored with the official PQ
   definition (IoU > 0.5, pooled over every annotator view).
5. **Ensemble:** mean of the 15 probability maps, horizontal-flip TTA.
6. **Instances (post-processing B):** threshold 0.6 -> connected components; components touched by the same predicted
   spine are merged when spine end points are < 40 px apart and point at each other (a filament broken by a faint gap
   stays one instance); main candidates >= 400 px.
7. **Faint candidates + false-positive filter:** extra candidates = blobs with p > 0.3 outside existing instances;
   every candidate gets 13 features (log area, mean/max p, spine p, elongation, contrast vs. a 15-px ring, radial position on
   the disk, distance to the nearest large filament, ...) and a gradient-boosting regressor trained out-of-fold to predict
   the fraction of annotators that would match it; candidates with score >= 0.35 are kept (stage 1; in the final pipeline this rule is replaced by step 8). Output instances never overlap.

8. **CNN judge** (`stage2/cnn_filter.py`): a small CNN looks at a 7-channel crop around every candidate (image,
   probability, candidate mask, 512-px context with the other candidates) and predicts the share of annotators that would
   match it; 5 backbones x 5 folds are averaged and blended with the GBM score (0.6 / 0.4), keep >= 0.4.
9. **Boundary refiner** (`stage2/refine.py`): every kept filament is re-segmented by a U-Net on a 256-px crop (image,
   probability, own mask, other candidates) trained to reproduce the annotators' outline (pycocotools raster, as the scorer); overlaps resolved by area.

## Stage-1 results (single encoders and ensemble, before the CNN judge and the refiner; OOF PQ, 707 observations)

| model | OOF PQ |
|---|---|
| ResNet34 (Kaggle notebook V16) | 0.4233 (LB 0.38) |
| ConvNeXt-Tiny | 0.4319 (LB 0.38) |
| EfficientNet-B4 | 0.4287 |
| RegNetY-032 | 0.4332 |
| **ensemble ConvNeXt-T + EffNet-B4 + RegNetY (stage 1)** | **0.4410 (LB 0.39)** |

These stage-1 numbers use our own bookkeeping (0.4406 with the organizers' code); the final pipeline (stage 1 + CNN judge + refiner) reaches 0.4541 / LB 0.40, see the table at the top.

Tried and rejected (all OOF, details in `docs/`): neighbouring GONG frames with transferred labels (-0.004),
multi-scale TTA (-0.003), thinner masks for small filaments (-0.001 to -0.008), longer spine links (-0.006),
learned global ensemble weights (-0.005), a stacking CNN on the member maps (-0.005), ConvNeXt-Small / 60 epochs
(0.4258 single, no ensemble gain), scale augmentation (0.4303 single), 4- and 5-model ensembles (0.4402–0.4410).

**Error analysis** (`docs/ERROR_ANALYSIS_ENS3.md`): annotator-vs-annotator PQ on the 296 multi-annotator images is
0.340, our model reaches 0.440 on the same images. Filaments drawn by only one annotator are recalled at 0.35, by all
annotators at 0.87; 871 of 2403 false positives match another annotator's filament. The largest remaining actionable
error is "near misses" (IoU 0.3–0.5: different splitting or extent of long filaments).

## Repository layout

```
inference/predict_final.py  FINAL pipeline, inference only: loads weights/, writes submission.csv (no training)
inference/demo.ipynb     demo notebook: whole pipeline step by step on one image, full test run, check against the
                         submitted file, OOF evaluation with the organizers' metric, optional retraining
inference/infer_ens.py   stage 1 only (previous final, GBM filter only)
tools/oof_viewer.py      organizer-consistent PQ scorer + overlay viewer for the OOF predictions
weights/                 15 fold models (fp16 state dicts); stage2_filter_table.npz (out-of-fold candidate table of the
                         GBM filter used by the final pipeline); filter_export.{json,npz} (stage-1 table)
                         cnn_judge/<backbone>/fold{0-4}.pth (25 CNN judges), refine/fold{0-4}.pth (boundary refiner)
stage2/                  CNN judge (cnn_filter.py, cnn_test.py), boundary refiner (refine.py, refine_test.py),
                         vis_prep.py (rebuilds the out-of-fold candidates and filter features)
stage2/experiments/      conflict graph / component tree, keep-threshold theory, paired bootstrap, uncertainty features
src/run.py               full pipeline for ONE encoder: preprocessing, targets, 5-fold training, OOF evaluation,
                         post-processing search, candidates + FP filter, test predictions
src/run_ens.py           same pipeline on the mean of several trained encoders (--ensemble enc:tag,...)
src/run_sa.py, run_ctx.py, lowres.py   experiments: scale augmentation; two-resolution model (global 1024-px context)
analysis/                error analysis, visualisations, weight search, stacking, post-processing experiments
docs/                    experiment logs and analyses
evaluation_b1_b2_b3_2026-11-12/   evaluation with the organizers' notebook (B1), example gallery (B2), sensitivity (B3)
results/                 submission_final.csv (submitted file, public LB 0.40), oof_final_predictions.csv + folds.csv,
                         per-run OOF result tables (oof_results.json)
```

## Reproduce

```bash
pip install -r requirements.txt
# final pipeline with the stored weights (GPU recommended; ~7.3 s per image on an RTX 4080 SUPER)
python inference/predict_final.py --test_dir <competition data>/test/test_images --out submission.csv
# or open inference/demo.ipynb
# training one encoder (5 folds, ~1.5–2 h on one 32 GB GPU, 2 folds in parallel)
python src/run.py --encoder tu-regnety_032 --tag regy --per_gpu 2
# ensemble + export of the FP-filter table used by inference
python src/run_ens.py --tag ens3 --eval_only --ensemble tu-convnext_tiny:cnxt,tu-efficientnet_b4:effb4,tu-regnety_032:regy
```

Stage 2 (after the out-of-fold maps of the 3 encoders exist):

```bash
S=128 TAG= ARCH=resnet34 python stage2/cnn_filter.py crops && S=128 TAG= ARCH=resnet34 python stage2/cnn_filter.py train
python stage2/refine.py crops && python stage2/refine.py train && python stage2/refine.py eval
MEMBERS=':resnet34:128,_cnxt:convnext_tiny:128,_effb3:efficientnet_b3:128,_r50:resnet50:128,_r34_192:resnet34:192' \
  W_CNN=0.6 THR=0.4 python stage2/refine_test.py
```

The stage-2 training scripts write their weights to `runs/cnnf/` and `/root/refine_coco/`; the fp16 copies in `weights/` are what
`inference/predict_final.py` and the demo notebook load.

Note: the training scripts were run on an AutoDL GPU server and assume the base directory `/root/autodl-tmp/`
(data in `/root/autodl-tmp/data`, outputs in `/root/autodl-tmp/runs/<tag>/`); adjust these paths for another machine.

## Licence

Code: MIT (see `LICENSE`). Pre-trained encoder weights come from timm (Apache-2.0 / respective model licences).
Only the official competition training annotations were used for supervision.
