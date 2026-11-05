# Solar Filament Segmentation 2026 — mask + spine U-Net ensemble

Instance segmentation of solar filaments on GONG H-alpha full-disk images
([Kaggle: Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)).

**Final model:** equal-weight ensemble of 3 encoders x 5 folds (15 U-Nets) + spine-guided instance post-processing +
a learned false-positive filter. Out-of-fold (OOF) PQ **0.4410** on all 707 training observations
(5-fold GroupKFold by month), public leaderboard 0.39.

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
6. **Instances (post-processing B):** threshold 0.5 -> connected components; components touched by the same predicted
   spine are merged when spine end points are < 30–40 px apart and point at each other (a filament broken by a faint gap
   stays one instance); min area 400 px.
7. **Faint candidates + false-positive filter:** extra candidates = blobs with p > 0.3 outside existing instances;
   every candidate gets 13 features (log area, mean/max p, spine p, elongation, contrast vs. a 15-px ring, radial position on
   the disk, distance to the nearest large filament, ...) and a gradient-boosting regressor trained out-of-fold to predict
   the fraction of annotators that would match it; candidates with score >= 0.35 are kept. Output instances never overlap.

## Results (OOF PQ, 707 observations)

| model | OOF PQ |
|---|---|
| ResNet34 (Kaggle notebook V16) | 0.4233 (LB 0.38) |
| ConvNeXt-Tiny | 0.4319 (LB 0.38) |
| EfficientNet-B4 | 0.4287 |
| RegNetY-032 | 0.4332 |
| **ensemble ConvNeXt-T + EffNet-B4 + RegNetY (final)** | **0.4410 (LB 0.39)** |

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
inference/infer_ens.py   inference only: loads weights/, writes submission.csv (no training)
inference/demo.ipynb     demo notebook (runs infer_ens.py and shows predictions)
weights/                 15 fold models (fp16 state dicts) + filter_export.{json,npz} (out-of-fold candidate table)
src/run.py               full pipeline for ONE encoder: preprocessing, targets, 5-fold training, OOF evaluation,
                         post-processing search, candidates + FP filter, test predictions
src/run_ens.py           same pipeline on the mean of several trained encoders (--ensemble enc:tag,...)
src/run_sa.py, run_ctx.py, lowres.py   experiments: scale augmentation; two-resolution model (global 1024-px context)
analysis/                error analysis, visualisations, weight search, stacking, post-processing experiments
docs/                    experiment logs and analyses
results/                 per-run OOF result tables (oof_results.json)
```

## Reproduce

```bash
pip install -r requirements.txt
# inference with the stored weights (GPU recommended; ~3 s per image on an RTX 4080)
python inference/infer_ens.py --test_dir <competition data>/test/test_images --out submission.csv
# training one encoder (5 folds, ~1.5–2 h on one 32 GB GPU, 2 folds in parallel)
python src/run.py --encoder tu-regnety_032 --tag regy --per_gpu 2
# ensemble + export of the FP-filter table used by inference
python src/run_ens.py --tag ens3 --eval_only --ensemble tu-convnext_tiny:cnxt,tu-efficientnet_b4:effb4,tu-regnety_032:regy
```

Note: the training scripts were run on an AutoDL GPU server and assume the base directory `/root/autodl-tmp/`
(data in `/root/autodl-tmp/data`, outputs in `/root/autodl-tmp/runs/<tag>/`); adjust these paths for another machine.

## Licence

Code: MIT (see `LICENSE`). Pre-trained encoder weights come from timm (Apache-2.0 / respective model licences).
Only the official competition training annotations were used for supervision.
