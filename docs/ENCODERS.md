# Solar Filament — encoder comparison (see RESULTS.md for everything else)
## Encoder comparison on AutoDL (vGPU 32 GB, Nov 4) — same model v2 pipeline, only the encoder changes
Script `run.py` (port of the notebook; per-fold subprocesses, 2 folds in parallel). Folds use a newer sklearn than Kaggle, so views per fold differ slightly ([225,236,235,230,228] vs Kaggle [219,232,235,240,228]); compare encoders with each other here, not with V16 exactly.

| encoder | A | B | best candidates + filter (OOF) | per fold 0–4 | LB |
|---|---|---|---|---|---|
| ResNet34 (= V16 model) | 0.3882 | 0.3956 | 0.4238 (B min_area 250) | 0.4302 / 0.4281 / 0.4025 / 0.4293 / 0.4297 | (0.38 = V16) |
| **ConvNeXt-Tiny** (tu-convnext_tiny) | 0.3968 | 0.4036 | **0.4319** (faint p>0.5) | 0.4287 / 0.4354 / 0.4200 / 0.4286 / 0.4483 | **0.38** |
| ResNet50 | 0.3875 | 0.3939 | 0.4265 (faint 0.4) | 0.4443 / 0.4233 / 0.4108 / 0.4245 / 0.4291 | – |
| EfficientNet-B4 | 0.4069 | 0.4128 | 0.4287 (B min_area 250) | 0.4354 / 0.4272 / 0.4148 / 0.4357 / 0.4309 | – |
| RegNetY-032 (tu-regnety_032) | 0.4052 | 0.4102 | **0.4332** (B min_area 150) | 0.4370 / 0.4414 / 0.4096 / 0.4395 / 0.4392 | – |

ConvNeXt inference notebook: https://www.kaggle.com/code/wentaozhu626/solar-filament-convnext-inference (weights dataset `solar-filament-convnext-tiny`: 5 fp16 folds + filter_export.npz/json; FP filter refit at inference). Test: 983 filaments on 177/180 images, 3.8 s/image on T4, run 12.6 min. LB 0.38 = same as V16 (LB shows 2 decimals; OOF +0.008 is below its resolution).
Reading: EffB4 has the best raw maps (A/B +0.010 over ConvNeXt) but gains less from candidates+filter; folds are mixed (EffB4 better on folds 0 and 3). Next idea: ensemble ConvNeXt + EffB4 probability maps (needs OOF maps from both runs).

## Probability-map ensembles (run_ens.py = run.py + `--ensemble enc:tag,...`; mean of all members' fold models, then the full pipeline re-tuned)
| ensemble | A | B | best candidates + filter (OOF) | per fold 0–4 | test |
|---|---|---|---|---|---|
| ens2 = ConvNeXt + EffB4 (10 models) | 0.4093 (hyst.) | 0.4155 | 0.4369 (faint p>0.4) | 0.4384 / 0.4427 / 0.4171 / 0.4458 / 0.4406 | 1055 filaments / 176 imgs |
| **ens3 = ConvNeXt + EffB4 + RegNetY (15 models)** | – | 0.4153 | **0.4410** (faint p>0.3; all 7 candidate configs 0.4356–0.4410) | 0.4476 / 0.4510 / 0.4143 / 0.4439 / 0.4495 | 1103 filaments / 178 imgs, 3.27 s/img |
| ens5 = all 5 encoders (25 models, + ResNet34 + ResNet50) | – | 0.4106 | 0.4375 (faint p>0.3) | 0.4440 / 0.4407 / 0.4160 / 0.4433 / 0.4440 | – |

ens5 < ens3 (-0.0035, worse in 4/5 folds): the two weaker ResNets dilute the equal-weight average. => use ens3.
ens3 vs best single (RegNetY 0.4332): +0.008; vs V16 (0.4233): +0.018. Submission file: AutoDL /root/autodl-tmp/runs/ens3/submission.csv (test inference done on AutoDL; the competition accepts CSV upload).
Kaggle CLI works from AutoDL WITHOUT network_turbo (unset proxy env vars); with turbo it fails on SSL.

## Global ensemble weights (wsearch.py on per-model OOF maps cnxt / effb4 / regy) — NOT adopted
Stage 1 (post-processing B only, 66 weights on a 0.1 grid): best (0.1, 0.6, 0.3) B PQ 0.4173 vs equal 0.4153; top-10 within 0.0017.
Nested (weights chosen on 4 folds, scored on the 5th): 0.4158 vs equal 0.4150 (better in 3/5 folds).
Full pipeline (candidates + FP filter): equal 0.4412 (faint p>0.5; = ens3), (0.1,0.6,0.3) 0.4353, (0.1,0.5,0.4) 0.4365, (0.2,0.5,0.3) 0.4366.
=> the small B-stage gain disappears / reverses after the FP filter (weights favouring EffB4 give more but worse candidates). Keep equal weights.
Per-model OOF maps saved: runs/m_{cnxt,effb4,regy}/oof_maps.npy (707x2x2048x2048 uint8) for stacking.

## Stacking judge (stack.py) — NOT adopted
Small fully-convolutional net (5 dilated 3x3 conv layers, 32 ch, RF ~63 px) on [cnxt mask, cnxt spine, effb4 mask, effb4 spine, regy mask, regy spine, image]; output = logit(member mean) + learned correction (zero-init head); BCE + soft Dice (mask) + 0.5 BCE (spine), 10 x 400 crops of 512 px per fold; trained on the other folds' OOF maps.
Result: B PQ 0.4076 (equal mean 0.4153); full pipeline 0.4363 (faint p>0.3) vs equal 0.4412; per fold 0.4411/0.4365/0.4209/0.4412/0.4428 (better only in fold 2). The judge re-calibrates the maps toward the soft consensus targets, which costs PQ after thresholding/post-processing tuned for the averaged maps.

## ConvNeXt-Small (tu-convnext_small, 60 epochs)
Training 5 folds ~4.3 h (folds 2/3 slowed by CPU contention). B-stage OOF 0.4004 (ConvNeXt-Tiny 0.4036) — no better than Tiny before candidates/filter.
Final (candidates + filter): 0.4258 (faint p>0.5), folds 0.4257/0.4368/0.4082/0.4253/0.4348 — below ConvNeXt-Tiny 0.4319; test 875 filaments / 172 imgs.

## ens4 = equal mean of cnxt + effb4 + regy + cnxs (weq.py)
B 0.4150; full 0.4410 (faint p>0.3), folds 0.4459/0.4459/0.4226/0.4431/0.4485 vs ens3 0.4412 (0.4487/0.4486/0.4161/0.4466/0.4472): equal overall, better in 2/5 folds. => ConvNeXt-Small adds nothing.

## RegNetY seed 7 + 60 epochs + scale augmentation 0.8–1.25 ('regy_sa', run_sa.py)
Single model final 0.4303 (faint p>0.3), folds 0.4332/0.4384/0.4066/0.4347/0.4383 (regy 0.4332). B 0.4042.
ens_sa4 = cnxt+effb4+regy+regy_sa equal: 0.4402 (0.4412/0.4434/0.4211/0.4491/0.4466).
ens_sa3 = cnxt+effb4+regy_sa equal: 0.4380 (0.4420/0.4440/0.4190/0.4383/0.4472).

## Overnight conclusion (Nov 5)
Nothing beat ens3 (equal mean cnxt+effb4+regy, OOF 0.4410/0.4412, LB 0.39): thinning small filaments, longer spine links, global weights, stacking judge, ConvNeXt-Small 60 ep, 4-model ensembles, scale augmentation + longer training. OOF plateau ~0.441; the remaining error is dominated by annotator disagreement (see ERROR_ANALYSIS_ENS3.md). Submission kept: ens3.
