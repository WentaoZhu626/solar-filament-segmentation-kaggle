"""Out-of-fold predictions of the final pipeline: loading, scoring and overlay figures.

Data needed:
  * competition data (Kaggle: filament-segmentation-2026): train/train_images/*.jpeg and
    train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  * results/oof_final_predictions.csv (this repo): one row per predicted filament on the 707 training observations,
    out of fold (each image predicted by models that never saw its month-fold).
    columns: file_name, fold, instance, keep_score, source (B = spine-merged instance, faint = faint blob),
             rle_refined (final mask, COCO RLE counts at 2048x2048), rle_original (mask before the boundary refiner)

Example:
    from tools.oof_viewer import load_gt, load_pred, score_image, show
    GT = load_gt(DATA + '/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    PR = load_pred('results/oof_final_predictions.csv')
    print(score_all(GT, PR))                       # organizers' pooled PQ: 0.4541 (0.4481 with rle_original)
    show('20140609195854Bh.jpeg', GT, PR, DATA + '/train/train_images/')
"""
import json
from collections import defaultdict
import numpy as np, pandas as pd
from pycocotools import mask as mu

H = W = 2048


def load_gt(json_path):
    """-> {file_name: [view0, view1, ...]}, each view = list of COCO RLEs (one per annotated filament; decoded lazily).
    One "view" = one annotator's annotation of the observation (several image entries share a file_name)."""
    d = json.load(open(json_path))
    by_id = defaultdict(list)
    for a in d['annotations']:
        by_id[a['image_id']].append(a)
    out = defaultdict(list)
    for im in d['images']:
        rles = []
        for a in by_id.get(im['id'], []):
            r = mu.merge(mu.frPyObjects(a['segmentation'], H, W))
            if mu.area(r) > 0:
                rles.append(r)
        out[im['file_name']].append(rles)
    return dict(out)


def load_pred(csv_path, which='rle_refined'):
    """-> {file_name: list of COCO RLEs} (which = 'rle_refined' final, or 'rle_original' before refinement)."""
    df = pd.read_csv(csv_path); out = defaultdict(list)
    for fn, rle in zip(df.file_name, df[which]):
        out[fn].append({'size': [H, W], 'counts': rle.encode()})
    return dict(out)


def decode(rle):
    """COCO RLE -> boolean 2048 x 2048 mask."""
    return mu.decode(rle).astype(bool)


def iou_matrix(preds, gts):
    """IoU between every prediction and every annotation (computed on the RLEs, no full masks needed)."""
    if not preds or not gts:
        return np.zeros((len(preds), len(gts)))
    return np.asarray(mu.iou(preds, gts, [0] * len(gts)))


def score_image(fn, GT, PR):
    """-> array [n_views, 4]: (sum of matched IoU, TP, FP, FN) per annotator view (IoU > 0.5 = match)."""
    preds = PR.get(fn, []); rows = []
    for gts in GT[fn]:
        T = iou_matrix(preds, gts); hit = T > 0.5; tp = int(hit.sum())   # organizers' notebook: every pair with IoU > 0.5 is a TP
        fp = int((~hit.any(1)).sum()) if len(preds) else 0            # predictions without any hit
        fn = int((~hit.any(0)).sum()) if len(gts) else 0              # annotations without any hit
        rows.append([T[hit].sum(), tp, fp, fn])
    return np.array(rows).reshape(-1, 4)


def score_all(GT, PR, files=None):
    S = sum(score_image(fn, GT, PR).sum(0) for fn in (files or GT))
    return dict(PQ=S[0] / (S[1] + 0.5 * S[2] + 0.5 * S[3]), SQ=S[0] / max(S[1], 1), TP=int(S[1]), FP=int(S[2]), FN=int(S[3]))


def show(fn, GT, PR, image_dir, crop=None, PR0=None, save=None):
    """Image + one panel per annotator. green = our filament matched to this annotator, red = not matched,
    cyan = annotated filament we matched, blue = annotated filament we missed, yellow dotted = mask before refinement."""
    import cv2, matplotlib.pyplot as plt
    img = cv2.imread(image_dir + fn, cv2.IMREAD_GRAYSCALE); preds = PR.get(fn, []); views = GT[fn]
    x0, y0, w = crop or (0, 0, W)
    fig, axs = plt.subplots(1, 1 + len(views), figsize=(6 * (1 + len(views)), 6)); axs = np.atleast_1d(axs)
    cont = lambda m: [c[:, 0, :] for c in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]]
    def draw(ax, m, col, ls='-', lw=1.2):
        for c in cont(m):
            c = np.vstack([c, c[:1]]); ax.plot(c[:, 0], c[:, 1], color=col, ls=ls, lw=lw)
    for k, ax in enumerate(axs):
        ax.imshow(img, cmap='gray'); ax.set_xlim(x0, x0 + w); ax.set_ylim(y0 + w, y0); ax.axis('off')
    axs[0].set_title(fn)
    for v, gts in enumerate(views):
        ax = axs[v + 1]; T = iou_matrix(preds, gts); S = score_image(fn, GT, PR)[v]
        ax.set_title(f'annotator {v + 1}: TP {int(S[1])} FP {int(S[2])} FN {int(S[3])}')
        for b, g in enumerate(gts):
            draw(ax, decode(g), 'cyan' if (T[:, b] > 0.5).any() else 'blue', lw=1.6)
        for a, p in enumerate(preds):
            draw(ax, decode(p), 'lime' if (T[a] > 0.5).any() else 'red')
            if PR0 is not None and fn in PR0:
                draw(ax, decode(PR0[fn][a]), 'yellow', ls=':', lw=0.8)
    plt.tight_layout()
    if save: fig.savefig(save, dpi=100); plt.close(fig)
    else: plt.show()
