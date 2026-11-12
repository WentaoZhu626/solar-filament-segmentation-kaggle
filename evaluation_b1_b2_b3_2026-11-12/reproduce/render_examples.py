"""Render selected fold-out predictions and annotator contours on raw H-alpha images."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pycocotools import mask as coco

from evaluate_oof import ROOT, GT_NAME, iou_matrix, plt, read_data, write_csv


CATEGORY = {
    "good": ("Good match", "Good"),
    "small_thin": ("Small, thin filament", "Difficult"),
    "near_limb": ("Near the solar limb", "Difficult"),
    "faint": ("Faint candidate", "Difficult"),
    "miss": ("Missed annotation", "Failure"),
    "false_positive": ("Unmatched prediction", "Failure"),
    "merge": ("Merged annotations", "Failure"),
    "split": ("Fragmented prediction", "Failure"),
    "annotator_disagreement": ("Annotator disagreement", "Disagreement"),
}
ORDER = list(CATEGORY)


def crop_for(masks, size):
    boxes = np.asarray([coco.toBbox(m) for m in masks], dtype=int)
    x, y = boxes[:, :2].min(0)
    x1, y1 = (boxes[:, :2] + boxes[:, 2:]).max(0)
    side = min(max(144, int(max(x1-x, y1-y)*1.55)), min(size))
    left = int(np.clip((x+x1-side)//2, 0, size[1]-side))
    top = int(np.clip((y+y1-side)//2, 0, size[0]-side))
    return left, top, side


def draw_outline(ax, rle, crop, color, style="-", width=1.15):
    x, y, side = crop
    local = coco.decode(rle)[y:y+side, x:x+side]
    contours, _ = cv2.findContours(local, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = contour[:, 0]
        points = np.vstack((points, points[:1]))
        ax.plot(points[:, 0]+x, points[:, 1]+y, style, color=color, linewidth=width)


def local_contrast(image, disk, focus):
    fg = coco.decode(coco.merge(focus)).astype(bool)
    ring = cv2.dilate(fg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(bool)
    ring &= ~fg & disk
    if not ring.any():
        return None
    inner, outer = float(np.median(image[fg])), float(np.median(image[ring]))
    return (outer-inner) / outer if outer else None


def prepare(case, image_dir, views, pred):
    name = case["file_name"]
    path = image_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing original H-alpha image: {path}")
    with Image.open(path) as img:
        image = np.asarray(img.convert("L"))
    group = views[name]
    if any(image.shape != tuple(item["size"]) for item in group.values()):
        raise ValueError(f"Unexpected raw image dimensions: {name}")
    selected = [p["rle_refined"] for p in pred[name] if int(p["instance"]) in case["pred_ids"]]
    focus = selected + [mask for key, mask in group[case["view_id"]]["masks"] if key in case["gt_ids"]]
    if not focus:
        raise ValueError(f"No focus masks: {name}")
    crop = crop_for(focus, image.shape)
    cx, cy, radius = 1024.0, 1024.0, 0.0
    blurred = cv2.GaussianBlur(image, (0, 0), 5)
    _, binary = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, labels, stats, ctr = cv2.connectedComponentsWithStats(binary)
    disk_index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    disk = labels == disk_index
    cx, cy = ctr[disk_index]
    radius = np.sqrt(stats[disk_index, cv2.CC_STAT_AREA] / np.pi)
    box = coco.toBbox(coco.merge(focus))
    radial = float(np.hypot(box[0]+box[2]/2-cx, box[1]+box[3]/2-cy) / radius)
    contrast = local_contrast(image, disk, selected or focus)
    return image, group, selected, crop, radial, contrast


def crop_image(ax, image, crop):
    x, y, side = crop
    local = image[y:y+side, x:x+side]
    low, high = np.percentile(local, [3, 97])
    ax.imshow(local, cmap="gray", vmin=low, vmax=max(high, low+1),
              extent=(x-.5, x+side-.5, y+side-.5, y-.5))
    ax.set(xlim=(x-.5, x+side-.5), ylim=(y+side-.5, y-.5))
    ax.set_axis_off()


def annotate(ax, preds, gt, crop):
    x, y, side = crop
    for _, rle in gt:
        draw_outline(ax, rle, crop, "#63dbff", width=1.25)
    for rle in preds:
        draw_outline(ax, rle, crop, "#ffce55", style="--")
    pixels = round(side / 5 / 10) * 10
    ax.plot([x+8, x+8+pixels], [y+side-10]*2, color="white", linewidth=1.5)
    ax.text(x+8, y+side-14, f"{pixels} px", color="white", fontsize=7)


def run(args):
    _, views, pred = read_data(args.gt, args.folds, args.pred)
    out = args.out
    cases = sorted(json.loads((out / "cases.json").read_text(encoding="utf-8")),
                   key=lambda c: ORDER.index(c["category"]))
    data = []
    for case in cases:
        data.append(prepare(case, args.images, views, pred))
    fig_all, all_axes = plt.subplots(3, 3, figsize=(11.5, 11.8), layout="constrained")
    legend = [Line2D([0], [0], color="#1686a7", label="Annotation"),
              Line2D([0], [0], color="#bd7900", linestyle="--", label="Prediction")]
    catalog = []
    for number, (case, prepared, grid) in enumerate(zip(cases, data, all_axes.flat), 1):
        image, group, selected, crop, radial, contrast = prepared
        category = case["category"]
        title, family = CATEGORY[category]
        name = case["file_name"]
        view_ids = list(group)
        focus_view = case["view_id"]
        index = view_ids.index(focus_view) + 1
        x, y, side = crop
        fig, axes = plt.subplots(1, 1 + len(group), figsize=(3.7*(1+len(group)), 4.5),
                                 layout="constrained")
        full = axes[0]
        full.imshow(image, cmap="gray", vmin=np.percentile(image, 2), vmax=np.percentile(image, 99))
        full.add_patch(Rectangle((x, y), side, side, fill=False, linewidth=1.35, edgecolor="#ffce55"))
        full.set_title("Raw H-alpha · crop location")
        full.set_axis_off()
        pred_in_crop = [p["rle_refined"] for p in pred[name]]
        for k, (ax, view) in enumerate(zip(axes[1:], view_ids), 1):
            crop_image(ax, image, crop)
            annotate(ax, pred_in_crop, group[view]["masks"], crop)
            gt_rles = [item[1] for item in group[view]["masks"]]
            best = iou_matrix(gt_rles, selected).max(initial=0)
            stat = f" · focus best IoU {best:.3f}" if selected else ""
            ax.set_title(f"Annotator {k}" + stat + (" *" if k == index else ""))
        fig.suptitle(f"{number:02d}  {title}   ·   {name}", fontsize=13)
        fig.supxlabel("Cyan solid: annotation   ·   amber dashed: refined prediction   ·   * selection view",
                      fontsize=9)
        filename = f"{number:02d}_{category}.png"
        fig.savefig(out / filename, dpi=185, facecolor="white")
        plt.close(fig)
        crop_image(grid, image, crop)
        annotate(grid, pred_in_crop, group[focus_view]["masks"], crop)
        grid.set_title(f"{number:02d}  {title}\n{Path(name).stem} · view {index}/{len(group)}", fontsize=10)
        catalog.append(dict(number=number, group=family, category=category, file_name=name,
                            annotator_view=focus_view, gt_ids="|".join(case["gt_ids"]),
                            pred_ids="|".join(map(str, case["pred_ids"])), crop_x=x, crop_y=y, crop_side=side,
                            radial_fraction=radial, local_dark_contrast=contrast, evidence=case["evidence"],
                            details=json.dumps({k: v for k, v in case.items() if k not in
                                                ("category", "rank_score", "file_name", "view_id", "gt_ids",
                                                 "pred_ids", "evidence")}),
                            panel=filename))
    for grid in list(all_axes.flat)[len(cases):]:
        grid.set_axis_off()
    fig_all.suptitle("Solar filament segmentation · OOF examples on raw H-alpha", fontsize=15)
    fig_all.supxlabel("Cyan solid: annotation   ·   amber dashed: refined prediction", fontsize=11)
    fig_all.savefig(out / "example_gallery.png", dpi=185, facecolor="white")
    plt.close(fig_all)
    write_csv(out / "case_catalog.csv", catalog)
    print(f"Rendered {len(catalog)} panels and contact sheet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=ROOT / "inputs" / GT_NAME)
    parser.add_argument("--pred", type=Path, default=ROOT / "inputs/oof_final_predictions.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "inputs/folds.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "inputs" / "train_images")
    parser.add_argument("--out", type=Path, default=ROOT / "B2_examples")
    run(parser.parse_args())
