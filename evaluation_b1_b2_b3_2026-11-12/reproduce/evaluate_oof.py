"""Evaluate OOF COCO masks and select cases for an H-alpha figure panel."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as coco


ROOT = Path(__file__).resolve().parent
GT_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
MASK_COLUMNS = ("rle_original", "rle_refined")
COLORS = ("#146e81", "#c36b35")
METRIC_COLUMNS = ("iou_sum", "tp", "fp", "fn")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_data(gt_path, folds_path, pred_path):
    manifest = read_csv(folds_path)
    folds = {row["file_name"]: row for row in manifest}
    if len(folds) != len(manifest) or not folds:
        raise ValueError("Fold manifest is empty or contains duplicate observations")
    source = json.loads(gt_path.read_bytes())
    views, annotations = defaultdict(dict), defaultdict(list)
    for image in source["images"]:
        name, view = image["file_name"], str(image["id"])
        if name not in folds or view in annotations:
            raise ValueError(f"Unexpected observation or duplicate view: {name} {view}")
        group = views[name]
        group[view] = dict(size=[image["height"], image["width"]], masks=annotations[view])
    for ann in source["annotations"]:
        view = str(ann["image_id"])
        if view not in annotations:
            raise ValueError(f"Unknown annotation image id: {view}")
        annotations[view].append(ann)
    for name, manifest_row in folds.items():
        if len(views[name]) != int(manifest_row["n_annotator_views"]):
            raise ValueError(f"Incorrect number of annotator views: {name}")
        for data in views[name].values():
            h, w = data["size"]
            masks = []
            for ann in data["masks"]:
                parts = coco.frPyObjects(ann["segmentation"], h, w)
                rle = coco.merge(parts)
                if coco.area(rle) > 0:
                    masks.append((str(ann["id"]), rle))
            data["masks"] = masks

    pred = defaultdict(list)
    seen = set()
    for row in read_csv(pred_path):
        name = row["file_name"]
        key = (name, int(row["instance"]))
        if key in seen or name not in folds or int(row["fold"]) != int(folds[name]["fold"]):
            raise ValueError(f"Duplicate prediction or incorrect fold: {key}")
        seen.add(key)
        sizes = {tuple(v["size"]) for v in views[name].values()}
        if len(sizes) != 1:
            raise ValueError(f"Inconsistent dimensions: {name}")
        size = list(sizes.pop())
        for col in MASK_COLUMNS:
            row[col] = {"size": size, "counts": row[col].encode("ascii")}
            if not coco.area(row[col]):
                raise ValueError(f"Empty prediction: {key} ({col})")
        row["keep_score"] = float(row["keep_score"])
        pred[name].append(row)
    return folds, views, pred


def iou_matrix(gt, pred):
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    return np.asarray(coco.iou(gt, pred, [0] * len(pred)), dtype=np.float64)


def pq_counts(matrix):
    hit = matrix > 0.5
    return (matrix[hit].sum(), int(hit.sum()), int((~hit.any(axis=0)).sum()),
            int((~hit.any(axis=1)).sum()))


def pq_result(totals):
    summed, tp, fp, fn = totals
    denominator = tp + (fp + fn) / 2
    return {"pq": float(summed / denominator) if denominator else 0.0,
            "sq": float(summed / tp) if tp else 0.0,
            "rq": float(tp / denominator) if denominator else 0.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "iou_sum": float(summed)}


def torch_dice(a, b):
    import torch
    from torchmetrics.functional.segmentation import dice_score

    a = torch.from_numpy(np.ascontiguousarray(a != 0))[None, None]
    b = torch.from_numpy(np.ascontiguousarray(b != 0))[None, None]
    return float(dice_score(a, b, num_classes=1, input_format="one-hot",
                            include_background=True, average="micro",
                            aggregation_level="samplewise").reshape(-1)[0])


def union_mask(rles, size):
    return coco.decode(coco.merge(rles)) if rles else np.zeros(size, dtype=np.uint8)


def cropped_pair(first, second):
    boxes = np.array([coco.toBbox(first), coco.toBbox(second)], dtype=int)
    x, y = boxes[:, :2].min(axis=0)
    x1, y1 = (boxes[:, :2] + boxes[:, 2:]).max(axis=0)
    return tuple(coco.decode(mask)[y:y1, x:x1] for mask in (first, second))


def add_case(options, category, score, name, view, gt_ids, pred_ids, reason, **evidence):
    options[category].append(dict(category=category, rank_score=float(score),
                                  file_name=name, view_id=view, gt_ids=gt_ids,
                                  pred_ids=pred_ids, evidence=reason, **evidence))


def case_options(options, name, view, ids, gt, pred_rows, matrices):
    mat = matrices[view]
    preds = [p["rle_refined"] for p in pred_rows]
    pred_ids = [int(p["instance"]) for p in pred_rows]
    areas = [int(coco.area(g)) for g in gt]
    for i, j in zip(*np.nonzero(mat > 0.5)):
        iou = float(mat[i, j])
        g, p = gt[i], pred_ids[j]
        x, y, w, h = coco.toBbox(g)
        radius = float(np.hypot(x+w/2-1024, y+h/2-1024))
        thickness = areas[i] / max(w, h)
        pair = dict(iou=iou, area_px=areas[i], radius_px=radius, area_over_bbox_length=thickness)
        if 1200 <= areas[i] <= 10000:
            add_case(options, "good", iou, name, view, [ids[i]], [p], "High-IoU match", **pair)
        if 70 <= areas[i] < 450 and thickness < 7:
            add_case(options, "small_thin", iou, name, view, [ids[i]], [p],
                     "Small, thin GT instance", **pair)
        if radius >= 730:
            add_case(options, "near_limb", radius, name, view, [ids[i]], [p],
                     "Large radial distance from image centre", **pair)
        if pred_rows[j]["source"] == "faint":
            add_case(options, "faint", iou, name, view, [ids[i]], [p],
                     "Matched instance from the low-probability candidate pool", **pair)
    for i, g in enumerate(gt):
        row = mat[i]
        if not (row > 0).any() and areas[i] >= 300:
            add_case(options, "miss", min(areas[i], 4500), name, view, [ids[i]], [],
                     "GT has zero overlap with every prediction", area_px=areas[i])
        jx = np.flatnonzero(row >= 0.10)
        if len(jx) >= 2 and row.max() <= 0.5:
            covered = float(coco.area(coco.merge([g, coco.merge([preds[j] for j in jx])], intersect=True))) / areas[i]
            if covered >= 0.5:
                add_case(options, "split", covered, name, view, [ids[i]], [pred_ids[j] for j in jx],
                         "At least two predictions overlap GT and jointly cover at least half", covered_fraction=covered)
    for j, p in enumerate(preds):
        column = mat[:, j]
        bests = {v: float(t[:, j].max(initial=0)) for v, t in matrices.items()}
        if max(bests.values()) == 0:
            add_case(options, "false_positive", min(int(coco.area(p)), 4500), name, view, [], [pred_ids[j]],
                     "No overlap with any annotation in the available views", area_px=int(coco.area(p)))
        elif len(bests) > 1 and bests[view] > 0.5 and min(bests.values()) == 0:
            add_case(options, "annotator_disagreement", bests[view], name, view,
                     [ids[int(column.argmax())]], [pred_ids[j]],
                     "Matches one view and has zero overlap in another", best_iou_by_view=bests)
        ix = np.flatnonzero(column >= 0.10)
        if len(ix) >= 2 and column.max() <= 0.5:
            coverage = [float(coco.area(coco.merge([p, gt[i]], intersect=True))) / areas[i] for i in ix]
            if min(coverage) >= 0.5:
                add_case(options, "merge", min(coverage), name, view, [ids[i] for i in ix], [pred_ids[j]],
                         "Single prediction covers at least half of each of two or more GT instances",
                         gt_coverage=coverage, iou_by_gt=column[ix].tolist())


def select_cases(options):
    chosen, used = [], set()
    categories = ("annotator_disagreement", "near_limb", "faint", "small_thin",
                  "merge", "split", "miss", "false_positive", "good")
    for category in categories:
        ordered = sorted(options[category], key=lambda c: (-c["rank_score"], c["file_name"], c["view_id"]))
        for case in ordered:
            if case["file_name"] not in used:
                chosen.append(case)
                used.add(case["file_name"])
                break
    return chosen


def plot_histograms(out, pairs, degrees):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), layout="constrained")
    hist_rows = []
    edges = np.linspace(0, 1, 51)
    for ax, metric in zip(axes, ("dice", "iou")):
        for version, color in zip(MASK_COLUMNS, COLORS):
            values = np.array([p[metric] for p in pairs if p["version"] == version])
            counts, _ = np.histogram(values, edges)
            ax.stairs(counts / max(len(values), 1), edges, color=color, linewidth=1.8,
                      label=f"{version[4:].title()} (n={len(values):,})")
            for j, count in enumerate(counts):
                hist_rows.append(dict(version=version, metric=metric, left=edges[j], right=edges[j+1],
                                      count=int(count), probability=count / max(len(values), 1)))
        ax.set(xlim=(0, 1), xlabel={"dice": "Dice", "iou": "IoU"}[metric],
               ylabel="Probability per bin")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("OOF instance overlap · all pairs with IoU > 0")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"dice_iou.{suffix}", dpi=200)
    plt.close(fig)
    write_csv(out / "histogram_bins.csv", hist_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4), layout="constrained")
    for ax, direction, title in zip(axes, ("gt_to_pred", "pred_to_gt"),
                                    ("Ground truth → prediction", "Prediction → ground truth")):
        subset = [d for d in degrees if d["direction"] == direction]
        labels = list(range(max((d["degree"] for d in subset), default=0) + 1))
        for offset, version, color in zip((-0.19, 0.19), MASK_COLUMNS, COLORS):
            counts = {d["degree"]: d["count"] for d in subset if d["version"] == version}
            ax.bar(np.array(labels)+offset, [counts.get(k, 0) for k in labels], 0.37,
                   label=version[4:].title(), color=color)
        ax.set(title=title, xticks=labels, xlabel="Number of overlapping partners", ylabel="Instance-view count",
               yscale="symlog", ylim=(0, max((d["count"] for d in subset), default=1) * 1.4))
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False)
    fig.suptitle("OOF overlap relations · edge defined by IoU > 0")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"split_merge.{suffix}", dpi=200)
    plt.close(fig)


def evaluate(args):
    import torch
    torch.set_num_threads(1)
    folds, views, pred = read_data(args.gt, args.folds, args.pred)
    out = args.out / "B1_metrics"
    out.mkdir(parents=True, exist_ok=True)
    scores, pairs, options, per_view_degrees = [], [], defaultdict(list), []
    sweeps = defaultdict(lambda: np.zeros(4))
    pooled = defaultdict(lambda: np.zeros(4))
    by_degree = Counter()
    checked_pairs = 0
    for index, (name, manifest) in enumerate(folds.items()):
        fold = int(manifest["fold"])
        pred_rows = pred[name]
        for version in MASK_COLUMNS:
            preds = [p[version] for p in pred_rows]
            matrices = {}
            size = next(iter(views[name].values()))["size"]
            p_union = union_mask(preds, size)
            for view, v in views[name].items():
                gt = [m[1] for m in v["masks"]]
                matrix = iou_matrix(gt, preds)
                matrices[view] = matrix
                stats = pq_counts(matrix)
                hit = matrix > 0.5
                record = dict(version=version, file_name=name, fold=fold, view_id=view)
                scores.append(dict(record, n_gt=len(gt), n_pred=len(preds),
                                   foreground_dice_torchmetrics=torch_dice(p_union, union_mask(gt, size)),
                                   multiple_pq_links=int((hit.sum(0)>1).sum()+(hit.sum(1)>1).sum()),
                                   **pq_result(stats)))
                pooled[version, "all"] += stats
                pooled[version, fold] += stats
                gt_degree = (matrix > 0).sum(1)
                pred_degree = (matrix > 0).sum(0)
                for direction, degrees, keys in (
                    ("gt_to_pred", gt_degree, [g[0] for g in v["masks"]]),
                    ("pred_to_gt", pred_degree, [p["instance"] for p in pred_rows]),
                ):
                    for key, degree in zip(keys, degrees):
                        by_degree[version, direction, int(degree)] += 1
                        per_view_degrees.append(dict(record, direction=direction, instance=key, degree=int(degree)))
                for i, j in zip(*np.nonzero(matrix > 0)):
                    iou = float(matrix[i, j])
                    dice = torch_dice(*cropped_pair(gt[i], preds[j]))
                    if abs(dice - 2 * iou / (1 + iou)) > 2e-7:
                        raise ValueError(f"Dice verification failed for {name} {view} {i} {j}")
                    checked_pairs += 1
                    pairs.append(dict(record, gt_id=v["masks"][i][0], pred_instance=pred_rows[j]["instance"],
                                      dice=dice, iou=iou, pq_match=int(iou > 0.5)))
                for t in (0.40, 0.425, 0.45, 0.475, 0.50, 0.55, 0.60):
                    keep = np.array([p["keep_score"] >= t for p in pred_rows], dtype=bool)
                    sweeps[version, "keep_score", t] += pq_counts(matrix[:, keep])
                for t in (400, 450, 500, 600, 800):
                    keep = np.array([p["source"] != "B" or coco.area(p["rle_original"]) >= t
                                     for p in pred_rows], dtype=bool)
                    sweeps[version, "fixed_pool_B_area", t] += pq_counts(matrix[:, keep])
            if version == "rle_refined":
                for view, v in views[name].items():
                    ids, gt = ([item[k] for item in v["masks"]] for k in (0, 1))
                    case_options(options, name, view, ids, gt, pred_rows, matrices)
        if (index+1) % 100 == 0:
            print(f"Evaluated {index+1}/{len(folds)} observations", flush=True)

    totals = [dict(version=version, fold=fold, **pq_result(stats))
              for (version, fold), stats in pooled.items()]
    degree_rows = [dict(version=v, direction=d, degree=k, count=n)
                   for (v, d, k), n in sorted(by_degree.items())]
    summaries = []
    for version in MASK_COLUMNS:
        per_view = [s for s in scores if s["version"] == version]
        population = [p for p in pairs if p["version"] == version]
        for metric, sample in (
            ("foreground_dice_torchmetrics", [s["foreground_dice_torchmetrics"] for s in per_view]),
            ("dice_positive_pairs", [p["dice"] for p in population]),
            ("iou_positive_pairs", [p["iou"] for p in population]),
            ("dice_PQ_matches", [p["dice"] for p in population if p["pq_match"]]),
            ("iou_PQ_matches", [p["iou"] for p in population if p["pq_match"]]),
        ):
            p10, p25, med, p75, p90 = np.percentile(sample, [10, 25, 50, 75, 90]) if sample else [None]*5
            summaries.append(dict(version=version, metric=metric, n=len(sample),
                                  mean=float(np.mean(sample)) if sample else None,
                                  p10=p10, p25=p25, median=med, p75=p75, p90=p90))
    write_csv(out / "pq_by_fold.csv", totals)
    write_csv(out / "view_scores.csv", scores)
    write_csv(out / "positive_pairs.csv", pairs)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "overlap_degrees.csv", degree_rows)
    write_csv(out / "instance_degrees.csv", per_view_degrees)
    plot_histograms(out, pairs, degree_rows)

    sens = args.out / "B3_sensitivity"
    sens.mkdir(exist_ok=True)
    sweep_rows = []
    for (version, variable, setting), stats in sweeps.items():
        baseline = 0.4 if variable == "keep_score" else 400
        sweep_rows.append(dict(version=version, variable=variable, setting=setting,
                               baseline=baseline, **pq_result(stats),
                               delta_pq=pq_result(stats)["pq"]-pq_result(sweeps[version, variable, baseline])["pq"],
                               scope="saved masks and candidate pool held fixed"))
    write_csv(sens / "fixed_pool_sweeps.csv", sweep_rows)
    examples = args.out / "B2_examples"
    examples.mkdir(exist_ok=True)
    chosen = select_cases(options)
    (examples / "cases.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    (examples / "image_list.txt").write_text("".join(f"{c['file_name']}\n" for c in chosen), encoding="utf-8")
    run = dict(observations=len(folds), views=sum(len(v) for v in views.values()),
               predictions=sum(len(p) for p in pred.values()),
               no_prediction_observations=sum(not p for p in pred.values()),
               multiple_pq_links={v: sum(s["multiple_pq_links"] for s in scores if s["version"] == v)
                                  for v in MASK_COLUMNS},
               torchmetrics_pair_dice_checks=checked_pairs,
               inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (args.gt, args.folds, args.pred)})
    (out / "run_manifest.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps([t for t in totals if t["fold"] == "all"], indent=2), flush=True)
    print(f"Selected {len(chosen)} examples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=ROOT / "inputs" / GT_NAME)
    parser.add_argument("--pred", type=Path, default=ROOT / "inputs/oof_final_predictions.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "inputs/folds.csv")
    parser.add_argument("--out", type=Path, default=ROOT)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
