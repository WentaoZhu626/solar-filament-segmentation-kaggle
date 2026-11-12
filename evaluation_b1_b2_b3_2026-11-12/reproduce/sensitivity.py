"""Consolidate recorded and recomputed sensitivity results and score new OOF exports."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_oof import ROOT, GT_NAME, MASK_COLUMNS, iou_matrix, pq_counts, pq_result, read_csv, read_data, write_csv


REFINER_DOC = "docs/BOUNDARY_REFINER.md"
JUDGE_DOC = "docs/CNN_JUDGE.md"


def build_rows(out, folds, views, predictions):
    sweep_file = out / "fixed_pool_sweeps.csv"
    fixed = read_csv(sweep_file)
    baseline = next(float(r["pq"]) for r in fixed if r["version"] == "rle_refined"
                    and r["variable"] == "keep_score" and float(r["setting"]) == 0.4)
    rows = []

    def add(variable, setting, pq, reference, status, scope, source, baseline_pq=baseline):
        rows.append(dict(variable=variable, setting=setting, pq=pq, reference=reference,
                         reference_pq=baseline_pq if pq is not None else None,
                         delta_pq=round(float(pq)-baseline_pq, 8) if pq is not None else None,
                         status=status, scope=scope, source=source))

    for row in fixed:
        version = row["version"]
        variable = row["variable"]
        setting = float(row["setting"])
        score = float(row["pq"])
        ref_score = score-float(row["delta_pq"])
        is_keep = variable == "keep_score"
        add("keep threshold" if is_keep else "minimum B area", setting, score,
            f"{version}, {'0.40' if is_keep else '400 px'}", "recomputed",
            ("Score threshold applied to all saved masks" if is_keep else
             "Prune saved B masks by original area; weak blobs and overlap resolution held fixed"),
            "fixed_pool_sweeps.csv", ref_score)
    add("keep threshold", 0.35, None, "0.40, final OOF pipeline", "needs export",
        "Includes candidates rejected at 0.40", "All candidate scores and masks")

    for threshold, pq in ((0.40, 0.4531), (0.50, 0.4542), (0.60, 0.4530)):
        add("refiner threshold", threshold, pq, "0.50, CNN-judge selection",
            "repository experiment", "Same CNN keep set, pycocotools raster", REFINER_DOC, 0.4542)
    for threshold, pq in ((0.40, 0.4443), (0.50, 0.4453), (0.60, 0.4446)):
        add("refiner threshold", threshold, pq, "0.50, GBM selection",
            "repository experiment", "Separate GBM keep set, pycocotools raster", REFINER_DOC, 0.4453)

    before = next(float(r["pq"]) for r in fixed if r["version"] == "rle_original"
                  and r["variable"] == "keep_score" and float(r["setting"]) == 0.4)
    add("CNN blend weight", 0.60, before, "0.60, five judges, before refinement",
        "recomputed baseline", "Final CNN keep set with original masks", "fixed_pool_sweeps.csv", before)
    for value in (0.30, 0.50, 0.70):
        add("CNN blend weight", value, None, "0.60, five judges, keep threshold 0.40",
            "needs export", "Keep five judge scores, GBM and threshold fixed",
            "Unfiltered five judge OOF scores and GBM score")

    add("minimum B area", 250, None, "400 px, final OOF pipeline", "needs export",
        "Regenerate main and faint candidates; rerun judge and refiner", "Full-pipeline variant export")
    for threshold in (0.20, 0.30, 0.40, 0.50):
        if threshold == 0.30:
            add("faint probability threshold", threshold, baseline, "0.30, final OOF pipeline",
                "recomputed baseline", "Baseline export with p > 0.30 weak-candidate generation",
                "fixed_pool_sweeps.csv")
        else:
            add("faint probability threshold", threshold, None, "0.30, final OOF pipeline",
                "needs export", "Recompute faint connected components and downstream predictions",
                "Full-pipeline variant export")
    for version in MASK_COLUMNS:
        totals = np.zeros(4)
        for name in folds:
            masks = [p[version] for p in predictions[name] if p["source"] == "B"]
            for view in views[name].values():
                totals += pq_counts(iou_matrix([r for _, r in view["masks"]], masks))
        add("faint candidates", "remove kept faint instances", pq_result(totals)["pq"],
            f"all saved masks, {version}", "recomputed proxy",
            "Remove saved faint masks with all other masks fixed; this is not a probability-threshold sweep",
            "oof_final_predictions.csv", baseline if version == "rle_refined" else before)

    add("judge count", 5, before, "five judges, before refinement", "recomputed baseline",
        "Final blend weight 0.60, keep threshold 0.40", "fixed_pool_sweeps.csv", before)
    for value in (1, 2, 3, 4):
        add("judge count", value, None, "five judges, blend 0.60, keep threshold 0.40",
            "needs export", "Use a declared judge order and fix the other controls", "Unfiltered per-judge OOF scores")
    for judge, pq in (
        ("ResNet34 128", 0.4464), ("ConvNeXt-Tiny 128", 0.4469), ("EfficientNet-B3 128", 0.4448),
        ("ResNet50 128", 0.4475), ("ResNet34 192", 0.4475),
    ):
        add("historical single judge", judge, pq, "five judges with blend 0.60, reported 0.4481",
            "repository experiment", "Single judge with blend 0.50; judge count and blend both differ",
            JUDGE_DOC, 0.4481)
    add("historical GBM only", 0, 0.4406, "five judges with blend 0.60, reported 0.4481",
        "repository experiment", "Historical comparison; thresholds and judge mix are not controlled",
        JUDGE_DOC, 0.4481)
    return rows


def score_variants(manifest, folds_path, gt_path, baseline):
    """Each CSV is a full OOF export with the same schema and folds as the baseline."""
    seen = set()
    rows = []
    for row in read_csv(manifest):
        key = (row["variable"], row["setting"])
        if key in seen:
            raise ValueError(f"Duplicate sweep condition: {key}")
        seen.add(key)
        path = manifest.parent / row["prediction_csv"]
        folds, views, pred = read_data(gt_path, folds_path, path)
        for version in MASK_COLUMNS:
            totals = np.zeros(4)
            for name in folds:
                masks = [p[version] for p in pred[name]]
                for view in views[name].values():
                    totals += pq_counts(iou_matrix([r for _, r in view["masks"]], masks))
            metrics = pq_result(totals)
            rows.append(dict(variable=key[0], setting=key[1], version=version, **metrics,
                             delta_pq=metrics["pq"]-baseline[version], prediction_csv=row["prediction_csv"]))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=ROOT / "inputs" / GT_NAME)
    parser.add_argument("--pred", type=Path, default=ROOT / "inputs/oof_final_predictions.csv")
    parser.add_argument("--folds", type=Path, default=ROOT / "inputs/folds.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "B3_sensitivity")
    parser.add_argument("--variants", type=Path, help="CSV with variable,setting,prediction_csv")
    args = parser.parse_args()
    folds, views, predictions = read_data(args.gt, args.folds, args.pred)
    rows = build_rows(args.out, folds, views, predictions)
    write_csv(args.out / "sensitivity_table.csv", rows)
    if args.variants:
        base = {r["version"]: float(r["pq"]) for r in read_csv(args.out / "fixed_pool_sweeps.csv")
                if r["variable"] == "keep_score" and float(r["setting"]) == 0.4}
        write_csv(args.out / "variant_results.csv", score_variants(args.variants, args.folds, args.gt, base))
    print(f"Saved {len(rows)} sensitivity rows")
