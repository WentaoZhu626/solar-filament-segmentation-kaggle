"""Cross-check local outputs against the downloaded organizer notebook."""
import ast
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import torch

from evaluate_oof import ROOT, GT_NAME, MASK_COLUMNS, coco, read_data, pq_counts, iou_matrix


def notebook_functions(path):
    names = {"get_overlap_matrices", "get_pq_score", "fn_count_hit", "fp_count_hit"}
    namespace = dict(torch=torch, pd=SimpleNamespace(DataFrame=object))
    notebook = json.loads(path.read_bytes())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
                exec(code, namespace)
    if not names <= namespace.keys():
        raise ValueError("Organizer notebook does not contain the reviewed functions")
    return namespace


def check_dense(function, gt, predictions, matrix):
    boxes = np.array([coco.toBbox(r) for r in gt + predictions], dtype=int)
    x, y = boxes[:, :2].min(0)
    x1, y1 = (boxes[:, :2] + boxes[:, 2:]).max(0)

    def layers(masks):
        array = np.stack([coco.decode(r)[y:y1, x:x1] for r in masks]).astype(np.float32)
        return torch.from_numpy(array)

    iou, dice = function(layers(gt), layers(predictions))
    np.testing.assert_allclose(iou.numpy(), matrix, atol=1e-7)
    np.testing.assert_allclose(dice.numpy(), 2 * matrix / (1 + matrix), atol=1e-7)


def main():
    torch.set_num_threads(1)
    inputs = ROOT / "inputs"
    notebook = inputs / "official/self-evaluation-notebook.ipynb"
    checksum = hashlib.sha256(notebook.read_bytes()).hexdigest()
    if checksum != "ae7c9445b4eee2b77b216df5c9772fdf92fda15611e98abd7ef5f02a4e380801":
        raise ValueError("Organizer notebook differs from the reviewed version")
    official = notebook_functions(notebook)
    folds, views, pred = read_data(inputs / GT_NAME, inputs / "folds.csv",
                                  inputs / "oof_final_predictions.csv")
    cases = json.loads((ROOT / "B2_examples/cases.json").read_bytes())
    selected = {c["file_name"] for c in cases} | {"20160810234034Lh.jpeg"}
    rows = []
    for version in MASK_COLUMNS:
        full_tables, overlap_checks, multi = [], 0, []
        for name in folds:
            predictions = [r[version] for r in pred[name]]
            for view_id, view in views[name].items():
                gt = [r for _, r in view["masks"]]
                matrix = iou_matrix(gt, predictions)
                full_tables.append(SimpleNamespace(iou_matrix=torch.from_numpy(matrix),
                                                   n_gt=len(gt), n_pred=len(predictions)))
                hit = matrix > 0.5
                if (hit.sum(0) > 1).any() or (hit.sum(1) > 1).any():
                    hits = [dict(gt_id=view["masks"][i][0], prediction=pred[name][j]["instance"],
                                 iou=float(matrix[i, j])) for i, j in zip(*np.nonzero(hit))]
                    multi.append(dict(file_name=name, view_id=view_id, n_gt=len(gt), n_pred=len(predictions),
                                      hits=hits, fp_unmatched=pq_counts(matrix)[2],
                                      fp_n_pred_minus_tp=len(predictions)-int(hit.sum())))
                if name in selected and gt and predictions:
                    check_dense(official["get_overlap_matrices"], gt, predictions, matrix)
                    overlap_checks += 1
        score = official["get_pq_score"](SimpleNamespace(itertuples=lambda index: iter(full_tables)))
        rows.append(dict(version=version, pq_via_notebook=score,
                         checked_dense_matrices=overlap_checks, multiple_matches=multi))
    output = ROOT / "B1_metrics/official_notebook_check.json"
    output.write_text(json.dumps(dict(notebook_sha256=checksum, checks=rows), indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
