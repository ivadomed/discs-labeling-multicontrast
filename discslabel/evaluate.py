"""Compare predicted disc-label NIfTIs with ground-truth NIfTIs.

For each paired ``<stem>`` we read:
  * GT     : ``<gt_dir>/<stem>.nii.gz``             — single-voxel disc labels (1..25)
  * Pred   : ``<pred_dir>/<stem>_discs.nii.gz``     — same format, output of
             ``discslabel_inference`` / ``discslabel_inference_3d``

Matching is by LABEL VALUE: each disc identity (1..25) is either present or
absent in each volume. For labels in both, we compute the voxel centroid in
each, project to world-space mm via the respective affine, and take the
Euclidean distance. Labels only in GT are false negatives (missed); labels
only in prediction are false positives (hallucinated or mislabeled).

Writes to ``<out_dir>``:
  * ``metrics.json`` — summary + per-subject / per-contrast / per-image breakdown
  * ``metrics.csv``  — one row per (image, label) for easy pandas slicing

Each run writes to its own ``--out-dir`` (``<eval>/metrics_<run_name>``), so
several evaluations can run concurrently. ``discslabel_aggregate_evaluation``
then gathers every ``metrics_*/metrics.json`` into ``evaluation.json`` plus
comparison CSVs (overall, per subject, per contrast).

Image stems follow ``<dataset>__sub-XXX[_ses-YY..]_<CONTRAST>`` (see
``prepare_evaluation.sh``): the subject is ``<dataset>__sub-XXX`` and the
contrast is the last ``_``-separated token of the stem (e.g. T2w, PSIR).

Both GT and prediction live in the input image's NATIVE space. Distances are
affine-invariant, so differing voxel grids between GT and prediction (e.g.
if a dataset keeps labels in a slightly different frame from the image) do
not corrupt the comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np


NIFTI_EXTS: Tuple[str, ...] = (".nii.gz", ".nii")
PRED_SUFFIX = "_discs"

SUMMARY_FILES: Dict[str, str] = {
    "overall": "metrics.csv",
    "subject": "metrics_per_subject.csv",
    "contrast": "metrics_per_contrast.csv",
}

METRIC_FIELDS: Tuple[str, ...] = (
    "n_images",
    "n_gt_discs",
    "n_pred_discs",
    "n_matched",
    "n_false_negative",
    "n_false_positive",
    "recall",
    "precision",
    "distance_mm_mean",
    "distance_mm_median",
    "distance_mm_p95",
    "distance_mm_max",
)


def _is_nifti(name: str) -> bool:
    return any(name.endswith(ext) for ext in NIFTI_EXTS)


def _strip_nifti_ext(name: str) -> str:
    for ext in NIFTI_EXTS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def _extract_disc_voxels(label_path: str) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Return ``({label_value: (3,) voxel centroid}, affine)``.

    Labels are read as rounded int. A label value may appear at more than one
    voxel (shouldn't for point annotations, but we tolerate it) — in that case
    we take the voxel centroid, which collapses to the single voxel in the
    normal case.
    """
    img = nib.load(label_path)
    data = np.rint(np.asarray(img.dataobj)).astype(np.int32)
    data = np.squeeze(data)
    discs: Dict[int, np.ndarray] = {}
    for value in np.unique(data):
        if value <= 0:
            continue
        coords = np.argwhere(data == int(value))
        if len(coords) == 0:
            continue
        discs[int(value)] = coords.mean(axis=0)
    return discs, np.asarray(img.affine)


def _voxel_to_world(voxel: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert a (3,) voxel coord to a (3,) world-space mm position."""
    v = np.array([voxel[0], voxel[1], voxel[2], 1.0], dtype=np.float64)
    return (affine @ v)[:3]


def _discover_pairs(gt_dir: str, pred_dir: str) -> List[Tuple[str, str, str]]:
    """Return ``[(stem, gt_path, pred_path), ...]`` for every stem with BOTH files.

    GT files are named ``<stem>.nii[.gz]``; pred files ``<stem>_discs.nii[.gz]``.
    Prints a warning for lonely predictions or lonely GTs so the user can
    see when the eval folder and the predictions folder are out of sync.
    """
    gt_by_stem: Dict[str, str] = {}
    for name in sorted(os.listdir(gt_dir)):
        if not _is_nifti(name) or name.startswith("."):
            continue
        gt_by_stem[_strip_nifti_ext(name)] = os.path.join(gt_dir, name)

    pred_by_stem: Dict[str, str] = {}
    for name in sorted(os.listdir(pred_dir)):
        if not _is_nifti(name) or name.startswith("."):
            continue
        stem = _strip_nifti_ext(name)
        if stem.endswith(PRED_SUFFIX):
            stem = stem[: -len(PRED_SUFFIX)]
        pred_by_stem[stem] = os.path.join(pred_dir, name)

    pairs: List[Tuple[str, str, str]] = []
    for stem in sorted(set(gt_by_stem) | set(pred_by_stem)):
        gt_path = gt_by_stem.get(stem)
        pred_path = pred_by_stem.get(stem)
        if gt_path is None:
            print(f"[eval] WARN: prediction without GT: {stem}")
            continue
        if pred_path is None:
            print(f"[eval] WARN: GT without prediction: {stem}")
            continue
        pairs.append((stem, gt_path, pred_path))
    return pairs


def _evaluate_one(
    stem: str, gt_path: str, pred_path: str
) -> Tuple[dict, List[dict]]:
    """Return ``(per_image_summary, per_disc_rows)`` for one image."""
    gt_discs, gt_affine = _extract_disc_voxels(gt_path)
    pred_discs, pred_affine = _extract_disc_voxels(pred_path)

    matched: List[Tuple[int, float]] = []
    fn_labels: List[int] = []
    fp_labels: List[int] = []
    rows: List[dict] = []

    for label in sorted(gt_discs):
        gt_vox = gt_discs[label]
        pred_vox = pred_discs.get(label)
        if pred_vox is None:
            fn_labels.append(label)
            rows.append(
                {"image": stem, "label": label, "status": "false_negative", "distance_mm": ""}
            )
            continue
        gt_w = _voxel_to_world(gt_vox, gt_affine)
        pred_w = _voxel_to_world(pred_vox, pred_affine)
        dist = float(np.linalg.norm(gt_w - pred_w))
        matched.append((label, dist))
        rows.append(
            {"image": stem, "label": label, "status": "matched", "distance_mm": f"{dist:.3f}"}
        )

    for label in sorted(pred_discs):
        if label in gt_discs:
            continue
        fp_labels.append(label)
        rows.append(
            {"image": stem, "label": label, "status": "false_positive", "distance_mm": ""}
        )

    summary = {
        "image": stem,
        "n_gt": len(gt_discs),
        "n_pred": len(pred_discs),
        "n_matched": len(matched),
        "n_false_negative": len(fn_labels),
        "n_false_positive": len(fp_labels),
        "false_negative_labels": sorted(fn_labels),
        "false_positive_labels": sorted(fp_labels),
        "matched": [{"label": lbl, "distance_mm": d} for lbl, d in matched],
    }
    return summary, rows


def _aggregate(per_image: List[dict]) -> dict:
    """Compute overall summary across every image."""
    all_distances = [m["distance_mm"] for p in per_image for m in p["matched"]]
    total_gt = sum(p["n_gt"] for p in per_image)
    total_pred = sum(p["n_pred"] for p in per_image)
    total_matched = sum(p["n_matched"] for p in per_image)
    total_fn = sum(p["n_false_negative"] for p in per_image)
    total_fp = sum(p["n_false_positive"] for p in per_image)

    # Per-label breakdown: how often is each disc label detected / missed?
    per_label: Dict[int, Dict[str, int]] = {}
    per_label_distances: Dict[int, List[float]] = {}
    for p in per_image:
        for m in p["matched"]:
            lbl = m["label"]
            per_label.setdefault(lbl, {"matched": 0, "fn": 0, "fp": 0})["matched"] += 1
            per_label_distances.setdefault(lbl, []).append(m["distance_mm"])
        for lbl in p["false_negative_labels"]:
            per_label.setdefault(lbl, {"matched": 0, "fn": 0, "fp": 0})["fn"] += 1
        for lbl in p["false_positive_labels"]:
            per_label.setdefault(lbl, {"matched": 0, "fn": 0, "fp": 0})["fp"] += 1

    per_label_stats = {}
    for lbl, counts in sorted(per_label.items()):
        dists = per_label_distances.get(lbl, [])
        per_label_stats[str(lbl)] = {
            **counts,
            "mean_distance_mm": float(np.mean(dists)) if dists else None,
            "median_distance_mm": float(np.median(dists)) if dists else None,
        }

    return {
        "n_images": len(per_image),
        "n_gt_discs": total_gt,
        "n_pred_discs": total_pred,
        "n_matched": total_matched,
        "n_false_negative": total_fn,
        "n_false_positive": total_fp,
        "recall": (total_matched / total_gt) if total_gt else None,
        "precision": (total_matched / total_pred) if total_pred else None,
        "distance_mm": {
            "n": len(all_distances),
            "mean": float(np.mean(all_distances)) if all_distances else None,
            "median": float(np.median(all_distances)) if all_distances else None,
            "p95": float(np.percentile(all_distances, 95)) if all_distances else None,
            "max": float(np.max(all_distances)) if all_distances else None,
        },
        "per_label": per_label_stats,
    }


def _fmt(value) -> str:
    return "" if value is None else f"{value:.4f}"


def _split_stem(stem: str) -> Tuple[str, str]:
    """Return ``(dataset_prefix, bids_name)``; prefix is ``""`` when absent."""
    if "__" in stem:
        prefix, name = stem.split("__", 1)
        return prefix + "__", name
    return "", stem


def _subject_of(stem: str) -> str:
    """``<dataset>__sub-XXX_ses-YY_T2w`` -> ``<dataset>__sub-XXX``."""
    prefix, name = _split_stem(stem)
    return prefix + name.split("_", 1)[0]


def _contrast_of(stem: str) -> str:
    """``<dataset>__sub-XXX_ses-YY_T2w`` -> ``T2w``."""
    _, name = _split_stem(stem)
    return name.rsplit("_", 1)[-1] if "_" in name else "unknown"


def _metric_row(overall: dict) -> dict:
    """Flatten an ``_aggregate`` result into the ``METRIC_FIELDS`` columns."""
    d = overall["distance_mm"]
    return {
        "n_images": overall["n_images"],
        "n_gt_discs": overall["n_gt_discs"],
        "n_pred_discs": overall["n_pred_discs"],
        "n_matched": overall["n_matched"],
        "n_false_negative": overall["n_false_negative"],
        "n_false_positive": overall["n_false_positive"],
        "recall": _fmt(overall["recall"]),
        "precision": _fmt(overall["precision"]),
        "distance_mm_mean": _fmt(d["mean"]),
        "distance_mm_median": _fmt(d["median"]),
        "distance_mm_p95": _fmt(d["p95"]),
        "distance_mm_max": _fmt(d["max"]),
    }


def _group_metrics(per_image: List[dict], group_fn) -> Dict[str, dict]:
    """``{group_name: _aggregate(images in group)}`` with groups from ``group_fn(stem)``."""
    groups: Dict[str, List[dict]] = {}
    for p in per_image:
        groups.setdefault(group_fn(p["image"]), []).append(p)
    return {name: _aggregate(groups[name]) for name in sorted(groups)}


def _write_json_atomic(path: str, obj: dict) -> None:
    """Write via a temp file + rename so concurrent readers never see a partial file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _write_csv_atomic(path: str, fieldnames: List[str], rows: List[dict]) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def evaluate(
    gt_dir: str,
    pred_dir: str,
    out_dir: str,
    run_name: str = "",
) -> None:
    """Compare every GT/pred pair under the two directories, write metrics."""
    os.makedirs(out_dir, exist_ok=True)
    run_name = run_name or os.path.basename(os.path.normpath(pred_dir))
    pairs = _discover_pairs(gt_dir, pred_dir)
    print(f"[eval] {len(pairs)} GT/pred pairs")

    per_image: List[dict] = []
    all_rows: List[dict] = []
    for stem, gt_path, pred_path in pairs:
        summary, rows = _evaluate_one(stem, gt_path, pred_path)
        per_image.append(summary)
        all_rows.extend(rows)

    overall = _aggregate(per_image)
    out = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pred_dir": os.path.abspath(pred_dir),
        "summary": overall,
        "per_subject": _group_metrics(per_image, _subject_of),
        "per_contrast": _group_metrics(per_image, _contrast_of),
        "per_image": per_image,
    }

    _write_json_atomic(os.path.join(out_dir, "metrics.json"), out)
    _write_csv_atomic(
        os.path.join(out_dir, "metrics.csv"),
        ["image", "label", "status", "distance_mm"],
        all_rows,
    )

    print(
        f"[eval] n_images={overall['n_images']}  "
        f"recall={overall['recall']:.3f}  "
        f"precision={overall['precision']:.3f}  "
        f"FN={overall['n_false_negative']}  FP={overall['n_false_positive']}"
        if overall["recall"] is not None and overall["precision"] is not None
        else f"[eval] n_images={overall['n_images']} (nothing to score)"
    )
    d = overall["distance_mm"]
    if d["n"]:
        print(
            f"[eval] distance mm: mean={d['mean']:.2f}  "
            f"median={d['median']:.2f}  p95={d['p95']:.2f}  max={d['max']:.2f}  n={d['n']}"
        )
    print(f"[eval] wrote {out_dir}/metrics.json + metrics.csv")


def aggregate_runs(eval_dir: str, out_dir: str = "") -> None:
    """Collect every ``<eval_dir>/metrics_*/metrics.json`` into one comparison.

    Writes to ``out_dir`` (defaults to ``eval_dir``):
      * ``evaluation.json``          — ``{run_name: {summary, per_subject, per_contrast, ...}}``
      * ``metrics.csv``              — one row per run
      * ``metrics_per_subject.csv``  — one row per (run, subject)
      * ``metrics_per_contrast.csv`` — one row per (run, contrast)

    Everything is rebuilt from the per-run folders on each call, so it is safe
    to re-run while (or after) several evaluations write their own folders.
    """
    out_dir = out_dir or os.path.join(eval_dir, "comparisons")
    os.makedirs(out_dir, exist_ok=True)

    runs: Dict[str, dict] = {}
    for name in sorted(os.listdir(eval_dir)):
        metrics_path = os.path.join(eval_dir, name, "metrics.json")
        if not name.startswith("metrics_") or not os.path.isfile(metrics_path):
            continue
        with open(metrics_path) as f:
            run = json.load(f)
        run_name = run.get("run_name") or name[len("metrics_"):]
        runs[run_name] = {
            "timestamp": run.get("timestamp"),
            "pred_dir": run.get("pred_dir"),
            "metrics_dir": os.path.abspath(os.path.join(eval_dir, name)),
            "summary": run["summary"],
            "per_subject": run.get("per_subject", {}),
            "per_contrast": run.get("per_contrast", {}),
        }
    print(f"[aggregate] {len(runs)} runs found in {eval_dir}")

    _write_json_atomic(os.path.join(out_dir, "evaluation.json"), runs)

    metrics = list(METRIC_FIELDS)
    _write_csv_atomic(
        os.path.join(out_dir, SUMMARY_FILES["overall"]),
        ["run_name", "timestamp"] + metrics,
        [
            {"run_name": rn, "timestamp": r["timestamp"], **_metric_row(r["summary"])}
            for rn, r in runs.items()
        ],
    )
    for key in ("subject", "contrast"):
        _write_csv_atomic(
            os.path.join(out_dir, SUMMARY_FILES[key]),
            [key, "run_name"] + metrics,
            sorted(
                (
                    {key: group, "run_name": rn, **_metric_row(agg)}
                    for rn, r in runs.items()
                    for group, agg in r[f"per_{key}"].items()
                ),
                key=lambda row: (row[key], row["run_name"]),
            ),
        )
    print(
        f"[aggregate] wrote {os.path.join(out_dir, 'evaluation.json')} + "
        + ", ".join(SUMMARY_FILES.values())
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare predicted disc-label NIfTIs with ground-truth NIfTIs."
    )
    parser.add_argument(
        "--gt-dir",
        required=True,
        help="Flat folder of ground-truth label NIfTIs named <stem>.nii[.gz].",
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Flat folder of prediction NIfTIs named <stem>_discs.nii[.gz].",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for metrics.json + metrics.csv (typically <eval>/metrics_<run_name>).",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Run identifier stored in metrics.json (defaults to the pred-dir name).",
    )
    args = parser.parse_args()
    evaluate(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
    )


def aggregate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate every metrics_*/metrics.json into one evaluation.json + comparison CSVs."
    )
    parser.add_argument(
        "--eval-dir",
        default=os.path.join(os.environ.get("DISCSLABEL_DATA", "."), "evaluation"),
        help="Folder holding the metrics_<run_name>/ folders (default: $DISCSLABEL_DATA/evaluation).",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Where to write evaluation.json + CSVs (default: --eval-dir/comparisons).",
    )
    args = parser.parse_args()
    aggregate_runs(eval_dir=args.eval_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
