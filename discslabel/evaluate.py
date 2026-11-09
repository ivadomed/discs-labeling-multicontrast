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
  * ``errors/<stem>.png`` — for every image with at least one error (missed
    disc, extra/mislabelled disc, or a disc farther than ``--tolerance-mm``),
    a sagittal preview with the ground truth and the prediction side by side.
    Needs the input images (``--image-dir``, default ``<gt_dir>/../images``).

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

from .parallel import make_executor, resolve_num_workers


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
    data = np.squeeze(np.rint(np.asarray(img.dataobj)).astype(np.int32))
    # One pass over the volume: gather every labelled voxel, then average the
    # coordinates per label value (instead of one full-volume scan per label).
    coords = np.argwhere(data > 0)
    discs: Dict[int, np.ndarray] = {}
    if len(coords):
        values = data[tuple(coords.T)]
        uniq, inverse = np.unique(values, return_inverse=True)
        inverse = inverse.ravel()
        counts = np.bincount(inverse).astype(np.float64)
        means = np.stack(
            [np.bincount(inverse, weights=coords[:, a]) for a in range(coords.shape[1])],
            axis=1,
        ) / counts[:, None]
        discs = {int(v): means[i] for i, v in enumerate(uniq)}
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


def _evaluate_pair(pair: Tuple[str, str, str]) -> Tuple[dict, List[dict]]:
    """Worker job: ``_evaluate_one`` on a ``(stem, gt_path, pred_path)`` tuple."""
    return _evaluate_one(*pair)


def _evaluate_one(
    stem: str, gt_path: str, pred_path: str
) -> Tuple[dict, List[dict]]:
    """Return ``(per_image_summary, per_disc_rows)`` for one image."""
    gt_discs, gt_affine = _extract_disc_voxels(gt_path)
    pred_discs, pred_affine = _extract_disc_voxels(pred_path)
    return compare_discs(stem, gt_discs, gt_affine, pred_discs, pred_affine)


def compare_discs(
    stem: str,
    gt_discs: Dict[int, np.ndarray],
    gt_affine: np.ndarray,
    pred_discs: Dict[int, np.ndarray],
    pred_affine: np.ndarray,
) -> Tuple[dict, List[dict]]:
    """Match ``{label: (3,) voxel}`` dicts by label value.

    Same output as ``_evaluate_one``; split out so training validation and
    post-processing tuning can score in-memory predictions without NIfTI I/O.
    """
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


def f1_at_tolerance(per_image: List[dict], tolerance_mm: float) -> float:
    """Label-aware F1 where a disc only counts when its label is right AND it
    lies within ``tolerance_mm`` of the GT. Off-by-one labelings therefore
    score as both a miss and a false positive, unlike plain recall/precision.
    """
    n_gt = sum(p["n_gt"] for p in per_image)
    n_pred = sum(p["n_pred"] for p in per_image)
    if n_gt + n_pred == 0:
        return 0.0
    hits = sum(
        1 for p in per_image for m in p["matched"] if m["distance_mm"] <= tolerance_mm
    )
    return 2.0 * hits / (n_gt + n_pred)


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


PREVIEW_MODES: Tuple[str, ...] = ("errors", "all", "none")
ERRORS_DIR = "errors"
PREVIEWS_DIR = "previews"


def _image_has_error(p: dict, tolerance_mm: float) -> bool:
    """Any missed disc, extra/mislabelled disc, or disc beyond ``tolerance_mm``."""
    return bool(
        p["n_false_negative"]
        or p["n_false_positive"]
        or any(m["distance_mm"] > tolerance_mm for m in p["matched"])
    )


def _discs_in_image_voxels(
    discs: Dict[int, np.ndarray], affine: np.ndarray, image_affine_inv: np.ndarray
) -> Dict[int, np.ndarray]:
    """Move ``{label: voxel}`` from a label volume's grid to the image's grid."""
    return {
        lbl: (image_affine_inv @ np.append(_voxel_to_world(v, affine), 1.0))[:3]
        for lbl, v in discs.items()
    }


def _draw_discs(ax, discs, colors, shape_yz) -> None:
    """Plot ``{label: (x, y, z)}`` RAS voxels on a sagittal panel (superior up, anterior left)."""
    Y, Z = shape_yz
    for lbl in sorted(discs):
        _x, y, z = discs[lbl]
        row, col = (Z - 1) - z, (Y - 1) - y
        color = colors.get(lbl, "lime")
        ax.plot(col, row, "o", color=color, markersize=8,
                markeredgecolor="black", markeredgewidth=0.8)
        ax.text(col + 6, row, str(lbl), color=color, fontsize=11,
                fontweight="bold", va="center")


def _save_error_preview(job: tuple) -> str:
    """Worker job: GT | prediction sagittal panels for one image, saved as PNG.

    Everything is drawn in the image's closest-canonical (RAS) voxel grid, so
    GT and prediction land on the same picture whatever their own grids. The
    slice is the median R-L position of the GT discs (prediction discs when
    there is no GT); discs off that slice are projected onto it, as in the
    inference previews.

    Colours — GT panel: green = found, orange = missed. Prediction panel:
    green = correct label within tolerance, red = beyond tolerance (number in
    the title), magenta = label absent from GT (extra / mislabelled).
    """
    stem, image_path, gt_path, pred_path, summary, tolerance_mm, save_path = job
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = nib.as_closest_canonical(nib.load(image_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    while data.ndim > 3:
        data = data[..., 0]
    inv = np.linalg.inv(img.affine)
    zooms = img.header.get_zooms()[:3]

    gt_discs, gt_aff = _extract_disc_voxels(gt_path)
    pred_discs, pred_aff = _extract_disc_voxels(pred_path)
    gt_v = _discs_in_image_voxels(gt_discs, gt_aff, inv)
    pred_v = _discs_in_image_voxels(pred_discs, pred_aff, inv)

    ref = gt_v or pred_v
    if ref:
        x = int(np.clip(np.rint(np.median([v[0] for v in ref.values()])), 0, data.shape[0] - 1))
    else:
        x = data.shape[0] // 2
    sl = data[x]  # (Y = P->A, Z = I->S)
    disp = np.ascontiguousarray(sl.T[::-1, ::-1])
    vmin, vmax = float(np.percentile(disp, 1.0)), float(np.percentile(disp, 99.0))
    if vmax <= vmin:
        vmin, vmax = float(disp.min()), float(disp.max() + 1e-6)

    dist = {m["label"]: m["distance_mm"] for m in summary["matched"]}
    gt_colors = {l: ("orange" if l not in dist else "lime") for l in gt_v}
    pred_colors = {
        l: ("magenta" if l not in dist else "red" if dist[l] > tolerance_mm else "lime")
        for l in pred_v
    }
    far = [f"{l} ({dist[l]:.0f}mm)" for l in sorted(dist) if dist[l] > tolerance_mm]

    aspect = float(zooms[2]) / float(zooms[1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    for ax, discs, colors, name in (
        (axes[0], gt_v, gt_colors, f"ground truth ({len(gt_v)})"),
        (axes[1], pred_v, pred_colors, f"prediction ({len(pred_v)})"),
    ):
        ax.imshow(disp, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
        _draw_discs(ax, discs, colors, sl.shape)
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    lines = [stem, f"F1@{tolerance_mm:g}mm = {_image_f1(summary, tolerance_mm):.2f}"]
    if summary["false_negative_labels"]:
        lines.append(f"missed: {summary['false_negative_labels']}")
    if summary["false_positive_labels"]:
        lines.append(f"extra: {summary['false_positive_labels']}")
    if far:
        more = f" (+{len(far) - 8} more)" if len(far) > 8 else ""
        lines.append(f"> {tolerance_mm:g}mm: {', '.join(far[:8])}{more}")
    lines.append("green = correct, orange = missed, red = too far, magenta = extra/mislabelled")
    fig.suptitle("\n".join(lines), fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _find_image(image_dir: str, stem: str) -> str:
    for ext in NIFTI_EXTS:
        path = os.path.join(image_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return ""


def write_previews(
    per_image: List[dict],
    pairs: List[Tuple[str, str, str]],
    image_dir: str,
    out_dir: str,
    tolerance_mm: float,
    mode: str = "errors",
    num_workers="auto",
) -> int:
    """Save GT | prediction previews into ``<out_dir>/errors`` (or ``previews`` for mode=all)."""
    if mode == "none":
        return 0
    if not image_dir or not os.path.isdir(image_dir):
        print(f"[eval] WARN: no image folder ({image_dir or 'unset'}); skipping previews")
        return 0
    folder = os.path.join(out_dir, ERRORS_DIR if mode == "errors" else PREVIEWS_DIR)
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):  # drop previews of a previous run
        if name.endswith(".png"):
            os.remove(os.path.join(folder, name))

    paths = {stem: (gt, pred) for stem, gt, pred in pairs}
    jobs = []
    for p in per_image:
        if mode == "errors" and not _image_has_error(p, tolerance_mm):
            continue
        image_path = _find_image(image_dir, p["image"])
        if not image_path:
            print(f"[eval] WARN: no image for {p['image']} in {image_dir}; no preview")
            continue
        gt, pred = paths[p["image"]]
        jobs.append((p["image"], image_path, gt, pred, p, tolerance_mm,
                     os.path.join(folder, f"{p['image']}.png")))

    num_workers = min(resolve_num_workers(num_workers), len(jobs))
    executor = make_executor(num_workers)
    try:
        results = map(_save_error_preview, jobs) if executor is None else executor.map(
            _save_error_preview, jobs
        )
        n = sum(1 for _ in results)
    finally:
        if executor is not None:
            executor.shutdown()
    print(f"[eval] wrote {n} preview(s) to {folder}")
    return n


def evaluate(
    gt_dir: str,
    pred_dir: str,
    out_dir: str,
    run_name: str = "",
    num_workers="auto",
    image_dir: str = "",
    previews: str = "errors",
    tolerance_mm: float = 10.0,
) -> None:
    """Compare every GT/pred pair under the two directories, write metrics.

    Pairs are scored in parallel over ``num_workers`` processes (0 = inline).
    """
    os.makedirs(out_dir, exist_ok=True)
    run_name = run_name or os.path.basename(os.path.normpath(pred_dir))
    pairs = _discover_pairs(gt_dir, pred_dir)
    num_workers = min(resolve_num_workers(num_workers), len(pairs))
    print(f"[eval] {len(pairs)} GT/pred pairs (num_workers={num_workers})")

    per_image: List[dict] = []
    all_rows: List[dict] = []
    executor = make_executor(num_workers)
    try:
        if executor is None:
            results = map(_evaluate_pair, pairs)
        else:
            chunksize = max(1, len(pairs) // (4 * num_workers))
            results = executor.map(_evaluate_pair, pairs, chunksize=chunksize)
        for summary, rows in results:
            per_image.append(summary)
            all_rows.extend(rows)
    finally:
        if executor is not None:
            executor.shutdown()

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

    if previews != "none":
        if not image_dir:
            image_dir = os.path.join(os.path.dirname(os.path.normpath(gt_dir)), "images")
        write_previews(per_image, pairs, image_dir, out_dir, tolerance_mm, previews, num_workers)


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


def _image_f1(p: dict, tolerance_mm: float) -> float:
    """``f1_at_tolerance`` of one image (1.0 when there is nothing to find or predict)."""
    denom = p["n_gt"] + p["n_pred"]
    if denom == 0:
        return 1.0
    hits = sum(1 for m in p["matched"] if m["distance_mm"] <= tolerance_mm)
    return 2.0 * hits / denom


def compare_runs(metrics_dirs: List[str], tolerance_mm: float = 10.0, out_dir: str = "") -> dict:
    """Side-by-side comparison of several ``metrics_<run>/metrics.json``.

    Prints the overall metrics (plus ``f1_at_tolerance``) of every run, with
    the difference to the FIRST run. Per image, ``f1_at_tolerance`` is
    compared with the first run on the images both runs scored; the counts of
    improved / worse images and the largest changes are printed, and
    ``comparison.csv`` (one row per image) is written to ``out_dir`` when
    given. The first run is the reference (e.g. without skeleton).
    """
    runs = []
    for d in metrics_dirs:
        with open(os.path.join(d, "metrics.json")) as f:
            m = json.load(f)
        per_image = m.get("per_image", [])
        runs.append(
            {
                "name": m.get("run_name") or os.path.basename(os.path.normpath(d)),
                "summary": m["summary"],
                "f1": f1_at_tolerance(per_image, tolerance_mm),
                "per_image": {p["image"]: p for p in per_image},
            }
        )

    def _overall(run: dict) -> Dict[str, float]:
        s = run["summary"]
        d = s["distance_mm"]
        return {
            f"f1_at_{tolerance_mm:g}mm": run["f1"],
            "recall": s["recall"],
            "precision": s["precision"],
            "n_pred_discs": s["n_pred_discs"],
            "n_matched": s["n_matched"],
            "n_false_negative": s["n_false_negative"],
            "n_false_positive": s["n_false_positive"],
            "distance_mm_mean": d["mean"],
            "distance_mm_median": d["median"],
            "distance_mm_p95": d["p95"],
            "distance_mm_max": d["max"],
        }

    tables = [_overall(r) for r in runs]
    width = max(14, *(len(r["name"]) for r in runs))
    header = f"{'metric':<20}" + "".join(f"{r['name']:>{width + 2}}" for r in runs)
    header += "".join(f"{'Δ ' + r['name'][-width + 2:]:>{width + 2}}" for r in runs[1:])
    print(f"[compare] {runs[0]['name']} is the reference; n_gt_discs={runs[0]['summary']['n_gt_discs']}")
    print(header)
    for key in tables[0]:
        vals = [t[key] for t in tables]
        line = f"{key:<20}" + "".join(
            f"{'' if v is None else (f'{v:.4f}' if isinstance(v, float) else v):>{width + 2}}" for v in vals
        )
        for v in vals[1:]:
            delta = None if v is None or vals[0] is None else v - vals[0]
            line += f"{'' if delta is None else (f'{delta:+.4f}' if isinstance(delta, float) else f'{delta:+d}'):>{width + 2}}"
        print(line)

    rows: List[dict] = []
    ref = runs[0]
    summary = {"runs": [r["name"] for r in runs], "overall": dict(zip([r["name"] for r in runs], tables))}
    for other in runs[1:]:
        common = sorted(set(ref["per_image"]) & set(other["per_image"]))
        changes = []
        for image in common:
            a = _image_f1(ref["per_image"][image], tolerance_mm)
            b = _image_f1(other["per_image"][image], tolerance_mm)
            changes.append((b - a, image, a, b))
            rows.append(
                {
                    "image": image,
                    "reference": ref["name"],
                    "run": other["name"],
                    "f1_reference": f"{a:.4f}",
                    "f1_run": f"{b:.4f}",
                    "delta": f"{b - a:+.4f}",
                }
            )
        better = [c for c in changes if c[0] > 1e-9]
        worse = [c for c in changes if c[0] < -1e-9]
        print(
            f"[compare] {other['name']} vs {ref['name']} on {len(common)} images: "
            f"{len(better)} better, {len(worse)} worse, {len(common) - len(better) - len(worse)} unchanged "
            f"(per-image F1@{tolerance_mm:g}mm)"
        )
        for title, items in (("worse", sorted(worse)[:10]), ("better", sorted(better, reverse=True)[:10])):
            for delta, image, a, b in items:
                print(f"    {title:6s} {delta:+.3f}  {a:.3f} -> {b:.3f}  {image}")
        summary[other["name"]] = {"n_images": len(common), "better": len(better), "worse": len(worse)}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        _write_csv_atomic(
            os.path.join(out_dir, "comparison.csv"),
            ["image", "reference", "run", "f1_reference", "f1_run", "delta"],
            rows,
        )
        _write_json_atomic(os.path.join(out_dir, "comparison.json"), summary)
        print(f"[compare] wrote {out_dir}/comparison.csv + comparison.json")
    return summary


def compare_main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the metrics of several evaluation runs (first one = reference)."
    )
    parser.add_argument(
        "--metrics-dirs",
        nargs="+",
        required=True,
        help="metrics_<run_name>/ folders written by discslabel_evaluate; the first is the reference.",
    )
    parser.add_argument("--tolerance-mm", type=float, default=10.0, help="F1 tolerance (default: 10 mm).")
    parser.add_argument("--out-dir", default="", help="Where to write comparison.csv / comparison.json.")
    args = parser.parse_args()
    compare_runs(args.metrics_dirs, args.tolerance_mm, args.out_dir)


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
    parser.add_argument(
        "--num-workers",
        default="auto",
        help="Worker processes used to score the pairs (default: auto; 0 = run inline).",
    )
    parser.add_argument(
        "--image-dir",
        default="",
        help="Input images <stem>.nii[.gz] for the previews (default: <gt-dir>/../images).",
    )
    parser.add_argument(
        "--previews",
        choices=PREVIEW_MODES,
        default="errors",
        help="GT | prediction sagittal PNGs: 'errors' (default) -> <out-dir>/errors for images "
        "with any error, 'all' -> <out-dir>/previews for every image, 'none' to skip.",
    )
    parser.add_argument(
        "--tolerance-mm",
        type=float,
        default=10.0,
        help="A matched disc farther than this counts as an error for the previews (default: 10).",
    )
    args = parser.parse_args()
    evaluate(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        num_workers=args.num_workers,
        image_dir=args.image_dir,
        previews=args.previews,
        tolerance_mm=args.tolerance_mm,
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
