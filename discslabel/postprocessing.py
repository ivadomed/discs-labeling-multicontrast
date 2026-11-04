"""Validation scoring and post-processing tuning, shared by 2D and 3D.

* ``collect_cases`` runs the model once per validation image (stitched
  sliding window, same as inference) and keeps only the decoding candidates.
  With a worker pool, image loading and peak extraction run in worker
  processes while the main process keeps the GPU busy.
* ``score_cases`` decodes those candidates with a given ``peak_threshold`` /
  ``min_peak_distance`` and scores them with ``discslabel.evaluate``.
* ``tune`` grid-searches both parameters on the validation split (nnU-Net
  determines its post-processing empirically the same way) and writes
  ``postprocessing.json`` next to the weights; inference picks it up
  automatically through ``resolve_postprocessing``. The grid points are
  scored in parallel across the worker pool.

Scores use ``f1_at_tolerance``: a disc is a hit only when its label is right
AND it lies within ``tolerance_mm`` of the ground truth.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tqdm

from .dataset import _load_nifti, _z_score_normalize
from .decoding import extract_candidates, label_discs
from .evaluate import _aggregate, _extract_disc_voxels, compare_discs, f1_at_tolerance
from .parallel import BoundedSubmitter, imap_ordered, make_executor, resolve_num_workers, submit


POSTPROCESSING_FILE = "postprocessing.json"
DEFAULT_PEAK_THRESHOLDS: Tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(2, 13))
DEFAULT_MIN_PEAK_DISTANCES: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
DEFAULT_TOLERANCE_MM = 10.0


@dataclass
class ValidationCase:
    stem: str
    candidates: Dict[str, object]
    volume_shape: Tuple[int, int, int]
    gt_discs: Dict[int, np.ndarray]
    affine: np.ndarray


def _stem(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _load_case(entry: dict) -> Tuple[str, np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    """Worker job: z-scored volume + ground-truth discs for one manifest entry."""
    volume = _z_score_normalize(_load_nifti(entry["image"]))
    gt_discs, affine = _extract_disc_voxels(entry["label"])
    return _stem(entry["image"]), volume, gt_discs, affine


def collect_cases(
    manifest_path: str,
    split: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    min_threshold: float,
    max_images: Optional[int] = None,
    desc: str = "validation",
    executor=None,
    num_workers: int = 0,
) -> List[ValidationCase]:
    """Run ``predict_fn`` on every labelled image of ``split``.

    ``predict_fn(volume_norm)`` returns the stitched heatmap; candidates are
    then extracted with ``extract_candidates(heatmap, min_threshold)``.

    Images in the manifest are already reoriented + resampled, so only the
    whole-volume z-score of inference is applied. The ground-truth label
    shares the image grid, so one affine serves both.

    With ``executor`` (a pool of ``num_workers`` processes), loading the
    next images and extracting the peaks of the previous heatmaps run in
    the workers while the main process predicts the current image.
    """
    with open(manifest_path) as f:
        entries = [e for e in json.load(f)[split] if e.get("label")]
    if max_images is not None:
        entries = entries[: int(max_images)]

    lookahead = max(2, 2 * num_workers)
    extractor = BoundedSubmitter(executor, max_pending=lookahead)
    meta = []
    for stem, volume, gt_discs, affine in tqdm.tqdm(
        imap_ordered(executor, _load_case, entries, lookahead),
        total=len(entries),
        desc=desc,
    ):
        heatmap = predict_fn(volume)
        extractor.submit(extract_candidates, heatmap, min_threshold)
        meta.append((stem, tuple(int(s) for s in volume.shape), gt_discs, affine))
        del volume, heatmap

    candidates = extractor.drain()
    return [
        ValidationCase(
            stem=stem, candidates=cand, volume_shape=shape, gt_discs=gt, affine=affine
        )
        for (stem, shape, gt, affine), cand in zip(meta, candidates)
    ]


def score_cases(
    cases: Sequence[ValidationCase],
    to_voxels: Callable,
    peak_threshold: float,
    min_peak_distance: float,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
) -> dict:
    """``_aggregate`` summary of all cases plus ``f1_at_tolerance``."""
    per_image = []
    for case in cases:
        labeled = label_discs(case.candidates, peak_threshold, min_peak_distance)
        pred = to_voxels(labeled, case.volume_shape)
        summary, _rows = compare_discs(case.stem, case.gt_discs, case.affine, pred, case.affine)
        per_image.append(summary)
    overall = _aggregate(per_image)
    overall.pop("per_label", None)
    overall["f1_at_tolerance"] = f1_at_tolerance(per_image, tolerance_mm)
    overall["tolerance_mm"] = float(tolerance_mm)
    return overall


def _grid_row(
    cases: Sequence[ValidationCase],
    to_voxels: Callable,
    thr: float,
    dist: float,
    tolerance_mm: float,
) -> dict:
    """Worker job: score one ``(peak_threshold, min_peak_distance)`` pair."""
    s = score_cases(cases, to_voxels, thr, dist, tolerance_mm)
    return {
        "peak_threshold": float(thr),
        "min_peak_distance": float(dist),
        "f1_at_tolerance": s["f1_at_tolerance"],
        "recall": s["recall"],
        "precision": s["precision"],
        "distance_mm_mean": s["distance_mm"]["mean"],
        "n_false_negative": s["n_false_negative"],
        "n_false_positive": s["n_false_positive"],
    }


def grid_search(
    cases: Sequence[ValidationCase],
    to_voxels: Callable,
    peak_thresholds: Sequence[float],
    min_peak_distances: Sequence[float],
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    executor=None,
) -> Tuple[dict, List[dict]]:
    """Score every ``(peak_threshold, min_peak_distance)`` pair.

    Pairs are scored in parallel when ``executor`` is given (``to_voxels``
    must then be a module-level function so it can be pickled).
    Returns ``(best_row, all_rows)``; best = highest ``f1_at_tolerance``,
    ties broken by lower mean distance.
    """
    cases = list(cases)
    futures = [
        submit(executor, _grid_row, cases, to_voxels, thr, dist, tolerance_mm)
        for thr in peak_thresholds
        for dist in min_peak_distances
    ]
    rows: List[dict] = [f.result() for f in tqdm.tqdm(futures, desc="grid search")]

    def _key(row):
        mean = row["distance_mm_mean"]
        return (row["f1_at_tolerance"], -(mean if mean is not None else float("inf")))

    return max(rows, key=_key), rows


def tune(
    cfg: dict,
    manifest_path: str,
    out_dir: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    to_voxels: Callable,
    split: str = "valid",
    num_workers=None,
) -> dict:
    """Grid-search the decoding parameters and write ``postprocessing.json``.

    ``predict_fn(volume_norm)`` must return the stitched heatmap; candidates
    are extracted once per image at the lowest threshold of the grid.
    ``num_workers`` (default: ``cfg["inference_num_workers"]``, ``"auto"``)
    CPU processes load images, extract peaks and score the grid.
    """
    thresholds = [float(t) for t in cfg.get("tune_peak_thresholds", DEFAULT_PEAK_THRESHOLDS)]
    distances = [float(d) for d in cfg.get("tune_min_peak_distances", DEFAULT_MIN_PEAK_DISTANCES)]
    tolerance = float(cfg.get("val_tolerance_mm", DEFAULT_TOLERANCE_MM))
    min_thr = min(thresholds)
    if num_workers is None:
        num_workers = cfg.get("inference_num_workers", "auto")
    num_workers = resolve_num_workers(num_workers)
    print(f"[tune] num_workers={num_workers}")

    default_thr = float(cfg.get("peak_threshold", 0.3))
    default_dist = float(cfg.get("inference_min_peak_distance", 10.0))

    executor = make_executor(num_workers)
    try:
        cases = collect_cases(
            manifest_path,
            split,
            predict_fn,
            min_thr,
            desc=f"predict {split}",
            executor=executor,
            num_workers=num_workers,
        )
        if not cases:
            raise RuntimeError(f"No labelled images in split '{split}' of {manifest_path}.")
        best, rows = grid_search(
            cases, to_voxels, thresholds, distances, tolerance, executor=executor
        )
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    default = score_cases(cases, to_voxels, default_thr, default_dist, tolerance)

    os.makedirs(out_dir, exist_ok=True)
    result = {
        "peak_threshold": best["peak_threshold"],
        "min_peak_distance": best["min_peak_distance"],
        "f1_at_tolerance": best["f1_at_tolerance"],
        "recall": best["recall"],
        "precision": best["precision"],
        "distance_mm_mean": best["distance_mm_mean"],
        "tolerance_mm": tolerance,
        "split": split,
        "n_images": len(cases),
        "config_default": {
            "peak_threshold": default_thr,
            "min_peak_distance": default_dist,
            "f1_at_tolerance": default["f1_at_tolerance"],
        },
    }
    with open(os.path.join(out_dir, POSTPROCESSING_FILE), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "postprocessing_grid.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(
        f"[tune] best peak_threshold={result['peak_threshold']} "
        f"min_peak_distance={result['min_peak_distance']} "
        f"F1@{tolerance:g}mm={result['f1_at_tolerance']:.4f} "
        f"(config default {default_thr}/{default_dist}: {default['f1_at_tolerance']:.4f})"
    )
    print(f"[tune] wrote {os.path.join(out_dir, POSTPROCESSING_FILE)} + postprocessing_grid.csv")
    return result


def resolve_postprocessing(cfg: dict, weights_path: str) -> Tuple[float, float, str]:
    """``(peak_threshold, min_peak_distance, source)`` for inference.

    Uses ``postprocessing.json`` next to the weights when present (written
    by ``discslabel_tune`` / ``discslabel_tune_3d``), else the config values.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(weights_path)), POSTPROCESSING_FILE)
    if os.path.isfile(path):
        with open(path) as f:
            tuned = json.load(f)
        return float(tuned["peak_threshold"]), float(tuned["min_peak_distance"]), path
    return (
        float(cfg.get("peak_threshold", 0.3)),
        float(cfg.get("inference_min_peak_distance", 10.0)),
        "config",
    )
