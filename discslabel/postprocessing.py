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
* When ``skeleton.json`` (``discslabel_skeleton``) sits in the output folder,
  ``tune`` also scores the skeleton refinement (``discslabel.skeleton``) with
  the best decoding parameters for every ``tune_skeleton_z_thresholds`` value
  and stores the best one (and whether it beats plain decoding) under the
  ``"skeleton"`` key of ``postprocessing.json``.

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

from discslabel.dataset import _load_nifti, _z_score_normalize
from discslabel.decoding import extract_candidates, label_discs
from discslabel.evaluate import _aggregate, _extract_disc_voxels, compare_discs, f1_at_tolerance
from discslabel.parallel import BoundedSubmitter, imap_ordered, make_executor, resolve_num_workers, submit
from discslabel.skeleton import (
    SKELETON_FILE,
    SkeletonModel,
    affine_spacing,
    decode_with_skeleton,
    load_skeleton,
    skeleton_settings,
)


POSTPROCESSING_FILE = "postprocessing.json"
DEFAULT_PEAK_THRESHOLDS: Tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(2, 13))
DEFAULT_MIN_PEAK_DISTANCES: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0)
DEFAULT_TOLERANCE_MM = 10.0
DEFAULT_SKELETON_Z_THRESHOLDS: Tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0)


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


@dataclass
class PreloadedCase:
    """A validation case whose ground truth (and optionally model input) is
    loaded once, before training, instead of at every validation round."""

    stem: str
    entry: dict
    cached: bool
    data: Optional[np.ndarray]
    volume_shape: Tuple[int, int, int]
    gt_discs: Dict[int, np.ndarray]
    affine: np.ndarray


def _load_volume_norm(entry: dict, cache_root: Optional[str], split: str, cached: bool) -> np.ndarray:
    """Whole-volume z-scored image, from the ``.npy`` cache when available.

    The cached image is a rescaled float16 copy; the z-score is invariant to
    the scale, so only float16 rounding differs from the NIfTI path.
    """
    if cached:
        from discslabel.cache import case_paths

        image = np.load(case_paths(cache_root, split, entry)["image"]).astype(np.float32)
    else:
        image = _load_nifti(entry["image"])
    return _z_score_normalize(image)


def _preload_case(job) -> PreloadedCase:
    """Worker job for ``preload_cases``."""
    entry, cache_root, split, cached, reduce_fn, keep_in_memory = job
    volume = _load_volume_norm(entry, cache_root, split, cached)
    data = None
    if keep_in_memory:
        data = reduce_fn(volume) if reduce_fn is not None else volume
        data = np.ascontiguousarray(data)
    gt_discs, affine = _extract_disc_voxels(entry["label"])
    return PreloadedCase(
        stem=_stem(entry["image"]),
        entry=entry,
        cached=cached,
        data=data,
        volume_shape=tuple(int(s) for s in volume.shape),
        gt_discs=gt_discs,
        affine=affine,
    )


def preload_cases(
    manifest_path: str,
    split: str,
    max_images: Optional[int] = None,
    reduce_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    keep_in_memory: bool = True,
    cache_root: Optional[str] = None,
    spine_disc_min_label: int = 60,
    num_workers: int = 0,
    desc: str = "preload validation",
) -> List[PreloadedCase]:
    """Load the ground truth of every labelled image of ``split`` once.

    With ``keep_in_memory`` the z-scored model input is kept too, after
    ``reduce_fn`` (e.g. the 2D pipeline keeps only the mid-sagittal slice
    stack, a few MB per image); otherwise ``collect_preloaded_cases`` reloads
    the volume at each call (fast when it is in the ``.npy`` cache).
    ``reduce_fn`` must be picklable when ``num_workers > 0``.
    """
    from discslabel.cache import is_cached

    with open(manifest_path) as f:
        entries = [e for e in json.load(f)[split] if e.get("label")]
    if max_images is not None:
        entries = entries[: int(max_images)]

    jobs = []
    for entry in entries:
        cached = cache_root is not None and is_cached(cache_root, split, entry, spine_disc_min_label)
        jobs.append((entry, cache_root, split, cached, reduce_fn, keep_in_memory))

    executor = make_executor(num_workers)
    try:
        return list(
            tqdm.tqdm(
                imap_ordered(executor, _preload_case, jobs, max(2, 2 * num_workers)),
                total=len(jobs),
                desc=desc,
            )
        )
    finally:
        if executor is not None:
            executor.shutdown()


def collect_preloaded_cases(
    preloaded: Sequence[PreloadedCase],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    min_threshold: float,
    cache_root: Optional[str] = None,
    split: str = "valid",
    desc: str = "validation",
) -> List[ValidationCase]:
    """``collect_cases`` on ``preload_cases`` output: no NIfTI is re-read.

    ``predict_fn`` receives the kept model input (or the reloaded z-scored
    volume when it was not kept in memory).
    """
    cases = []
    for case in tqdm.tqdm(preloaded, desc=desc):
        data = case.data
        if data is None:
            data = _load_volume_norm(case.entry, cache_root, split, case.cached)
        heatmap = predict_fn(data)
        cases.append(
            ValidationCase(
                stem=case.stem,
                candidates=extract_candidates(heatmap, min_threshold),
                volume_shape=case.volume_shape,
                gt_discs=case.gt_discs,
                affine=case.affine,
            )
        )
        del data, heatmap
    return cases


def score_cases(
    cases: Sequence[ValidationCase],
    to_voxels: Callable,
    peak_threshold: float,
    min_peak_distance: float,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    skeleton: Optional[SkeletonModel] = None,
    skeleton_kwargs: Optional[dict] = None,
) -> dict:
    """``_aggregate`` summary of all cases plus ``f1_at_tolerance``.

    With ``skeleton``, decoding is followed by the skeleton correction
    (``skeleton_kwargs`` are passed to ``refine_chain``).
    """
    per_image = []
    for case in cases:
        if skeleton is None:
            labeled = label_discs(case.candidates, peak_threshold, min_peak_distance)
        else:
            labeled, _report = decode_with_skeleton(
                case.candidates,
                peak_threshold,
                min_peak_distance,
                affine_spacing(case.affine, skeleton.projection),
                skeleton,
                mode="correct",
                **(skeleton_kwargs or {}),
            )
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


def _skeleton_row(
    cases: Sequence[ValidationCase],
    to_voxels: Callable,
    thr: float,
    dist: float,
    tolerance_mm: float,
    skeleton: SkeletonModel,
    skeleton_kwargs: dict,
) -> dict:
    """Worker job: score the skeleton correction at one ``z_threshold``."""
    s = score_cases(cases, to_voxels, thr, dist, tolerance_mm, skeleton, skeleton_kwargs)
    return {
        "z_threshold": skeleton_kwargs["z_threshold"],
        "f1_at_tolerance": s["f1_at_tolerance"],
        "recall": s["recall"],
        "precision": s["precision"],
        "distance_mm_mean": s["distance_mm"]["mean"],
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
    projection: Optional[str] = None,
    skeleton_path: Optional[str] = None,
) -> dict:
    """Grid-search the decoding parameters and write ``postprocessing.json``.

    ``projection`` (``"3d"`` / ``"sagittal"``) enables the skeleton search
    when ``skeleton_path`` (default: ``<out_dir>/skeleton.json``) exists.

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
        skeleton_result = None
        skeleton_path = skeleton_path or os.path.join(out_dir, SKELETON_FILE)
        if projection is not None and os.path.isfile(skeleton_path):
            skeleton_result = _tune_skeleton(
                cfg, cases, to_voxels, best, tolerance,
                load_skeleton(skeleton_path, projection), skeleton_path, executor,
            )
        elif projection is not None:
            print(f"[tune] no skeleton at {skeleton_path}; skipping skeleton tuning")
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
    if skeleton_result is not None:
        result["skeleton"] = skeleton_result
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
    if skeleton_result is not None:
        print(
            f"[tune] skeleton z_threshold={skeleton_result['z_threshold']} "
            f"F1@{tolerance:g}mm={skeleton_result['f1_at_tolerance']:.4f} "
            f"(without skeleton: {result['f1_at_tolerance']:.4f}"
            f"{'' if skeleton_result['improves'] else ' -> skeleton does NOT help on ' + split})"
        )
    print(f"[tune] wrote {os.path.join(out_dir, POSTPROCESSING_FILE)} + postprocessing_grid.csv")
    return result


def _tune_skeleton(
    cfg: dict,
    cases: Sequence[ValidationCase],
    to_voxels: Callable,
    best: dict,
    tolerance: float,
    skeleton: SkeletonModel,
    skeleton_path: str,
    executor,
) -> dict:
    """Score the skeleton correction for each ``z_threshold`` at the best decoding parameters."""
    z_values = [
        float(z) for z in cfg.get("tune_skeleton_z_thresholds", DEFAULT_SKELETON_Z_THRESHOLDS)
    ]
    base = skeleton_settings(cfg)
    futures = [
        submit(
            executor, _skeleton_row, list(cases), to_voxels,
            best["peak_threshold"], best["min_peak_distance"], tolerance,
            skeleton, {**base, "z_threshold": z},
        )
        for z in z_values
    ]
    rows = [f.result() for f in tqdm.tqdm(futures, desc="skeleton search")]
    top = max(rows, key=lambda r: (r["f1_at_tolerance"], -(r["distance_mm_mean"] or float("inf"))))
    return {
        "path": os.path.abspath(skeleton_path),
        "projection": skeleton.projection,
        "z_threshold": top["z_threshold"],
        "f1_at_tolerance": top["f1_at_tolerance"],
        "recall": top["recall"],
        "precision": top["precision"],
        "distance_mm_mean": top["distance_mm_mean"],
        "improves": bool(top["f1_at_tolerance"] > best["f1_at_tolerance"]),
        "grid": rows,
    }


def resolve_skeleton(cfg: dict, weights_path: str, mode: str, skeleton_path: Optional[str], projection: str):
    """``(SkeletonModel, refine kwargs, source)`` for inference, or ``(None, None, None)`` when off.

    ``skeleton_path`` defaults to ``skeleton.json`` next to the weights; the
    ``z_threshold`` tuned by ``discslabel_tune`` (``postprocessing.json``)
    wins over the config value.
    """
    if mode == "off":
        return None, None, None
    weights_dir = os.path.dirname(os.path.abspath(weights_path))
    path = skeleton_path or os.path.join(weights_dir, SKELETON_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"--skeleton {mode} needs a skeleton file; {path} not found (run discslabel_skeleton)."
        )
    tuned = None
    pp_path = os.path.join(weights_dir, POSTPROCESSING_FILE)
    if os.path.isfile(pp_path):
        with open(pp_path) as f:
            tuned = json.load(f).get("skeleton")
        if tuned and not tuned.get("improves", True):
            print(
                f"[skeleton] WARNING: on the tuning split the skeleton correction did not "
                f"improve F1 ({pp_path})."
            )
    return load_skeleton(path, projection), skeleton_settings(cfg, tuned), path


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
