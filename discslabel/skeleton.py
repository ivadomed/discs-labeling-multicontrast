"""Anatomical skeleton prior on inter-disc distances, shared by 2D and 3D.

Fit (``discslabel_skeleton``, on the training labels)
-----------------------------------------------------
For every pair of adjacent discs ``(l, l+1)`` we collect the distance between
them over the training subjects and build a *template*: the median gap
``mu_l`` of a median-sized spine. Spines differ in size (children, short or
tall adults), so each subject gets a scale ``s = median_l(d_l / mu_l)`` and
the spread of every gap is measured on the scale-normalised log ratio
``log(d_l / (s * mu_l))`` (robust sigma from the MAD, leave-one-out scale so
a gap does not normalise itself). We also store the distribution of the
turning angle of the chain at each disc and of the subject scale. The
result is ``skeleton.json`` next to the weights, with two projections:
``3d`` (3D pipeline) and ``sagittal`` (2D pipeline, the R-L component of the
gaps is dropped since the 2D model only sees the mid-sagittal plane).

Refinement (inference, optional)
--------------------------------
The first labelling step (``decoding.label_discs``) numbers the peaks one by
one away from the anchor, so a missed disc shifts every label past it and a
spurious peak does the opposite. ``refine_chain`` instead looks for the most
plausible labelling of the peak chain under the skeleton, with dynamic
programming over ``(peak, label)`` states:

* the anchor peak keeps the anchor label (it is the trusted reference);
* each kept pair of peaks labelled ``l < m`` costs ``z^2 / 2`` where ``z`` is
  the deviation of their distance from the template distance between ``l``
  and ``m`` (scaled to the subject);
* every skipped label (missed disc) and every dropped peak (off / spurious
  disc) costs a fixed penalty, derived from ``z_threshold`` so that a
  correction is only made when it explains the geometry better than a
  ``z_threshold``-sigma deviation would.

Missed discs are then placed between their kept neighbours by monotone
interpolation along the chain, parameterised by the template position of
each label.

Soft matching (``soft_floor``, ``"correct"`` mode only)
-------------------------------------------------------
A disc whose heatmap blob stays just under ``peak_threshold`` never reaches
the chain above, and can only come back as an interpolated guess. With
``soft_floor`` the chain is built from the *soft candidates* instead
(``decoding.soft_disc_candidates``: local-mass score, down to
``soft_floor``) and each peak carries a confidence weight ``w`` in [0, 1]
(0 at the floor, 1 from ``peak_threshold`` up). The fixed penalties become
evidence-dependent: dropping a peak costs ``w * drop_penalty`` and keeping
it costs ``(1 - w) * keep_penalty``. A confident peak behaves exactly as
before; a weak one is kept only where the skeleton expects a disc (it is
never worse than imputing that disc, so a weak peak in a hole wins over
interpolation, while a weak peak past the end of the chain needs a
substantial score). Kept weak peaks are reported as ``recovered`` and raise
the ``weak_discs`` flag.

Quality control flags are reported for each image (``no_anchor``,
``too_few_discs``, ``abnormal_gap``, ``abnormal_angle``, ``abnormal_scale``,
``missed_discs``, ``outlier_discs``, ``relabeled``, ``weak_discs``) so problematic
predictions can be reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tqdm

from discslabel.decoding import (
    DEFAULT_SOFT_FLOOR,
    MAX_DISC_LABEL,
    MIN_DISC_LABEL,
    decode_chain,
    decode_chain_soft,
)
from discslabel.evaluate import _extract_disc_voxels
from discslabel.parallel import imap_ordered, make_executor, resolve_num_workers


SKELETON_FILE = "skeleton.json"
PROJECTIONS = ("3d", "sagittal")
# Grid axes kept by each projection; the prepared grid is RPI: (R-L, P-A, I-S).
PROJECTION_AXES = {"3d": (0, 1, 2), "sagittal": (1, 2)}

SKELETON_MODES = ("off", "flag", "correct")
DEFAULT_Z_THRESHOLD = 3.0
DEFAULT_MAX_MISSING = 3
DEFAULT_MAX_SKIPPED = 3
# Cost of a missed disc relative to a dropped peak. Below 1 so a disc beyond a
# single missed one is kept (and the gap filled) rather than dropped; above
# 0.5 so a far-away end peak is dropped rather than explained by 2+ misses.
MISS_TO_DROP_RATIO = 0.75
# Soft matching: cost of keeping a zero-confidence peak relative to dropping a
# confident one. Equal to the miss ratio, so using a weak peak where a disc
# is expected never costs more than imputing that disc.
KEEP_TO_DROP_RATIO = MISS_TO_DROP_RATIO
# Soft matching: most consecutive WEAK peaks (weight < 1) that can be dropped
# between two kept ones, on top of ``max_skipped`` confident peaks.
SOFT_MAX_WEAK_SKIPPED = 6

MIN_SAMPLES = 10
MIN_LOG_SIGMA = 0.04
MAD_TO_SIGMA = 1.4826
# Per-image false-flag rate of the remaining-gap check on a normal spine: the
# flag threshold is Bonferroni-corrected for the number of gaps tested.
GAP_FLAG_ALPHA = 0.01
# Normal subject scales: median +- this many robust sigmas (in log space).
SCALE_N_SIGMA = 4.0


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------


def _subject_discs(label_path: str) -> Dict[int, List[float]]:
    """Worker job: ``{label: [d, h, w] in mm on the prepared grid}``."""
    discs, affine = _extract_disc_voxels(label_path)
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    return {
        int(l): (np.asarray(v, dtype=np.float64) * spacing).tolist()
        for l, v in discs.items()
        if MIN_DISC_LABEL <= int(l) <= MAX_DISC_LABEL
    }


def _gap_table(subjects: Sequence[Dict[int, np.ndarray]]) -> np.ndarray:
    """``(n_subjects, MAX+1)`` distances ``l -> l+1`` (NaN when not observed)."""
    table = np.full((len(subjects), MAX_DISC_LABEL + 1), np.nan)
    for i, discs in enumerate(subjects):
        for l in range(MIN_DISC_LABEL, MAX_DISC_LABEL):
            if l in discs and l + 1 in discs:
                table[i, l] = float(np.linalg.norm(discs[l + 1] - discs[l]))
    return table


def _column_medians(table: np.ndarray) -> np.ndarray:
    """Median of the finite values of every column (NaN for empty columns)."""
    out = np.full(table.shape[1], np.nan)
    for l in range(table.shape[1]):
        col = table[:, l][np.isfinite(table[:, l])]
        if len(col):
            out[l] = np.median(col)
    return out


def _fill_template(mu: np.ndarray) -> np.ndarray:
    """Interpolate (in log space) the template gaps never observed in training."""
    labels = np.arange(MIN_DISC_LABEL, MAX_DISC_LABEL)
    known = np.isfinite(mu[labels])
    if not known.any():
        raise RuntimeError("No pair of adjacent discs found in the training labels.")
    out = mu.copy()
    out[labels] = np.exp(np.interp(labels, labels[known], np.log(mu[labels][known])))
    return out


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def _fit_projection(subjects_mm: Sequence[Dict[int, np.ndarray]], projection: str) -> dict:
    axes = list(PROJECTION_AXES[projection])
    subjects = [{l: np.asarray(p)[axes] for l, p in s.items()} for s in subjects_mm]
    table = _gap_table(subjects)
    observed = np.isfinite(table)

    # Template = median gap of a median-sized spine; two passes so the subject
    # scales and the template agree with each other.
    mu = _fill_template(_column_medians(table))
    for _ in range(2):
        ratios = table / mu[None, :]
        scales = np.array(
            [np.median(r[np.isfinite(r)]) if np.isfinite(r).any() else np.nan for r in ratios]
        )
        with np.errstate(invalid="ignore"):
            normalised = table / scales[:, None]
        mu = _fill_template(_column_medians(normalised))

    # Spread of each gap around the template, with a leave-one-out scale.
    log_dev: Dict[int, List[float]] = {l: [] for l in range(MIN_DISC_LABEL, MAX_DISC_LABEL)}
    ratios = table / mu[None, :]
    for r in ratios:
        idx = np.flatnonzero(np.isfinite(r))
        if len(idx) < 3:
            continue
        for l in idx:
            others = r[idx[idx != l]]
            log_dev[int(l)].append(float(np.log(r[l] / np.median(others))))
    pooled = np.concatenate([np.asarray(v) for v in log_dev.values() if v] or [np.zeros(1)])
    pooled_sigma = max(MAD_TO_SIGMA * float(np.median(np.abs(pooled - np.median(pooled)))), MIN_LOG_SIGMA)

    gaps = {}
    for l in range(MIN_DISC_LABEL, MAX_DISC_LABEL):
        col = table[:, l][observed[:, l]]
        dev = np.asarray(log_dev[l])
        if len(dev) >= MIN_SAMPLES:
            sigma = max(MAD_TO_SIGMA * float(np.median(np.abs(dev - np.median(dev)))), MIN_LOG_SIGMA)
        else:
            # Rarely observed gap: be lenient.
            sigma = 1.5 * pooled_sigma
        gaps[str(l)] = {
            "n": int(len(col)),
            "template_mm": float(mu[l]),
            "median_mm": float(np.median(col)) if len(col) else None,
            "p05_mm": float(np.percentile(col, 5)) if len(col) else None,
            "p95_mm": float(np.percentile(col, 95)) if len(col) else None,
            "log_sigma": float(sigma),
        }

    angles = {}
    for l in range(MIN_DISC_LABEL + 1, MAX_DISC_LABEL):
        vals = [
            _angle_deg(s[l] - s[l - 1], s[l + 1] - s[l])
            for s in subjects
            if l - 1 in s and l in s and l + 1 in s
        ]
        angles[str(l)] = {
            "n": len(vals),
            "p50_deg": float(np.percentile(vals, 50)) if vals else None,
            "p95_deg": float(np.percentile(vals, 95)) if vals else None,
            "p99_deg": float(np.percentile(vals, 99)) if vals else None,
            # Flag limit: beyond any training spine (robust to the ~24 tests per image).
            "max_deg": float(np.max(vals)) if vals else None,
        }

    valid_scales = scales[np.isfinite(scales)]
    if not len(valid_scales):
        raise RuntimeError("No training subject with two adjacent labelled discs.")
    log_s = np.log(valid_scales)
    log_med = float(np.median(log_s))
    log_sig = max(MAD_TO_SIGMA * float(np.median(np.abs(log_s - log_med))), MIN_LOG_SIGMA)
    return {
        "gaps": gaps,
        "angles": angles,
        "scale": {
            "n": int(len(valid_scales)),
            "p01": float(np.percentile(valid_scales, 1)),
            "p50": float(np.percentile(valid_scales, 50)),
            "p99": float(np.percentile(valid_scales, 99)),
            "low": float(np.exp(log_med - SCALE_N_SIGMA * log_sig)),
            "high": float(np.exp(log_med + SCALE_N_SIGMA * log_sig)),
        },
        "pooled_log_sigma": float(pooled_sigma),
    }


def fit_skeleton(
    manifest_path: str,
    splits: Sequence[str] = ("train",),
    num_workers=0,
) -> dict:
    """Build the skeleton from the ground-truth labels of ``splits``."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    label_paths = [e["label"] for split in splits for e in manifest.get(split, []) if e.get("label")]
    if not label_paths:
        raise RuntimeError(f"No labelled images in splits {list(splits)} of {manifest_path}.")

    num_workers = min(resolve_num_workers(num_workers), len(label_paths))
    executor = make_executor(num_workers)
    try:
        subjects = [
            {l: np.asarray(p) for l, p in d.items()}
            for d in tqdm.tqdm(
                imap_ordered(executor, _subject_discs, label_paths, max(2, 4 * num_workers)),
                total=len(label_paths),
                desc="skeleton",
            )
        ]
    finally:
        if executor is not None:
            executor.shutdown()

    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "manifest": os.path.abspath(manifest_path),
        "splits": list(splits),
        "n_images": len(subjects),
        "n_discs": int(sum(len(s) for s in subjects)),
        "label_range": [MIN_DISC_LABEL, MAX_DISC_LABEL],
        "projections": {p: _fit_projection(subjects, p) for p in PROJECTIONS},
    }


# --------------------------------------------------------------------------
# Model used at decoding time
# --------------------------------------------------------------------------


@dataclass
class SkeletonModel:
    """Arrays indexed by disc label; picklable so it can go to worker pools."""

    projection: str
    cum: np.ndarray  # template position of each label (cum[l+1] - cum[l] = mu_l)
    cum_var: np.ndarray  # cumulative (mu_l * sigma_l)^2
    angle_limit: np.ndarray  # turning-angle limit per label (NaN = unknown)
    scale_range: Tuple[float, float]

    @property
    def axes(self) -> Tuple[int, ...]:
        return PROJECTION_AXES[self.projection]

    @classmethod
    def from_dict(cls, data: dict, projection: str) -> "SkeletonModel":
        proj = data["projections"][projection]
        n = MAX_DISC_LABEL + 2
        mu = np.zeros(n)
        sig = np.zeros(n)
        for l in range(MIN_DISC_LABEL, MAX_DISC_LABEL):
            g = proj["gaps"][str(l)]
            mu[l] = g["template_mm"]
            sig[l] = g["log_sigma"]
        cum = np.zeros(n)
        cum_var = np.zeros(n)
        for l in range(MIN_DISC_LABEL, MAX_DISC_LABEL + 1):
            cum[l + 1] = cum[l] + mu[l]
            cum_var[l + 1] = cum_var[l] + (mu[l] * sig[l]) ** 2
        angle_limit = np.full(n, np.nan)
        for l, a in proj["angles"].items():
            limit = a.get("max_deg")
            if limit is not None and a["n"] >= MIN_SAMPLES:
                angle_limit[int(l)] = limit
        scale = proj["scale"]
        return cls(projection, cum, cum_var, angle_limit, (scale["low"], scale["high"]))

    def template_distance(self, l: int, m: int) -> float:
        return float(self.cum[m] - self.cum[l])

    def gap_z(self, distance_mm: float, l: int, m: int, scale: float) -> float:
        """Deviation of a measured distance between discs ``l < m`` (log-sigma units)."""
        span = self.template_distance(l, m)
        sigma = np.sqrt(self.cum_var[m] - self.cum_var[l]) / span
        return float((np.log(max(distance_mm, 1e-3)) - np.log(scale * span)) / sigma)


def load_skeleton(path: str, projection: str) -> SkeletonModel:
    with open(path) as f:
        return SkeletonModel.from_dict(json.load(f), projection)


# --------------------------------------------------------------------------
# Refinement
# --------------------------------------------------------------------------


def _estimate_scale(model: SkeletonModel, labels: Sequence[int], pts_mm: np.ndarray) -> Optional[float]:
    """Median ratio of measured to template distance over consecutive labelled peaks."""
    ratios = [
        float(np.linalg.norm(pts_mm[k + 1] - pts_mm[k])) / model.template_distance(labels[k], labels[k + 1])
        for k in range(len(labels) - 1)
    ]
    ratios = [r for r in ratios if r > 0]
    return float(np.median(ratios)) if ratios else None


def _best_labelling(
    pts_mm: np.ndarray,
    anchor_idx: int,
    anchor_label: int,
    model: SkeletonModel,
    scale: float,
    drop_penalty: float,
    miss_penalty: float,
    max_missing: int,
    max_skipped: int,
    weights: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
    """Viterbi over ``(peak, label)``; returns the kept ``[(peak_idx, label), ...]``.

    ``weights`` (soft matching, see the module docstring) is the confidence
    of each peak in [0, 1]; ``None`` = every peak fully trusted.
    """
    n = len(pts_mm)
    lo, hi = MIN_DISC_LABEL, MAX_DISC_LABEL
    dist = np.linalg.norm(pts_mm[:, None, :] - pts_mm[None, :, :], axis=-1)
    cost = np.full((n, hi + 1), np.inf)
    back: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    keep_cost = KEEP_TO_DROP_RATIO * drop_penalty * (1.0 - w)
    # dropped[k] = cost of dropping peaks 0..k-1; n_strong[k] = confident peaks among them.
    dropped = np.concatenate([[0.0], np.cumsum(drop_penalty * w)])
    n_strong = np.concatenate([[0], np.cumsum(w >= 1.0)])

    for j in range(n):
        # Earliest previous kept peak: at most ``max_skipped`` confident and
        # ``SOFT_MAX_WEAK_SKIPPED`` weak peaks dropped in between, never the anchor.
        i_min = j
        while i_min > 0:
            i = i_min - 1
            skipped_strong = n_strong[j] - n_strong[i + 1]
            if (
                i < anchor_idx < j
                or skipped_strong > max_skipped
                or (j - i - 1) - skipped_strong > SOFT_MAX_WEAK_SKIPPED
            ):
                break
            i_min = i
        labels = [anchor_label] if j == anchor_idx else range(lo, hi + 1)
        for m in labels:
            # Start the chain here: every peak above is dropped (not the anchor).
            best = dropped[j] if j <= anchor_idx else np.inf
            bp = None
            for i in range(i_min, j):
                between = dropped[j] - dropped[i + 1]
                for l in range(max(lo, m - 1 - max_missing), m):
                    c = cost[i, l]
                    if not np.isfinite(c):
                        continue
                    z = model.gap_z(dist[i, j], l, m, scale)
                    c += between + miss_penalty * (m - l - 1) + 0.5 * z * z
                    if c < best:
                        best, bp = c, (i, l)
            cost[j, m] = best + keep_cost[j]
            back[(j, m)] = bp

    total = cost + (dropped[n] - dropped[1:])[:, None]
    total[:anchor_idx] = np.inf
    j, m = np.unravel_index(int(np.argmin(total)), total.shape)
    path: List[Tuple[int, int]] = []
    state: Optional[Tuple[int, int]] = (int(j), int(m))
    while state is not None:
        path.append(state)
        state = back[state]
    return path[::-1]


def _impute(
    model: SkeletonModel, kept: List[Tuple[int, np.ndarray]]
) -> List[Tuple[int, np.ndarray]]:
    """Grid coordinates of every label missing BETWEEN two kept discs."""
    if len(kept) < 2:
        return []
    labels = np.array([l for l, _ in kept])
    coords = np.stack([c for _, c in kept])
    t = model.cum[labels]
    present = set(labels.tolist())
    missing = [l for l in range(int(labels[0]) + 1, int(labels[-1])) if l not in present]
    if not missing:
        return []
    tq = model.cum[missing]
    if len(kept) >= 3:
        from scipy.interpolate import PchipInterpolator

        est = PchipInterpolator(t, coords, axis=0)(tq)
    else:
        est = np.stack([np.interp(tq, t, coords[:, a]) for a in range(coords.shape[1])], axis=1)
    return [(int(l), est[k]) for k, l in enumerate(missing)]


def _assess(
    model: SkeletonModel,
    labels: Sequence[int],
    pts_mm: np.ndarray,
    scale: float,
    z_threshold: float,
) -> Tuple[List[dict], List[dict]]:
    """Gap z-scores and turning angles of an observed, label-sorted chain.

    A gap is abnormal beyond ``z_threshold``, raised to the Bonferroni bound
    for ``GAP_FLAG_ALPHA`` over the gaps of the chain.
    """
    from scipy.stats import norm

    n_gaps = max(len(labels) - 1, 1)
    flag_z = max(z_threshold, float(norm.isf(GAP_FLAG_ALPHA / (2 * n_gaps))))
    gaps = []
    for k in range(len(labels) - 1):
        l, m = int(labels[k]), int(labels[k + 1])
        d = float(np.linalg.norm(pts_mm[k + 1] - pts_mm[k]))
        z = model.gap_z(d, l, m, scale)
        gaps.append(
            {
                "from": l,
                "to": m,
                "distance_mm": round(d, 2),
                "expected_mm": round(scale * model.template_distance(l, m), 2),
                "z": round(z, 2),
                "abnormal": bool(abs(z) > flag_z),
            }
        )
    angles = []
    for k in range(1, len(labels) - 1):
        l = int(labels[k])
        if labels[k - 1] != l - 1 or labels[k + 1] != l + 1:
            continue
        limit = model.angle_limit[l]
        a = _angle_deg(pts_mm[k] - pts_mm[k - 1], pts_mm[k + 1] - pts_mm[k])
        angles.append(
            {
                "label": l,
                "angle_deg": round(a, 1),
                "limit_deg": None if np.isnan(limit) else round(float(limit), 1),
                "abnormal": bool(not np.isnan(limit) and a > limit),
            }
        )
    return gaps, angles


def _round(coords) -> List[float]:
    return [round(float(c), 2) for c in coords]


def refine_chain(
    chain: List[Tuple[float, ...]],
    anchor_label: int,
    anchor_idx: int,
    spacing: Sequence[float],
    model: SkeletonModel,
    mode: str = "correct",
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    max_missing: int = DEFAULT_MAX_MISSING,
    max_skipped: int = DEFAULT_MAX_SKIPPED,
    weights: Optional[Sequence[float]] = None,
) -> Tuple[List[Tuple[float, ...]], dict]:
    """Check (and in ``"correct"`` mode repair) a decoded peak chain.

    ``chain`` / ``anchor_label`` / ``anchor_idx`` come from
    ``decoding.decode_chain``; ``spacing`` is the voxel size in mm of the
    chain's coordinate axes. Returns ``(labeled, report)`` where ``labeled``
    is ``[(label, *coords), ...]`` like ``label_discs`` (the uncorrected
    labelling in ``"flag"`` mode) and ``report`` holds the QC flags.

    ``weights`` (from ``decoding.decode_chain_soft``) enables soft matching:
    the confidence of each chain peak in [0, 1]. Peaks below 1 are *weak*:
    they are ignored by the first labelling step and only enter the result
    when the skeleton keeps them.
    """
    if mode not in ("flag", "correct"):
        raise ValueError(f"mode must be 'flag' or 'correct', got {mode!r}")
    spacing = np.asarray(spacing, dtype=np.float64)
    coords = np.array([p[:-1] for p in chain], dtype=np.float64)
    pts_mm = coords * spacing

    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    # First labelling step, identical to decoding._assign_labels (on the
    # confident peaks only when soft matching is on).
    strong = range(len(chain)) if w is None else [i for i in range(len(chain)) if w[i] >= 1.0]
    anchor_rank = list(strong).index(anchor_idx)
    initial = {
        i: anchor_label + (rank - anchor_rank)
        for rank, i in enumerate(strong)
        if MIN_DISC_LABEL <= anchor_label + (rank - anchor_rank) <= MAX_DISC_LABEL
    }
    initial_idx = sorted(initial)
    initial_labeled = [(initial[i],) + tuple(coords[i]) for i in initial_idx]

    report: dict = {"mode": mode, "anchor_label": int(anchor_label), "flags": []}
    if len(initial_idx) < 2:
        report["flags"].append("too_few_discs")
        report["status"] = "flagged"
        return initial_labeled, report

    scale = _estimate_scale(model, [initial[i] for i in initial_idx], pts_mm[initial_idx])
    drop_penalty = 0.5 * z_threshold ** 2
    miss_penalty = MISS_TO_DROP_RATIO * drop_penalty
    path: List[Tuple[int, int]] = []
    for _ in range(2):  # re-estimate the scale on the repaired chain
        path = _best_labelling(
            pts_mm, anchor_idx, anchor_label, model, scale,
            drop_penalty, miss_penalty, max_missing, max_skipped, w,
        )
        new_scale = _estimate_scale(model, [l for _, l in path], pts_mm[[i for i, _ in path]])
        if new_scale is None or abs(new_scale - scale) < 1e-3 * scale:
            break
        scale = new_scale

    kept = {i: l for i, l in path}
    removed = [
        {"initial_label": initial[i], "coords": _round(coords[i]), "peak_value": round(float(chain[i][-1]), 3)}
        for i in initial_idx
        if i not in kept
    ]
    relabeled = [
        {"from": initial[i], "to": l, "coords": _round(coords[i])}
        for i, l in path
        if i in initial and initial[i] != l
    ]
    imputed = _impute(model, [(l, coords[i]) for i, l in path])
    recovered = [
        {"label": l, "coords": _round(coords[i]), "score": round(float(chain[i][-1]), 3)}
        for i, l in path
        if w is not None and w[i] < 1.0
    ]
    report["changes"] = {
        "applied": mode == "correct",
        "removed": removed,
        "relabeled": relabeled,
        "imputed": [{"label": l, "coords": _round(c)} for l, c in imputed],
        "recovered": recovered,
    }
    if recovered:
        report["flags"].append("weak_discs")
    if imputed:
        report["flags"].append("missed_discs")
    if removed:
        report["flags"].append("outlier_discs")
    if relabeled:
        report["flags"].append("relabeled")

    if mode == "correct":
        obs_idx = [i for i, _ in path]
        obs_labels = [l for _, l in path]
        labeled = sorted(
            [(l,) + tuple(coords[i]) for i, l in path] + [(l,) + tuple(c) for l, c in imputed],
            key=lambda d: d[0],
        )
    else:
        obs_idx = initial_idx
        obs_labels = [initial[i] for i in initial_idx]
        labeled = initial_labeled

    gaps, angles = _assess(model, obs_labels, pts_mm[obs_idx], scale, z_threshold)
    report["scale"] = round(float(scale), 3)
    report["scale_range"] = [round(float(s), 3) for s in model.scale_range]
    report["gaps"] = gaps
    report["angles"] = angles
    if any(g["abnormal"] for g in gaps):
        report["flags"].append("abnormal_gap")
    if any(a["abnormal"] for a in angles):
        report["flags"].append("abnormal_angle")
    if not model.scale_range[0] <= scale <= model.scale_range[1]:
        report["flags"].append("abnormal_scale")
    report["status"] = "flagged" if report["flags"] else "ok"
    return labeled, report


def decode_with_skeleton(
    candidates: Dict[str, object],
    peak_threshold: float,
    min_peak_distance: float,
    spacing: Sequence[float],
    model: SkeletonModel,
    mode: str = "correct",
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    max_missing: int = DEFAULT_MAX_MISSING,
    max_skipped: int = DEFAULT_MAX_SKIPPED,
    soft_floor: Optional[float] = None,
) -> Tuple[List[Tuple[float, ...]], dict]:
    """``label_discs`` followed by ``refine_chain`` (``spacing`` in mm per coordinate axis).

    With ``soft_floor`` (and ``"soft_discs"`` in ``candidates``, ``"correct"``
    mode only) the chain is built from the soft candidates down to that
    score, weighted by confidence — see the module docstring. ``"flag"`` mode
    always reports on the plain thresholded chain, since it leaves the
    prediction unchanged.
    """
    weights = None
    if mode == "correct" and soft_floor is not None and "soft_discs" in candidates:
        decoded = decode_chain_soft(candidates, peak_threshold, min_peak_distance, soft_floor)
        if decoded is not None:
            *decoded, weights = decoded
    else:
        decoded = decode_chain(candidates, peak_threshold, min_peak_distance)
    if decoded is None:
        return [], {"mode": mode, "flags": ["no_anchor"], "status": "flagged"}
    chain, anchor_label, anchor_idx = decoded
    labeled, report = refine_chain(
        chain, anchor_label, anchor_idx, spacing, model,
        mode=mode, z_threshold=z_threshold, max_missing=max_missing, max_skipped=max_skipped,
        weights=weights,
    )
    report["soft_floor"] = soft_floor if weights is not None else None
    return labeled, report


def affine_spacing(affine: np.ndarray, projection: str) -> np.ndarray:
    """Voxel size (mm) of the axes a projection's coordinates live on."""
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)
    return spacing[list(PROJECTION_AXES[projection])]


def skeleton_settings(cfg: dict, tuned: Optional[dict] = None) -> dict:
    """Refinement keyword arguments: tuned ``z_threshold`` / ``soft_floor`` first, then the config.

    ``soft_floor`` is ``None`` when soft matching is off (``skeleton_soft_floor: null``).
    """
    z = cfg.get("skeleton_z_threshold", DEFAULT_Z_THRESHOLD)
    if tuned and tuned.get("z_threshold") is not None:
        z = tuned["z_threshold"]
    soft_floor = cfg.get("skeleton_soft_floor", DEFAULT_SOFT_FLOOR)
    if tuned and "soft_floor" in tuned:
        soft_floor = tuned["soft_floor"]
    return {
        "soft_floor": None if soft_floor is None else float(soft_floor),
        "z_threshold": float(z),
        "max_missing": int(cfg.get("skeleton_max_missing", DEFAULT_MAX_MISSING)),
        "max_skipped": int(cfg.get("skeleton_max_skipped", DEFAULT_MAX_SKIPPED)),
    }


QC_FIELDS = (
    "image", "status", "flags", "n_discs", "n_imputed", "n_recovered", "n_removed",
    "n_relabeled", "max_abs_gap_z", "scale",
)


def qc_row(report: dict) -> dict:
    """Flatten a skeleton QC report into one ``qc_summary.csv`` row."""
    changes = report.get("changes", {})
    gap_z = [abs(g["z"]) for g in report.get("gaps", [])]
    return {
        "image": report["image"],
        "status": report["status"],
        "flags": ";".join(report["flags"]),
        "n_discs": report["n_discs"],
        "n_imputed": len(changes.get("imputed", [])),
        "n_recovered": len(changes.get("recovered", [])),
        "n_removed": len(changes.get("removed", [])),
        "n_relabeled": len(changes.get("relabeled", [])),
        "max_abs_gap_z": round(max(gap_z), 2) if gap_z else "",
        "scale": report.get("scale", ""),
    }


def write_qc_summary(out_dir: str, rows: List[dict]) -> None:
    path = os.path.join(out_dir, "qc_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(QC_FIELDS))
        w.writeheader()
        w.writerows(rows)
    n_flagged = sum(r["status"] != "ok" for r in rows)
    print(f"[qc] {n_flagged}/{len(rows)} images flagged; see {path} and <name>_discs_qc.json")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the inter-disc distance skeleton on the training labels (skeleton.json)."
    )
    parser.add_argument("--manifest", required=True, help="Prepared manifest.json.")
    parser.add_argument("--out-dir", required=True, help="Where to write skeleton.json (the weights folder).")
    parser.add_argument(
        "--splits", nargs="+", default=["train"], help="Manifest splits to fit on (default: train)."
    )
    parser.add_argument(
        "--num-workers", default="auto", help="CPU worker processes for reading labels (0 = inline)."
    )
    args = parser.parse_args()

    skeleton = fit_skeleton(args.manifest, args.splits, args.num_workers)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, SKELETON_FILE)
    with open(out, "w") as f:
        json.dump(skeleton, f, indent=2)

    proj = skeleton["projections"]["3d"]
    print(f"[skeleton] {skeleton['n_images']} images, {skeleton['n_discs']} discs")
    print("[skeleton] template gaps (3d, mm): " + " ".join(
        f"{l}-{int(l) + 1}:{g['template_mm']:.1f}(n={g['n']})" for l, g in proj["gaps"].items()
    ))
    print(f"[skeleton] subject scale p01/p50/p99: {proj['scale']['p01']:.2f}/"
          f"{proj['scale']['p50']:.2f}/{proj['scale']['p99']:.2f}")
    print(f"[skeleton] wrote {out}")


if __name__ == "__main__":
    main()
