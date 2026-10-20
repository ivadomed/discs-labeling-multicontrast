"""Landmark-detection metrics and visualization helpers."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from skimage import draw as sk_draw
from skimage import io as sk_io

from .. import HEATMAP_CHANNELS, NUM_CLASSES


def heatmap_argmax(heatmap: np.ndarray) -> Tuple[float, float, float]:
    """Return ``(x, y, peak)`` of the max response of a 2D heatmap.

    Multiple pixels tying for the max are averaged (which smooths the output
    for plateau peaks).
    """
    peak = float(heatmap.max())
    yy, xx = np.where(heatmap == peak)
    return float(np.mean(xx)), float(np.mean(yy)), peak


def decode_predictions(heatmaps: np.ndarray, peak_threshold: float = 0.3) -> Dict[str, dict]:
    """Decode a (NUM_CLASSES, H, W) heatmap stack into per-channel landmarks.

    Returns a dict keyed by ``HEATMAP_CHANNELS``:
        all_discs -> {"peaks": [(x, y, intensity), ...]} (local maxima)
        others    -> {"x": float, "y": float, "peak": float, "present": bool}
    """
    assert heatmaps.ndim == 3 and heatmaps.shape[0] == NUM_CLASSES, (
        f"Expected ({NUM_CLASSES}, H, W) heatmap stack, got {heatmaps.shape}"
    )

    result: Dict[str, dict] = {}

    # Channel 0: multi-peak detection on the "all discs" heatmap.
    peaks = _local_maxima(heatmaps[0], min_distance=5, threshold=peak_threshold)
    result[HEATMAP_CHANNELS[0]] = {"peaks": peaks}

    # Channels 1..4: single-peak anchors.
    for ch in range(1, NUM_CLASSES):
        x, y, peak = heatmap_argmax(heatmaps[ch])
        result[HEATMAP_CHANNELS[ch]] = {
            "x": x,
            "y": y,
            "peak": peak,
            "present": peak >= peak_threshold,
        }
    return result


def _local_maxima(heatmap: np.ndarray, min_distance: int, threshold: float) -> List[Tuple[float, float, float]]:
    """Return [(x, y, intensity), ...] for local maxima above ``threshold``."""
    try:
        from skimage.feature import peak_local_max
    except ImportError:  # pragma: no cover - skimage is a hard dep
        peak_local_max = None
    if peak_local_max is None:
        x, y, peak = heatmap_argmax(heatmap)
        return [(x, y, peak)] if peak >= threshold else []
    coords = peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False,
    )
    return [(float(c[1]), float(c[0]), float(heatmap[c[0], c[1]])) for c in coords]


def mean_radial_error(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    spacing_mm: float = 1.0,
) -> Tuple[float, float]:
    """Compute MRE and 2 mm success-detection rate.

    ``predictions`` and ``ground_truth`` are both (N, 2) arrays of (x, y) pixels.
    """
    if predictions.size == 0:
        return float("nan"), float("nan")
    errors = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=1)) * spacing_mm
    mre = float(errors.mean())
    sdr_2mm = float((errors < 2.0).mean())
    return mre, sdr_2mm


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        arr = np.zeros_like(arr)
    else:
        arr = (arr - lo) / (hi - lo)
    arr = (arr * 255.0).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def visualize_prediction(
    image: np.ndarray,
    pred_landmarks: List[Tuple[int, float, float]],
    gt_landmarks: List[Tuple[int, float, float]],
    save_path: str,
) -> None:
    """Overlay ground-truth (green) and predicted (red) landmarks on the slice."""
    canvas = _to_rgb_uint8(image)
    h, w = canvas.shape[:2]

    for _label, x, y in gt_landmarks:
        rr, cc = sk_draw.disk((int(y), int(x)), radius=4, shape=(h, w))
        canvas[rr, cc, :] = [0, 255, 0]

    for _label, x, y in pred_landmarks:
        rr, cc = sk_draw.disk((int(y), int(x)), radius=4, shape=(h, w))
        canvas[rr, cc, :] = [255, 0, 0]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    sk_io.imsave(save_path, canvas)
