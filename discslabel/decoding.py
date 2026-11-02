"""Sliding-window heatmap stitching and disc decoding, shared by 2D and 3D.

nnU-Net-style inference: every window prediction is weighted by a Gaussian
importance map (centre voxels trusted more than border voxels) and
accumulated into one full-size heatmap; peaks are then decoded ONCE on the
stitched map instead of per window.

Decoding is split in two stages so post-processing can be grid-searched
without re-running the model:

    1. ``extract_candidates(heatmap, min_threshold)`` — every local maximum of
       the all-discs channel above ``min_threshold`` plus the argmax of each
       anchor channel.
    2. ``label_discs(candidates, peak_threshold, min_peak_distance)`` —
       threshold, Euclidean NMS, anchor choice and superior → inferior
       labelling.

Coordinates are array indices of the heatmap's spatial axes: ``(d, h, w)`` in
3D, ``(h, w)`` for the 2D mid-sagittal slice. In both cases the LAST axis is
I-S (RPI orientation, higher = superior).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import HEATMAP_CHANNELS, LANDMARK_CHANNEL_LABELS, NUM_CLASSES


# C2C3 (3) > L5S1 (25) > C7T1 (8) > T12L1 (20).
ANCHOR_PRIORITY: Tuple[str, ...] = ("C2C3", "L5S1", "C7T1", "T12L1")

# Labels outside this range are dropped when propagating from the anchor.
MIN_DISC_LABEL = 1
MAX_DISC_LABEL = 25

# Candidate peaks are at least this many voxels apart (Chebyshev). Keeps the
# candidate list short; the real spacing is enforced by ``min_peak_distance``.
CANDIDATE_MIN_DISTANCE = 2


def gaussian_importance_map(
    patch_shape: Sequence[int], device: torch.device, sigma_scale: float = 1.0 / 8
) -> torch.Tensor:
    """Separable Gaussian centred on the patch, peak 1, ``sigma = patch * 1/8``.

    Floored at ``1e-4`` (nnU-Net also floors its map) so voxels that only
    window corners cover keep a usable normalisation weight.
    """
    weight = torch.ones((), device=device)
    for axis, size in enumerate(patch_shape):
        coords = torch.arange(size, device=device, dtype=torch.float32)
        sigma = max(size * sigma_scale, 1e-3)
        g = torch.exp(-((coords - size // 2) ** 2) / (2 * sigma * sigma))
        shape = [1] * len(patch_shape)
        shape[axis] = size
        weight = weight * g.view(shape)
    return (weight / weight.max()).clamp_min(1e-4)


def _pad_to_patch(
    image: np.ndarray, patch_shape: Sequence[int]
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Zero-pad the spatial axes of ``(C, *S)`` to at least ``patch_shape``."""
    pads = [(0, 0)]
    before: List[int] = []
    for size, patch in zip(image.shape[1:], patch_shape):
        total = max(0, int(patch) - int(size))
        pads.append((total // 2, total - total // 2))
        before.append(total // 2)
    return np.pad(image, pads, mode="constant"), tuple(before)


def _axis_origins(src_len: int, patch_len: int, stride: int) -> List[int]:
    """Window origins along one axis, with the last one pinned to the edge."""
    if src_len <= patch_len:
        return [0]
    origins = list(range(0, src_len - patch_len + 1, stride))
    if origins[-1] + patch_len < src_len:
        origins.append(src_len - patch_len)
    return origins


def sliding_window_origins(
    shape: Sequence[int], patch_shape: Sequence[int], overlap: float = 0.5
) -> List[Tuple[int, ...]]:
    """Every window origin covering ``shape`` with the requested overlap."""
    per_axis = [
        _axis_origins(int(s), int(p), max(1, int(p * (1 - overlap))))
        for s, p in zip(shape, patch_shape)
    ]
    grids = np.meshgrid(*per_axis, indexing="ij")
    return [tuple(int(v) for v in o) for o in np.stack([g.ravel() for g in grids], axis=1)]


def predict_heatmap(
    image: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_shape: Sequence[int],
    overlap: float,
    batch_size: int,
) -> np.ndarray:
    """Gaussian-weighted sliding-window prediction over a whole image.

    ``image`` is ``(C_in, *S)`` (C_in = 1 for 3D, the slice stack for 2D).
    Returns the stitched ``(NUM_CLASSES, *S)`` heatmap as float32 on CPU.
    The caller is responsible for ``model.eval()``.
    """
    patch_shape = tuple(int(p) for p in patch_shape)
    padded, before = _pad_to_patch(image, patch_shape)
    spatial = padded.shape[1:]
    origins = sliding_window_origins(spatial, patch_shape, overlap)

    image_t = torch.from_numpy(np.ascontiguousarray(padded)).float().to(device)
    weight = gaussian_importance_map(patch_shape, device)
    accum = torch.zeros((NUM_CLASSES,) + tuple(spatial), device=device)
    norm = torch.zeros(tuple(spatial), device=device)

    def _window(origin):
        return tuple(slice(o, o + p) for o, p in zip(origin, patch_shape))

    with torch.no_grad():
        for i in range(0, len(origins), batch_size):
            chunk = origins[i : i + batch_size]
            batch = torch.stack([image_t[(slice(None),) + _window(o)] for o in chunk])
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(batch)
            pred = pred.float()
            for origin, p in zip(chunk, pred):
                win = _window(origin)
                accum[(slice(None),) + win] += p * weight
                norm[win] += weight

    heatmap = accum / norm.clamp_min(1e-8)
    crop = tuple(slice(b, b + s) for b, s in zip(before, image.shape[1:]))
    return heatmap[(slice(None),) + crop].cpu().numpy()


def extract_candidates(heatmap: np.ndarray, min_threshold: float) -> Dict[str, object]:
    """Stage 1 of decoding: raw peaks, independent of the final threshold.

    Returns ``{"discs": [(*coords, value), ...], "anchors": {name: (*coords, value)}}``.
    """
    from skimage.feature import peak_local_max

    assert heatmap.shape[0] == NUM_CLASSES, (
        f"Expected ({NUM_CLASSES}, ...) heatmap stack, got {heatmap.shape}"
    )
    disc_map = heatmap[0]
    coords = peak_local_max(
        disc_map,
        min_distance=CANDIDATE_MIN_DISTANCE,
        threshold_abs=min_threshold,
        exclude_border=False,
    )
    discs = [tuple(float(c) for c in pt) + (float(disc_map[tuple(pt)]),) for pt in coords]

    anchors: Dict[str, Tuple[float, ...]] = {}
    for ch in range(1, NUM_CLASSES):
        channel = heatmap[ch]
        peak = float(channel.max())
        where = np.nonzero(channel == peak)
        anchors[HEATMAP_CHANNELS[ch]] = tuple(float(np.mean(w)) for w in where) + (peak,)
    return {"discs": discs, "anchors": anchors}


def _merge_peaks(
    peaks: List[Tuple[float, ...]], min_distance: float
) -> List[Tuple[float, ...]]:
    """Greedy Euclidean NMS: strongest peak kept, drop anything within ``min_distance``."""
    kept: List[Tuple[float, ...]] = []
    kept_xyz = np.empty((0, len(peaks[0]) - 1)) if peaks else None
    for peak in sorted(peaks, key=lambda p: -p[-1]):
        xyz = np.asarray(peak[:-1])
        if len(kept) and np.min(np.sum((kept_xyz - xyz) ** 2, axis=1)) < min_distance ** 2:
            continue
        kept.append(peak)
        kept_xyz = np.vstack([kept_xyz, xyz])
    return kept


def _choose_anchor(
    anchors: Dict[str, Tuple[float, ...]], peak_threshold: float
) -> Optional[Tuple[str, int, Tuple[float, ...]]]:
    """Highest-priority anchor whose peak clears ``peak_threshold``.

    Returns ``(channel_name, disc_label, coords)`` or ``None``.
    """
    for name in ANCHOR_PRIORITY:
        hit = anchors.get(name)
        if hit is None or hit[-1] < peak_threshold:
            continue
        anchor_label = LANDMARK_CHANNEL_LABELS[HEATMAP_CHANNELS.index(name)]
        return (name, int(anchor_label), tuple(hit[:-1]))
    return None


def _assign_labels(
    discs: List[Tuple[float, ...]],
    anchor_label: int,
    anchor_coords: Tuple[float, ...],
) -> List[Tuple[float, ...]]:
    """Label disc peaks superior → inferior from a known anchor.

    Sort by the last (I-S) coordinate descending so index 0 is the most
    superior disc; the disc closest to the anchor gets the anchor's label and
    labels move by ±1 per step. Returns ``[(label, *coords), ...]``.
    """
    if not discs:
        return []
    sorted_discs = sorted(discs, key=lambda p: (-p[-2],) + tuple(p[:-2]))
    anchor = np.asarray(anchor_coords)
    anchor_idx = int(
        np.argmin([np.sum((np.asarray(p[:-1]) - anchor) ** 2) for p in sorted_discs])
    )
    out: List[Tuple[float, ...]] = []
    for i, peak in enumerate(sorted_discs):
        label = anchor_label + (i - anchor_idx)
        if MIN_DISC_LABEL <= label <= MAX_DISC_LABEL:
            out.append((label,) + tuple(peak[:-1]))
    return out


def label_discs(
    candidates: Dict[str, object], peak_threshold: float, min_peak_distance: float
) -> List[Tuple[float, ...]]:
    """Stage 2 of decoding: ``[(label, *coords), ...]`` (empty when no anchor)."""
    discs = [p for p in candidates["discs"] if p[-1] >= peak_threshold]
    merged = _merge_peaks(discs, min_distance=min_peak_distance)
    chosen = _choose_anchor(candidates["anchors"], peak_threshold)
    if chosen is None:
        return []
    _name, anchor_label, anchor_coords = chosen
    return _assign_labels(merged, anchor_label, anchor_coords)
