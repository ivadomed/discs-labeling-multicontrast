"""Whole-image prediction for the 2D (mid-sagittal) pipeline.

Shared by inference, training validation and post-processing tuning so all
three decode exactly the same way.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from discslabel.decoding import extract_candidates, predict_heatmap

# Which skeleton projection matches the decoded coordinates (see discslabel.skeleton).
SKELETON_PROJECTION = "sagittal"


INPUT_MODES = ("mid", "avg")


def slab_indices(D: int, d_mid: int, n_slices: int) -> List[int]:
    """``n_slices`` adjacent R-L indices centered on ``d_mid``, clamped to ``[0, D-1]``.

    Clamping lets a volume thinner than the slab degrade gracefully by
    replicating the boundary slice.
    """
    half = n_slices // 2
    return [max(0, min(D - 1, d_mid + o)) for o in range(-half, -half + n_slices)]


def slab_num_slices(cfg: dict) -> int:
    """Number of R-L slices averaged in "avg" mode: ``slab_thickness_mm`` / R-L spacing."""
    from discslabel.gpu_artifacts import parse_spacing

    spacing_rl = parse_spacing(cfg.get("resolution", "1x1x1"))[0]
    thickness = float(cfg.get("slab_thickness_mm", 10.0))
    return max(1, int(round(thickness / spacing_rl)))


def model_input(volume: np.ndarray, input_mode: str, n_slab: int) -> np.ndarray:
    """The ``(1, H, W)`` model input from a ``(D, H, W)`` volume.

    ``"mid"``: the mid-sagittal slice. ``"avg"``: the mean of the ``n_slab``
    R-L slices around it (a thick sagittal slab).
    """
    d_mid = volume.shape[0] // 2
    if input_mode == "mid":
        return np.ascontiguousarray(volume[d_mid : d_mid + 1])
    if input_mode == "avg":
        idx = slab_indices(volume.shape[0], d_mid, n_slab)
        return volume[idx].mean(axis=0, keepdims=True, dtype=np.float32)
    raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")


def predict_volume_heatmap(
    volume_norm: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_hw: Tuple[int, int],
    overlap: float,
    batch_size: int,
    input_mode: str,
    n_slab: int,
) -> np.ndarray:
    """Stitched ``(NUM_CLASSES, H, W)`` heatmap of the 2D model input."""
    image_2d = model_input(volume_norm, input_mode, n_slab)
    return predict_heatmap(image_2d, model, device, patch_hw, overlap, batch_size)


def predict_candidates(
    volume_norm: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_hw: Tuple[int, int],
    overlap: float,
    batch_size: int,
    input_mode: str,
    n_slab: int,
    min_threshold: float,
) -> Dict[str, object]:
    """Stitched heatmap of the 2D model input → decoding candidates.

    Candidate coordinates are ``(h, w)`` on the slice.
    """
    heatmap = predict_volume_heatmap(
        volume_norm, model, device, patch_hw, overlap, batch_size, input_mode, n_slab
    )
    return extract_candidates(heatmap, min_threshold)


def labeled_to_voxels(
    labeled_discs: List[Tuple[float, ...]], volume_shape: Tuple[int, int, int]
) -> Dict[int, np.ndarray]:
    """``[(label, h, w), ...]`` → ``{label: (d_mid, h, w)}`` on the volume grid."""
    d_mid = float(volume_shape[0] // 2)
    return {int(l): np.array([d_mid, h, w], dtype=np.float64) for l, h, w in labeled_discs}
