"""Whole-image prediction for the 2D (mid-sagittal) pipeline.

Shared by inference, training validation and post-processing tuning so all
three decode exactly the same way.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from ..decoding import extract_candidates, predict_heatmap


def extract_slice_stack(volume: np.ndarray, d_mid: int, n_slices: int) -> np.ndarray:
    """Pull ``n_slices`` adjacent R-L slices centered on ``d_mid`` from a 3D volume.

    Returns a ``(n_slices, H, W)`` array. Indices are clamped to ``[0, D-1]``
    so a volume with fewer slices than requested degrades gracefully by
    replicating the boundary slice.
    """
    D = volume.shape[0]
    half = n_slices // 2
    offsets = range(-half, -half + n_slices)
    indices = [max(0, min(D - 1, d_mid + o)) for o in offsets]
    return volume[indices]


def mid_slice_stack(volume: np.ndarray, n_slices: int) -> np.ndarray:
    """The ``n_slices`` R-L slices around the volume centre (the model input)."""
    return extract_slice_stack(volume, volume.shape[0] // 2, n_slices)


def predict_volume_heatmap(
    volume_norm: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_hw: Tuple[int, int],
    overlap: float,
    batch_size: int,
    n_mid_slices: int,
) -> np.ndarray:
    """Stitched ``(NUM_CLASSES, H, W)`` heatmap of the mid-sagittal slice stack."""
    stack = mid_slice_stack(volume_norm, n_mid_slices)
    return predict_heatmap(stack, model, device, patch_hw, overlap, batch_size)


def predict_candidates(
    volume_norm: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_hw: Tuple[int, int],
    overlap: float,
    batch_size: int,
    n_mid_slices: int,
    min_threshold: float,
) -> Dict[str, object]:
    """Stitched heatmap of the mid-sagittal slice stack → decoding candidates.

    Candidate coordinates are ``(h, w)`` on the slice.
    """
    heatmap = predict_volume_heatmap(
        volume_norm, model, device, patch_hw, overlap, batch_size, n_mid_slices
    )
    return extract_candidates(heatmap, min_threshold)


def labeled_to_voxels(
    labeled_discs: List[Tuple[float, ...]], volume_shape: Tuple[int, int, int]
) -> Dict[int, np.ndarray]:
    """``[(label, h, w), ...]`` → ``{label: (d_mid, h, w)}`` on the volume grid."""
    d_mid = float(volume_shape[0] // 2)
    return {int(l): np.array([d_mid, h, w], dtype=np.float64) for l, h, w in labeled_discs}
