"""Whole-volume prediction for the 3D pipeline.

Shared by inference, training validation and post-processing tuning so all
three decode exactly the same way.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from ..decoding import extract_candidates, predict_heatmap

# Which skeleton projection matches the decoded coordinates (see discslabel.skeleton).
SKELETON_PROJECTION = "3d"


def predict_volume_heatmap(
    volume_norm: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_dhw: Tuple[int, int, int],
    overlap: float,
    batch_size: int,
) -> np.ndarray:
    """Stitched ``(NUM_CLASSES, D, H, W)`` heatmap of a z-scored volume."""
    return predict_heatmap(volume_norm[None], model, device, patch_dhw, overlap, batch_size)


def predict_candidates(
    volume_norm: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_dhw: Tuple[int, int, int],
    overlap: float,
    batch_size: int,
    min_threshold: float,
) -> Dict[str, object]:
    """Stitched 3D heatmap of a z-scored ``(D, H, W)`` volume → decoding candidates."""
    heatmap = predict_volume_heatmap(volume_norm, model, device, patch_dhw, overlap, batch_size)
    return extract_candidates(heatmap, min_threshold)


def labeled_to_voxels(
    labeled_discs: List[Tuple[float, ...]], volume_shape: Tuple[int, int, int]
) -> Dict[int, np.ndarray]:
    """``[(label, d, h, w), ...]`` → ``{label: (d, h, w)}`` on the volume grid."""
    return {int(l): np.array([d, h, w], dtype=np.float64) for l, d, h, w in labeled_discs}
