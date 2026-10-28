"""GPU-side batch processor for full-3D heatmap regression.

Takes a batch of 3D ``(image, label, seg)`` volumes, optionally runs the
SmaugLab 3D GPU augmentations, and builds the 5-channel 3D Gaussian heatmap
target from the point-wise disc label. No mid-sagittal projection, no 2D
elastic transform — the model consumes the full volume.

Expected tensor layout:
    image : (B, 1, D, H, W) float, intensities in [0, 1]
    label : (B, 1, D, H, W) float, integer disc values (0 = background, 1..25 = discs)
    seg   : (B, NUM_SEG_CHANNELS, D, H, W) binary tensor — one channel per
            anatomical structure (``discslabel.SEG_CHANNELS``):
                0: vertebrae, 1: discs, 2: spinal cord, 3: spinal canal.
            Used as the SECOND parameter of ``AugTransformsGPU``. The
            point-wise disc landmark volume is NOT fed to the augmentation
            pipeline, so the aug config should stay intensity-only unless you
            add a dedicated spatial-replay path for the disc label.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import LANDMARK_CHANNEL_LABELS


def _resolve_aug_config(name: str) -> str:
    """Resolve ``name`` to an existing SmaugLab aug-config file.

    Same lookup rules as ``pipeline_2d.gpu_pipeline._resolve_aug_config``:
    absolute path if it exists, else ``discslabel/augmentations/<name>[.json]``.
    """
    if os.path.isfile(name):
        return name

    import discslabel as _pkg

    pkg_dir = os.path.dirname(os.path.abspath(_pkg.__file__))
    base = os.path.basename(name)
    candidate = os.path.join(pkg_dir, "augmentations", base)
    if os.path.isfile(candidate):
        return candidate
    if not candidate.endswith(".json"):
        with_ext = candidate + ".json"
        if os.path.isfile(with_ext):
            return with_ext
    raise FileNotFoundError(
        f"Could not resolve aug_config '{name}'. Tried the literal path and "
        f"{candidate}. Point it at an absolute path or drop the file under "
        f"discslabel/augmentations/."
    )


def _gaussian_kernel_3d(
    sigma: float, radius: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """A (1, 1, K, K, K) isotropic 3D Gaussian kernel normalised so the peak = 1."""
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    kernel = torch.exp(-(zz * zz + yy * yy + xx * xx) / (2.0 * sigma * sigma))
    kernel = kernel.clamp(min=0.0)
    k = 2 * radius + 1
    return kernel.view(1, 1, k, k, k)


def build_heatmaps_3d(
    label_3d: torch.Tensor,
    sigma: float,
    radius: int,
    anchor_labels: Tuple[int, ...] = tuple(v for v in LANDMARK_CHANNEL_LABELS if v is not None),
) -> torch.Tensor:
    """Convert a 3D disc-label volume into a 5-channel heatmap stack.

    ``label_3d`` : (B, 1, D, H, W) with integer disc values (0..25). The output
    is (B, NUM_CLASSES, D, H, W) with ``clamp(0, 1)`` Gaussian peaks:

        channel 0 : union of every visible disc (any label > 0)
        channel i : the specific anchor disc (``LANDMARK_CHANNEL_LABELS[i]``)
    """
    assert label_3d.dim() == 5 and label_3d.size(1) == 1, (
        f"Expected (B, 1, D, H, W), got {tuple(label_3d.shape)}"
    )
    device = label_3d.device
    dtype = torch.float32

    label_int = label_3d.to(torch.int32).squeeze(1)  # (B, D, H, W)

    masks = [(label_int > 0).to(dtype).unsqueeze(1)]  # all-discs channel
    for anchor in anchor_labels:
        masks.append((label_int == int(anchor)).to(dtype).unsqueeze(1))
    stacked = torch.cat(masks, dim=1)  # (B, NUM_CLASSES, D, H, W)

    kernel = _gaussian_kernel_3d(sigma=sigma, radius=radius, device=device, dtype=dtype)
    B, C, D, H, W = stacked.shape
    # Depthwise 3D convolution: one Gaussian kernel broadcast across every channel.
    weight = kernel.expand(C, 1, kernel.size(-3), kernel.size(-2), kernel.size(-1))
    heatmap = F.conv3d(stacked, weight, padding=radius, groups=C)
    return heatmap.clamp_(0.0, 1.0)


class GpuBatchProcessor(nn.Module):
    """Augment on 3D and build 5-channel 3D heatmaps.

    The augmentation module is lazy-constructed on the first forward so a
    missing SmaugLab install stays harmless when ``aug_config`` is ``None``.
    """

    def __init__(
        self,
        aug_config: Optional[str] = None,
        heatmap_sigma: float = 3.0,
        heatmap_radius: int = 6,
    ):
        super().__init__()
        self.aug_config = aug_config
        self.heatmap_sigma = float(heatmap_sigma)
        self.heatmap_radius = int(heatmap_radius)
        self._aug: Optional[nn.Module] = None

    def _build_aug(self) -> nn.Module:
        if self.aug_config is None:
            return nn.Identity()
        from smauglab.transforms.gpu.transforms import AugTransformsGPU

        return AugTransformsGPU(_resolve_aug_config(self.aug_config))

    def _ensure_aug(self) -> nn.Module:
        if self._aug is None:
            self._aug = self._build_aug()
        return self._aug

    def _augment_3d(
        self, image: torch.Tensor, seg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aug = self._ensure_aug()
        if isinstance(aug, nn.Identity):
            return image, seg
        out = aug(image, seg)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            return out[0], out[1]
        raise RuntimeError(f"Unexpected AugTransformsGPU output: {type(out)}")

    def forward(
        self,
        image_3d: torch.Tensor,
        label_3d: Optional[torch.Tensor] = None,
        seg_3d: Optional[torch.Tensor] = None,
        apply_augmentation: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return ``(image_3d, heatmap_3d, label_3d, seg_3d)``.

        ``label_3d`` may be ``None`` at inference time (no ground truth); then
        ``heatmap_3d`` is ``None`` and only the (optionally augmented) image is
        returned. The disc label is left untouched by augmentation — configure
        ``aug_config`` with intensity-only transforms unless you wire a
        spatial-replay path for the label.
        """
        if image_3d.dim() != 5:
            raise ValueError(f"image_3d must be (B, 1, D, H, W), got {tuple(image_3d.shape)}")

        if apply_augmentation and seg_3d is not None:
            image_3d, seg_3d = self._augment_3d(image_3d, seg_3d)

        if label_3d is None:
            return image_3d, None, None, seg_3d

        heatmap_3d = build_heatmaps_3d(
            label_3d,
            sigma=self.heatmap_sigma,
            radius=self.heatmap_radius,
        )
        return image_3d, heatmap_3d, label_3d, seg_3d
