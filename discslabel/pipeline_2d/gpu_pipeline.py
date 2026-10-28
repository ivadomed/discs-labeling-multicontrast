"""GPU-side batch processor.

Takes a batch of 3D ``(image, label, seg)`` volumes, optionally runs the
SmaugLab 3D GPU augmentations, extracts the mid-sagittal slice, projects the
3D point-wise disc label onto the mid-sagittal plane, and builds the
5-channel Gaussian heatmap target. When ``elastic_config`` is set, a 2D
kornia random elastic transform is then applied jointly to the image and
heatmap to simulate spine curvature.

All operations run on the GPU with no CPU round-trips.

Expected tensor layout:
    image : (B, 1, D, H, W) float, intensities in [0, 1]
    label : (B, 1, D, H, W) float, integer disc values (0 = background, 1..25 = discs)
    seg   : (B, NUM_SEG_CHANNELS, D, H, W) binary tensor — one channel per
            anatomical structure (``discslabel.SEG_CHANNELS``):
                0: vertebrae, 1: discs, 2: spinal cord, 3: spinal canal.
            Used as the SECOND parameter of ``AugTransformsGPU`` — the
            point-wise disc landmark volume is NOT fed to the augmentation
            pipeline, so the aug config should stay intensity-only (or add
            a dedicated spatial-replay path for the disc label if you want
            to use spatial transforms).

Prepared NIfTIs are RPI, so axis D = R→L, axis H = P→A, axis W = I→S. The
mid-sagittal slice is ``image[..., D // 2, :, :]`` — that is axis 2 in the
batched (B, C, D, H, W) tensor.
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

    Mirrors the ``data_json`` lookup used by the pipeline scripts:
        1. if ``name`` already points to a real file, use it as-is;
        2. otherwise look for its basename under the bundled
           ``discslabel/augmentations/`` folder (with and without a ``.json``
           suffix).
    Raises ``FileNotFoundError`` with both tried paths when neither resolves.
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


_SAGITTAL_AXIS = 2  # (B, C, D, H, W) with D = R-L in RPI-oriented volumes


def _gaussian_kernel_2d(sigma: float, radius: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A (1, 1, K, K) isotropic Gaussian kernel normalised so the peak = 1."""
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    kernel = kernel.clamp(min=0.0)
    return kernel.view(1, 1, 2 * radius + 1, 2 * radius + 1)


def _project_label_to_2d(label_3d: torch.Tensor) -> torch.Tensor:
    """Collapse the R-L axis of a 3D point-wise label volume onto the mid-sagittal plane.

    ``label_3d`` : (B, 1, D, H, W) with discrete integer-valued voxels. The max
    along the R-L axis preserves the disc value when the point is on the midline
    (which is where the posterior-tip annotations sit).
    """
    return label_3d.amax(dim=_SAGITTAL_AXIS)  # (B, 1, H, W)


def build_heatmaps_2d(
    label_2d: torch.Tensor,
    sigma: float,
    radius: int,
    anchor_labels: Tuple[int, ...] = tuple(v for v in LANDMARK_CHANNEL_LABELS if v is not None),
) -> torch.Tensor:
    """Convert a 2D disc-label map into a 5-channel heatmap stack.

    ``label_2d`` : (B, 1, H, W) with integer disc values (0..25). The output
    is (B, NUM_CLASSES, H, W) with ``clamp(0, 1)`` Gaussian peaks:

        channel 0 : union of every visible disc (any label > 0)
        channel i : the specific anchor disc (``LANDMARK_CHANNEL_LABELS[i]``)
    """
    assert label_2d.dim() == 4 and label_2d.size(1) == 1, (
        f"Expected (B, 1, H, W), got {tuple(label_2d.shape)}"
    )
    device = label_2d.device
    dtype = torch.float32

    label_int = label_2d.to(torch.int32).squeeze(1)  # (B, H, W)

    masks = [(label_int > 0).to(dtype).unsqueeze(1)]  # all-discs channel
    for anchor in anchor_labels:
        masks.append((label_int == int(anchor)).to(dtype).unsqueeze(1))
    stacked = torch.cat(masks, dim=1)  # (B, NUM_CLASSES, H, W)

    kernel = _gaussian_kernel_2d(sigma=sigma, radius=radius, device=device, dtype=dtype)
    B, C, H, W = stacked.shape
    # Depthwise convolution: one Gaussian kernel broadcast across every channel.
    weight = kernel.expand(C, 1, kernel.size(-2), kernel.size(-1))
    heatmap = F.conv2d(stacked, weight, padding=radius, groups=C)
    return heatmap.clamp_(0.0, 1.0)


class GpuBatchProcessor(nn.Module):
    """Augment on 3D, slice to mid-sagittal, build 5-channel heatmaps.

    The augmentation module is lazy-constructed on the first forward so a
    missing SmaugLab install stays harmless when ``aug_config`` is ``None``.
    """

    def __init__(
        self,
        aug_config: Optional[str] = None,
        heatmap_sigma: float = 8.0,
        heatmap_radius: int = 20,
        elastic_config: Optional[dict] = None,
    ):
        super().__init__()
        self.aug_config = aug_config
        self.heatmap_sigma = float(heatmap_sigma)
        self.heatmap_radius = int(heatmap_radius)
        self.elastic_config = elastic_config
        self._aug: Optional[nn.Module] = None
        self._elastic: Optional[nn.Module] = None

    def _build_aug(self) -> nn.Module:
        if self.aug_config is None:
            return nn.Identity()
        # SmaugLab's AugTransformsGPU is a kornia AugmentationSequential wrapper that
        # takes paired (image, mask) tensors through the same spatial params.
        from smauglab.transforms.gpu.transforms import AugTransformsGPU

        return AugTransformsGPU(_resolve_aug_config(self.aug_config))

    def _ensure_aug(self) -> nn.Module:
        if self._aug is None:
            self._aug = self._build_aug()
        return self._aug

    def _build_elastic(self) -> Optional[nn.Module]:
        """Build the 2D random elastic transform used to simulate spine curvature.

        Kornia is only pulled in when ``elastic_config`` is set, keeping it off
        the inference import path.
        """
        if not self.elastic_config:
            return None
        import kornia.augmentation as K

        cfg = self.elastic_config
        return K.RandomElasticTransform(
            kernel_size=tuple(cfg.get("kernel_size", (63, 63))),
            sigma=tuple(cfg.get("sigma", (32.0, 32.0))),
            alpha=tuple(cfg.get("alpha", (1.0, 1.0))),
            align_corners=cfg.get("align_corners", False),
            resample=cfg.get("resample", 1),
            padding_mode=cfg.get("padding_mode", "zeros"),
            same_on_batch=cfg.get("same_on_batch", False),
            p=cfg.get("p", 0.5),
        )

    def _ensure_elastic(self) -> Optional[nn.Module]:
        if self._elastic is None:
            self._elastic = self._build_elastic()
        return self._elastic

    def _apply_elastic_2d(
        self, image_2d: torch.Tensor, heatmap_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Warp image and heatmap with the same random displacement field.

        Concatenating along the channel axis guarantees kornia samples one
        displacement field per batch item and applies it to both tensors in a
        single call — no replay plumbing needed.
        """
        elastic = self._ensure_elastic()
        if elastic is None:
            return image_2d, heatmap_2d

        n_img = image_2d.size(1)
        n_hm = heatmap_2d.size(1)
        stacked = torch.cat([image_2d, heatmap_2d], dim=1)
        deformed = elastic(stacked)
        return deformed[:, :n_img], deformed[:, n_img : n_img + n_hm]

    def _augment_3d(
        self, image: torch.Tensor, seg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        aug = self._ensure_aug()
        if isinstance(aug, nn.Identity):
            return image, seg
        # AugmentationSequential expects paired inputs; seg stays discrete via
        # kornia's nearest resampling because the entry's data_key is "mask".
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
        """Return ``(image_2d, heatmap_2d, label_2d, seg_2d)``.

        ``label_3d`` may be ``None`` at inference time (no ground truth); then
        ``heatmap_2d`` and ``label_2d`` are also ``None``. ``seg_3d`` is
        passed through ``AugTransformsGPU`` as the "mask" tensor; the disc
        label is left untouched so it is only sensible to configure
        ``aug_config`` with intensity-only transforms unless you add a
        separate spatial-replay path for the label.
        """
        if image_3d.dim() != 5:
            raise ValueError(f"image_3d must be (B, 1, D, H, W), got {tuple(image_3d.shape)}")

        if apply_augmentation and seg_3d is not None:
            image_3d, seg_3d = self._augment_3d(image_3d, seg_3d)

        mid = image_3d.size(_SAGITTAL_AXIS) // 2
        image_2d = image_3d.narrow(_SAGITTAL_AXIS, mid, 1).squeeze(_SAGITTAL_AXIS)  # (B, 1, H, W)
        seg_2d = (
            seg_3d.narrow(_SAGITTAL_AXIS, mid, 1).squeeze(_SAGITTAL_AXIS)
            if seg_3d is not None
            else None
        )

        if label_3d is None:
            return image_2d, None, None, seg_2d

        label_2d = _project_label_to_2d(label_3d)
        heatmap_2d = build_heatmaps_2d(
            label_2d,
            sigma=self.heatmap_sigma,
            radius=self.heatmap_radius,
        )

        if apply_augmentation:
            image_2d, heatmap_2d = self._apply_elastic_2d(image_2d, heatmap_2d)

        return image_2d, heatmap_2d, label_2d, seg_2d
