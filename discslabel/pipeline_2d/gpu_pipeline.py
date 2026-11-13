"""GPU-side batch processor.

Takes a batch of 3D ``(image, label, seg)`` volumes, optionally runs the
SmaugLab 3D GPU augmentations, extracts the mid-sagittal slice, projects the
3D point-wise disc label onto the mid-sagittal plane, and builds the
5-channel Gaussian heatmap target. When ``elastic_config`` is set, a 2D
kornia random elastic transform warps the image and moves the disc points
with the same field (simulating spine curvature) BEFORE the heatmap is
rendered, so the Gaussians stay circular. ``artifact_config`` /
``lowres_config`` add image-only MRI artifacts and an S-I low-resolution
simulation on the 3D batch right after the SmaugLab augmentations (see
``discslabel.gpu_artifacts``).

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

from discslabel import LANDMARK_CHANNEL_LABELS
from discslabel.gpu_artifacts import build_image_augmentations
from discslabel.pipeline_2d.predictor import INPUT_MODES, slab_indices


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


_NEWTON_ITERS = 3


def _invert_sampling_map(src: torch.Tensor, b: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Solve ``g(p) = q`` for each point, where ``g = src[b]`` maps output → input pixels.

    ``src`` : (B, 2, H, W) input (h, w) sampled by each output pixel.
    ``b`` / ``q`` : (N,) batch indices and (N, 2) input positions.
    Returns the (N, 2) output positions ``p``.

    Global start (the output pixel whose ``g`` is closest to ``q``; robust to
    large or locally compressive fields where fixed-point iteration fails),
    then a few Newton steps with a finite-difference Jacobian for sub-pixel
    precision.
    """
    _, _, H, W = src.shape
    flat = src.flatten(2)  # (B, 2, H*W)
    p = torch.empty_like(q)
    for bi in b.unique().tolist():  # one (n_b, H*W) distance matrix per sample
        m = b == bi
        d2 = (flat[bi].T.unsqueeze(0) - q[m].unsqueeze(1)).square().sum(-1)  # (n_b, H*W)
        idx = d2.argmin(dim=1)
        p[m] = torch.stack([idx // W, idx % W], dim=1).to(q.dtype)
    eh = q.new_tensor([0.5, 0.0])
    ew = q.new_tensor([0.0, 0.5])
    for _ in range(_NEWTON_ITERS):
        r = q - _sample_bilinear(src, b, p)  # (N, 2)
        jh = _sample_bilinear(src, b, p + eh) - _sample_bilinear(src, b, p - eh)  # dg/dh
        jw = _sample_bilinear(src, b, p + ew) - _sample_bilinear(src, b, p - ew)  # dg/dw
        J = torch.stack([jh, jw], dim=2)  # (N, 2, 2): columns = d/dh, d/dw
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        ok = det.abs() > 1e-3  # skip near-singular (folded) spots: keep the grid start
        step = torch.stack(
            [J[:, 1, 1] * r[:, 0] - J[:, 0, 1] * r[:, 1], -J[:, 1, 0] * r[:, 0] + J[:, 0, 0] * r[:, 1]],
            dim=1,
        ) / det.where(ok, torch.ones_like(det)).unsqueeze(1)
        p = p + step.where(ok.unsqueeze(1), torch.zeros_like(step)).clamp(-1.0, 1.0)
    return p


def _sample_bilinear(field: torch.Tensor, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``field`` (B, C, H, W) at points ``p`` (N, 2) = (h, w) of batch items ``b``.

    Points outside the grid are clamped to the border. Returns (N, C).
    """
    _, _, H, W = field.shape
    h = p[:, 0].clamp(0, H - 1)
    w = p[:, 1].clamp(0, W - 1)
    h0 = h.floor().long().clamp(max=H - 2) if H > 1 else h.floor().long()
    w0 = w.floor().long().clamp(max=W - 2) if W > 1 else w.floor().long()
    h1 = (h0 + 1).clamp(max=H - 1)
    w1 = (w0 + 1).clamp(max=W - 1)
    th = (h - h0).unsqueeze(1)
    tw = (w - w0).unsqueeze(1)
    f00 = field[b, :, h0, w0]
    f01 = field[b, :, h0, w1]
    f10 = field[b, :, h1, w0]
    f11 = field[b, :, h1, w1]
    return (f00 * (1 - tw) + f01 * tw) * (1 - th) + (f10 * (1 - tw) + f11 * tw) * th


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
        slab_slices: int = 1,
        slab_prob: float = 0.5,
        eval_input_mode: str = "avg",
        artifact_config: Optional[dict] = None,
        lowres_config: Optional[dict] = None,
        voxel_spacing=(1.0, 1.0, 1.0),
    ):
        super().__init__()
        if slab_slices < 1:
            raise ValueError(f"slab_slices must be >= 1, got {slab_slices}")
        if eval_input_mode not in INPUT_MODES:
            raise ValueError(f"eval_input_mode must be one of {INPUT_MODES}, got {eval_input_mode!r}")
        self.aug_config = aug_config
        self.heatmap_sigma = float(heatmap_sigma)
        self.heatmap_radius = int(heatmap_radius)
        self.elastic_config = elastic_config
        # Single-channel model input: either the mid-sagittal slice ("mid") or
        # the mean of the ``slab_slices`` R-L slices around it ("avg", a thick
        # slab, e.g. 10 mm). With augmentation on, each sample independently
        # gets "avg" with probability ``slab_prob``; otherwise
        # ``eval_input_mode`` is used for the whole batch.
        self.slab_slices = int(slab_slices)
        self.slab_prob = float(slab_prob)
        self.eval_input_mode = eval_input_mode
        self._aug: Optional[nn.Module] = None
        self._elastic: Optional[nn.Module] = None
        # Image-only GPU augmentations (S-I low resolution, MRI artifacts),
        # applied on the 3D batch after the SmaugLab pipeline.
        self.image_aug = build_image_augmentations(artifact_config, lowres_config, voxel_spacing)

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
        self, image_2d: torch.Tensor, label_2d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Warp the image and move the disc points with the same random displacement field.

        The heatmap is NOT warped: resampling a rendered Gaussian stretches it
        into an ellipse. Instead the disc POINTS are moved by the field and the
        heatmap is rendered afterwards (``build_heatmaps_2d``), so the Gaussians
        stay circular.

        Two pixel-coordinate channels (+ a ones channel) go through the same
        kornia call as the image, which yields the sampling map ``g``: output
        pixel ``p`` reads input location ``g(p)``. A disc at input location
        ``q`` therefore lands at the ``p`` solving ``g(p) = q`` (see
        ``_invert_sampling_map``). Points are rounded to the nearest pixel, so the targets keep
        the exact shape of non-augmented ones (peak = 1). Discs pushed out of
        the field of view are dropped.
        """
        elastic = self._ensure_elastic()
        if elastic is None:
            return image_2d, label_2d

        B, n_img, H, W = image_2d.shape
        device = image_2d.device
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=image_2d.dtype),
            torch.arange(W, device=device, dtype=image_2d.dtype),
            indexing="ij",
        )
        coords = torch.stack([yy, xx, torch.ones_like(yy)]).expand(B, 3, H, W)
        deformed = elastic(torch.cat([image_2d, coords], dim=1))
        image_out = deformed[:, :n_img]
        # Bilinear sampling near a zero-padded border scales every channel by
        # the in-bounds weight; dividing by the ones channel undoes it.
        weight = deformed[:, n_img + 2 : n_img + 3]
        src = deformed[:, n_img : n_img + 2] / weight.clamp_min(1e-6)  # (B, 2, H, W)

        label_out = torch.zeros_like(label_2d)
        b, h, w = label_2d[:, 0].nonzero(as_tuple=True)
        if b.numel() == 0:
            return image_out, label_out
        values = label_2d[b, 0, h, w]
        q = torch.stack([h, w], dim=1).to(src.dtype)  # (N, 2) input positions
        p = _invert_sampling_map(src, b, q)
        p_round = p.round().long()
        keep = (
            (p_round[:, 0] >= 0) & (p_round[:, 0] < H)
            & (p_round[:, 1] >= 0) & (p_round[:, 1] < W)
        )
        b, p_round, values = b[keep], p_round[keep], values[keep]
        keep = weight[b, 0, p_round[:, 0], p_round[:, 1]] > 0.5  # landed inside the warped image
        b, p_round, values = b[keep], p_round[keep], values[keep]
        label_out[b, 0, p_round[:, 0], p_round[:, 1]] = values
        return image_out, label_out

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
        if apply_augmentation and self.image_aug is not None:
            image_3d = self.image_aug(image_3d)

        D = image_3d.size(_SAGITTAL_AXIS)
        mid = D // 2
        # (B, 1, D, H, W) → (B, 1, H, W)
        mid_2d = image_3d.select(_SAGITTAL_AXIS, mid)
        if apply_augmentation:
            use_avg = torch.rand(image_3d.size(0), device=image_3d.device) < self.slab_prob
        else:
            use_avg = torch.full(
                (image_3d.size(0),), self.eval_input_mode == "avg", device=image_3d.device
            )
        if bool(use_avg.any()):
            slice_indices = torch.as_tensor(
                slab_indices(D, mid, self.slab_slices), device=image_3d.device
            )
            avg_2d = image_3d.index_select(_SAGITTAL_AXIS, slice_indices).mean(dim=_SAGITTAL_AXIS)
            image_2d = torch.where(use_avg.view(-1, 1, 1, 1), avg_2d, mid_2d)
        else:
            image_2d = mid_2d
        seg_2d = (
            seg_3d.narrow(_SAGITTAL_AXIS, mid, 1).squeeze(_SAGITTAL_AXIS)
            if seg_3d is not None
            else None
        )

        if label_3d is None:
            return image_2d, None, None, seg_2d

        label_2d = _project_label_to_2d(label_3d)
        if apply_augmentation:
            # Warp points, not the rendered heatmap, so the Gaussians stay circular.
            image_2d, label_2d = self._apply_elastic_2d(image_2d, label_2d)
        heatmap_2d = build_heatmaps_2d(
            label_2d,
            sigma=self.heatmap_sigma,
            radius=self.heatmap_radius,
        )

        return image_2d, heatmap_2d, label_2d, seg_2d
