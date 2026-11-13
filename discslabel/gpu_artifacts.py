"""GPU MRI-artifact augmentations, shared by the 2D and 3D training pipelines.

Pure-torch ports of the torchio transforms used by SmaugLab's CPU
``ArtifactTransform`` (``smauglab/transforms/cpu/artifact.py``), so they run on
the training batch already sitting on the GPU instead of in DataLoader
workers:

    =============  ===================================================
    name           what it simulates (same maths as ``tio.Random*``)
    =============  ===================================================
    motion         patient motion: k-space assembled from rigidly
                   moved copies of the volume (``RandomMotion``)
    ghosting       periodic k-space line corruption along one axis
                   (``RandomGhosting``)
    spike          k-space spike → stripe / herringbone pattern
                   (``RandomSpike(intensity=(1, 2))``)
    bias_field     smooth multiplicative polynomial field
                   (``RandomBiasField``)
    blur           anisotropic Gaussian blur (``RandomBlur``)
    noise          additive Gaussian noise (``RandomNoise``)
    swap           random patch swaps (``RandomSwap``)
    =============  ===================================================

plus ``RandomLowResAxisGPU``: downsample one axis (default: S-I, the last
axis of RPI volumes) to a random spacing and upsample back, simulating thick
slices / partial volume along that axis.

Every transform takes a ``(B, C, D, H, W)`` float tensor and draws its random
parameters independently for each batch item. Intensities are processed in a
min-shifted (non-negative) space so multiplicative effects behave on z-scored
inputs, and the per-sample mean / std are restored afterwards
(``retain_stats``) so the network keeps seeing z-scored intensities. Noise
std is expressed in units of the sample's std for the same reason. Only the
image is touched: none of these transforms moves anatomy, so the disc labels
and heatmaps stay valid.

Config (pipeline JSON)::

    "artifact_config": {
        "p": 0.3,              # probability that a batch item is corrupted
        "random_pick": true,   # one random enabled artifact (else all, in order)
        "retain_stats": true,
        "motion": true,        # true → defaults below, false → disabled,
        "ghosting": true,      # or a dict overriding the defaults, e.g.
        "spike": {"intensity": [1, 2]},
        ...
    },
    "lowres_si_config": {"p": 0.25, "spacing": [1.0, 5.0]}   # final S-I spacing, mm
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SPATIAL_DIMS: Tuple[int, int, int] = (-3, -2, -1)


# ---------------------------------------------------------------------------
# Random helpers (torch RNG so ``torch.manual_seed`` makes runs reproducible)
# ---------------------------------------------------------------------------


def _as_range(value, symmetric: bool = False) -> Tuple[float, float]:
    """``a`` → ``(-a, a)`` when ``symmetric`` else ``(0, a)``; pairs kept as-is."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"Expected a (min, max) pair, got {value!r}")
        return float(value[0]), float(value[1])
    v = float(value)
    return (-abs(v), abs(v)) if symmetric else (0.0, v)


def _uniform(lo: float, hi: float, shape=(), device=None) -> torch.Tensor:
    return torch.rand(shape, device=device) * (hi - lo) + lo


def _randint(lo: int, hi: int) -> int:
    """Integer in ``[lo, hi]`` (inclusive)."""
    return int(torch.randint(int(lo), int(hi) + 1, (1,)).item())


# ---------------------------------------------------------------------------
# k-space helpers
# ---------------------------------------------------------------------------


def _fft(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftn(x.float(), dim=SPATIAL_DIMS), dim=SPATIAL_DIMS)


def _ifft(k: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftn(torch.fft.ifftshift(k, dim=SPATIAL_DIMS), dim=SPATIAL_DIMS).real


# ---------------------------------------------------------------------------
# Per-sample artifact functions: ``x`` is one sample ``(C, D, H, W)``,
# non-negative (min-shifted). They return a tensor of the same shape.
# ---------------------------------------------------------------------------


def _rotation_matrix(radians: torch.Tensor) -> torch.Tensor:
    """``Rz @ Ry @ Rx`` from three Euler angles."""
    a, b, c = (float(r) for r in radians)
    rx = torch.tensor([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = torch.tensor([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = torch.tensor([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _rigid_resample(x: torch.Tensor, rotation: torch.Tensor, translation_vox: torch.Tensor) -> torch.Tensor:
    """Rotate about the volume centre and translate (voxels), linear interpolation.

    ``affine_grid`` works in normalised ``(x, y, z) = (W, H, D)`` coordinates;
    the rigid motion is defined in voxel units so the rotation stays rigid on
    anisotropic patch shapes: ``theta = N @ [R | t] @ N^-1`` with
    ``N = diag(2 / size)``.
    """
    C, D, H, W = x.shape
    size = torch.tensor([W, H, D], dtype=torch.float32)
    n = torch.diag(2.0 / size)
    n_inv = torch.diag(size / 2.0)
    theta = torch.zeros(1, 3, 4)
    theta[0, :, :3] = n @ rotation @ n_inv
    theta[0, :, 3] = n @ translation_vox
    theta = theta.to(device=x.device, dtype=torch.float32)
    grid = F.affine_grid(theta, (1, C, D, H, W), align_corners=False)
    # zeros padding == torchio's "min value" padding because x is min-shifted.
    return F.grid_sample(x[None].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]


def motion(
    x: torch.Tensor,
    degrees=10.0,
    translation=10.0,
    num_transforms: int = 2,
    perturbation: float = 0.3,
) -> torch.Tensor:
    """``tio.RandomMotion``: k-space segments from rigidly moved copies.

    ``num_transforms`` rigid motions happen at random times; k-space is split
    along the last axis into the matching segments, each filled from the
    corresponding moved copy. The original (unmoved) spectrum is placed in
    the segment that contains the k-space centre, as torchio does.
    Translation is in voxels (= mm at the 1 mm training resolution).
    """
    deg_lo, deg_hi = _as_range(degrees, symmetric=True)
    tr_lo, tr_hi = _as_range(translation, symmetric=True)
    step = 1.0 / (num_transforms + 1)
    times = torch.arange(1, num_transforms + 1, dtype=torch.float32) * step
    times += _uniform(-step * perturbation, step * perturbation, (num_transforms,))

    images = [x]
    for _ in range(num_transforms):
        radians = torch.deg2rad(_uniform(deg_lo, deg_hi, (3,)))
        trans = _uniform(tr_lo, tr_hi, (3,))
        images.append(_rigid_resample(x, _rotation_matrix(radians), trans))
    spectra = [_fft(img) for img in images]

    # sort_spectra: the original spectrum fills the centre of k-space.
    above = torch.nonzero(times > 0.5)
    index = int(above.min()) if len(above) else len(spectra) - 1
    spectra[0], spectra[index] = spectra[index], spectra[0]

    last = x.shape[-1]
    bounds = [int(last * float(t)) for t in times] + [last]
    result = torch.empty_like(spectra[0])
    ini = 0
    for spectrum, fin in zip(spectra, bounds):
        result[..., ini:fin] = spectrum[..., ini:fin]
        ini = fin
    return _ifft(result)


def ghosting(
    x: torch.Tensor,
    num_ghosts=(4, 10),
    axes: Sequence[int] = (0, 1, 2),
    intensity=(0.5, 1.0),
    restore: Optional[float] = None,
) -> torch.Tensor:
    """``tio.RandomGhosting``: damp every ``num_ghosts``-th k-space plane."""
    ng_lo, ng_hi = (int(v) for v in _as_range(num_ghosts))
    n_ghosts = _randint(ng_lo, ng_hi)
    axis = int(axes[_randint(0, len(axes) - 1)])
    amount = float(_uniform(*_as_range(intensity)))
    if n_ghosts == 0 or amount == 0:
        return x
    k = _fft(x)
    dim = 1 + axis  # skip channel axis
    size = k.shape[dim]
    mid = size // 2
    if restore is None:
        keep = slice(mid, mid + 1)
    else:
        n_restore = int(round(float(restore) * size))
        keep = slice(mid - n_restore // 2, mid + n_restore // 2)
    idx_keep = [slice(None)] * k.ndim
    idx_keep[dim] = keep
    saved = k[tuple(idx_keep)].clone()
    idx_planes = [slice(None)] * k.ndim
    idx_planes[dim] = slice(None, None, n_ghosts)
    k[tuple(idx_planes)] *= 1.0 - amount
    k[tuple(idx_keep)] = saved
    return _ifft(k)


def spike(x: torch.Tensor, num_spikes=(1, 1), intensity=(1.0, 2.0)) -> torch.Tensor:
    """``tio.RandomSpike``: add ``max|k| * intensity`` at random k-space points."""
    ns_lo, ns_hi = (int(v) for v in _as_range(num_spikes))
    n_spikes = _randint(ns_lo, ns_hi)
    factor = float(_uniform(*_as_range(intensity)))
    if n_spikes == 0 or factor == 0:
        return x
    k = _fft(x)
    shape = torch.tensor(k.shape[1:])
    positions = torch.floor(torch.rand(n_spikes, 3) * shape).long()
    artifact = k.abs().amax(dim=SPATIAL_DIMS) * factor  # (C,)
    for i, j, l in positions.tolist():
        k[:, i, j, l] += artifact
    return _ifft(k)


def _bias_field(shape: Sequence[int], coefficients: torch.Tensor, order: int, device) -> torch.Tensor:
    axes = []
    for n in shape:
        r = torch.arange(n, device=device, dtype=torch.float32) - n / 2 + 0.5
        axes.append(r / r.max().clamp_min(1e-6))
    zd, yh, xw = torch.meshgrid(*axes, indexing="ij")
    field = torch.zeros(tuple(shape), device=device)
    i = 0
    for a in range(order + 1):
        for b in range(order + 1 - a):
            for c in range(order + 1 - (a + b)):
                field += coefficients[i] * zd**a * yh**b * xw**c
                i += 1
    return torch.exp(field)


def bias_field(x: torch.Tensor, coefficients=0.5, order: int = 3) -> torch.Tensor:
    """``tio.RandomBiasField``: multiply by ``exp(polynomial of degree order)``."""
    lo, hi = _as_range(coefficients, symmetric=True)
    n_coeffs = sum(
        1 for a in range(order + 1) for b in range(order + 1 - a) for _ in range(order + 1 - (a + b))
    )
    coeffs = _uniform(lo, hi, (n_coeffs,)).tolist()
    return x * _bias_field(x.shape[1:], coeffs, order, x.device)


def _gaussian_kernel_1d(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(coords**2) / (2 * sigma * sigma))
    return k / k.sum()


def blur(x: torch.Tensor, std=(0.0, 2.0)) -> torch.Tensor:
    """``tio.RandomBlur``: separable Gaussian, one random std (voxels) per axis."""
    lo, hi = _as_range(std)
    sigmas = _uniform(lo, hi, (3,)).tolist()
    out = x[:, None].float()  # (C, 1, D, H, W) — channels as batch
    for axis, sigma in enumerate(sigmas):
        if sigma < 0.1:
            continue
        k = _gaussian_kernel_1d(sigma, x.device)
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        r = k.numel() // 2
        # F.pad order is (W_l, W_r, H_l, H_r, D_l, D_r).
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = r
        if out.shape[2 + axis] > r:
            out = F.pad(out, pad, mode="reflect")
        else:
            out = F.pad(out, pad, mode="replicate")
        out = F.conv3d(out, k.view(shape))
    return out[:, 0]


def noise(x: torch.Tensor, mean=0.0, std=(0.0, 0.25)) -> torch.Tensor:
    """``tio.RandomNoise``; ``std`` is relative to the sample's own std."""
    m = float(_uniform(*_as_range(mean, symmetric=True))) if isinstance(mean, (list, tuple)) else float(mean)
    s = float(_uniform(*_as_range(std)))
    scale = x.float().std().clamp_min(1e-6)
    return x + (torch.randn_like(x, dtype=torch.float32) * s + m) * scale


def swap(x: torch.Tensor, patch_size=15, num_iterations: int = 100) -> torch.Tensor:
    """``tio.RandomSwap``: swap ``num_iterations`` pairs of random cubic patches."""
    shape = x.shape[1:]
    if isinstance(patch_size, (list, tuple)):
        ps = [int(p) for p in patch_size]
    else:
        ps = [int(patch_size)] * 3
    ps = [min(p, s) for p, s in zip(ps, shape)]
    out = x.clone()
    for _ in range(int(num_iterations)):
        while True:
            a = [_randint(0, s - p) for s, p in zip(shape, ps)]
            b = [_randint(0, s - p) for s, p in zip(shape, ps)]
            # Reject a second patch fully inside the first (torchio's rule).
            if not all(bi >= ai and bi + p <= ai + p for ai, bi, p in zip(a, b, ps)) or ps == list(shape):
                break
        sa = (slice(None),) + tuple(slice(i, i + p) for i, p in zip(a, ps))
        sb = (slice(None),) + tuple(slice(i, i + p) for i, p in zip(b, ps))
        pa = out[sa].clone()
        out[sa] = out[sb]
        out[sb] = pa
    return out


#: Artifact name → per-sample function, in application order (same as
#: SmaugLab's ``ARTIFACTS`` table).
ARTIFACTS: Dict[str, Callable[..., torch.Tensor]] = {
    "motion": motion,
    "ghosting": ghosting,
    "spike": spike,
    "bias_field": bias_field,
    "blur": blur,
    "noise": noise,
    "swap": swap,
}


# ---------------------------------------------------------------------------
# Batch-level modules
# ---------------------------------------------------------------------------


def _sample_stats(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x.mean(), x.std().clamp_min(1e-6)


def apply_artifact(
    name: str,
    sample: torch.Tensor,
    params: Optional[dict] = None,
    retain_stats: bool = True,
) -> torch.Tensor:
    """Apply one artifact to one ``(C, D, H, W)`` sample.

    Works in a min-shifted (non-negative) space, then shifts back and, with
    ``retain_stats``, restores the sample's mean / std.
    """
    fn = ARTIFACTS[name]
    dtype = sample.dtype
    x = sample.float()
    mean, std = _sample_stats(x)
    lo = x.amin()
    out = fn(x - lo, **(params or {})) + lo
    if retain_stats:
        out_mean, out_std = _sample_stats(out)
        out = (out - out_mean) / out_std * std + mean
    return out.to(dtype)


def _parse_artifacts(cfg: dict) -> Dict[str, dict]:
    enabled: Dict[str, dict] = {}
    for name in ARTIFACTS:
        value = cfg.get(name, False)
        if value is True:
            enabled[name] = {}
        elif isinstance(value, dict):
            params = dict(value)
            if params.pop("enabled", True):
                enabled[name] = params
        elif value not in (False, None):
            raise ValueError(f"artifact_config.{name} must be a bool or a dict, got {value!r}")
    unknown = set(cfg) - set(ARTIFACTS) - {"p", "random_pick", "retain_stats"}
    if unknown:
        raise ValueError(f"Unknown artifact_config keys: {sorted(unknown)}")
    return enabled


class RandomArtifactsGPU(nn.Module):
    """GPU counterpart of SmaugLab's CPU ``ArtifactTransform``.

    Each batch item is corrupted with probability ``p``. With ``random_pick``
    exactly one enabled artifact is drawn for it, otherwise every enabled
    artifact is applied in ``ARTIFACTS`` order.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.p = float(cfg.get("p", 1.0))
        self.random_pick = bool(cfg.get("random_pick", True))
        self.retain_stats = bool(cfg.get("retain_stats", True))
        self.artifacts = _parse_artifacts(cfg)

    def extra_repr(self) -> str:
        return f"p={self.p}, random_pick={self.random_pick}, artifacts={list(self.artifacts)}"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if not self.artifacts or self.p <= 0:
            return image
        names = list(self.artifacts)
        out = image.clone()
        for b in range(image.shape[0]):
            if float(torch.rand(())) >= self.p:
                continue
            chosen = [names[_randint(0, len(names) - 1)]] if self.random_pick else names
            sample = out[b]
            for name in chosen:
                sample = apply_artifact(name, sample, self.artifacts[name], self.retain_stats)
            out[b] = sample
        return out


_AXIS_NAMES = {"RL": 0, "LR": 0, "PA": 1, "AP": 1, "SI": 2, "IS": 2}


def lowres_axis(
    sample: torch.Tensor, factor: float, axis: int = 2, down_mode: str = "area", up_mode: str = "linear"
) -> torch.Tensor:
    """Downsample ``axis`` of a ``(C, D, H, W)`` sample by ``factor`` and upsample back.

    ``down_mode="area"`` averages the voxels merged into each thick slice
    (partial-volume effect); ``"nearest"`` drops slices instead.
    ``up_mode`` is ``"linear"`` (trilinear) or ``"nearest"``.
    """
    shape = list(sample.shape[1:])
    low = list(shape)
    low[axis] = max(1, int(round(shape[axis] / float(factor))))
    if low == shape:
        return sample
    x = sample[None].float()
    x = F.interpolate(x, size=low, mode=down_mode)
    if up_mode == "nearest":
        x = F.interpolate(x, size=shape, mode="nearest")
    else:
        x = F.interpolate(x, size=shape, mode="trilinear", align_corners=False)
    return x[0].to(sample.dtype)


def parse_spacing(resolution) -> Tuple[float, float, float]:
    """``"1x1x1"`` / ``1.0`` / ``[1, 1, 3]`` → per-axis spacing in mm on ``(D, H, W)``."""
    if resolution is None:
        return (1.0, 1.0, 1.0)
    if isinstance(resolution, str):
        parts = resolution.lower().split("x")
    elif isinstance(resolution, (list, tuple)):
        parts = list(resolution)
    else:
        parts = [resolution] * 3
    if len(parts) != 3:
        raise ValueError(f"resolution must have 3 values (mm), got {resolution!r}")
    return tuple(float(v) for v in parts)  # type: ignore[return-value]


class RandomLowResAxisGPU(nn.Module):
    """Simulate a low resolution along one axis (default: S-I).

    With probability ``p`` per batch item, a target spacing is drawn
    uniformly in ``spacing`` (mm, e.g. ``[1, 5]``), the axis is downsampled
    from its current spacing (``voxel_spacing``, i.e. the config
    ``resolution``) to that target and upsampled back to the original size.
    Targets at or below the current spacing leave the sample unchanged.
    ``axis`` is ``"SI"`` / ``"PA"`` / ``"RL"`` or 0..2 on ``(D, H, W)``;
    prepared volumes are RPI, so S-I is the last axis.
    """

    def __init__(self, cfg: dict, voxel_spacing=(1.0, 1.0, 1.0)):
        super().__init__()
        if "factor" in cfg:
            raise ValueError(
                "lowres_si_config.factor was replaced by 'spacing': the [min, max] "
                "final spacing in mm along the axis (e.g. [1, 5])."
            )
        self.p = float(cfg.get("p", 1.0))
        self.spacing = _as_range(cfg.get("spacing", (1.0, 5.0)))
        if not 0 < self.spacing[0] <= self.spacing[1]:
            raise ValueError(f"lowres spacing must be 0 < min <= max (mm), got {self.spacing}")
        axis = cfg.get("axis", "SI")
        self.axis = _AXIS_NAMES[axis.upper()] if isinstance(axis, str) else int(axis)
        self.current_spacing = float(parse_spacing(voxel_spacing)[self.axis])
        self.down_mode = str(cfg.get("down_mode", "area"))
        self.up_mode = str(cfg.get("up_mode", "linear"))

    def extra_repr(self) -> str:
        return (
            f"p={self.p}, spacing={self.spacing} mm (current {self.current_spacing} mm), "
            f"axis={self.axis}, down={self.down_mode}, up={self.up_mode}"
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return image
        out = image.clone()
        for b in range(image.shape[0]):
            if float(torch.rand(())) >= self.p:
                continue
            target = float(_uniform(*self.spacing))
            factor = target / self.current_spacing
            if factor <= 1.0:
                continue
            out[b] = lowres_axis(out[b], factor, self.axis, self.down_mode, self.up_mode)
        return out


def build_image_augmentations(
    artifact_config: Optional[dict] = None,
    lowres_config: Optional[dict] = None,
    voxel_spacing=(1.0, 1.0, 1.0),
) -> Optional[nn.Module]:
    """Image-only GPU augmentations run after the SmaugLab pipeline.

    Order: S-I low resolution first (acquisition geometry), then artifacts.
    ``voxel_spacing`` is the spacing (mm) of the training volumes, i.e. the
    config ``resolution``. Returns ``None`` when both configs are empty / disabled.
    """
    modules = []
    if lowres_config:
        modules.append(RandomLowResAxisGPU(lowres_config, voxel_spacing))
    if artifact_config:
        modules.append(RandomArtifactsGPU(artifact_config))
    return nn.Sequential(*modules) if modules else None
