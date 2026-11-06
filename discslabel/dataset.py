"""Dataset reader for prepared 3D NIfTI volumes.

The dataset returns 3D ``(image, label, seg)`` tensors with no 2D projection
applied: all augmentation, mid-sagittal extraction and heatmap generation
runs **on the GPU** via ``discslabel.pipeline_2d.gpu_pipeline.GpuBatchProcessor``
(or its 3D counterpart in ``discslabel.pipeline_3d``).

Per-sample pipeline:

    1. Load the case: from the memory-mapped ``.npy`` cache
       (``discslabel.cache``, only the crop is read) or from the NIfTIs
       (image, point-wise disc label, SEG_* files).
    2. Pick a 3D crop origin so that the extracted mid-sagittal slice of the
       crop contains **disc** voxels. The disc mask is the dense disc
       segmentation from SEG_SPINE (voxels with value ``>= spine_disc_min_label``);
       when SEG_SPINE is unavailable we fall back to the point-wise ``LABEL``
       (any non-zero disc landmark). In train mode the origin follows
       nnU-Net's sampling (``_pick_crop_origin_nnunet``): foreground-forced
       crops are centred on a voxel of a uniformly drawn disc, the others are
       uniformly random. In valid / inference mode we use the median disc
       slice and the exact disc centroid.
       2D pipeline, entries without SEG_SPINE: the point-wise LABEL does not
       say which sagittal slices cross the discs, so the mid-sagittal slice is
       the volume centre (as at inference), drawn uniformly in
       ``centre ± midline_jitter`` (default 2, i.e. 5 slices) in train mode.
    3. Apply the SAME crop (plus zero-padding if the volume is smaller than
       ``patch_size``) to every volume so every tensor stays co-registered.
    4. Return compact tensors; ``batch_to_device`` moves the batch to the
       GPU, z-score normalises the image (non-background voxels) and unpacks
       the seg bit mask there. This keeps the worker -> main process -> GPU
       traffic ~8x smaller than float32 tensors.

Returned sample dict:

    image : (1, D, H, W) float16, raw intensities rescaled to max |x| = 1000
    label : (1, D, H, W) uint8, integer disc labels (0 = background)
    seg   : (1, D, H, W) uint8 bit mask, bit ``c`` = anatomy channel ``c``
            (``discslabel.SEG_CHANNELS``: 0 vertebrae, 1 discs, 2 spinal cord,
            3 spinal canal). Missing SEG_* files leave their bits at zero.
    name  : basename of the volume

After ``batch_to_device``: image (B, 1, D, H, W) float32 z-scored, label
(B, 1, D, H, W) float32, seg (B, NUM_SEG_CHANNELS, D, H, W) float32 binary.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from . import (
    NUM_SEG_CHANNELS,
    SEG_CH_CANAL,
    SEG_CH_CORD,
    SEG_CH_DISCS,
    SEG_CH_VERTEBRAE,
    SEG_SPINE_DISC_MIN_LABEL,
)


def _load_nifti(path: str) -> np.ndarray:
    """Load a NIfTI as a 3D float32 array in whatever orientation it is on disk.

    Preprocessing (``scripts/prepare_datasets.sh``) has already reoriented
    everything to RPI, so axis 0 = R→L, axis 1 = P→A, axis 2 = I→S.
    """
    arr = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {arr.shape} from {path}")
    return arr


def _z_score_normalize(arr: np.ndarray) -> np.ndarray:
    """Z-score normalise ``arr`` using the mean and std of the non-background voxels.

    Background voxels (which are typically zero in MRI) would otherwise drag
    the mean toward zero and inflate the std, so we restrict the statistics to
    strictly positive voxels. The output is returned as float32 with no
    clipping — SmaugLab's downstream augmentations tolerate any float range.
    """
    arr = arr.astype(np.float32)
    nonzero = arr[arr > 0]
    if nonzero.size < 10:
        return np.zeros_like(arr, dtype=np.float32)
    mean = float(nonzero.mean())
    std = float(nonzero.std())
    if std < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mean) / std).astype(np.float32)


def _clamp_start(start: int, src_len: int, dst_len: int) -> int:
    """Keep a crop window inside the volume; center-pad when the volume is smaller."""
    if src_len >= dst_len:
        return int(max(0, min(start, src_len - dst_len)))
    return -((dst_len - src_len) // 2)


def _crop_or_pad_at(
    arr: np.ndarray,
    origin: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Extract the window ``arr[d0:d0+pD, h0:h0+pH, w0:w0+pW]``.

    The window may extend past the volume (negative origin or beyond the end);
    those voxels are filled with ``pad_value``. The origin pickers use
    ``_clamp_start`` to keep the window inside the volume (or center-pad it),
    except on the R-L axis of the 2D pipeline where the chosen sagittal slice
    must land exactly on the patch centre.

    No interpolation: discrete label values pass through unchanged.
    """
    out_shape = tuple(int(s) for s in patch_size)
    out = np.full(out_shape, pad_value, dtype=arr.dtype)

    src_slices = []
    dst_slices = []
    for src_len, dst_len, start in zip(arr.shape[:3], out_shape, origin):
        start = int(start)
        src_lo = max(0, start)
        src_hi = min(src_len, start + dst_len)
        if src_hi <= src_lo:
            return out
        src_slices.append(slice(src_lo, src_hi))
        dst_slices.append(slice(src_lo - start, src_hi - start))

    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def _pick_crop_origin_nnunet(
    vol_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    fg_coords: np.ndarray,
    fg_ids: np.ndarray,
    rng: np.random.Generator,
    force_fg: bool,
    midline_on_disc: bool,
    fixed_midline: Optional[int] = None,
) -> Tuple[int, int, int]:
    """nnU-Net-style training crop origin.

    ``fg_coords`` ``(N, 3)`` are the disc voxels and ``fg_ids`` ``(N,)`` the
    disc each of them belongs to.

    * ``force_fg`` (nnU-Net ``oversample_foreground``): draw one disc id
      uniformly — so small cervical discs are as likely as large lumbar ones —
      then one of its voxels, and centre the crop on it.
    * otherwise: uniform random origin on every axis; the crop may hold no disc.

    ``midline_on_disc`` (2D pipeline): the model only sees the crop's
    mid-sagittal slice, and inference always runs on a slice through the
    spine, so the R-L centre is always a disc-bearing slice; the
    foreground / random choice then applies to H and W only. The R-L origin
    is not clamped, so the chosen slice always lands on the patch centre.

    ``fixed_midline`` (2D pipeline, no dense disc segmentation): use this
    sagittal slice as the R-L centre instead of a disc-bearing one; the
    foreground H/W position is then drawn from the discs of every slice
    (``fg_coords`` hold the point-wise ``LABEL``).
    """
    D, H, W = vol_shape[:3]
    pD, pH, pW = (int(s) for s in patch_size)

    def _axis_uniform(src_len: int, dst_len: int) -> int:
        if src_len <= dst_len:
            return _clamp_start(0, src_len, dst_len)
        return int(rng.integers(0, src_len - dst_len + 1))

    coords, ids = fg_coords, fg_ids

    d_mid: Optional[int] = None
    if fixed_midline is not None:
        d_mid = int(fixed_midline)
    elif midline_on_disc and len(coords) > 0:
        d_mid = int(rng.choice(np.unique(coords[:, 0])))
        keep = coords[:, 0] == d_mid
        coords, ids = coords[keep], ids[keep]

    if force_fg and len(coords) > 0:
        chosen_id = rng.choice(np.unique(ids))
        candidates = coords[ids == chosen_id]
        d_fg, h_fg, w_fg = candidates[rng.integers(0, len(candidates))]
        d0 = _clamp_start(int(d_fg) - pD // 2, D, pD)
        h0 = _clamp_start(int(h_fg) - pH // 2, H, pH)
        w0 = _clamp_start(int(w_fg) - pW // 2, W, pW)
    else:
        d0, h0, w0 = _axis_uniform(D, pD), _axis_uniform(H, pH), _axis_uniform(W, pW)

    if d_mid is not None:
        d0 = d_mid - pD // 2

    return (d0, h0, w0)


def _pick_crop_origin(
    vol_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    fg_coords: np.ndarray,
    fixed_midline: Optional[int] = None,
    exact_midline: bool = False,
) -> Tuple[int, int, int]:
    """Deterministic (valid / inference) 3D crop origin ``(d0, h0, w0)``.

    With disc voxels ``fg_coords`` ``(N, 3)``, the crop's mid-sagittal slice
    (``cropped[pD // 2, :, :]``) lands on the median disc-bearing sagittal
    slice and the H/W window is centred on the disc centroid of that slice.

    ``fixed_midline`` forces the mid-sagittal slice (used when no dense disc
    segmentation exists); the H/W window is then centred on the centroid of
    ``fg_coords`` projected over every slice.

    ``exact_midline`` (2D pipeline) leaves the R-L origin unclamped so the
    chosen slice always lands on the patch centre.

    Falls back to a plain center crop when nothing is available.
    """
    D, H, W = vol_shape[:3]
    pD, pH, pW = (int(s) for s in patch_size)

    if fixed_midline is not None:
        d_mid = int(fixed_midline)
        hw_coords = fg_coords[:, 1:]
    else:
        if len(fg_coords) == 0:
            return (
                _clamp_start((D - pD) // 2, D, pD),
                _clamp_start((H - pH) // 2, H, pH),
                _clamp_start((W - pW) // 2, W, pW),
            )
        sag_hits = np.unique(fg_coords[:, 0])
        d_mid = int(sag_hits[len(sag_hits) // 2])
        hw_coords = fg_coords[fg_coords[:, 0] == d_mid][:, 1:]

    if len(hw_coords) == 0:
        h_center, w_center = H // 2, W // 2
    else:
        h_center = int(np.median(hw_coords[:, 0]))
        w_center = int(np.median(hw_coords[:, 1]))

    d0 = d_mid - pD // 2
    if not exact_midline:
        d0 = _clamp_start(d0, D, pD)
    return (
        d0,
        _clamp_start(h_center - pH // 2, H, pH),
        _clamp_start(w_center - pW // 2, W, pW),
    )


def _image_to_float16(image: np.ndarray) -> np.ndarray:
    """Rescale to ``max |x| = IMAGE_SCALE_TARGET`` and cast to float16.

    The z-score normalisation applied on the GPU is invariant to a positive
    scale, so this only costs float16 rounding while halving the bytes read
    from the cache and moved to the GPU.
    """
    from .cache import IMAGE_SCALE_TARGET

    max_abs = float(np.abs(image).max()) if image.size else 0.0
    if max_abs > 0:
        image = image * (IMAGE_SCALE_TARGET / max_abs)
    return image.astype(np.float16)


def load_case_nifti(entry: dict, spine_disc_min_label: int = SEG_SPINE_DISC_MIN_LABEL) -> Dict[str, object]:
    """Load one manifest entry from its NIfTIs into the compact case format.

    Shared by the dataset (no cache) and the cache builder:

        shape        : (D, H, W)
        image        : (D, H, W) float16, rescaled (see ``_image_to_float16``)
        label_coords : (N, 3) int32 voxels of the point-wise disc label
        label_values : (N,) uint8 disc values (1..25)
        seg_bits     : (D, H, W) uint8 bit mask (bit ``c`` = anatomy channel
                       ``c``), or None when no SEG_* file is listed
        fg_coords    : (M, 3) int32 disc voxels for the crop sampler — the
                       dense SEG_SPINE discs when available, else the label
        fg_ids       : (M,) int32 disc id of each ``fg_coords`` voxel
        has_disc_seg : whether ``fg_coords`` come from SEG_SPINE
    """
    image = _load_nifti(entry["image"])
    shape = image.shape

    label_coords = np.empty((0, 3), dtype=np.int32)
    label_values = np.empty((0,), dtype=np.uint8)
    if entry.get("label"):
        label = np.rint(_load_nifti(entry["label"])).astype(np.int32)
        label_coords = np.argwhere(label > 0).astype(np.int32)
        label_values = label[tuple(label_coords.T)].astype(np.uint8)

    seg_bits: Optional[np.ndarray] = None
    fg_coords, fg_ids, has_disc_seg = label_coords, label_values.astype(np.int32), False
    if entry.get("seg_spine") or entry.get("seg_cord") or entry.get("seg_canal"):
        seg_bits = np.zeros(shape, dtype=np.uint8)
    if entry.get("seg_spine"):
        spine = np.rint(_load_nifti(entry["seg_spine"])).astype(np.int32)
        is_disc = spine >= spine_disc_min_label
        seg_bits |= (((spine > 0) & ~is_disc).astype(np.uint8) << SEG_CH_VERTEBRAE)
        seg_bits |= (is_disc.astype(np.uint8) << SEG_CH_DISCS)
        disc_coords = np.argwhere(is_disc).astype(np.int32)
        if len(disc_coords):
            fg_coords, fg_ids, has_disc_seg = disc_coords, spine[tuple(disc_coords.T)], True
    for key, ch in (("seg_cord", SEG_CH_CORD), ("seg_canal", SEG_CH_CANAL)):
        if entry.get(key):
            mask = np.rint(_load_nifti(entry[key])) > 0
            seg_bits |= (mask.astype(np.uint8) << ch)

    return {
        "shape": tuple(int(s) for s in shape),
        "image": _image_to_float16(image),
        "label_coords": label_coords,
        "label_values": label_values,
        "seg_bits": seg_bits,
        "fg_coords": fg_coords.astype(np.int32),
        "fg_ids": np.asarray(fg_ids, dtype=np.int32),
        "has_disc_seg": has_disc_seg,
    }


def _rasterize_points(
    coords: np.ndarray,
    values: np.ndarray,
    origin: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> np.ndarray:
    """Dense ``uint8`` crop of a sparse point-wise label."""
    out = np.zeros(patch_size, dtype=np.uint8)
    if len(coords) == 0:
        return out
    local = coords - np.asarray(origin, dtype=np.int64)
    inside = np.all((local >= 0) & (local < np.asarray(patch_size)), axis=1)
    local = local[inside]
    out[tuple(local.T)] = values[inside]
    return out


class DiscVolumeDataset(Dataset):
    """Reads a prepared split (``manifest.json`` + per-split ``images/`` /
    ``labels/`` / ``segs_{spine,cord,canal}/`` folders) and yields 3D crops
    so the extracted mid-sagittal slice always lands on a disc.

    With ``cache_root`` (see ``discslabel.cache``) the cases are read from
    the memory-mapped ``.npy`` cache, so only the crop is read from disk;
    entries missing from the cache fall back to the NIfTIs.

    See the module docstring for the per-sample pipeline and output format.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        patch_size: Tuple[int, int, int] = (96, 320, 320),
        require_label: bool = True,
        spine_disc_min_label: int = SEG_SPINE_DISC_MIN_LABEL,
        random_crop: bool = False,
        oversample_foreground_percent: float = 0.33,
        midline_on_disc: bool = False,
        midline_jitter: int = 2,
        cache_root: Optional[str] = None,
    ):
        super().__init__()
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if split not in manifest:
            raise KeyError(f"Split '{split}' not in manifest {manifest_path}")
        self.split = split
        self.entries: List[Dict[str, str]] = list(manifest[split])
        if require_label:
            self.entries = [e for e in self.entries if e.get("label")]
        self.patch_size = tuple(int(s) for s in patch_size)
        self.require_label = require_label
        self.spine_disc_min_label = int(spine_disc_min_label)
        self.random_crop = bool(random_crop)
        # Training crops (``random_crop=True``) use ``_pick_crop_origin_nnunet``.
        # ``ForegroundBatchSampler`` passes ``(index, force_fg)`` keys; a plain
        # int index falls back to drawing ``force_fg`` with this probability.
        self.oversample_foreground_percent = float(oversample_foreground_percent)
        self.midline_on_disc = bool(midline_on_disc)
        # 2D pipeline, entries without a dense disc segmentation (no SEG_SPINE):
        # the mid-sagittal slice is the volume centre (as at inference); in
        # training it is drawn uniformly in ``centre ± midline_jitter``.
        self.midline_jitter = max(0, int(midline_jitter))

        self.cache_root = cache_root
        self._cached = [False] * len(self.entries)
        if cache_root is not None:
            from .cache import is_cached

            self._cached = [
                is_cached(cache_root, split, e, self.spine_disc_min_label) for e in self.entries
            ]
            n_miss = self._cached.count(False)
            if n_miss:
                print(
                    f"[dataset] {n_miss}/{len(self.entries)} '{split}' cases are not in the "
                    f"cache {cache_root}; they are read from the NIfTIs."
                )

    def __len__(self) -> int:
        return len(self.entries)

    def _rng(self) -> np.random.Generator:
        """Fresh RNG per __getitem__ call so DataLoader worker seeding sticks."""
        return np.random.default_rng()

    def _load_case(self, index: int) -> Dict[str, object]:
        entry = self.entries[index]
        if self._cached[index]:
            from .cache import load_case_cached

            return load_case_cached(self.cache_root, self.split, entry)
        return load_case_nifti(entry, self.spine_disc_min_label)

    def __getitem__(self, key) -> Dict[str, torch.Tensor | str]:
        rng = self._rng()
        if isinstance(key, (tuple, list)):
            index, force_fg = int(key[0]), bool(key[1])
        else:
            index = int(key)
            force_fg = bool(rng.random() < self.oversample_foreground_percent)
        entry = self.entries[index]
        if self.require_label and not entry.get("label"):
            raise KeyError(f"Entry {entry} has no label but require_label=True")

        # 1. Load the case: from the memory-mapped cache nothing but the
        #    metadata is read until the crop below.
        case = self._load_case(index)
        shape = case["shape"]

        # 2. Pick a crop origin keyed on the disc voxels so the extracted
        #    mid-sagittal slice always intersects a disc. ``fg_coords`` are the
        #    dense SEG_SPINE discs (value >= threshold) when available, else
        #    the point-wise LABEL; ``fg_ids`` keep one id per disc so the
        #    foreground sampler can pick discs uniformly.
        #
        #    2D pipeline without a dense disc mask: the point-wise LABEL does
        #    not tell which sagittal slices cross the discs, so the
        #    mid-sagittal slice is the volume centre (same slice as
        #    inference), jittered in training.
        fixed_midline: Optional[int] = None
        if self.midline_on_disc and not case["has_disc_seg"]:
            D = shape[0]
            fixed_midline = D // 2
            if self.random_crop and self.midline_jitter > 0:
                fixed_midline += int(rng.integers(-self.midline_jitter, self.midline_jitter + 1))
            fixed_midline = int(np.clip(fixed_midline, 0, D - 1))

        if self.random_crop:
            origin = _pick_crop_origin_nnunet(
                shape,
                self.patch_size,
                case["fg_coords"],
                case["fg_ids"],
                rng=rng,
                force_fg=force_fg,
                midline_on_disc=self.midline_on_disc,
                fixed_midline=fixed_midline,
            )
        else:
            origin = _pick_crop_origin(
                shape,
                self.patch_size,
                case["fg_coords"],
                fixed_midline=fixed_midline,
                exact_midline=self.midline_on_disc,
            )

        # 3. Crop every volume at the same origin. The tensors stay compact
        #    (float16 image, uint8 label and seg bit mask); ``batch_to_device``
        #    z-scores the image and unpacks the seg channels on the GPU.
        image = _crop_or_pad_at(case["image"], origin, self.patch_size, pad_value=0.0)
        sample: Dict[str, torch.Tensor | str] = {
            "image": torch.from_numpy(image).unsqueeze(0),
            "name": os.path.basename(entry["image"]),
        }
        if entry.get("label"):
            label = _rasterize_points(
                case["label_coords"], case["label_values"], origin, self.patch_size
            )
            sample["label"] = torch.from_numpy(label).unsqueeze(0)

        # Missing segmentations stay empty (all-zero bits), so every sample
        # carries a ``seg`` of the same shape: batches mixing entries with and
        # without SEG_* still collate, and all of them go through the
        # SmaugLab augmentations.
        if case["seg_bits"] is not None:
            seg_bits = _crop_or_pad_at(case["seg_bits"], origin, self.patch_size, pad_value=0)
        else:
            seg_bits = np.zeros(self.patch_size, dtype=np.uint8)
        sample["seg"] = torch.from_numpy(seg_bits).unsqueeze(0)
        return sample


# ---------------------------------------------------------------------------
# GPU side of the sample format
# ---------------------------------------------------------------------------


def z_score_normalize_torch(image: torch.Tensor, min_voxels: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Batched ``_z_score_normalize``: per-sample mean / std of the strictly
    positive voxels; samples with fewer than ``min_voxels`` such voxels or a
    std below ``eps`` become all zeros. ``image`` is ``(B, C, *S)``."""
    x = image.float()
    dims = tuple(range(1, x.dim()))
    mask = (x > 0).float()
    n = mask.sum(dim=dims, keepdim=True)
    mean = (x * mask).sum(dim=dims, keepdim=True) / n.clamp_min(1)
    var = (((x - mean) * mask) ** 2).sum(dim=dims, keepdim=True) / n.clamp_min(1)
    std = var.sqrt()
    valid = (n >= min_voxels) & (std >= eps)
    return torch.where(valid, (x - mean) / std.clamp_min(eps), torch.zeros_like(x))


def unpack_seg_bits(seg_bits: torch.Tensor) -> torch.Tensor:
    """``(B, 1, *S)`` uint8 bit mask → ``(B, NUM_SEG_CHANNELS, *S)`` float {0, 1}."""
    shifts = torch.arange(NUM_SEG_CHANNELS, device=seg_bits.device, dtype=torch.uint8)
    shifts = shifts.view(1, NUM_SEG_CHANNELS, *([1] * (seg_bits.dim() - 2)))
    return ((seg_bits >> shifts) & 1).float()


def batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Move a ``DiscVolumeDataset`` batch to ``device`` and decode it there.

    Returns ``(image, label, seg)``: the z-scored float32 image
    ``(B, 1, D, H, W)``, the float32 disc label ``(B, 1, D, H, W)`` and the
    float32 anatomy channels ``(B, NUM_SEG_CHANNELS, D, H, W)``.
    """
    image = z_score_normalize_torch(batch["image"].to(device, non_blocking=True))
    label = batch.get("label")
    if label is not None:
        label = label.to(device, non_blocking=True).float()
    seg = batch.get("seg")
    if seg is not None:
        seg = unpack_seg_bits(seg.to(device, non_blocking=True))
    return image, label, seg
