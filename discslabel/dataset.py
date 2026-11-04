"""Dataset reader for prepared 3D NIfTI volumes.

The dataset returns 3D ``(image, label, seg)`` tensors with no 2D projection
applied: all augmentation, mid-sagittal extraction and heatmap generation
runs **on the GPU** via ``discslabel.pipeline_2d.gpu_pipeline.GpuBatchProcessor``
(or its 3D counterpart in ``discslabel.pipeline_3d``).

Per-sample pipeline:

    1. Load the image, the point-wise disc label, and the three SEG_* NIfTIs
       raw (whatever shape they have on disk after reorient + resample).
    2. Pick a 3D crop origin so that the extracted mid-sagittal slice of the
       crop contains **disc** voxels. The disc mask is the dense disc
       segmentation from SEG_SPINE (voxels with value ``>= spine_disc_min_label``);
       when SEG_SPINE is unavailable we fall back to the point-wise ``LABEL``
       (any non-zero disc landmark). In train mode the origin follows
       nnU-Net's sampling (``_pick_crop_origin_nnunet``): foreground-forced
       crops are centred on a voxel of a uniformly drawn disc, the others are
       uniformly random. In valid / inference mode we use the median disc
       slice and the exact disc centroid.
    3. Apply the SAME crop (plus zero-padding if the volume is smaller than
       ``patch_size``) to every volume so every tensor stays co-registered.
    4. Z-score normalise the cropped image using the non-background voxels.

Returned sample dict:

    image : (1, D, H, W) float tensor, z-scored
    label : (1, D, H, W) float tensor, integer disc labels (0 = background)
    seg   : (NUM_SEG_CHANNELS, D, H, W) binary tensor — one channel per
            anatomical structure (see ``discslabel.SEG_CHANNELS``):
                0: vertebrae, 1: discs, 2: spinal cord, 3: spinal canal.
            Omitted when none of the three SEG_* files is listed.
    name  : basename of the volume
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


def _crop_or_pad_at(
    arr: np.ndarray,
    origin: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Extract ``arr[d0:d0+pD, h0:h0+pH, w0:w0+pW]``, zero-padding on the
    axes where the source is smaller than the patch.

    No interpolation: discrete label values pass through unchanged.
    """
    out_shape = tuple(int(s) for s in patch_size)
    out = np.full(out_shape, pad_value, dtype=arr.dtype)

    pD, pH, pW = out_shape
    D, H, W = arr.shape[:3]

    src_slices = []
    dst_slices = []
    for src_len, dst_len, start in zip((D, H, W), (pD, pH, pW), origin):
        if src_len >= dst_len:
            start = int(max(0, min(start, src_len - dst_len)))
            src_slices.append(slice(start, start + dst_len))
            dst_slices.append(slice(0, dst_len))
        else:
            # Volume is smaller than the patch on this axis: center-pad.
            dst_start = (dst_len - src_len) // 2
            src_slices.append(slice(0, src_len))
            dst_slices.append(slice(dst_start, dst_start + src_len))

    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def _pick_crop_origin_nnunet(
    vol_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    disc_ids: Optional[np.ndarray],
    rng: np.random.Generator,
    force_fg: bool,
    midline_on_disc: bool,
) -> Tuple[int, int, int]:
    """nnU-Net-style training crop origin.

    ``disc_ids`` is an int volume where each disc carries its own id
    (0 = background).

    * ``force_fg`` (nnU-Net ``oversample_foreground``): draw one disc id
      uniformly — so small cervical discs are as likely as large lumbar ones —
      then one of its voxels, and centre the crop on it.
    * otherwise: uniform random origin on every axis; the crop may hold no disc.

    ``midline_on_disc`` (2D pipeline): the model only sees the crop's
    mid-sagittal slice, and inference always runs on a slice through the
    spine, so the R-L centre is always a disc-bearing slice; the
    foreground / random choice then applies to H and W only.
    """
    D, H, W = vol_shape[:3]
    pD, pH, pW = (int(s) for s in patch_size)

    def _axis_uniform(src_len: int, dst_len: int) -> int:
        if src_len <= dst_len:
            return 0
        return int(rng.integers(0, src_len - dst_len + 1))

    def _clamp(start: int, src_len: int, dst_len: int) -> int:
        if src_len >= dst_len:
            return int(max(0, min(start, src_len - dst_len)))
        return 0

    coords = np.argwhere(disc_ids > 0) if disc_ids is not None else np.empty((0, 3), int)
    if len(coords) == 0:
        return (_axis_uniform(D, pD), _axis_uniform(H, pH), _axis_uniform(W, pW))

    if midline_on_disc:
        d_mid = int(rng.choice(np.unique(coords[:, 0])))
        coords = coords[coords[:, 0] == d_mid]

    if force_fg:
        ids = disc_ids[coords[:, 0], coords[:, 1], coords[:, 2]]
        chosen_id = rng.choice(np.unique(ids))
        candidates = coords[ids == chosen_id]
        d_fg, h_fg, w_fg = candidates[rng.integers(0, len(candidates))]
        d0, h0, w0 = int(d_fg) - pD // 2, int(h_fg) - pH // 2, int(w_fg) - pW // 2
    else:
        d0, h0, w0 = _axis_uniform(D, pD), _axis_uniform(H, pH), _axis_uniform(W, pW)

    if midline_on_disc:
        d0 = d_mid - pD // 2

    return (_clamp(d0, D, pD), _clamp(h0, H, pH), _clamp(w0, W, pW))


def _pick_crop_origin(
    vol_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    disc_presence: Optional[np.ndarray],
    random_crop: bool,
    rng: np.random.Generator,
    hw_jitter_frac: float = 0.125,
) -> Tuple[int, int, int]:
    """Pick a 3D crop origin ``(d0, h0, w0)``.

    When ``disc_presence`` is given, it is a 3D bool mask of disc voxels.
    The origin is chosen so that the crop's mid-sagittal slice
    (``cropped[pD // 2, :, :]``) lands on a sagittal slice with disc voxels,
    and so that the crop's H/W window overlaps the disc centroid on that slice.

    When ``disc_presence`` is None, falls back to a plain center crop.

    Train mode (``random_crop=True``) draws:
        - a random disc-bearing R-L slice as the mid-sag anchor;
        - H/W jitter of up to ``hw_jitter_frac * patch_size`` around the disc
          centroid on that slice.
    Valid / inference (``random_crop=False``) uses the median disc slice and
    the disc centroid exactly, so the output is deterministic.
    """
    D, H, W = vol_shape[:3]
    pD, pH, pW = (int(s) for s in patch_size)

    def _axis_center_fallback(src_len: int, dst_len: int) -> int:
        return max(0, (src_len - dst_len) // 2)

    if disc_presence is None:
        return (
            _axis_center_fallback(D, pD),
            _axis_center_fallback(H, pH),
            _axis_center_fallback(W, pW),
        )

    sag_hits = np.where(disc_presence.any(axis=(1, 2)))[0]
    if len(sag_hits) == 0:
        return (
            _axis_center_fallback(D, pD),
            _axis_center_fallback(H, pH),
            _axis_center_fallback(W, pW),
        )

    # 1) R-L anchor: a sagittal slice index where discs are visible.
    if random_crop:
        d_mid = int(rng.choice(sag_hits))
    else:
        d_mid = int(sag_hits[len(sag_hits) // 2])

    # 2) H/W centroid of the discs ON that chosen slice.
    hw_slice = disc_presence[d_mid]
    hw_coords = np.argwhere(hw_slice)
    if len(hw_coords) == 0:
        h_center, w_center = H // 2, W // 2
    else:
        h_center = int(np.median(hw_coords[:, 0]))
        w_center = int(np.median(hw_coords[:, 1]))

    if random_crop:
        h_amp = max(1, int(hw_jitter_frac * pH))
        w_amp = max(1, int(hw_jitter_frac * pW))
        h_center += int(rng.integers(-h_amp, h_amp + 1))
        w_center += int(rng.integers(-w_amp, w_amp + 1))

    # 3) Convert from (d_mid, h_center, w_center) to origins and clamp.
    d0 = d_mid - pD // 2
    h0 = h_center - pH // 2
    w0 = w_center - pW // 2
    if D >= pD:
        d0 = int(max(0, min(d0, D - pD)))
    else:
        d0 = 0
    if H >= pH:
        h0 = int(max(0, min(h0, H - pH)))
    else:
        h0 = 0
    if W >= pW:
        w0 = int(max(0, min(w0, W - pW)))
    else:
        w0 = 0
    return (d0, h0, w0)


class DiscVolumeDataset(Dataset):
    """Reads a prepared split (``manifest.json`` + per-split ``images/`` /
    ``labels/`` / ``segs_{spine,cord,canal}/`` folders) and yields 3D tensors
    cropped so the extracted mid-sagittal slice always lands on a disc.

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
    ):
        super().__init__()
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        if split not in manifest:
            raise KeyError(f"Split '{split}' not in manifest {manifest_path}")
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

    def __len__(self) -> int:
        return len(self.entries)

    def _rng(self) -> np.random.Generator:
        """Fresh RNG per __getitem__ call so DataLoader worker seeding sticks."""
        return np.random.default_rng()

    def __getitem__(self, key) -> Dict[str, torch.Tensor | str]:
        rng = self._rng()
        if isinstance(key, (tuple, list)):
            index, force_fg = int(key[0]), bool(key[1])
        else:
            index = int(key)
            force_fg = bool(rng.random() < self.oversample_foreground_percent)
        entry = self.entries[index]

        # 1. Load every 3D volume we need (raw, pre-crop).
        image_raw = _load_nifti(entry["image"])

        label_raw: Optional[np.ndarray] = None
        label_path = entry.get("label")
        if label_path:
            label_raw = np.rint(_load_nifti(label_path)).astype(np.float32)
        elif self.require_label:
            raise KeyError(f"Entry {entry} has no label but require_label=True")

        seg_spine_raw: Optional[np.ndarray] = None
        if entry.get("seg_spine"):
            seg_spine_raw = np.rint(_load_nifti(entry["seg_spine"])).astype(np.int32)

        seg_cord_raw: Optional[np.ndarray] = None
        if entry.get("seg_cord"):
            seg_cord_raw = (np.rint(_load_nifti(entry["seg_cord"])) > 0).astype(np.float32)

        seg_canal_raw: Optional[np.ndarray] = None
        if entry.get("seg_canal"):
            seg_canal_raw = (np.rint(_load_nifti(entry["seg_canal"])) > 0).astype(np.float32)

        # 2. Pick a crop origin keyed on the DISC mask so the extracted
        #    mid-sagittal slice always intersects a disc — which is what the
        #    heatmap-regression task actually cares about.
        #    Preferred source: dense disc mask from SEG_SPINE (value >= threshold).
        #    Fallback: the point-wise LABEL (any non-zero disc landmark).
        #    ``disc_ids`` keeps one id per disc (SEG_SPINE value or LABEL value)
        #    so the foreground sampler can pick discs uniformly.
        disc_ids: Optional[np.ndarray] = None
        if seg_spine_raw is not None:
            dense_ids = np.where(seg_spine_raw >= self.spine_disc_min_label, seg_spine_raw, 0)
            if dense_ids.any():
                disc_ids = dense_ids
        if disc_ids is None and label_raw is not None:
            label_ids = label_raw.astype(np.int32)
            if label_ids.any():
                disc_ids = label_ids
        if self.random_crop:
            origin = _pick_crop_origin_nnunet(
                image_raw.shape,
                self.patch_size,
                disc_ids,
                rng=rng,
                force_fg=force_fg,
                midline_on_disc=self.midline_on_disc,
            )
        else:
            origin = _pick_crop_origin(
                image_raw.shape,
                self.patch_size,
                disc_ids > 0 if disc_ids is not None else None,
                random_crop=False,
                rng=rng,
            )

        # 3. Crop + z-score normalise the image.
        image_cropped = _crop_or_pad_at(image_raw, origin, self.patch_size, pad_value=0.0)
        image_norm = _z_score_normalize(image_cropped)
        image_t = torch.from_numpy(image_norm).unsqueeze(0).contiguous()  # (1, D, H, W)

        sample: Dict[str, torch.Tensor | str] = {
            "image": image_t,
            "name": os.path.basename(entry["image"]),
        }

        # 4. Crop the disc label volume at the same origin.
        if label_raw is not None:
            label_cropped = _crop_or_pad_at(label_raw, origin, self.patch_size, pad_value=0.0)
            sample["label"] = torch.from_numpy(label_cropped).unsqueeze(0).contiguous()

        # 5. Crop the three SEG_* masks and pack them into the 4-channel tensor.
        if seg_spine_raw is not None or seg_cord_raw is not None or seg_canal_raw is not None:
            seg_out = np.zeros((NUM_SEG_CHANNELS,) + self.patch_size, dtype=np.float32)
            if seg_spine_raw is not None:
                spine_cropped = _crop_or_pad_at(seg_spine_raw, origin, self.patch_size, pad_value=0)
                seg_out[SEG_CH_VERTEBRAE] = (
                    (spine_cropped > 0) & (spine_cropped < self.spine_disc_min_label)
                ).astype(np.float32)
                seg_out[SEG_CH_DISCS] = (spine_cropped >= self.spine_disc_min_label).astype(np.float32)
            if seg_cord_raw is not None:
                cord_cropped = _crop_or_pad_at(seg_cord_raw, origin, self.patch_size, pad_value=0.0)
                seg_out[SEG_CH_CORD] = (cord_cropped > 0).astype(np.float32)
            if seg_canal_raw is not None:
                canal_cropped = _crop_or_pad_at(seg_canal_raw, origin, self.patch_size, pad_value=0.0)
                seg_out[SEG_CH_CANAL] = (canal_cropped > 0).astype(np.float32)
            sample["seg"] = torch.from_numpy(seg_out).contiguous()

        return sample
