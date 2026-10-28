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
       (any non-zero disc landmark). In train mode the origin is drawn
       randomly from the set of disc-bearing sagittal slices (plus a small
       H/W jitter around the disc centroid); in valid / inference mode we use
       the median disc slice and the exact disc centroid.
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


def _pick_crop_origin_random_fg(
    vol_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    disc_presence: Optional[np.ndarray],
    rng: np.random.Generator,
    foreground_prob: float,
) -> Tuple[int, int, int]:
    """Random 3D crop origin with a configurable minimum-foreground guarantee.

    Used by the 3D pipeline where the full volume (not a 2D slice) is fed to
    the model — a crop that misses every disc is a valid negative example,
    but we still want the majority of crops to contain at least one disc so
    the heatmap-regression loss has a positive target to pull toward.

    With probability ``foreground_prob``, picks a random disc voxel from
    ``disc_presence`` and places the crop so that voxel lands at a uniformly
    random position WITHIN the crop — the disc is guaranteed to be inside
    the output. With probability ``1 - foreground_prob``, draws the origin
    uniformly from the valid range on every axis independently; those crops
    may or may not contain discs.

    Falls back to a uniformly random crop when no disc voxels are available
    (empty volume), instead of raising.
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

    want_fg = (
        disc_presence is not None
        and disc_presence.any()
        and rng.random() < foreground_prob
    )

    if want_fg:
        disc_voxels = np.argwhere(disc_presence)  # (N, 3) in (D, H, W)
        d_fg, h_fg, w_fg = disc_voxels[rng.integers(0, len(disc_voxels))]
        # Uniform random position of the chosen disc inside the crop. Any
        # clamping afterwards keeps the disc inside (just at the edge).
        od = int(rng.integers(0, pD))
        oh = int(rng.integers(0, pH))
        ow = int(rng.integers(0, pW))
        d0 = int(d_fg) - od
        h0 = int(h_fg) - oh
        w0 = int(w_fg) - ow
    else:
        d0 = _axis_uniform(D, pD)
        h0 = _axis_uniform(H, pH)
        w0 = _axis_uniform(W, pW)

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
        foreground_prob: Optional[float] = None,
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
        # None  → use the 2D midline-tight sampler (``_pick_crop_origin``).
        # float → use the 3D random-with-foreground sampler
        #         (``_pick_crop_origin_random_fg``) when ``random_crop`` is on.
        self.foreground_prob = (
            float(foreground_prob) if foreground_prob is not None else None
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _rng(self) -> np.random.Generator:
        """Fresh RNG per __getitem__ call so DataLoader worker seeding sticks."""
        return np.random.default_rng()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
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
        disc_presence: Optional[np.ndarray] = None
        if seg_spine_raw is not None:
            dense_discs = seg_spine_raw >= self.spine_disc_min_label
            if dense_discs.any():
                disc_presence = dense_discs
        if disc_presence is None and label_raw is not None:
            label_points = label_raw > 0
            if label_points.any():
                disc_presence = label_points
        if self.random_crop and self.foreground_prob is not None:
            origin = _pick_crop_origin_random_fg(
                image_raw.shape,
                self.patch_size,
                disc_presence,
                rng=self._rng(),
                foreground_prob=self.foreground_prob,
            )
        else:
            origin = _pick_crop_origin(
                image_raw.shape,
                self.patch_size,
                disc_presence,
                random_crop=self.random_crop,
                rng=self._rng(),
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
