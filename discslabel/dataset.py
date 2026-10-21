"""Dataset reader for prepared 3D NIfTI volumes.

The dataset returns 3D ``(image, label, seg)`` tensors with no 2D projection
applied: all augmentation, mid-sagittal extraction and heatmap generation
runs **on the GPU** via ``discslabel.gpu_pipeline.GpuBatchProcessor``.

Volumes carried per sample:

    image : the T1w / T2w / … MRI volume
    label : the point-wise disc landmark volume (background = 0, discs = 1..25)
    seg   : the four anatomy BINARY channels, stacked:
              ch 0 : vertebrae (from SEG_SPINE values < spine_disc_min_label)
              ch 1 : discs     (from SEG_SPINE values >= spine_disc_min_label)
              ch 2 : spinal cord (from SEG_CORD > 0)
              ch 3 : spinal canal (from SEG_CANAL > 0)
            This 4-channel tensor is fed as the second parameter of
            SmaugLab's ``AugTransformsGPU`` so that any 3D spatial
            augmentation keeps the image aligned with each anatomy mask.

All volumes are padded/center-cropped to a configurable ``patch_size`` so a
batch of volumes has a uniform shape — which is what SmaugLab's GPU
augmentations expect.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

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


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    """Min/max normalise to [0, 1] per volume. Flat volumes map to zero."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _center_crop_or_pad(arr: np.ndarray, patch_size: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    """Center-crop then zero-pad ``arr`` to ``patch_size``.

    No interpolation: the data stays on its original grid, discrete label
    values are preserved exactly.
    """
    out_shape = tuple(int(s) for s in patch_size)
    out = np.full(out_shape, pad_value, dtype=arr.dtype)

    src_slices = []
    dst_slices = []
    for src_len, dst_len in zip(arr.shape, out_shape):
        if src_len >= dst_len:
            start = (src_len - dst_len) // 2
            src_slices.append(slice(start, start + dst_len))
            dst_slices.append(slice(0, dst_len))
        else:
            start = (dst_len - src_len) // 2
            src_slices.append(slice(0, src_len))
            dst_slices.append(slice(start, start + src_len))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


def _build_binary_seg_channels(
    entry: Dict[str, str],
    patch_size: Tuple[int, int, int],
    spine_disc_min_label: int = SEG_SPINE_DISC_MIN_LABEL,
) -> np.ndarray | None:
    """Build the 4-channel binary anatomy tensor fed to AugTransformsGPU.

    Channels (indexing matches ``discslabel.SEG_CH_*``):
        0  vertebrae    : SEG_SPINE voxels with value in (0, spine_disc_min_label)
        1  discs        : SEG_SPINE voxels with value >= spine_disc_min_label
        2  spinal cord  : SEG_CORD > 0
        3  spinal canal : SEG_CANAL > 0

    Every channel is a {0, 1} mask so kornia's mask data-key treats them as
    label masks with nearest resampling.

    Returns ``None`` if none of the SEG_* files is present for this entry.
    """
    patch = tuple(int(s) for s in patch_size)
    out = np.zeros((NUM_SEG_CHANNELS,) + patch, dtype=np.float32)
    present = False

    spine_path = entry.get("seg_spine")
    if spine_path:
        spine = np.rint(_load_nifti(spine_path)).astype(np.int32)
        vert = ((spine > 0) & (spine < int(spine_disc_min_label))).astype(np.float32)
        disc = (spine >= int(spine_disc_min_label)).astype(np.float32)
        out[SEG_CH_VERTEBRAE] = _center_crop_or_pad(vert, patch, pad_value=0.0)
        out[SEG_CH_DISCS] = _center_crop_or_pad(disc, patch, pad_value=0.0)
        present = True

    cord_path = entry.get("seg_cord")
    if cord_path:
        cord = (np.rint(_load_nifti(cord_path)) > 0).astype(np.float32)
        out[SEG_CH_CORD] = _center_crop_or_pad(cord, patch, pad_value=0.0)
        present = True

    canal_path = entry.get("seg_canal")
    if canal_path:
        canal = (np.rint(_load_nifti(canal_path)) > 0).astype(np.float32)
        out[SEG_CH_CANAL] = _center_crop_or_pad(canal, patch, pad_value=0.0)
        present = True

    return out if present else None


class DiscVolumeDataset(Dataset):
    """Reads a prepared split (``manifest.json`` + per-split ``images/`` /
    ``labels/`` / ``segs_{spine,cord,canal}/`` folders) and yields padded 3D
    tensors.

    Each sample is a dict:
        image : (1, D, H, W) float tensor, intensities in [0, 1]
        label : (1, D, H, W) float tensor, integer disc labels (0 = background)
        seg   : (NUM_SEG_CHANNELS, D, H, W) binary tensor — one channel per
                anatomical structure (see ``SEG_CHANNELS``). Omitted when
                none of the three SEG_* files is listed for this entry.
        name  : basename of the volume (useful for debugging / inference output)
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        patch_size: Tuple[int, int, int] = (96, 320, 320),
        require_label: bool = True,
        spine_disc_min_label: int = SEG_SPINE_DISC_MIN_LABEL,
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

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        image_path = entry["image"]
        image = _normalize_image(_load_nifti(image_path))
        image = _center_crop_or_pad(image, self.patch_size, pad_value=0.0)
        image_t = torch.from_numpy(image).unsqueeze(0).contiguous()  # (1, D, H, W)

        sample: Dict[str, torch.Tensor | str] = {
            "image": image_t,
            "name": os.path.basename(image_path),
        }

        label_path = entry.get("label")
        if label_path:
            label_arr = _load_nifti(label_path)
            # Point-wise labels are small integers; round to defend against
            # resample-induced float drift and clamp into the known range.
            label_arr = np.rint(label_arr).astype(np.float32)
            label_arr = _center_crop_or_pad(label_arr, self.patch_size, pad_value=0.0)
            sample["label"] = torch.from_numpy(label_arr).unsqueeze(0).contiguous()
        elif self.require_label:
            raise KeyError(f"Entry {entry} has no label but require_label=True")

        seg_arr = _build_binary_seg_channels(
            entry, self.patch_size, spine_disc_min_label=self.spine_disc_min_label
        )
        if seg_arr is not None:
            # Already (NUM_SEG_CHANNELS, D, H, W); no extra unsqueeze.
            sample["seg"] = torch.from_numpy(seg_arr).contiguous()
        return sample
