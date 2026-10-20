"""Dataset reader for mid-sagittal disc-landmark CSVs."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .transforms import landmarks_from_csv_row


def _load_slice(path: str) -> np.ndarray:
    """Load a mid-sagittal slice saved as either .npy or a 2D NIfTI."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".nii", ".gz"):
        import nibabel as nib

        img = nib.load(path)
        arr = np.asarray(img.get_fdata(), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported slice format: {path}")
    arr = np.squeeze(arr).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {arr.shape} from {path}")
    return arr


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min/max normalise to [0, 1]. Flat slices (constant intensity) map to zero."""
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


class DiscLandmarkDataset(Dataset):
    """Reads a landmark CSV written by ``discslabel prepare``.

    Each row is one subject-contrast pair: a path to its mid-sagittal slice
    plus the per-disc (label, x, y) triples projected onto that slice.
    """

    def __init__(self, csv_file_path: str, transform=None):
        super().__init__()
        self.frame = pd.read_csv(csv_file_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        slice_path = str(row.iloc[0])
        image = _normalize(_load_slice(slice_path))
        landmarks = landmarks_from_csv_row(row)

        sample = {"image": image, "landmarks": landmarks}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
