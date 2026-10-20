"""Data transforms: rescale, flip, convert to tensor, and generate Gaussian heatmaps.

Samples flow through transforms as dicts with keys:
    image     -> (H, W) float32 numpy array (single-channel 2D slice)
    landmarks -> list of (label, x, y) tuples, with ``label`` in 1..25
                 and ``(x, y)`` in image pixel coordinates (x = column, y = row)
"""

from __future__ import annotations

import random
from typing import Sequence, Tuple

import numpy as np
from skimage import transform as sk_transform

from . import LANDMARK_CHANNEL_LABELS, NUM_CLASSES


def generate_2d_gaussian_heatmap(
    heatmap: np.ndarray,
    center: Tuple[int, int],
    sigma: float = 8.0,
    radius: int = 20,
) -> np.ndarray:
    """Paint a 2D isotropic Gaussian into ``heatmap`` at ``center = (x0, y0)``.

    Multiple landmarks on the same channel are combined by element-wise
    ``np.maximum`` so overlapping peaks don't cancel.
    """
    x0, y0 = center
    xx, yy = np.ogrid[-radius:radius + 1, -radius:radius + 1]

    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0

    height, width = heatmap.shape
    left, right = min(x0, radius), min(width - x0, radius + 1)
    top, bottom = min(y0, radius), min(height - y0, radius + 1)

    masked_heatmap = heatmap[y0 - top:y0 + bottom, x0 - left:x0 + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


class Rescale:
    """Rescale the 2D slice to ``output_size`` and rescale landmark coords with it."""

    def __init__(self, output_size: Tuple[int, int]):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]
        new_h, new_w = int(self.output_size[0]), int(self.output_size[1])

        image = sk_transform.resize(image, (new_h, new_w), mode="constant", preserve_range=False)

        rescaled = []
        for label, x, y in landmarks:
            rescaled.append((label, x * new_w / w, y * new_h / h))
        return {"image": image, "landmarks": rescaled}


class RandomVerticalFlip:
    """Flip the slice up/down.

    Mid-sagittal spine slices are oriented with superior/inferior along the
    vertical axis, so horizontal flipping would swap left/right (not useful on
    a mid-sagittal cut) and vertical flipping would swap cranial/caudal — which
    scrambles the C2C3 / L5S1 anchor identities. We only expose it here as an
    optional augmentation; keep ``p = 0`` for the default disc-labeling task.
    """

    def __init__(self, p: float):
        self.prob = float(p)

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        if random.random() < self.prob:
            h = image.shape[0]
            image = image[::-1, :].copy()
            landmarks = [(label, x, h - 1 - y) for (label, x, y) in landmarks]
        return {"image": image, "landmarks": landmarks}


class RandomIntensityJitter:
    """Multiplicative + additive intensity jitter on the normalised slice."""

    def __init__(self, scale: float = 0.1, shift: float = 0.1):
        self.scale = float(scale)
        self.shift = float(shift)

    def __call__(self, sample):
        image = sample["image"].astype(np.float32)
        s = 1.0 + random.uniform(-self.scale, self.scale)
        b = random.uniform(-self.shift, self.shift)
        image = np.clip(s * image + b, 0.0, 1.0)
        return {"image": image, "landmarks": sample["landmarks"]}


class ToTensor:
    """Convert ``image`` to CHW float32 and build the 5-channel heatmap target.

    Output tensors (as numpy arrays — the DataLoader converts to torch):
        image   -> (1, H, W)
        heatmap -> (NUM_CLASSES, H, W)
    """

    def __init__(self, sigma: float = 8.0, radius: int = 20):
        self.sigma = float(sigma)
        self.radius = int(radius)

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]

        heatmap = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
        for label, x, y in landmarks:
            cx, cy = int(x + 0.5), int(y + 0.5)
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            # Channel 0: union of every visible disc
            generate_2d_gaussian_heatmap(heatmap[0], (cx, cy), self.sigma, self.radius)
            # Channels 1..4: specific anchor landmarks
            for ch in range(1, NUM_CLASSES):
                if LANDMARK_CHANNEL_LABELS[ch] == label:
                    generate_2d_gaussian_heatmap(heatmap[ch], (cx, cy), self.sigma, self.radius)

        image = image.astype(np.float32)[np.newaxis, :, :]
        return image, heatmap


def landmarks_from_csv_row(row: Sequence) -> list:
    """Parse one dataset-CSV row's landmark columns into [(label, x, y), ...].

    CSV layout: ``image_path, slice_shape_h, slice_shape_w, label_1, x_1, y_1,
                 label_2, x_2, y_2, ...``. Columns past the shape are read in
    triples until we run out of data. Missing (NaN) triples are skipped.
    """
    result = []
    data = list(row)[3:]
    for i in range(0, len(data), 3):
        if i + 2 >= len(data):
            break
        label, x, y = data[i], data[i + 1], data[i + 2]
        try:
            label = int(label)
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            continue
        if np.isnan(x) or np.isnan(y):
            continue
        result.append((label, x, y))
    return result
