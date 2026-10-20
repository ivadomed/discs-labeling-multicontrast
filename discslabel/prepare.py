"""Prepare mid-sagittal 2D slices + landmark CSVs from 3D BIDS NIfTIs.

Reads a dataset JSON of the same shape as SmaugLab-benchmark's dataset JSONs
(``TRAINING`` / ``VALIDATION`` / ``TESTING`` lists of ``{IMAGE, LABEL}`` paths
relative to ``$DISCSLABEL_DATA/bids``). For every entry it:

  1. Loads the 3D image and the 3D point-wise disc label (values 1..25 placed
     at the posterior tip of each disc, with background = 0).
  2. Reorients both to the same canonical orientation.
  3. Picks the mid-sagittal slice along the right-left axis.
  4. Projects each labelled voxel onto that slice by dropping the R-L
     coordinate (keeps anterior-posterior and superior-inferior).
  5. Writes the slice as ``.npy`` under ``$DISCSLABEL_DATA/prepared/<split>/images/``
     and appends one row to ``train.csv`` / ``valid.csv`` / ``test.csv``.

CSV columns:
    slice_path, slice_h, slice_w, label_1, x_1, y_1, label_2, x_2, y_2, ...
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from .utils.io import check_and_make_dir


_CANONICAL_ORIENTATION = ("R", "A", "S")  # x=R-L, y=A-P, z=S-I in RAS+


def _reorient_to_ras(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient a NIfTI to RAS+ so axis roles are deterministic across subjects."""
    return nib.as_closest_canonical(img)


def _mid_sagittal_slice(volume: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return the mid-sagittal slice (middle index along the R-L axis = axis 0 in RAS)."""
    mid = volume.shape[0] // 2
    return volume[mid, :, :], mid


def _project_labels(label_vol: np.ndarray) -> List[Tuple[int, int, int]]:
    """Collect every non-zero voxel of ``label_vol`` as ``(label, x_col, y_row)``.

    ``label_vol`` is in RAS+ with axes (R-L, A-P, S-I). Mid-sagittal view is
    rendered with A-P on the horizontal axis (columns) and S-I on the vertical
    axis (rows). We drop the R-L coordinate — point labels sit on the midline
    of the disc anyway, so collapsing across R-L yields one 2D landmark per
    disc in practice.
    """
    coords = np.argwhere(label_vol > 0)
    out: List[Tuple[int, int, int]] = []
    seen = {}  # label -> (x_col, y_row) : keep the median-R-L voxel per label
    buckets: dict = {}
    for r, a, s in coords:
        label = int(label_vol[r, a, s])
        buckets.setdefault(label, []).append((int(a), int(s)))
    for label, pts in buckets.items():
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        x_col = xs[len(xs) // 2]
        y_row = ys[len(ys) // 2]
        seen[label] = (x_col, y_row)
    for label in sorted(seen):
        x_col, y_row = seen[label]
        out.append((label, x_col, y_row))
    return out


def _slice_to_npy_name(image_rel_path: str) -> str:
    base = os.path.basename(image_rel_path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base + ".npy"


def _render_mid_sagittal(image_path: str, label_path: str):
    """Return (mid_sagittal_image_slice, landmarks_on_slice).

    The slice is oriented so that ``slice[row, col]`` has:
        row -> S-I axis in RAS (so top row = superior)
        col -> A-P axis in RAS (so left col = anterior)
    Landmarks come out as ``(label, x_col, y_row)``.
    """
    img = _reorient_to_ras(nib.load(image_path))
    img_arr = np.asarray(img.get_fdata(), dtype=np.float32)
    img_slice, mid_rl = _mid_sagittal_slice(img_arr)

    lbl = _reorient_to_ras(nib.load(label_path))
    lbl_arr = np.asarray(lbl.get_fdata())
    lbl_arr = np.rint(lbl_arr).astype(np.int32)
    landmarks = _project_labels(lbl_arr)

    # Reorient the slice so superior is up and anterior is left when displayed.
    # argwhere returns (A, S) above; our landmarks are already in (col=A, row=S).
    # Flip rows so S increases upward -> row 0 = top = superior:
    img_slice = np.ascontiguousarray(img_slice.T[::-1, :])
    h = img_slice.shape[0]
    landmarks = [(label, x, h - 1 - y) for (label, x, y) in landmarks]
    return img_slice, landmarks


def _max_landmarks(entries: Iterable[dict]) -> int:
    return max((len(e.get("_landmarks", [])) for e in entries), default=0)


def _write_split_csv(entries: List[dict], out_path: str, n_landmark_slots: int) -> None:
    columns = ["slice_path", "slice_h", "slice_w"]
    for i in range(n_landmark_slots):
        columns.extend([f"label_{i + 1}", f"x_{i + 1}", f"y_{i + 1}"])
    rows = []
    for e in entries:
        row = [e["_slice_path"], e["_slice_h"], e["_slice_w"]]
        for i in range(n_landmark_slots):
            if i < len(e["_landmarks"]):
                label, x, y = e["_landmarks"][i]
                row.extend([label, x, y])
            else:
                row.extend(["", "", ""])
        rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(out_path, index=False)


def _data_root() -> str:
    root = os.environ.get("DISCSLABEL_DATA")
    if not root:
        raise SystemExit(
            "Please set the DISCSLABEL_DATA environment variable before running "
            "`discslabel_prepare` (it points to the root folder that holds `bids/` "
            "and will receive `prepared/`)."
        )
    return root


def _process_split(
    split_name: str,
    entries: List[dict],
    data_root: str,
    out_root: str,
) -> List[dict]:
    """Materialise the mid-sagittal slices for one split and attach metadata to each entry."""
    images_dir = os.path.join(out_root, split_name, "images")
    check_and_make_dir(images_dir)

    processed = []
    for entry in entries:
        image_rel = entry.get("IMAGE")
        label_rel = entry.get("LABEL")
        if image_rel is None or label_rel is None:
            print(f"[prepare] skipping entry without IMAGE/LABEL: {entry}")
            continue
        image_path = os.path.join(data_root, "bids", image_rel)
        label_path = os.path.join(data_root, "bids", label_rel)
        if not os.path.isfile(image_path):
            print(f"[prepare] missing image: {image_path}")
            continue
        if not os.path.isfile(label_path):
            print(f"[prepare] missing label: {label_path}")
            continue

        slice_img, landmarks = _render_mid_sagittal(image_path, label_path)
        slice_name = _slice_to_npy_name(image_rel)
        slice_path = os.path.join(images_dir, slice_name)
        np.save(slice_path, slice_img.astype(np.float32))

        processed.append(
            {
                **entry,
                "_slice_path": slice_path,
                "_slice_h": int(slice_img.shape[0]),
                "_slice_w": int(slice_img.shape[1]),
                "_landmarks": landmarks,
            }
        )
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mid-sagittal slices and landmark CSVs.")
    parser.add_argument(
        "--data-json",
        required=True,
        help="Path to the dataset JSON (same schema as SmaugLab-benchmark).",
    )
    parser.add_argument(
        "--out-name",
        default=None,
        help="Subfolder under $DISCSLABEL_DATA/prepared/ (defaults to the data JSON basename).",
    )
    args = parser.parse_args()

    with open(args.data_json, "r") as f:
        spec = json.load(f)

    data_root = _data_root()
    out_name = args.out_name or os.path.splitext(os.path.basename(args.data_json))[0]
    out_root = os.path.join(data_root, "prepared", out_name)
    check_and_make_dir(out_root)

    all_splits = {
        "train": spec.get("TRAINING", []),
        "valid": spec.get("VALIDATION", []),
        "test": spec.get("TESTING", []),
    }

    processed = {}
    for split, entries in all_splits.items():
        print(f"[prepare] processing split '{split}' ({len(entries)} entries)")
        processed[split] = _process_split(split, entries, data_root, out_root)

    all_processed = [e for s in processed.values() for e in s]
    n_slots = _max_landmarks(all_processed)
    print(f"[prepare] widest CSV row uses {n_slots} landmark triples")

    for split, entries in processed.items():
        out_csv = os.path.join(out_root, f"{split}.csv")
        _write_split_csv(entries, out_csv, n_slots)
        print(f"[prepare] wrote {out_csv} ({len(entries)} rows)")


if __name__ == "__main__":
    main()
