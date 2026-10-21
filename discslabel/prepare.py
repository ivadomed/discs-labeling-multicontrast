"""Build the per-split manifest after reorient + resample has run on the BIDS folder.

Preprocessing itself (reorient to RPI, resample to 1 mm isotropic) is done by
``scripts/prepare_datasets.sh`` through the ``discslabel_reorient_image`` /
``discslabel_resample_image`` CLIs. This script only copies/renames files into
the per-split folders and emits ``manifest.json``, which the dataset reader
consumes at training time.

The 2D mid-sagittal slice and the Gaussian heatmap targets are built **inside
the data loader on the GPU** — not here. We keep both the image and the
point-wise label as full 3D NIfTIs so the 3D GPU augmentations from SmaugLab
can operate on them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List

from . import SEG_KEYS
from .utils.io import check_and_make_dir


_SPLITS = ("train", "valid", "test")
_SPLIT_SRC_KEYS = {"train": "TRAINING", "valid": "VALIDATION", "test": "TESTING"}


def _data_root() -> str:
    root = os.environ.get("DISCSLABEL_DATA")
    if not root:
        raise SystemExit(
            "Please set the DISCSLABEL_DATA environment variable before running "
            "`discslabel_prepare` (it points to the root folder that holds `bids/` "
            "and will receive `prepared/`)."
        )
    return root


def _strip_ext(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def _collect_split(entries: List[dict], split_name: str, data_root: str, out_root: str) -> List[dict]:
    images_dst = os.path.join(out_root, split_name, "images")
    labels_dst = os.path.join(out_root, split_name, "labels")
    check_and_make_dir(images_dst)
    check_and_make_dir(labels_dst)
    seg_dsts = {}
    for json_key, folder, _val in SEG_KEYS:
        seg_dsts[json_key] = os.path.join(out_root, split_name, folder)
        check_and_make_dir(seg_dsts[json_key])

    # Manifest key used in the output JSON for each SEG field (SEG_SPINE -> seg_spine).
    seg_manifest_keys = {json_key: json_key.lower() for json_key, _folder, _val in SEG_KEYS}

    manifest = []
    for entry in entries:
        image_rel = entry.get("IMAGE")
        label_rel = entry.get("LABEL")
        if image_rel is None:
            print(f"[prepare] skipping entry without IMAGE: {entry}")
            continue

        image_src = os.path.join(data_root, "bids", image_rel)
        if not os.path.isfile(image_src):
            print(f"[prepare] missing image: {image_src}")
            continue

        base = _strip_ext(os.path.basename(image_rel))
        image_dst = os.path.join(images_dst, base + ".nii.gz")
        shutil.copyfile(image_src, image_dst)

        row: Dict[str, str] = {"image": image_dst}

        if label_rel is not None:
            label_src = os.path.join(data_root, "bids", label_rel)
            if not os.path.isfile(label_src):
                print(f"[prepare] missing label: {label_src}")
                continue
            label_dst = os.path.join(labels_dst, base + ".nii.gz")
            shutil.copyfile(label_src, label_dst)
            row["label"] = label_dst

        for json_key, _folder, _val in SEG_KEYS:
            seg_rel = entry.get(json_key)
            if seg_rel is None:
                continue
            seg_src = os.path.join(data_root, "bids", seg_rel)
            if not os.path.isfile(seg_src):
                print(f"[prepare] missing {json_key.lower()}: {seg_src}")
                continue
            seg_dst = os.path.join(seg_dsts[json_key], base + ".nii.gz")
            shutil.copyfile(seg_src, seg_dst)
            row[seg_manifest_keys[json_key]] = seg_dst

        manifest.append(row)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy BIDS images+labels into prepared/<name>/{train,valid,test} and emit manifest.json."
    )
    parser.add_argument("--data-json", required=True, help="Dataset JSON (SmaugLab-benchmark schema).")
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

    manifest = {}
    for split in _SPLITS:
        src_key = _SPLIT_SRC_KEYS[split]
        entries = spec.get(src_key, [])
        print(f"[prepare] copying split '{split}' ({len(entries)} entries) from {src_key}")
        manifest[split] = _collect_split(entries, split, data_root, out_root)

    manifest_path = os.path.join(out_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[prepare] wrote {manifest_path}")
    print(
        "[prepare] NOTE: run scripts/prepare_datasets.sh (not this script directly) if you "
        "want the data reoriented to RPI and resampled to 1 mm iso before manifest generation."
    )


if __name__ == "__main__":
    main()
