"""Render the GPU MRI artifacts / S-I low-resolution augmentation for debugging.

For each input volume, every artifact of ``discslabel.gpu_artifacts`` is
applied on its own (forced, ``p = 1``) a few times with different random
draws, plus the S-I low-resolution step (final spacing drawn in
``lowres_si_config.spacing`` from the volume's own spacing) and the full training mix
(``lowres_si_config`` + ``artifact_config`` with their configured
probabilities). Writes to ``--out-dir``:

  * ``<stem>_augmentations.png``: one row per augmentation, column 0 =
    original, then one column per random draw; mid-sagittal slice shown
    like the training previews (superior up, anterior left);
  * with ``--save-nifti``: ``<stem>/<augmentation>_draw<k>.nii.gz`` volumes
    (preprocessed grid) to inspect in a viewer;
  * the mean GPU/CPU time of each augmentation, printed at the end.

Artifact parameters come from ``artifact_config`` / ``lowres_si_config`` in
``--config`` when given (only the enabled artifacts are shown); otherwise
every artifact is shown with its default parameters.

Examples::

    discslabel_debug_augmentations --manifest $DISCSLABEL_DATA/prepared/<name>/manifest.json \\
        --config discslabel/pipeline_2d/configs/default.json --out-dir debug_aug
    discslabel_debug_augmentations --input sub-01_T2w.nii.gz --out-dir debug_aug --save-nifti
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch

logging.getLogger("discslabel.utils.image").setLevel(logging.ERROR)

from .dataset import _z_score_normalize
from .gpu_artifacts import (
    ARTIFACTS,
    RandomLowResAxisGPU,
    _parse_artifacts,
    apply_artifact,
    build_image_augmentations,
)

DEFAULT_LOWRES = {"p": 1.0, "spacing": [1.0, 5.0], "axis": "SI"}


def _stem(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _load_inputs(args) -> List[Tuple[str, np.ndarray, np.ndarray, Tuple[float, float, float]]]:
    """``[(stem, z-scored volume, affine, voxel spacing mm), ...]`` on the training grid."""
    out = []
    if args.manifest:
        with open(args.manifest) as f:
            entries = json.load(f)[args.split]
        for entry in entries[: args.n_images]:
            img = nib.load(entry["image"])
            data = np.squeeze(np.asarray(img.get_fdata(), dtype=np.float32))
            spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
            out.append((_stem(entry["image"]), _z_score_normalize(data), np.asarray(img.affine), spacing))
        return out

    from .pipeline_2d.inference import _discover_inputs, _parse_resolution, _preprocess_image

    orientation, resolution = "RPI", (1.0, 1.0, 1.0)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        orientation = str(cfg.get("orientation", orientation))
        resolution = _parse_resolution(str(cfg.get("resolution", "1x1x1")))
    for path in _discover_inputs(args.input)[: args.n_images]:
        data, affine, _ = _preprocess_image(path, orientation, resolution)
        out.append((_stem(path), _z_score_normalize(data), affine, tuple(resolution)))
    return out


def _variants(cfg: dict, voxel_spacing) -> Dict[str, callable]:
    """Augmentation name → ``fn(volume (1, 1, D, H, W)) -> same shape``, all forced."""
    artifact_cfg = cfg.get("artifact_config")
    lowres_cfg = cfg.get("lowres_si_config")
    artifacts = _parse_artifacts(artifact_cfg) if artifact_cfg else {n: {} for n in ARTIFACTS}
    retain = bool((artifact_cfg or {}).get("retain_stats", True))

    variants: Dict[str, callable] = {}
    for name, params in artifacts.items():
        variants[name] = lambda x, n=name, p=params: apply_artifact(n, x[0], p, retain)[None]
    lowres_forced = RandomLowResAxisGPU({**(lowres_cfg or DEFAULT_LOWRES), "p": 1.0}, voxel_spacing)
    variants["lowres_si"] = lowres_forced

    pipeline = build_image_augmentations(artifact_cfg, lowres_cfg, voxel_spacing)
    if pipeline is not None:
        variants["training_mix"] = pipeline
    return variants


def _display(slice_2d: np.ndarray) -> np.ndarray:
    """Mid-sagittal (P-A, I-S) → superior up, anterior left (training preview layout)."""
    return np.ascontiguousarray(slice_2d.T[::-1, :])


def _window(img: np.ndarray) -> Tuple[float, float]:
    vmin, vmax = float(np.percentile(img, 1.0)), float(np.percentile(img, 99.0))
    if vmax <= vmin:
        vmin, vmax = float(img.min()), float(img.max() + 1e-6)
    return vmin, vmax


def _save_grid(path: str, title: str, rows: Dict[str, List[np.ndarray]], original: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_cols = 1 + max(len(v) for v in rows.values())
    h, w = original.shape
    cell_w = 2.6
    cell_h = cell_w * w / max(h, 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell_w * n_cols, cell_h * n_rows + 0.6), squeeze=False)
    orig_disp = _display(original)
    for r, (name, slices) in enumerate(rows.items()):
        panels = [orig_disp] + [_display(s) for s in slices]
        for c in range(n_cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(panels):
                ax.axis("off")
                continue
            ax.imshow(panels[c], cmap="gray", vmin=_window(panels[c])[0], vmax=_window(panels[c])[1])
            if r == 0:
                ax.set_title("original" if c == 0 else f"draw {c}", fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the GPU MRI-artifact / S-I low-resolution augmentations for debugging."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="Prepared manifest.json (images already RPI / 1 mm).")
    src.add_argument(
        "--input",
        help="A .nii/.nii.gz file or flat folder of them (reoriented + resampled like inference).",
    )
    parser.add_argument("--split", default="train", help="Manifest split (default: train).")
    parser.add_argument("--config", default=None, help="Pipeline JSON with artifact_config / lowres_si_config.")
    parser.add_argument("--out-dir", required=True, help="Where to write the PNGs (and NIfTIs).")
    parser.add_argument("--n-images", type=int, default=3, help="Number of volumes to render (default: 3).")
    parser.add_argument("--n-draws", type=int, default=4, help="Random draws per augmentation (default: 4).")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed (default: 0).")
    parser.add_argument("--save-nifti", action="store_true", help="Also save every augmented volume.")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda when available).")
    args = parser.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[debug-aug] device={device}")

    os.makedirs(args.out_dir, exist_ok=True)
    timings: Dict[str, List[float]] = defaultdict(list)
    for i_img, (stem, volume, affine, spacing) in enumerate(_load_inputs(args)):
        variants = _variants(cfg, spacing)
        if i_img == 0:
            print(f"[debug-aug] augmentations={list(variants)}")
        x = torch.from_numpy(volume)[None, None].to(device)
        d_mid = volume.shape[0] // 2
        rows: Dict[str, List[np.ndarray]] = {}
        for i_var, (name, fn) in enumerate(variants.items()):
            rows[name] = []
            for k in range(args.n_draws):
                torch.manual_seed(args.seed + 1000 * i_img + 100 * i_var + k)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    y = fn(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                timings[name].append(time.perf_counter() - t0)
                y_np = y[0, 0].float().cpu().numpy()
                rows[name].append(y_np[d_mid])
                if args.save_nifti:
                    nii_dir = os.path.join(args.out_dir, stem)
                    os.makedirs(nii_dir, exist_ok=True)
                    nib.save(
                        nib.Nifti1Image(y_np.astype(np.float32), affine),
                        os.path.join(nii_dir, f"{name}_draw{k}.nii.gz"),
                    )
        if args.save_nifti:
            nib.save(
                nib.Nifti1Image(volume.astype(np.float32), affine),
                os.path.join(args.out_dir, stem, "original.nii.gz"),
            )
        png = os.path.join(args.out_dir, f"{stem}_augmentations.png")
        spacing_str = "x".join(f"{v:g}" for v in spacing)
        _save_grid(png, f"{stem}  shape={tuple(volume.shape)}  spacing={spacing_str} mm", rows, volume[d_mid])
        print(f"[debug-aug] wrote {png}")

    print("[debug-aug] mean time per volume:")
    for name, ts in timings.items():
        print(f"  {name:<14s} {1000 * float(np.mean(ts)):8.1f} ms")


if __name__ == "__main__":
    main()
