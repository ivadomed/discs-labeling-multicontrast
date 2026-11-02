"""Image-only sliding-window inference for the disc-landmark UNet.

Input is either a single ``.nii`` / ``.nii.gz`` file or a flat folder of
them. For each image we:

    0. reorient to ``cfg["orientation"]`` and resample to
       ``cfg["resolution"]`` (same preprocessing as training inputs — the
       model only ever saw volumes on that grid)
    1. z-score normalise the full 3D volume
    2. extract the mid-sagittal slice (axis R-L, center)
    3. slide ``(patch_H, patch_W)`` windows over the 2D slice and stitch the
       5-channel predictions into ONE slice-sized heatmap, each window
       weighted by a Gaussian importance map (nnU-Net style)
    4. decode once on the stitched heatmap: disc peaks from channel 0
       (threshold + Euclidean NMS) and the argmax of anchor channels 1..4
    5. pick the highest-priority detected anchor
       (``C2C3`` > ``L5S1`` > ``C7T1`` > ``T12L1``) and label every disc
       in superior → inferior order relative to that anchor
    6. project the labeled discs back onto the input image's native grid
       via pure affine arithmetic (points, no interpolation)
    7. write a 3D NIfTI with single-voxel discs on the mid-sagittal plane
       (shape and affine match the ORIGINAL input, not the preprocessed
       volume) plus a PNG preview of the preprocessed slice rendered with
       the same ``p1..p99`` contrast window used by the training previews.

``peak_threshold`` / ``min_peak_distance`` come from ``postprocessing.json``
next to the weights when ``discslabel_tune`` has been run, else from the
config.

No label or seg files are required. No augmentation is applied.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
import tqdm

logging.getLogger("discslabel.utils.image").setLevel(logging.ERROR)

from ..dataset import _z_score_normalize
from ..decoding import _choose_anchor, label_discs
from ..postprocessing import resolve_postprocessing
from ..training import load_inference_weights
from .model import load_model
from .predictor import predict_candidates
from .train import (
    _display_slice,
    _intensity_window,
    _load_config,
    _resolve_batch_size,
)
from ..utils.image import Image, resample_nib
from ..utils.io import check_and_make_dir


NIFTI_EXTS: Tuple[str, ...] = (".nii.gz", ".nii")


def _is_nifti(name: str) -> bool:
    return any(name.endswith(ext) for ext in NIFTI_EXTS)


def _discover_inputs(path: str) -> List[str]:
    """Return the list of NIfTI files for ``path``.

    * A file ending in ``.nii`` / ``.nii.gz`` → ``[path]``.
    * A directory → every ``*.nii`` / ``*.nii.gz`` directly inside it (sorted,
      flat, no recursion). Hidden files are skipped.
    * Anything else → ``FileNotFoundError``.
    """
    if os.path.isfile(path):
        if not _is_nifti(path):
            raise ValueError(f"--input file is not a NIfTI: {path}")
        return [path]
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if _is_nifti(name) and not name.startswith(".")
        )
        if not files:
            raise FileNotFoundError(f"No .nii / .nii.gz files found in {path}")
        return files
    raise FileNotFoundError(f"--input must be a .nii/.nii.gz file or a folder: {path}")


def _build_preview_png(
    slice_2d: np.ndarray,
    labeled_discs: List[Tuple[int, float, float]],
    title: str,
    save_path: str,
) -> None:
    """Mid-sagittal PNG with disc labels overlaid.

    Reorientation and contrast window match the training preview
    (``_display_slice`` + ``_intensity_window``: superior up, anterior left,
    gray-level between the 1st and 99th image percentile).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = _display_slice(slice_2d)
    H_orig, W_orig = slice_2d.shape

    vmin, vmax = _intensity_window(img)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    for label, x, y in labeled_discs:
        # slice (y=row=P-A, x=col=I-S) → display (row = W-1-x, col = y)
        disp_row = (W_orig - 1) - x
        disp_col = y
        ax.plot(
            disp_col,
            disp_row,
            "o",
            color="red",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.8,
        )
        ax.text(
            disp_col + 8,
            disp_row,
            str(int(label)),
            color="red",
            fontsize=14,
            fontweight="bold",
            va="center",
        )
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _resample_discs_to_native(
    labeled_discs: List[Tuple[int, float, float]],
    preproc_affine: np.ndarray,
    d_mid: int,
    original_nib: nib.Nifti1Image,
) -> np.ndarray:
    """Project disc labels from the preprocessed grid back to the native grid.

    Each disc is a POINT in physical space, so going back to the native
    grid is pure affine arithmetic — no interpolation needed. For every
    disc at preprocessed voxel ``(d_mid, round(y), round(x))`` we compute::

        native_voxel = inv(native_affine) @ preproc_affine @ preproc_voxel

    and plant the label at the nearest native voxel. One matrix solve, one
    rounding, no per-label volume resample — orders of magnitude faster
    than ``_resample_label_pointwise`` and mathematically exact for
    single-voxel landmarks.
    """
    dest_shape = tuple(int(s) for s in original_nib.shape[:3])
    out = np.zeros(dest_shape, dtype=np.int16)
    if not labeled_discs:
        return out

    discs_arr = np.asarray(labeled_discs, dtype=np.float64)  # (N, 3): label, x, y
    N = discs_arr.shape[0]

    # Preprocessed voxel coords in homogeneous form (4, N):
    #   row 0 = d_mid, row 1 = y (= H axis), row 2 = x (= W axis), row 3 = 1.
    preproc_voxels = np.empty((4, N), dtype=np.float64)
    preproc_voxels[0, :] = float(d_mid)
    preproc_voxels[1, :] = discs_arr[:, 2]  # y → H axis
    preproc_voxels[2, :] = discs_arr[:, 1]  # x → W axis
    preproc_voxels[3, :] = 1.0

    world = np.asarray(preproc_affine) @ preproc_voxels
    native_affine_inv = np.linalg.inv(np.asarray(original_nib.affine))
    native_voxels = native_affine_inv @ world  # (4, N)
    native_rounded = np.rint(native_voxels[:3, :]).astype(np.int64).T  # (N, 3)

    Dn, Hn, Wn = dest_shape
    for (label, _x, _y), (nd, nh, nw) in zip(labeled_discs, native_rounded):
        if 0 <= nd < Dn and 0 <= nh < Hn and 0 <= nw < Wn:
            out[int(nd), int(nh), int(nw)] = int(label)
    return out


def _write_disc_nifti(
    out_path: str, label_volume: np.ndarray, affine: np.ndarray
) -> None:
    """Save ``label_volume`` as an int16 NIfTI with the given affine.

    Fresh header so the output dtype stays ``int16`` instead of inheriting
    the likely-float dtype of the input image header.
    """
    img_out = nib.Nifti1Image(label_volume.astype(np.int16), affine=affine)
    img_out.set_data_dtype(np.int16)
    nib.save(img_out, out_path)


def _stem(path: str) -> str:
    """Basename without the ``.nii`` / ``.nii.gz`` extension."""
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _parse_resolution(resolution: str) -> Tuple[float, float, float]:
    """``"1x1x1"`` → ``(1.0, 1.0, 1.0)``. Raises if the string is malformed."""
    parts = resolution.lower().split("x")
    if len(parts) != 3:
        raise ValueError(
            f"resolution must be formatted as 'AxBxC' in mm, got {resolution!r}"
        )
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _preprocess_image(
    image_path: str, orientation: str, resolution_mm: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Load, reorient, and resample a NIfTI to match the training preprocessing.

    Returns ``(data, affine, original_nib)``: the preprocessed 3D float32 data
    on the ``(orientation, resolution_mm)`` grid along with its affine, plus
    the untouched nibabel image on the user's original grid — used at the end
    to project the output disc map back onto that same native frame.
    """
    original_nib = nib.load(image_path)
    image = Image(image_path)
    image.change_orientation(orientation)
    image = resample_nib(
        image,
        new_size=list(resolution_mm),
        new_size_type="mm",
        interpolation="linear",
        verbose=False,
    )
    data = np.asarray(image.data, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D volume after preprocessing, got shape {data.shape} for {image_path}"
        )
    affine = np.asarray(image.hdr.get_best_affine())
    return data, affine, original_nib


def _infer_one_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    orientation: str,
    resolution_mm: Tuple[float, float, float],
    patch_hw: Tuple[int, int],
    overlap: float,
    peak_threshold: float,
    min_peak_distance: float,
    batch_size: int,
    n_mid_slices: int,
    out_dir: str,
) -> None:
    """Run the full sliding-window pipeline on one 3D image."""
    image_raw, affine, original_nib = _preprocess_image(
        image_path, orientation, resolution_mm
    )
    image_norm = _z_score_normalize(image_raw)

    d_mid = image_raw.shape[0] // 2
    # Center slice of the input stack — the display background of the preview.
    slice_2d = image_norm[d_mid]

    candidates = predict_candidates(
        image_norm,
        model,
        device,
        patch_hw,
        overlap,
        batch_size,
        n_mid_slices,
        peak_threshold,
    )
    if _choose_anchor(candidates["anchors"], peak_threshold) is None:
        # tqdm.write prints above the active bar instead of smearing it.
        tqdm.tqdm.write(
            f"[warn] no anchor detected for {os.path.basename(image_path)} "
            f"(found {len(candidates['discs'])} disc candidates); writing empty NIfTI."
        )
    # Decoding yields (label, h, w); the preview / native projection below use
    # the slice convention (label, x = I-S = w, y = P-A = h).
    labeled_discs = [
        (label, w, h)
        for label, h, w in label_discs(candidates, peak_threshold, min_peak_distance)
    ]

    stem = _stem(image_path)
    nifti_out = os.path.join(out_dir, f"{stem}_discs.nii.gz")
    png_out = os.path.join(out_dir, f"{stem}_discs.png")

    # Preview first — it shows what the model actually saw, so it is kept
    # in the preprocessed frame regardless of what the output NIfTI is in.
    _build_preview_png(
        slice_2d,
        labeled_discs,
        title=f"{stem} — {len(labeled_discs)} discs",
        save_path=png_out,
    )

    # Project the preprocessed-frame labels back onto the user's native grid
    # so the output NIfTI overlays cleanly on their input image.
    native_label_volume = _resample_discs_to_native(
        labeled_discs,
        preproc_affine=affine,
        d_mid=d_mid,
        original_nib=original_nib,
    )
    _write_disc_nifti(nifti_out, native_label_volume, original_nib.affine)


def predict(
    cfg: dict,
    weights_path: str,
    input_path: str,
    out_dir: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = _discover_inputs(input_path)

    model = load_model(cfg["model_name"], in_channels=cfg["in_channels"]).to(device)
    load_inference_weights(model, weights_path, device)
    model.eval()

    patch_size = tuple(int(s) for s in cfg["patch_size"])
    patch_hw = (patch_size[1], patch_size[2])
    overlap = float(cfg.get("inference_overlap", 0.5))
    peak_thr, min_peak_distance, pp_source = resolve_postprocessing(cfg, weights_path)
    batch_size = _resolve_batch_size(cfg.get("inference_batch_size"), device)
    orientation = str(cfg.get("orientation", "RPI"))
    resolution_mm = _parse_resolution(str(cfg.get("resolution", "1x1x1")))
    n_mid_slices = int(cfg["in_channels"])
    print(
        f"[inference] device={device} inputs={len(image_paths)} "
        f"orientation={orientation} resolution={resolution_mm} "
        f"patch_hw={patch_hw} overlap={overlap} batch_size={batch_size} "
        f"n_mid_slices={n_mid_slices}"
    )
    print(
        f"[inference] peak_threshold={peak_thr} min_peak_distance={min_peak_distance} "
        f"(from {pp_source})"
    )

    check_and_make_dir(out_dir)
    for image_path in tqdm.tqdm(image_paths, desc="inference"):
        _infer_one_image(
            image_path,
            model=model,
            device=device,
            orientation=orientation,
            resolution_mm=resolution_mm,
            patch_hw=patch_hw,
            overlap=overlap,
            peak_threshold=peak_thr,
            min_peak_distance=min_peak_distance,
            batch_size=batch_size,
            n_mid_slices=n_mid_slices,
            out_dir=out_dir,
        )
    print(f"Wrote per-image NIfTI + PNG into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image-only sliding-window inference with a trained disc-landmark UNet."
    )
    parser.add_argument("--weights", required=True, help="Path to best_model.pt.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a .nii/.nii.gz file or a flat folder of them.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory for per-image <name>_discs.nii.gz + <name>_discs.png outputs.",
    )
    parser.add_argument("--config", required=True, help="JSON config.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    predict(
        cfg,
        weights_path=args.weights,
        input_path=args.input,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
