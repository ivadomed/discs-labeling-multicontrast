"""Image-only 3D sliding-window inference for the disc-landmark 3D UNet.

Full-3D counterpart to ``discslabel.pipeline_2d.inference``. Input is either
a single ``.nii`` / ``.nii.gz`` file or a flat folder of them. For each image:

    0. reorient to ``cfg["orientation"]`` and resample to
       ``cfg["resolution"]`` (same preprocessing as training inputs)
    1. z-score normalise the full 3D volume
    2. slide ``(patch_D, patch_H, patch_W)`` windows over the whole volume and
       stitch the 5-channel predictions into ONE volume-sized heatmap, each
       window weighted by a Gaussian importance map (nnU-Net style)
    3. decode once on the stitched heatmap: disc peaks from channel 0
       (threshold + Euclidean NMS) and the argmax of anchor channels 1..4
    4. pick the highest-priority detected anchor
       (``C2C3`` > ``L5S1`` > ``C7T1`` > ``T12L1``) and label every disc
       superior → inferior relative to that anchor
    5. project the labeled discs back onto the input image's native grid via
       pure affine arithmetic (points, no interpolation)
    6. write a 3D NIfTI with single-voxel discs plus a PNG preview showing
       the preprocessed mid-sagittal slice with each labeled disc overlaid.

``peak_threshold`` / ``min_peak_distance`` come from ``postprocessing.json``
next to the weights when ``discslabel_tune_3d`` has been run, else from the
config.

Optional skeleton step (``--skeleton``, needs ``skeleton.json`` from
``discslabel_skeleton``): after the anchor-based labelling, the inter-disc
distances are checked against the training distribution
(``discslabel.skeleton``). ``flag`` only reports problems; ``correct`` also
removes off-chain peaks, fixes the labels and fills missed discs. Each image
then gets a ``<name>_discs_qc.json`` report and the run a ``qc_summary.csv``.

No label or seg files are required. No augmentation is applied.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import tqdm

logging.getLogger("discslabel.utils.image").setLevel(logging.ERROR)

from discslabel.dataset import _z_score_normalize
from discslabel.decoding import _choose_anchor, extract_candidates, label_discs
from discslabel.parallel import BoundedSubmitter, imap_ordered, make_executor, resolve_num_workers
from discslabel.postprocessing import resolve_postprocessing, resolve_skeleton
from discslabel.skeleton import (
    SKELETON_MODES,
    affine_spacing,
    decode_with_skeleton,
    qc_row as _qc_row,
    write_qc_summary as _write_qc_summary,
)
from discslabel.training import load_inference_weights
from discslabel.pipeline_3d.model import load_model
from discslabel.pipeline_3d.predictor import SKELETON_PROJECTION, predict_volume_heatmap
from discslabel.pipeline_3d.train import (
    _display_slice,
    _intensity_window,
    _load_config,
    _resolve_batch_size,
)
from discslabel.utils.image import Image, resample_nib
from discslabel.utils.io import check_and_make_dir


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
    labeled_discs: List[Tuple[int, float, float, float]],
    title: str,
    save_path: str,
    imputed_labels: frozenset = frozenset(),
    heatmap_2d: Optional[np.ndarray] = None,
) -> None:
    """Mid-sagittal PNG with every labeled disc overlaid.

    Discs imputed by the skeleton correction are drawn in yellow.

    ``slice_2d`` is the preprocessed volume sliced at D//2; markers are placed
    at each disc's (h, w) regardless of its own ``d`` so all detected discs are
    visible even when they sit off the midline (common in scoliosis).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = _display_slice(slice_2d)
    _H_orig, W_orig = slice_2d.shape

    vmin, vmax = _intensity_window(img)
    n_cols = 1 if heatmap_2d is None else 2
    # Column width follows the displayed image aspect so panels sit side by side.
    col_w = min(8.0, max(4.0, 10.0 * img.shape[1] / img.shape[0] + 1.0))
    fig, axes = plt.subplots(1, n_cols, figsize=(col_w * n_cols, 10), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    for label, _d, h, w in labeled_discs:
        disp_row = (W_orig - 1) - w
        disp_col = h
        color = "yellow" if int(label) in imputed_labels else "red"
        ax.plot(
            disp_col,
            disp_row,
            "o",
            color=color,
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.8,
        )
        ax.text(
            disp_col + 8,
            disp_row,
            str(int(label)),
            color=color,
            fontsize=14,
            fontweight="bold",
            va="center",
        )
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    if heatmap_2d is not None:
        # Raw sigmoid output (no peak threshold, no masking): max over all
        # channels (discs + anchors), overlaid on the same background.
        hm = _display_slice(heatmap_2d.max(axis=0))
        ax_hm = axes[0, 1]
        ax_hm.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        im = ax_hm.imshow(hm, cmap="hot", alpha=0.6, vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.02)
        ax_hm.set_title(f"predicted heatmap (max over channels, peak={hm.max():.2f})", fontsize=12)
        ax_hm.set_xticks([])
        ax_hm.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _resample_discs_to_native(
    labeled_discs: List[Tuple[int, float, float, float]],
    preproc_affine: np.ndarray,
    native_shape: Tuple[int, int, int],
    native_affine: np.ndarray,
) -> np.ndarray:
    """Project disc labels from the preprocessed grid back to the native grid.

    Each disc is a POINT in physical space, so projection is pure affine
    arithmetic — no interpolation needed. For every disc at preprocessed voxel
    ``(d, h, w)``::

        native_voxel = inv(native_affine) @ preproc_affine @ (d, h, w, 1)

    plant the label at the nearest native voxel. One matrix solve per batch,
    one rounding, no per-label resample.
    """
    dest_shape = tuple(int(s) for s in native_shape[:3])
    out = np.zeros(dest_shape, dtype=np.int16)
    if not labeled_discs:
        return out

    discs_arr = np.asarray(labeled_discs, dtype=np.float64)  # (N, 4): label, d, h, w
    N = discs_arr.shape[0]

    preproc_voxels = np.empty((4, N), dtype=np.float64)
    preproc_voxels[0, :] = discs_arr[:, 1]  # d
    preproc_voxels[1, :] = discs_arr[:, 2]  # h
    preproc_voxels[2, :] = discs_arr[:, 3]  # w
    preproc_voxels[3, :] = 1.0

    world = np.asarray(preproc_affine) @ preproc_voxels
    native_affine_inv = np.linalg.inv(np.asarray(native_affine))
    native_voxels = native_affine_inv @ world
    native_rounded = np.rint(native_voxels[:3, :]).astype(np.int64).T

    Dn, Hn, Wn = dest_shape
    for (label, _d, _h, _w), (nd, nh, nw) in zip(labeled_discs, native_rounded):
        if 0 <= nd < Dn and 0 <= nh < Hn and 0 <= nw < Wn:
            out[int(nd), int(nh), int(nw)] = int(label)
    return out


def _write_disc_nifti(
    out_path: str, label_volume: np.ndarray, affine: np.ndarray
) -> None:
    """Save ``label_volume`` as an int16 NIfTI with the given affine."""
    img_out = nib.Nifti1Image(label_volume.astype(np.int16), affine=affine)
    img_out.set_data_dtype(np.int16)
    nib.save(img_out, out_path)


def _stem(path: str) -> str:
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _parse_resolution(resolution: str) -> Tuple[float, float, float]:
    parts = resolution.lower().split("x")
    if len(parts) != 3:
        raise ValueError(
            f"resolution must be formatted as 'AxBxC' in mm, got {resolution!r}"
        )
    return tuple(float(p) for p in parts)  # type: ignore[return-value]


def _preprocess_image(
    image_path: str, orientation: str, resolution_mm: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Load, reorient, and resample a NIfTI to match the training preprocessing."""
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


def _load_and_preprocess(job: Tuple[str, str, Tuple[float, float, float]]) -> dict:
    """Worker job: preprocess + z-score one image (CPU only, no model)."""
    image_path, orientation, resolution_mm = job
    image_raw, affine, original_nib = _preprocess_image(image_path, orientation, resolution_mm)
    return {
        "path": image_path,
        "volume": _z_score_normalize(image_raw),
        "affine": affine,
        "native_shape": tuple(int(s) for s in original_nib.shape[:3]),
        "native_affine": np.asarray(original_nib.affine),
    }


def _finalize_one(
    image_path: str,
    heatmap: np.ndarray,
    slice_2d: np.ndarray,
    affine: np.ndarray,
    native_shape: Tuple[int, int, int],
    native_affine: np.ndarray,
    peak_threshold: float,
    min_peak_distance: float,
    out_dir: str,
    skeleton=None,
    skeleton_kwargs: Optional[dict] = None,
    skeleton_mode: str = "off",
    plain_out_dir: Optional[str] = None,
) -> Tuple[Optional[str], Optional[dict]]:
    """Worker job: decode the stitched heatmap and write the NIfTI + PNG.

    With a skeleton and ``plain_out_dir``, the decoding without the skeleton
    is also written there (same heatmap, no second forward pass).

    Returns ``(warning, qc_row)``: a message (or ``None``) the main process
    prints above its progress bar, and the QC summary row (``None`` when the
    skeleton step is off).
    """
    candidates = extract_candidates(heatmap, peak_threshold)
    # (C, D, H, W) → (C, H, W): max along R-L so off-midline peaks stay visible.
    heatmap_2d = heatmap.max(axis=1)
    del heatmap
    stem = _stem(image_path)
    warning = None
    if _choose_anchor(candidates["anchors"], peak_threshold) is None:
        warning = (
            f"[warn] no anchor detected for {os.path.basename(image_path)} "
            f"(found {len(candidates['discs'])} disc candidates); writing empty NIfTI."
        )

    qc_row = None
    imputed = frozenset()
    if skeleton is None:
        labeled_discs = label_discs(candidates, peak_threshold, min_peak_distance)
    else:
        labeled_discs, report = decode_with_skeleton(
            candidates,
            peak_threshold,
            min_peak_distance,
            affine_spacing(affine, skeleton.projection),
            skeleton,
            mode=skeleton_mode,
            **skeleton_kwargs,
        )
        if skeleton_mode == "correct":
            imputed = frozenset(d["label"] for d in report.get("changes", {}).get("imputed", []))
        report = {"image": stem, "n_discs": len(labeled_discs), **report}
        with open(os.path.join(out_dir, f"{stem}_discs_qc.json"), "w") as f:
            json.dump(report, f, indent=2)
        qc_row = _qc_row(report)
        if report["status"] != "ok" and warning is None:
            warning = f"[qc] {stem}: {', '.join(report['flags'])}"

    title = f"{stem} — {len(labeled_discs)} discs"
    if qc_row is not None and qc_row["status"] != "ok":
        title += f"\nQC: {qc_row['flags']}"
    _write_outputs(
        out_dir, stem, labeled_discs, title, imputed,
        slice_2d, affine, native_shape, native_affine, heatmap_2d,
    )
    if plain_out_dir is not None and skeleton is not None:
        # Same heatmap decoded without the skeleton, for side-by-side evaluation.
        plain = label_discs(candidates, peak_threshold, min_peak_distance)
        _write_outputs(
            plain_out_dir, stem, plain, f"{stem} — {len(plain)} discs", frozenset(),
            slice_2d, affine, native_shape, native_affine, heatmap_2d,
        )
    return warning, qc_row


def _write_outputs(
    out_dir: str,
    stem: str,
    labeled_discs: List[Tuple[int, float, float, float]],
    title: str,
    imputed: frozenset,
    slice_2d: np.ndarray,
    affine: np.ndarray,
    native_shape: Tuple[int, int, int],
    native_affine: np.ndarray,
    heatmap_2d: Optional[np.ndarray] = None,
) -> None:
    """Write ``<stem>_discs.png`` + ``<stem>_discs.nii.gz`` (native grid) into ``out_dir``."""
    _build_preview_png(
        slice_2d,
        labeled_discs,
        title=title,
        save_path=os.path.join(out_dir, f"{stem}_discs.png"),
        imputed_labels=imputed,
        heatmap_2d=heatmap_2d,
    )
    native_label_volume = _resample_discs_to_native(
        labeled_discs,
        preproc_affine=affine,
        native_shape=native_shape,
        native_affine=native_affine,
    )
    _write_disc_nifti(
        os.path.join(out_dir, f"{stem}_discs.nii.gz"), native_label_volume, native_affine
    )


def predict(
    cfg: dict,
    weights_path: str,
    input_path: str,
    out_dir: str,
    num_workers=None,
    skeleton_mode: str = "off",
    skeleton_path: Optional[str] = None,
    plain_out_dir: Optional[str] = None,
) -> None:
    """Sliding-window inference on every input volume.

    ``skeleton_mode`` (``off`` / ``flag`` / ``correct``) enables the skeleton
    QC / correction step (``skeleton_path`` defaults to ``skeleton.json``
    next to the weights). ``plain_out_dir`` additionally writes the
    predictions decoded WITHOUT the skeleton from the same heatmaps, so both
    can be evaluated with a single forward pass.

    Three-stage pipeline: ``num_workers`` CPU processes preprocess the next
    volumes and decode / write the previous ones while the main process runs
    the model on the current one.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = _discover_inputs(input_path)

    model = load_model(cfg["model_name"], in_channels=cfg["in_channels"]).to(device)
    load_inference_weights(model, weights_path, device)
    model.eval()

    patch_dhw = tuple(int(s) for s in cfg["patch_size"])
    overlap = float(cfg.get("inference_overlap", 0.5))
    peak_thr, min_peak_distance, pp_source = resolve_postprocessing(cfg, weights_path)
    batch_size = _resolve_batch_size(cfg.get("inference_batch_size"), device)
    orientation = str(cfg.get("orientation", "RPI"))
    resolution_mm = _parse_resolution(str(cfg.get("resolution", "1x1x1")))
    if num_workers is None:
        num_workers = cfg.get("inference_num_workers", "auto")
    num_workers = min(resolve_num_workers(num_workers), len(image_paths))
    print(
        f"[inference 3D] device={device} inputs={len(image_paths)} "
        f"orientation={orientation} resolution={resolution_mm} "
        f"patch_dhw={patch_dhw} overlap={overlap} batch_size={batch_size} "
        f"num_workers={num_workers}"
    )
    print(
        f"[inference 3D] peak_threshold={peak_thr} min_peak_distance={min_peak_distance} "
        f"(from {pp_source})"
    )
    skeleton, skeleton_kwargs, sk_source = resolve_skeleton(
        cfg, weights_path, skeleton_mode, skeleton_path, SKELETON_PROJECTION
    )
    if plain_out_dir is not None:
        if skeleton is None:
            raise ValueError("--plain-out-dir needs --skeleton flag|correct.")
        check_and_make_dir(plain_out_dir)
        print(f"[inference] predictions without skeleton also written to {plain_out_dir}")
    if skeleton is not None:
        print(f"[inference 3D] skeleton={skeleton_mode} {skeleton_kwargs} (from {sk_source})")

    check_and_make_dir(out_dir)
    # 3D volumes and heatmaps are large: keep fewer of them queued per worker.
    lookahead = max(2, num_workers + 1)
    jobs = [(p, orientation, resolution_mm) for p in image_paths]
    executor = make_executor(num_workers)
    qc_rows: List[dict] = []
    try:
        writer = BoundedSubmitter(executor, max_pending=lookahead)
        bar = tqdm.tqdm(
            imap_ordered(executor, _load_and_preprocess, jobs, lookahead),
            total=len(jobs),
            desc="inference 3D",
        )
        for item in bar:
            volume = item["volume"]
            heatmap = predict_volume_heatmap(
                volume, model, device, patch_dhw, overlap, batch_size
            )
            writer.submit(
                _finalize_one,
                item["path"],
                heatmap,
                # Preprocessed mid-sagittal slice — the preview background.
                np.ascontiguousarray(volume[volume.shape[0] // 2]),
                item["affine"],
                item["native_shape"],
                item["native_affine"],
                peak_thr,
                min_peak_distance,
                out_dir,
                skeleton,
                skeleton_kwargs,
                skeleton_mode,
                plain_out_dir,
            )
            del volume, heatmap, item
            for warning, qc_row in writer.pop_ready():
                if warning:
                    tqdm.tqdm.write(warning)
                if qc_row is not None:
                    qc_rows.append(qc_row)
        for warning, qc_row in writer.drain():
            if warning:
                tqdm.tqdm.write(warning)
            if qc_row is not None:
                qc_rows.append(qc_row)
    finally:
        if executor is not None:
            executor.shutdown(cancel_futures=True)
    if skeleton is not None:
        _write_qc_summary(out_dir, qc_rows)
    print(f"Wrote per-image NIfTI + PNG into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image-only 3D sliding-window inference with a trained 3D disc-landmark UNet."
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
    parser.add_argument(
        "--num-workers",
        default=None,
        help="CPU worker processes for preprocessing / decoding / writing "
        "(default: config inference_num_workers, else 'auto'; 0 = run inline).",
    )
    parser.add_argument(
        "--skeleton",
        choices=SKELETON_MODES,
        default=None,
        help="Skeleton QC step after labelling: 'off', 'flag' (report problematic "
        "predictions only) or 'correct' (also fix them). Default: config "
        "inference_skeleton, else 'off'.",
    )
    parser.add_argument(
        "--skeleton-path",
        default=None,
        help="skeleton.json from discslabel_skeleton (default: next to the weights).",
    )
    parser.add_argument(
        "--plain-out-dir",
        default=None,
        help="With --skeleton, also write the predictions decoded without the "
        "skeleton (same heatmaps) into this folder, for comparison.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    predict(
        cfg,
        weights_path=args.weights,
        input_path=args.input,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        skeleton_mode=args.skeleton or str(cfg.get("inference_skeleton", "off")),
        skeleton_path=args.skeleton_path,
        plain_out_dir=args.plain_out_dir,
    )


if __name__ == "__main__":
    main()
