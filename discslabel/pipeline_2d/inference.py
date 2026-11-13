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
       volume) and, with ``--save-preview``, a PNG preview of the
       preprocessed slice rendered with the same ``p1..p99`` contrast window
       used by the training previews.

Everything comes from ``--model-dir`` (the training output folder; default:
the pretrained weights, downloaded once from the GitHub release into
``data/weights``, see ``discslabel.weights``):
``config.json``, ``best_model.pt`` (or ``--checkpoint``), and optionally
``skeleton.json`` / ``postprocessing.json``.

``peak_threshold`` / ``min_peak_distance`` come from ``postprocessing.json``
next to the weights when ``discslabel_tune`` has been run, else from the
config.

Skeleton step (``--skeleton``, default ``correct`` when ``skeleton.json``
from ``discslabel_skeleton`` is in the model folder, else ``off``): after the anchor-based labelling, the inter-disc
distances on the mid-sagittal plane are checked against the training
distribution (``discslabel.skeleton``). ``flag`` only reports problems;
``correct`` also removes off-chain peaks, fixes the labels and fills missed
discs; with ``skeleton_soft_floor`` it can also keep a weak peak (below
``peak_threshold``) where the skeleton expects a disc. Each image then gets a ``<name>_discs_qc.json`` report and the run a
``qc_summary.csv``.

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
from discslabel.decoding import _choose_anchor, extract_candidates, label_discs, soft_settings
from discslabel.parallel import BoundedSubmitter, imap_ordered, make_executor, resolve_num_workers
from discslabel.postprocessing import (
    CHECKPOINT_FILE,
    resolve_model_dir,
    resolve_postprocessing,
    resolve_skeleton,
)
from discslabel.skeleton import (
    SKELETON_MODES,
    affine_spacing,
    decode_with_skeleton,
    qc_row as _qc_row,
    write_qc_summary as _write_qc_summary,
)
from discslabel.training import load_inference_weights
from discslabel.pipeline_2d.model import load_model
from discslabel.pipeline_2d.predictor import INPUT_MODES, SKELETON_PROJECTION, predict_volume_heatmap, slab_num_slices
from discslabel.pipeline_2d.train import (
    _display_slice,
    _intensity_window,
    _resolve_batch_size,
)
from discslabel.utils.image import Image, resample_nib
from discslabel.utils.io import check_and_make_dir
from discslabel.weights import WEIGHTS_URL_ENV, ensure_weights


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
    imputed_labels: frozenset = frozenset(),
    heatmap_2d: Optional[np.ndarray] = None,
) -> None:
    """Mid-sagittal PNG with disc labels overlaid (skeleton-imputed discs in yellow).

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
    n_cols = 1 if heatmap_2d is None else 2
    # Column width follows the displayed image aspect so panels sit side by side.
    col_w = min(8.0, max(4.0, 10.0 * img.shape[1] / img.shape[0] + 1.0))
    fig, axes = plt.subplots(1, n_cols, figsize=(col_w * n_cols, 10), squeeze=False)
    ax = axes[0, 0]
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    for label, x, y in labeled_discs:
        # slice (y=row=P-A, x=col=I-S) → display (row = W-1-x, col = y)
        disp_row = (W_orig - 1) - x
        disp_col = y
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
    labeled_discs: List[Tuple[int, float, float]],
    preproc_affine: np.ndarray,
    d_mid: int,
    native_shape: Tuple[int, int, int],
    native_affine: np.ndarray,
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
    dest_shape = tuple(int(s) for s in native_shape[:3])
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
    native_affine_inv = np.linalg.inv(np.asarray(native_affine))
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
    d_mid: int,
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
    soft: Optional[dict] = None,
    save_preview: bool = False,
) -> Tuple[Optional[str], Optional[dict]]:
    """Worker job: decode the stitched heatmap and write the NIfTI (+ PNG with ``save_preview``).

    With a skeleton and ``plain_out_dir``, the decoding without the skeleton
    is also written there (same heatmap, no second forward pass).

    Returns ``(warning, qc_row)``: a message (or ``None``) the main process
    prints above its progress bar, and the QC summary row (``None`` when the
    skeleton step is off).
    """
    candidates = extract_candidates(heatmap, peak_threshold, soft)
    warning = None
    if _choose_anchor(candidates["anchors"], peak_threshold) is None:
        warning = (
            f"[warn] no anchor detected for {os.path.basename(image_path)} "
            f"(found {len(candidates['discs'])} disc candidates); writing empty NIfTI."
        )
    stem = _stem(image_path)
    qc_row = None
    imputed = frozenset()
    if skeleton is None:
        decoded = label_discs(candidates, peak_threshold, min_peak_distance)
    else:
        decoded, report = decode_with_skeleton(
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
        report = {"image": stem, "n_discs": len(decoded), **report}
        with open(os.path.join(out_dir, f"{stem}_discs_qc.json"), "w") as f:
            json.dump(report, f, indent=2)
        qc_row = _qc_row(report)
        if report["status"] != "ok" and warning is None:
            warning = f"[qc] {stem}: {', '.join(report['flags'])}"
    title = f"{stem} — {len(decoded)} discs"
    if qc_row is not None and qc_row["status"] != "ok":
        title += f"\nQC: {qc_row['flags']}"
    _write_outputs(
        out_dir, stem, decoded, title, imputed,
        slice_2d, d_mid, affine, native_shape, native_affine, heatmap,
        save_preview,
    )
    if plain_out_dir is not None and skeleton is not None:
        # Same heatmap decoded without the skeleton, for side-by-side evaluation.
        plain = label_discs(candidates, peak_threshold, min_peak_distance)
        _write_outputs(
            plain_out_dir, stem, plain, f"{stem} — {len(plain)} discs", frozenset(),
            slice_2d, d_mid, affine, native_shape, native_affine, heatmap,
            save_preview,
        )
    return warning, qc_row


def _write_outputs(
    out_dir: str,
    stem: str,
    decoded: List[Tuple[int, float, float]],
    title: str,
    imputed: frozenset,
    slice_2d: np.ndarray,
    d_mid: int,
    affine: np.ndarray,
    native_shape: Tuple[int, int, int],
    native_affine: np.ndarray,
    heatmap_2d: Optional[np.ndarray] = None,
    save_preview: bool = False,
) -> None:
    """Write ``<stem>_discs.nii.gz`` (native grid) into ``out_dir``, plus
    ``<stem>_discs.png`` when ``save_preview`` is set.

    ``decoded`` is ``[(label, h, w), ...]`` as returned by the decoding.
    """
    # The preview / native projection use the slice convention
    # (label, x = I-S = w, y = P-A = h).
    labeled_discs = [(label, w, h) for label, h, w in decoded]
    # Preview first — it shows what the model actually saw, so it is kept
    # in the preprocessed frame regardless of what the output NIfTI is in.
    if save_preview:
        _build_preview_png(
            slice_2d,
            labeled_discs,
            title=title,
            save_path=os.path.join(out_dir, f"{stem}_discs.png"),
            imputed_labels=imputed,
            heatmap_2d=heatmap_2d,
        )
    # Project the preprocessed-frame labels back onto the user's native grid
    # so the output NIfTI overlays cleanly on their input image.
    native_label_volume = _resample_discs_to_native(
        labeled_discs,
        preproc_affine=affine,
        d_mid=d_mid,
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
    input_mode: Optional[str] = None,
    save_preview: bool = False,
) -> None:
    """Sliding-window inference on every input image.

    ``input_mode`` picks the 2D model input: ``"avg"`` (mean of the
    ``slab_thickness_mm`` R-L slab around the midline) or ``"mid"`` (the
    mid-sagittal slice). Default: config ``inference_input_mode``, else ``"avg"``.

    ``skeleton_mode`` (``off`` / ``flag`` / ``correct``) enables the skeleton
    QC / correction step (``skeleton_path`` defaults to ``skeleton.json``
    next to the weights). ``plain_out_dir`` additionally writes the
    predictions decoded WITHOUT the skeleton from the same heatmaps, so both
    can be evaluated with a single forward pass. ``save_preview`` also writes
    a ``<name>_discs.png`` preview per image.

    Three-stage pipeline: ``num_workers`` CPU processes preprocess the next
    images and decode / write the previous ones while the main process runs
    the model on the current one.
    """
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
    if input_mode is None:
        input_mode = str(cfg.get("inference_input_mode", "avg"))
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input_mode must be one of {INPUT_MODES}, got {input_mode!r}")
    n_slab = slab_num_slices(cfg)
    if num_workers is None:
        num_workers = cfg.get("inference_num_workers", "auto")
    num_workers = min(resolve_num_workers(num_workers), len(image_paths))
    print(
        f"[inference] device={device} inputs={len(image_paths)} "
        f"orientation={orientation} resolution={resolution_mm} "
        f"patch_hw={patch_hw} overlap={overlap} batch_size={batch_size} "
        f"input_mode={input_mode} slab_slices={n_slab} num_workers={num_workers}"
    )
    print(
        f"[inference] peak_threshold={peak_thr} min_peak_distance={min_peak_distance} "
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
    # Soft matching (skeleton "correct" only): also extract the weak disc candidates.
    soft = None
    if skeleton is not None and skeleton_mode == "correct":
        soft = soft_settings(cfg, skeleton_kwargs["soft_floor"])
    if skeleton is not None:
        print(f"[inference] skeleton={skeleton_mode} {skeleton_kwargs} (from {sk_source})")

    check_and_make_dir(out_dir)
    lookahead = max(2, 2 * num_workers)
    jobs = [(p, orientation, resolution_mm) for p in image_paths]
    executor = make_executor(num_workers)
    qc_rows: List[dict] = []
    try:
        writer = BoundedSubmitter(executor, max_pending=lookahead)
        bar = tqdm.tqdm(
            imap_ordered(executor, _load_and_preprocess, jobs, lookahead),
            total=len(jobs),
            desc="inference",
        )
        for item in bar:
            volume = item["volume"]
            d_mid = volume.shape[0] // 2
            heatmap = predict_volume_heatmap(
                volume, model, device, patch_hw, overlap, batch_size, input_mode, n_slab
            )
            writer.submit(
                _finalize_one,
                item["path"],
                heatmap,
                # Mid-sagittal slice — the (sharper) preview background.
                np.ascontiguousarray(volume[d_mid]),
                d_mid,
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
                soft,
                save_preview,
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
    print(f"Wrote per-image NIfTI{' + PNG' if save_preview else ''} into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image-only sliding-window inference with a trained disc-landmark UNet."
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Trained model folder with config.json, the checkpoint and (optionally) "
        "skeleton.json / postprocessing.json. Default: the pretrained weights, "
        "downloaded from the GitHub release on first use and cached in data/weights.",
    )
    parser.add_argument(
        "--weights-url",
        default=None,
        help=f"Zip to download when --model-dir is not given (default: ${WEIGHTS_URL_ENV}, "
        "else the release URL).",
    )
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_FILE,
        help=f"Checkpoint file name inside --model-dir (default: {CHECKPOINT_FILE}).",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a .nii/.nii.gz file or a flat folder of them.",
    )
    parser.add_argument(
        "--out_dir", "-o",
        required=True,
        help="Directory for the per-image <name>_discs.nii.gz outputs.",
    )
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Also write a <name>_discs.png preview (mid-sagittal slice + labels) per image.",
    )
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
        "predictions only) or 'correct' (also fix them). Default: 'correct' when "
        "skeleton.json is in --model-dir, else 'off'.",
    )
    parser.add_argument(
        "--plain-out-dir",
        default=None,
        help="With --skeleton, also write the predictions decoded without the "
        "skeleton (same heatmaps) into this folder, for comparison.",
    )
    parser.add_argument(
        "--input-mode",
        choices=INPUT_MODES,
        default=None,
        help="2D model input: 'avg' (mean over slab_thickness_mm around the "
        "mid-sagittal slice) or 'mid' (mid-sagittal slice only). Default: config "
        "inference_input_mode, else 'avg'.",
    )
    args = parser.parse_args()

    model_dir = args.model_dir or ensure_weights(args.weights_url)
    cfg, weights_path, has_skeleton = resolve_model_dir(model_dir, args.checkpoint)
    skeleton_mode = args.skeleton
    if skeleton_mode is None:
        skeleton_mode = "correct" if has_skeleton else "off"
        if not has_skeleton:
            print(f"[inference] no skeleton.json in {model_dir}; skeleton correction disabled.")
    predict(
        cfg,
        weights_path=weights_path,
        input_path=args.input,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        skeleton_mode=skeleton_mode,
        plain_out_dir=args.plain_out_dir,
        input_mode=args.input_mode,
        save_preview=args.save_preview,
    )


if __name__ == "__main__":
    main()
