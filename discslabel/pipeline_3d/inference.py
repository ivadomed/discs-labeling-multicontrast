"""Image-only 3D sliding-window inference for the disc-landmark 3D UNet.

Full-3D counterpart to ``discslabel.pipeline_2d.inference``. Input is either
a single ``.nii`` / ``.nii.gz`` file or a flat folder of them. For each image:

    0. reorient to ``cfg["orientation"]`` and resample to
       ``cfg["resolution"]`` (same preprocessing as training inputs)
    1. z-score normalise the full 3D volume
    2. slide ``(patch_D, patch_H, patch_W)`` windows over the whole volume,
       traversing I-S superior → inferior along W
    3. forward the windows through the 3D model in batches → 5-channel 3D heatmap
    4. collect disc peaks from channel 0 and anchor peaks from channels 1..4
       in GLOBAL volume coordinates, with duplicates merged across windows
    5. pick the highest-priority detected anchor
       (``C2C3`` > ``L5S1`` > ``C7T1`` > ``T12L1``) and label every disc
       superior → inferior relative to that anchor
    6. project the labeled discs back onto the input image's native grid via
       pure affine arithmetic (points, no interpolation)
    7. write a 3D NIfTI with single-voxel discs plus a PNG preview showing
       the preprocessed mid-sagittal slice with each labeled disc overlaid.

No label or seg files are required. No augmentation is applied.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import tqdm

logging.getLogger("discslabel.utils.image").setLevel(logging.ERROR)

from .. import HEATMAP_CHANNELS, LANDMARK_CHANNEL_LABELS, NUM_CLASSES
from ..dataset import _z_score_normalize
from .model import load_model
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


ANCHOR_PRIORITY: Tuple[str, ...] = ("C2C3", "L5S1", "C7T1", "T12L1")

MIN_DISC_LABEL = 1
MAX_DISC_LABEL = 25


def _pad_volume_to_patch(
    volume: np.ndarray, patch_dhw: Tuple[int, int, int]
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Zero-pad ``volume`` to at least ``patch_dhw`` on each axis."""
    D, H, W = volume.shape
    pD, pH, pW = patch_dhw
    pad_d, pad_h, pad_w = max(0, pD - D), max(0, pH - H), max(0, pW - W)
    pad_d_before, pad_h_before, pad_w_before = pad_d // 2, pad_h // 2, pad_w // 2
    padded = np.pad(
        volume,
        (
            (pad_d_before, pad_d - pad_d_before),
            (pad_h_before, pad_h - pad_h_before),
            (pad_w_before, pad_w - pad_w_before),
        ),
        mode="constant",
    )
    return padded, (pad_d_before, pad_h_before, pad_w_before)


def _axis_origins(src_len: int, patch_len: int, stride: int) -> List[int]:
    if src_len <= patch_len:
        return [0]
    origins = list(range(0, src_len - patch_len + 1, stride))
    if origins[-1] + patch_len < src_len:
        origins.append(src_len - patch_len)
    return origins


def _sliding_window_origins(
    volume_shape: Tuple[int, int, int],
    patch_dhw: Tuple[int, int, int],
    overlap: float = 0.5,
) -> List[Tuple[int, int, int]]:
    """Generate ``(d0, h0, w0)`` origins covering ``volume_shape``.

    W is the I-S axis in RPI: W origins are sorted descending so the traversal
    walks superior → inferior (purely for human-readable logs; decoding is
    per-window and insensitive to the order).
    """
    D, H, W = volume_shape
    pD, pH, pW = patch_dhw
    stride_d = max(1, int(pD * (1 - overlap)))
    stride_h = max(1, int(pH * (1 - overlap)))
    stride_w = max(1, int(pW * (1 - overlap)))
    d_origins = _axis_origins(D, pD, stride_d)
    h_origins = _axis_origins(H, pH, stride_h)
    w_origins = sorted(_axis_origins(W, pW, stride_w), reverse=True)
    return [(d, h, w) for d in d_origins for h in h_origins for w in w_origins]


def _local_maxima_3d(
    heatmap: np.ndarray, min_distance: int, threshold: float
) -> List[Tuple[float, float, float, float]]:
    """Return [(d, h, w, intensity), ...] for 3D local maxima above ``threshold``."""
    try:
        from skimage.feature import peak_local_max
    except ImportError:  # pragma: no cover
        peak_local_max = None
    if peak_local_max is None:
        peak = float(heatmap.max())
        if peak < threshold:
            return []
        d, h, w = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        return [(float(d), float(h), float(w), peak)]
    coords = peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold,
        exclude_border=False,
    )
    return [
        (float(c[0]), float(c[1]), float(c[2]), float(heatmap[c[0], c[1], c[2]]))
        for c in coords
    ]


def _argmax_3d(heatmap: np.ndarray) -> Tuple[float, float, float, float]:
    """``(d, h, w, peak)`` of the max response of a 3D heatmap."""
    peak = float(heatmap.max())
    dd, hh, ww = np.where(heatmap == peak)
    return float(np.mean(dd)), float(np.mean(hh)), float(np.mean(ww)), peak


def _decode_predictions_3d(
    heatmaps: np.ndarray, peak_threshold: float
) -> Dict[str, dict]:
    """Decode a (NUM_CLASSES, D, H, W) heatmap stack into per-channel landmarks.

    * ``HEATMAP_CHANNELS[0]`` (``all_discs``) → {"peaks": [(d, h, w, intensity), ...]}
    * other channels → {"d","h","w","peak","present"} for the single argmax.
    """
    assert heatmaps.ndim == 4 and heatmaps.shape[0] == NUM_CLASSES, (
        f"Expected ({NUM_CLASSES}, D, H, W) heatmap stack, got {heatmaps.shape}"
    )

    result: Dict[str, dict] = {}
    peaks = _local_maxima_3d(heatmaps[0], min_distance=5, threshold=peak_threshold)
    result[HEATMAP_CHANNELS[0]] = {"peaks": peaks}

    for ch in range(1, NUM_CLASSES):
        d, h, w, peak = _argmax_3d(heatmaps[ch])
        result[HEATMAP_CHANNELS[ch]] = {
            "d": d,
            "h": h,
            "w": w,
            "peak": peak,
            "present": peak >= peak_threshold,
        }
    return result


def _merge_peaks_3d(
    peaks: List[Tuple[float, float, float, float]], min_distance: float
) -> List[Tuple[float, float, float, float]]:
    """Greedy 3D NMS: strongest peak kept, drop anything within ``min_distance``."""
    kept: List[Tuple[float, float, float, float]] = []
    for pd, ph, pw, pp in sorted(peaks, key=lambda p: -p[3]):
        if any(
            (pd - kd) ** 2 + (ph - kh) ** 2 + (pw - kw) ** 2 < min_distance ** 2
            for (kd, kh, kw, _) in kept
        ):
            continue
        kept.append((pd, ph, pw, pp))
    return kept


def _choose_anchor(
    anchors_by_name: Dict[str, List[Tuple[float, float, float, float]]],
) -> Optional[Tuple[str, int, float, float, float]]:
    """Highest-priority anchor; tie-break by peak intensity.

    Returns ``(channel_name, disc_label, d, h, w)`` or ``None``.
    """
    for name in ANCHOR_PRIORITY:
        hits = anchors_by_name.get(name) or []
        if not hits:
            continue
        best = max(hits, key=lambda p: p[3])
        ch_idx = HEATMAP_CHANNELS.index(name)
        anchor_label = LANDMARK_CHANNEL_LABELS[ch_idx]
        return (name, int(anchor_label), best[0], best[1], best[2])
    return None


def _assign_labels(
    discs: List[Tuple[float, float, float, float]],
    anchor_label: int,
    anchor_d: float,
    anchor_h: float,
    anchor_w: float,
) -> List[Tuple[int, float, float, float]]:
    """Label disc peaks superior → inferior from a known anchor.

    RPI convention: W (axis 2) indexes I-S with higher values = superior. Sort
    by W descending so index 0 is the most superior disc; the disc closest to
    the anchor in 3D gets the anchor's label, and labels above/below move by
    ±1 per step. Values outside ``[1, 25]`` are dropped.
    """
    if not discs:
        return []
    sorted_discs = sorted(discs, key=lambda d: (-d[2], d[1], d[0]))
    anchor_idx = min(
        range(len(sorted_discs)),
        key=lambda i: (sorted_discs[i][0] - anchor_d) ** 2
        + (sorted_discs[i][1] - anchor_h) ** 2
        + (sorted_discs[i][2] - anchor_w) ** 2,
    )
    out: List[Tuple[int, float, float, float]] = []
    for i, (d, h, w, _p) in enumerate(sorted_discs):
        label = anchor_label + (i - anchor_idx)
        if MIN_DISC_LABEL <= label <= MAX_DISC_LABEL:
            out.append((label, d, h, w))
    return out


def _build_preview_png(
    volume_norm: np.ndarray,
    labeled_discs: List[Tuple[int, float, float, float]],
    title: str,
    save_path: str,
) -> None:
    """Mid-sagittal PNG with every labeled disc overlaid.

    The 3D volume is sliced at D//2 for the background; markers are placed at
    each disc's (h, w) regardless of its own ``d`` so all detected discs are
    visible even when they sit off the midline (common in scoliosis).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    D = volume_norm.shape[0]
    slice_2d = volume_norm[D // 2]
    img = _display_slice(slice_2d)
    _H_orig, W_orig = slice_2d.shape

    vmin, vmax = _intensity_window(img)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    for label, _d, h, w in labeled_discs:
        disp_row = (W_orig - 1) - w
        disp_col = h
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
    labeled_discs: List[Tuple[int, float, float, float]],
    preproc_affine: np.ndarray,
    original_nib: nib.Nifti1Image,
) -> np.ndarray:
    """Project disc labels from the preprocessed grid back to the native grid.

    Each disc is a POINT in physical space, so projection is pure affine
    arithmetic — no interpolation needed. For every disc at preprocessed voxel
    ``(d, h, w)``::

        native_voxel = inv(native_affine) @ preproc_affine @ (d, h, w, 1)

    plant the label at the nearest native voxel. One matrix solve per batch,
    one rounding, no per-label resample.
    """
    dest_shape = tuple(int(s) for s in original_nib.shape[:3])
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
    native_affine_inv = np.linalg.inv(np.asarray(original_nib.affine))
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


def _forward_windows(
    padded_volume: np.ndarray,
    origins: List[Tuple[int, int, int]],
    patch_dhw: Tuple[int, int, int],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Model-forward every window of one image, chunked by ``batch_size``.

    Returns ``(N, NUM_CLASSES, pD, pH, pW)`` heatmaps on CPU float32 in the
    same order as ``origins``.
    """
    pD, pH, pW = patch_dhw
    patches = np.stack(
        [padded_volume[d0 : d0 + pD, h0 : h0 + pH, w0 : w0 + pW] for (d0, h0, w0) in origins]
    )  # (N, pD, pH, pW)
    patches_t = torch.from_numpy(patches).float().unsqueeze(1).to(device)  # (N, 1, pD, pH, pW)

    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, patches_t.size(0), batch_size):
            chunk = patches_t[i : i + batch_size]
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                hm = model(chunk)
            chunks.append(hm.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


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


def _infer_one_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    orientation: str,
    resolution_mm: Tuple[float, float, float],
    patch_dhw: Tuple[int, int, int],
    overlap: float,
    peak_threshold: float,
    min_peak_distance: float,
    batch_size: int,
    out_dir: str,
) -> None:
    """Run the full 3D sliding-window pipeline on one image."""
    image_raw, affine, original_nib = _preprocess_image(
        image_path, orientation, resolution_mm
    )
    image_norm = _z_score_normalize(image_raw)

    padded_volume, (pad_d_before, pad_h_before, pad_w_before) = _pad_volume_to_patch(
        image_norm, patch_dhw
    )
    origins = _sliding_window_origins(padded_volume.shape, patch_dhw, overlap=overlap)
    heatmaps = _forward_windows(
        padded_volume, origins, patch_dhw, model, device, batch_size
    )

    all_discs: List[Tuple[float, float, float, float]] = []
    anchors_by_name: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for i, (d0, h0, w0) in enumerate(origins):
        decoded = _decode_predictions_3d(heatmaps[i], peak_threshold=peak_threshold)
        for dp, hp, wp, pp in decoded[HEATMAP_CHANNELS[0]]["peaks"]:
            gd = d0 + dp - pad_d_before
            gh = h0 + hp - pad_h_before
            gw = w0 + wp - pad_w_before
            all_discs.append((gd, gh, gw, pp))
        for ch_name in HEATMAP_CHANNELS[1:]:
            pred = decoded[ch_name]
            if pred["present"]:
                gd = d0 + pred["d"] - pad_d_before
                gh = h0 + pred["h"] - pad_h_before
                gw = w0 + pred["w"] - pad_w_before
                anchors_by_name.setdefault(ch_name, []).append(
                    (gd, gh, gw, pred["peak"])
                )

    merged_discs = _merge_peaks_3d(all_discs, min_distance=min_peak_distance)
    chosen = _choose_anchor(anchors_by_name)

    if chosen is None:
        tqdm.tqdm.write(
            f"[warn] no anchor detected for {os.path.basename(image_path)} "
            f"(found {len(merged_discs)} disc peaks); writing empty NIfTI."
        )
        labeled_discs: List[Tuple[int, float, float, float]] = []
    else:
        _ch_name, anchor_label, ad, ah, aw = chosen
        labeled_discs = _assign_labels(merged_discs, anchor_label, ad, ah, aw)

    stem = _stem(image_path)
    nifti_out = os.path.join(out_dir, f"{stem}_discs.nii.gz")
    png_out = os.path.join(out_dir, f"{stem}_discs.png")

    _build_preview_png(
        image_norm,
        labeled_discs,
        title=f"{stem} — {len(labeled_discs)} discs",
        save_path=png_out,
    )

    native_label_volume = _resample_discs_to_native(
        labeled_discs,
        preproc_affine=affine,
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
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    patch_dhw = tuple(int(s) for s in cfg["patch_size"])
    overlap = float(cfg.get("inference_overlap", 0.5))
    peak_thr = float(cfg.get("peak_threshold", 0.3))
    min_peak_distance = float(cfg.get("inference_min_peak_distance", 10.0))
    batch_size = _resolve_batch_size(cfg.get("inference_batch_size"), device)
    orientation = str(cfg.get("orientation", "RPI"))
    resolution_mm = _parse_resolution(str(cfg.get("resolution", "1x1x1")))
    print(
        f"[inference 3D] device={device} inputs={len(image_paths)} "
        f"orientation={orientation} resolution={resolution_mm} "
        f"patch_dhw={patch_dhw} overlap={overlap} batch_size={batch_size}"
    )

    check_and_make_dir(out_dir)
    for image_path in tqdm.tqdm(image_paths, desc="inference 3D"):
        _infer_one_image(
            image_path,
            model=model,
            device=device,
            orientation=orientation,
            resolution_mm=resolution_mm,
            patch_dhw=patch_dhw,
            overlap=overlap,
            peak_threshold=peak_thr,
            min_peak_distance=min_peak_distance,
            batch_size=batch_size,
            out_dir=out_dir,
        )
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
