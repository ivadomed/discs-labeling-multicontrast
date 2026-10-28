"""Image-only sliding-window inference for the disc-landmark UNet.

For each image we:

    1. z-score normalise the full 3D volume
    2. extract the mid-sagittal slice (axis R-L, center)
    3. slide ``(patch_H, patch_W)`` windows over the 2D slice, from superior
       to inferior along the I-S axis
    4. forward each window through the model → 5-channel heatmap
    5. collect disc peaks from channel 0 and anchor peaks from channels 1..4
       in GLOBAL slice coordinates, with duplicates merged across windows
    6. pick the highest-priority detected anchor
       (``C2C3`` > ``L5S1`` > ``C7T1`` > ``T12L1``) and label every disc
       in superior → inferior order relative to that anchor
    7. write a 3D NIfTI with single-voxel discs on the mid-sagittal plane
       (shape and affine match the input) plus a PNG preview rendered with
       the same ``p1..p99`` contrast window used by the training previews.

No label or seg files are required. No augmentation is applied.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import tqdm

from . import HEATMAP_CHANNELS, LANDMARK_CHANNEL_LABELS
from .dataset import _load_nifti, _z_score_normalize
from .model import load_model
from .train import _display_slice, _intensity_window, _load_config
from .utils.io import check_and_make_dir
from .utils.metrics import decode_predictions


# User-requested priority: C2C3 (3) > L5S1 (25) > C7T1 (8) > T12L1 (20).
ANCHOR_PRIORITY: Tuple[str, ...] = ("C2C3", "L5S1", "C7T1", "T12L1")

# Highest / lowest valid disc label — labels outside this range are dropped
# when propagating from the chosen anchor.
MIN_DISC_LABEL = 1
MAX_DISC_LABEL = 25


def _pad_slice_to_patch(
    slice_2d: np.ndarray, patch_hw: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Zero-pad ``slice_2d`` to at least ``patch_hw`` on each axis.

    Returns ``(padded, (pad_h_before, pad_w_before))`` so callers can translate
    coordinates back to the original slice frame.
    """
    H, W = slice_2d.shape
    pH, pW = patch_hw
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)
    pad_h_before = pad_h // 2
    pad_w_before = pad_w // 2
    padded = np.pad(
        slice_2d,
        ((pad_h_before, pad_h - pad_h_before), (pad_w_before, pad_w - pad_w_before)),
        mode="constant",
    )
    return padded, (pad_h_before, pad_w_before)


def _axis_origins(src_len: int, patch_len: int, stride: int) -> List[int]:
    """Window origins along one axis, with the last one pinned to the edge."""
    if src_len <= patch_len:
        return [0]
    origins = list(range(0, src_len - patch_len + 1, stride))
    if origins[-1] + patch_len < src_len:
        origins.append(src_len - patch_len)
    return origins


def _sliding_window_origins(
    slice_shape: Tuple[int, int],
    patch_hw: Tuple[int, int],
    overlap: float = 0.5,
) -> List[Tuple[int, int]]:
    """Generate ``(h0, w0)`` origins covering ``slice_shape``.

    W is the I-S axis in RPI: we sort W origins descending so the traversal
    goes top (superior, high W) → bottom (inferior, low W), matching the
    way the labeling story was described. Decoding is independent per window
    so the order has no effect on the final labels — it is kept for the
    tqdm log and for anyone reading the output.
    """
    H, W = slice_shape
    pH, pW = patch_hw
    stride_h = max(1, int(pH * (1 - overlap)))
    stride_w = max(1, int(pW * (1 - overlap)))
    h_origins = _axis_origins(H, pH, stride_h)
    w_origins = sorted(_axis_origins(W, pW, stride_w), reverse=True)
    return [(h, w) for h in h_origins for w in w_origins]


def _merge_peaks(
    peaks: List[Tuple[float, float, float]], min_distance: float
) -> List[Tuple[float, float, float]]:
    """Greedy NMS: strongest peak kept, drop anything within ``min_distance``."""
    kept: List[Tuple[float, float, float]] = []
    for px, py, pp in sorted(peaks, key=lambda p: -p[2]):
        if any(
            (px - kx) ** 2 + (py - ky) ** 2 < min_distance ** 2 for (kx, ky, _) in kept
        ):
            continue
        kept.append((px, py, pp))
    return kept


def _choose_anchor(
    anchors_by_name: Dict[str, List[Tuple[float, float, float]]],
) -> Optional[Tuple[str, int, float, float]]:
    """Pick the highest-priority anchor; break ties by peak intensity.

    Returns ``(channel_name, disc_label, x, y)`` or ``None`` when no anchor
    was detected in any window.
    """
    for name in ANCHOR_PRIORITY:
        hits = anchors_by_name.get(name) or []
        if not hits:
            continue
        best = max(hits, key=lambda p: p[2])
        ch_idx = HEATMAP_CHANNELS.index(name)
        anchor_label = LANDMARK_CHANNEL_LABELS[ch_idx]
        return (name, int(anchor_label), best[0], best[1])
    return None


def _assign_labels(
    discs: List[Tuple[float, float, float]],
    anchor_label: int,
    anchor_x: float,
    anchor_y: float,
) -> List[Tuple[int, float, float]]:
    """Label disc peaks superior → inferior from a known anchor.

    Slice coord convention: ``x`` indexes the I-S axis (higher = superior),
    ``y`` indexes the P-A axis. We sort by ``x`` descending so index 0 is
    the most superior disc; the disc closest to the anchor gets the anchor's
    label, and labels above/below move by ±1 per step. Values outside
    ``[1, 25]`` are dropped.
    """
    if not discs:
        return []
    sorted_discs = sorted(discs, key=lambda d: (-d[0], d[1]))
    anchor_idx = min(
        range(len(sorted_discs)),
        key=lambda i: (sorted_discs[i][0] - anchor_x) ** 2
        + (sorted_discs[i][1] - anchor_y) ** 2,
    )
    out: List[Tuple[int, float, float]] = []
    for i, (x, y, _p) in enumerate(sorted_discs):
        label = anchor_label + (i - anchor_idx)
        if MIN_DISC_LABEL <= label <= MAX_DISC_LABEL:
            out.append((label, x, y))
    return out


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
            color="yellow",
            markersize=5,
            markeredgecolor="black",
            markeredgewidth=0.6,
        )
        ax.text(
            disp_col + 4,
            disp_row,
            str(int(label)),
            color="yellow",
            fontsize=8,
            va="center",
        )
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _write_disc_nifti(
    out_path: str,
    labeled_discs: List[Tuple[int, float, float]],
    reference_nifti_path: str,
    d_mid: int,
) -> None:
    """Write a 3D NIfTI with one voxel per labeled disc on the mid-sag plane.

    Shape and affine are copied from the input image, so the output stays
    aligned with it in physical space.
    """
    ref = nib.load(reference_nifti_path)
    ref_shape = tuple(int(s) for s in ref.shape[:3])
    out = np.zeros(ref_shape, dtype=np.int16)
    D, H, W = ref_shape
    for label, x, y in labeled_discs:
        h = int(round(y))
        w = int(round(x))
        if 0 <= d_mid < D and 0 <= h < H and 0 <= w < W:
            out[d_mid, h, w] = int(label)
    # Fresh header so the output dtype matches the int16 we write, instead of
    # inheriting the (likely float32) dtype from the input image header.
    img_out = nib.Nifti1Image(out, affine=ref.affine)
    img_out.set_data_dtype(np.int16)
    nib.save(img_out, out_path)


def _stem(path: str) -> str:
    """Basename without the ``.nii`` / ``.nii.gz`` extension."""
    base = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            return base[: -len(ext)]
    return os.path.splitext(base)[0]


def _infer_one_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    patch_hw: Tuple[int, int],
    overlap: float,
    peak_threshold: float,
    min_peak_distance: float,
    out_dir: str,
) -> None:
    """Run the full sliding-window pipeline on one 3D image."""
    image_raw = _load_nifti(image_path)
    image_norm = _z_score_normalize(image_raw)

    D, _H, _W = image_raw.shape
    d_mid = D // 2
    slice_2d = image_norm[d_mid]  # (H, W)

    padded_slice, (pad_h_before, pad_w_before) = _pad_slice_to_patch(slice_2d, patch_hw)
    pH, pW = patch_hw
    origins = _sliding_window_origins(padded_slice.shape, patch_hw, overlap=overlap)

    all_discs: List[Tuple[float, float, float]] = []
    anchors_by_name: Dict[str, List[Tuple[float, float, float]]] = {}

    with torch.no_grad():
        for h0, w0 in origins:
            patch = padded_slice[h0 : h0 + pH, w0 : w0 + pW]
            patch_t = (
                torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
            )
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                heatmaps = model(patch_t)
            heatmaps_np = heatmaps[0].float().cpu().numpy()
            decoded = decode_predictions(heatmaps_np, peak_threshold=peak_threshold)

            # Translate window coords → padded-slice coords → original-slice coords.
            for xp, yp, pp in decoded[HEATMAP_CHANNELS[0]]["peaks"]:
                gx = w0 + xp - pad_w_before
                gy = h0 + yp - pad_h_before
                all_discs.append((gx, gy, pp))
            for ch_name in HEATMAP_CHANNELS[1:]:
                pred = decoded[ch_name]
                if pred["present"]:
                    gx = w0 + pred["x"] - pad_w_before
                    gy = h0 + pred["y"] - pad_h_before
                    anchors_by_name.setdefault(ch_name, []).append(
                        (gx, gy, pred["peak"])
                    )

    merged_discs = _merge_peaks(all_discs, min_distance=min_peak_distance)
    chosen = _choose_anchor(anchors_by_name)

    if chosen is None:
        print(
            f"[warn] no anchor detected for {os.path.basename(image_path)} "
            f"(found {len(merged_discs)} disc peaks); writing empty NIfTI."
        )
        labeled_discs: List[Tuple[int, float, float]] = []
    else:
        ch_name, anchor_label, ax, ay = chosen
        labeled_discs = _assign_labels(merged_discs, anchor_label, ax, ay)
        print(
            f"  {os.path.basename(image_path)}: anchor={ch_name}({anchor_label}) "
            f"@ ({ax:.1f}, {ay:.1f}); {len(labeled_discs)}/{len(merged_discs)} discs labeled"
        )

    stem = _stem(image_path)
    nifti_out = os.path.join(out_dir, f"{stem}_discs.nii.gz")
    png_out = os.path.join(out_dir, f"{stem}_discs.png")
    _write_disc_nifti(nifti_out, labeled_discs, image_path, d_mid)
    _build_preview_png(
        slice_2d,
        labeled_discs,
        title=f"{stem} — {len(labeled_discs)} discs",
        save_path=png_out,
    )


def predict(
    cfg: dict,
    weights_path: str,
    manifest_path: str,
    split: str,
    out_dir: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    if split not in manifest:
        raise KeyError(f"Split '{split}' not in manifest {manifest_path}")
    entries = list(manifest[split])

    model = load_model(cfg["model_name"], in_channels=cfg["in_channels"]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    patch_size = tuple(int(s) for s in cfg["patch_size"])
    patch_hw = (patch_size[1], patch_size[2])
    overlap = float(cfg.get("inference_overlap", 0.5))
    peak_thr = float(cfg.get("peak_threshold", 0.3))
    min_peak_distance = float(cfg.get("inference_min_peak_distance", 10.0))

    check_and_make_dir(out_dir)
    for entry in tqdm.tqdm(entries, desc=f"inference/{split}"):
        _infer_one_image(
            entry["image"],
            model=model,
            device=device,
            patch_hw=patch_hw,
            overlap=overlap,
            peak_threshold=peak_thr,
            min_peak_distance=min_peak_distance,
            out_dir=out_dir,
        )
    print(f"Wrote per-image NIfTI + PNG into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Image-only sliding-window inference with a trained disc-landmark UNet."
    )
    parser.add_argument("--weights", required=True, help="Path to best_model.pt.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json.")
    parser.add_argument(
        "--split", default="test", help="Split name inside the manifest (default: test)."
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
        manifest_path=args.manifest,
        split=args.split,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
