"""Run inference with a trained disc-landmark UNet and compute metrics."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torchvision import transforms

from . import HEATMAP_CHANNELS, LANDMARK_CHANNEL_LABELS
from .dataset import DiscLandmarkDataset, _load_slice, _normalize
from .model import load_model
from .transforms import Rescale, ToTensor, landmarks_from_csv_row
from .train import _load_config
from .utils.io import check_and_make_dir
from .utils.metrics import decode_predictions, visualize_prediction


def _predict_one(
    model: torch.nn.Module,
    slice_arr: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> np.ndarray:
    """Run the model on one 2D slice and return the 5-channel heatmap at input resolution."""
    orig_h, orig_w = slice_arr.shape
    sample = {"image": _normalize(slice_arr), "landmarks": []}
    sample = Rescale(output_size=(cfg["image_height"], cfg["image_width"]))(sample)
    image, _ = ToTensor(sigma=cfg["heatmap_sigma"], radius=cfg["heatmap_radius"])(sample)
    tensor = torch.from_numpy(image[np.newaxis, :, :, :]).float().to(device)
    with torch.no_grad():
        heatmap = model(tensor).cpu().numpy()[0]
    # Resize each channel back to the original slice resolution.
    from skimage import transform as sk_transform

    rescaled = np.zeros((heatmap.shape[0], orig_h, orig_w), dtype=np.float32)
    for ch in range(heatmap.shape[0]):
        rescaled[ch] = sk_transform.resize(
            heatmap[ch], (orig_h, orig_w), mode="constant", preserve_range=True
        )
    return rescaled


def _collect_gt_anchors(landmarks: List[Tuple[int, float, float]]) -> Dict[str, Tuple[float, float]]:
    """Return {channel_name: (x, y)} for each of the four anchor discs if present."""
    gt: Dict[str, Tuple[float, float]] = {}
    for ch_idx, ch_name in enumerate(HEATMAP_CHANNELS):
        anchor_label = LANDMARK_CHANNEL_LABELS[ch_idx]
        if anchor_label is None:
            continue
        for label, x, y in landmarks:
            if label == anchor_label:
                gt[ch_name] = (x, y)
                break
    return gt


def predict(
    cfg: dict,
    weights_path: str,
    test_csv: str,
    out_csv: str,
    cuda_id: int = 0,
    save_image_dir: str | None = None,
) -> None:
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    model = load_model(cfg["model_name"], in_channels=cfg["in_channels"]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    frame = pd.read_csv(test_csv)

    # Output columns: path + predicted (x, y) for each anchor + aggregated disc peaks column
    columns = ["slice_path"]
    for name in HEATMAP_CHANNELS[1:]:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_peak"])
    columns.append("all_discs_peaks")
    rows = []

    # Metrics buffers
    anchor_errors: Dict[str, List[float]] = {name: [] for name in HEATMAP_CHANNELS[1:]}

    for idx in tqdm.tqdm(range(len(frame)), total=len(frame)):
        row = frame.iloc[idx]
        slice_path = str(row.iloc[0])
        slice_arr = _load_slice(slice_path)
        heatmaps = _predict_one(model, slice_arr, cfg, device)
        decoded = decode_predictions(heatmaps, peak_threshold=cfg.get("peak_threshold", 0.3))

        gt_landmarks = landmarks_from_csv_row(row)
        gt_anchors = _collect_gt_anchors(gt_landmarks)

        out_row = [slice_path]
        for name in HEATMAP_CHANNELS[1:]:
            pred = decoded[name]
            out_row.extend([pred["x"], pred["y"], pred["peak"]])
            if name in gt_anchors and pred["present"]:
                gx, gy = gt_anchors[name]
                err = float(np.sqrt((pred["x"] - gx) ** 2 + (pred["y"] - gy) ** 2))
                anchor_errors[name].append(err)
        all_peaks = decoded[HEATMAP_CHANNELS[0]]["peaks"]
        out_row.append(";".join(f"{x:.2f},{y:.2f},{p:.3f}" for (x, y, p) in all_peaks))
        rows.append(out_row)

        if save_image_dir is not None:
            check_and_make_dir(save_image_dir)
            pred_landmarks = []
            for ch_idx, name in enumerate(HEATMAP_CHANNELS[1:], start=1):
                pred = decoded[name]
                if pred["present"]:
                    pred_landmarks.append((LANDMARK_CHANNEL_LABELS[ch_idx], pred["x"], pred["y"]))
            for x, y, _p in all_peaks:
                pred_landmarks.append((1, x, y))
            viz_path = os.path.join(save_image_dir, os.path.basename(slice_path) + ".png")
            visualize_prediction(slice_arr, pred_landmarks, gt_landmarks, viz_path)

    pd.DataFrame(rows, columns=columns).to_csv(out_csv, index=False)
    print(f"Wrote predictions to {out_csv}")

    print("\nPer-anchor pixel errors (ground truth vs. prediction, mid-sagittal pixels):")
    for name, errs in anchor_errors.items():
        if errs:
            errs_arr = np.array(errs)
            print(
                f"  {name:>6s}: n={len(errs):3d}  mean={errs_arr.mean():.2f}px  "
                f"median={np.median(errs_arr):.2f}px  <2px rate={(errs_arr < 2).mean():.2%}"
            )
        else:
            print(f"  {name:>6s}: no matching ground-truth anchors")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a trained disc-landmark UNet.")
    parser.add_argument("--weights", required=True, help="Path to best_model.pt")
    parser.add_argument("--test-csv", required=True, help="Path to test.csv (from `discslabel_prepare`).")
    parser.add_argument("--out-csv", required=True, help="Where to write the predictions.csv.")
    parser.add_argument("--config", default=None, help="Optional JSON config overriding defaults.")
    parser.add_argument("--save-image-dir", default=None, help="If set, write overlay PNGs here.")
    parser.add_argument("--cuda-id", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    predict(
        cfg,
        weights_path=args.weights,
        test_csv=args.test_csv,
        out_csv=args.out_csv,
        cuda_id=args.cuda_id,
        save_image_dir=args.save_image_dir,
    )


if __name__ == "__main__":
    main()
