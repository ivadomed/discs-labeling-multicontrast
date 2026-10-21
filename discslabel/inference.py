"""Run inference with a trained disc-landmark UNet.

Inference re-uses the same 3D-load → GPU mid-sagittal extraction path as
training, but turns augmentation off and decodes landmarks from the 2D
heatmap predictions.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader

from . import HEATMAP_CHANNELS, LANDMARK_CHANNEL_LABELS
from .dataset import DiscVolumeDataset
from .gpu_pipeline import GpuBatchProcessor
from .model import load_model
from .train import _load_config
from .utils.io import check_and_make_dir
from .utils.metrics import decode_predictions, visualize_prediction


def _anchors_from_label_2d(label_2d: torch.Tensor) -> Dict[str, Tuple[float, float]]:
    """Return ``{channel_name: (x, y)}`` for each anchor present in a (H, W) label slice."""
    anchors: Dict[str, Tuple[float, float]] = {}
    arr = label_2d.detach().cpu().numpy()
    for ch_idx, ch_name in enumerate(HEATMAP_CHANNELS):
        anchor_label = LANDMARK_CHANNEL_LABELS[ch_idx]
        if anchor_label is None:
            continue
        ys, xs = np.where(np.rint(arr).astype(np.int32) == int(anchor_label))
        if len(xs) == 0:
            continue
        anchors[ch_name] = (float(xs.mean()), float(ys.mean()))
    return anchors


def predict(
    cfg: dict,
    weights_path: str,
    manifest_path: str,
    split: str,
    out_csv: str,
    save_image_dir: str | None = None,
) -> None:
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    dataset = DiscVolumeDataset(
        manifest_path,
        split=split,
        patch_size=tuple(cfg["patch_size"]),
        require_label=False,
        spine_disc_min_label=int(cfg.get("spine_disc_min_label", 60)),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    processor = GpuBatchProcessor(
        aug_config=None,  # never augment at inference time
        heatmap_sigma=cfg["heatmap_sigma"],
        heatmap_radius=cfg["heatmap_radius"],
    ).to(device)

    model = load_model(cfg["model_name"], in_channels=cfg["in_channels"]).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    columns = ["name"]
    for name in HEATMAP_CHANNELS[1:]:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_peak"])
    columns.append("all_discs_peaks")
    rows = []

    anchor_errors: Dict[str, List[float]] = {name: [] for name in HEATMAP_CHANNELS[1:]}

    peak_thr = cfg.get("peak_threshold", 0.3)
    with torch.no_grad():
        for batch in tqdm.tqdm(loader):
            image_3d = batch["image"].to(device, non_blocking=True).float()
            label_3d = batch.get("label")
            if label_3d is not None:
                label_3d = label_3d.to(device, non_blocking=True).float()
            # Anatomy seg (SPINE+CORD+CANAL) is harmless to pass through at
            # inference time — augmentation is disabled, so the processor only
            # extracts its mid-sagittal slice.
            seg_3d = batch.get("seg")
            if seg_3d is not None:
                seg_3d = seg_3d.to(device, non_blocking=True).float()

            image_2d, _hm_target, label_2d, _seg_2d = processor(
                image_3d, label_3d, seg_3d=seg_3d, apply_augmentation=False
            )
            heatmaps = model(image_2d)[0].cpu().numpy()  # (NUM_CLASSES, H, W)
            decoded = decode_predictions(heatmaps, peak_threshold=peak_thr)

            name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else str(batch["name"])
            out_row = [name]
            for ch_name in HEATMAP_CHANNELS[1:]:
                pred = decoded[ch_name]
                out_row.extend([pred["x"], pred["y"], pred["peak"]])
            all_peaks = decoded[HEATMAP_CHANNELS[0]]["peaks"]
            out_row.append(";".join(f"{x:.2f},{y:.2f},{p:.3f}" for (x, y, p) in all_peaks))
            rows.append(out_row)

            if label_2d is not None:
                gt_anchors = _anchors_from_label_2d(label_2d[0, 0])
                for ch_name in HEATMAP_CHANNELS[1:]:
                    pred = decoded[ch_name]
                    if pred["present"] and ch_name in gt_anchors:
                        gx, gy = gt_anchors[ch_name]
                        anchor_errors[ch_name].append(
                            float(np.sqrt((pred["x"] - gx) ** 2 + (pred["y"] - gy) ** 2))
                        )

            if save_image_dir is not None:
                check_and_make_dir(save_image_dir)
                pred_landmarks = []
                for ch_idx, ch_name in enumerate(HEATMAP_CHANNELS[1:], start=1):
                    pred = decoded[ch_name]
                    if pred["present"]:
                        pred_landmarks.append((LANDMARK_CHANNEL_LABELS[ch_idx], pred["x"], pred["y"]))
                for x, y, _p in all_peaks:
                    pred_landmarks.append((1, x, y))
                gt_landmarks: List[Tuple[int, float, float]] = []
                if label_2d is not None:
                    arr = label_2d[0, 0].cpu().numpy()
                    ys, xs = np.where(arr > 0)
                    for yy, xx in zip(ys, xs):
                        gt_landmarks.append((int(arr[yy, xx]), float(xx), float(yy)))
                viz_path = os.path.join(save_image_dir, name + ".png")
                visualize_prediction(image_2d[0, 0].cpu().numpy(), pred_landmarks, gt_landmarks, viz_path)

    pd.DataFrame(rows, columns=columns).to_csv(out_csv, index=False)
    print(f"Wrote predictions to {out_csv}")

    print("\nPer-anchor pixel errors (mid-sagittal pixels):")
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
    parser.add_argument("--manifest", required=True, help="Path to manifest.json.")
    parser.add_argument("--split", default="test", help="Split name inside the manifest (default: test).")
    parser.add_argument("--out-csv", required=True, help="Where to write predictions.csv.")
    parser.add_argument("--config", default=None, help="Optional JSON config overriding defaults.")
    parser.add_argument("--save-image-dir", default=None, help="If set, write overlay PNGs here.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    predict(
        cfg,
        weights_path=args.weights,
        manifest_path=args.manifest,
        split=args.split,
        out_csv=args.out_csv,
        save_image_dir=args.save_image_dir,
    )


if __name__ == "__main__":
    main()
