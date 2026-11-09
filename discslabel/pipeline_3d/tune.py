"""Grid-search ``peak_threshold`` / ``min_peak_distance`` for a trained 3D model.

Runs the stitched sliding-window prediction once per validation volume, then
re-decodes the cached peak candidates for every grid point and writes
``postprocessing.json`` (best pair) + ``postprocessing_grid.csv`` (all pairs).
``discslabel_inference_3d`` picks up ``postprocessing.json`` automatically
when it sits next to the weights.

When ``skeleton.json`` (``discslabel_skeleton``) is present, the skeleton
correction's ``z_threshold`` is tuned too (see ``discslabel.postprocessing``).
"""

from __future__ import annotations

import argparse
import os

import torch

from ..postprocessing import tune
from ..training import load_inference_weights
from .model import load_model
from .predictor import SKELETON_PROJECTION, labeled_to_voxels, predict_volume_heatmap
from .train import _load_config, _resolve_batch_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune 3D decoding parameters (peak threshold, min peak distance) on validation."
    )
    parser.add_argument("--weights", required=True, help="Path to best_model.pt.")
    parser.add_argument("--manifest", required=True, help="Prepared manifest.json.")
    parser.add_argument("--config", required=True, help="JSON config.")
    parser.add_argument("--split", default="valid", help="Manifest split to tune on (default: valid).")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Where to write postprocessing.json (default: the weights folder).",
    )
    parser.add_argument(
        "--num-workers",
        default=None,
        help="CPU worker processes for loading / peak extraction / grid scoring "
        "(default: config inference_num_workers, else 'auto'; 0 = run inline).",
    )
    parser.add_argument(
        "--skeleton",
        default=None,
        help="skeleton.json from discslabel_skeleton (default: <out-dir>/skeleton.json). "
        "When present, the skeleton correction z_threshold is tuned too.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg["model_name"], in_channels=cfg["in_channels"]).to(device)
    load_inference_weights(model, args.weights, device)
    model.eval()

    patch_dhw = tuple(int(s) for s in cfg["patch_size"])
    overlap = float(cfg.get("inference_overlap", 0.5))
    batch_size = _resolve_batch_size(cfg.get("inference_batch_size"), device)

    tune(
        cfg,
        manifest_path=args.manifest,
        out_dir=args.out_dir or os.path.dirname(os.path.abspath(args.weights)),
        predict_fn=lambda v: predict_volume_heatmap(
            v, model, device, patch_dhw, overlap, batch_size
        ),
        to_voxels=labeled_to_voxels,
        split=args.split,
        num_workers=args.num_workers,
        projection=SKELETON_PROJECTION,
        skeleton_path=args.skeleton,
    )


if __name__ == "__main__":
    main()
