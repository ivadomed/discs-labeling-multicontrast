"""Train the UNet heatmap-regression model with GPU-side 3D augmentation.

The training loop pulls 3D volumes from the dataset, moves the batch to the
GPU, and lets ``GpuBatchProcessor`` do:

    1. optional SmaugLab 3D augmentations on the paired (image, label)
    2. mid-sagittal slice extraction (axis R-L)
    3. point-wise label projection + Gaussian heatmap synthesis

The 2D model then sees ``(image_2d, heatmap_2d)`` for the loss.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from .dataset import DiscVolumeDataset
from .gpu_pipeline import GpuBatchProcessor
from .losses import load_loss
from .model import load_model
from .utils.io import check_and_make_dir


DEFAULT_CONFIG = {
    "model_name": "UNet",
    "in_channels": 1,
    "patch_size": [96, 320, 320],
    "heatmap_sigma": 8.0,
    "heatmap_radius": 20,
    "batch_size": 2,
    "batch_size_valid": 1,
    "num_workers": 2,
    "train_max_epoch": 400,
    "epoch_patience": 10,
    "save_model_step": 10,
    "lr": 1e-4,
    "beta1": 0.9,
    "beta2": 0.999,
    "scheduler_step_size": 10,
    "scheduler_gamma": 0.9,
    "loss_name": "focalLoss",
    "peak_threshold": 0.3,
    # Path to a SmaugLab GPU augmentation config JSON. ``null`` disables augmentation.
    "aug_config": None,
}


def _load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r") as f:
            cfg.update(json.load(f))
    return cfg


def _one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    processor: GpuBatchProcessor,
    loss_fn,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> float:
    """Run one pass over ``loader``. When ``optimizer`` is None, operates in eval mode."""
    train = optimizer is not None
    model.train(train)
    losses = []
    with torch.set_grad_enabled(train):
        for batch in tqdm.tqdm(loader, desc=desc):
            image_3d = batch["image"].to(device, non_blocking=True).float()
            label_3d = batch["label"].to(device, non_blocking=True).float()

            image_2d, heatmap_2d, _label_2d = processor(image_3d, label_3d, apply_augmentation=train)
            pred = model(image_2d)
            loss = loss_fn(pred, heatmap_2d)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("inf")


def train(
    cfg: dict,
    manifest_path: str,
    save_model_dir: str,
    cuda_id: int = 0,
) -> None:
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    train_ds = DiscVolumeDataset(manifest_path, split="train", patch_size=tuple(cfg["patch_size"]))
    valid_ds = DiscVolumeDataset(manifest_path, split="valid", patch_size=tuple(cfg["patch_size"]))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["batch_size_valid"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
    )

    processor = GpuBatchProcessor(
        aug_config=cfg.get("aug_config"),
        heatmap_sigma=cfg["heatmap_sigma"],
        heatmap_radius=cfg["heatmap_radius"],
    ).to(device)

    model = load_model(cfg["model_name"], in_channels=cfg["in_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler_step_size"], gamma=cfg["scheduler_gamma"]
    )
    loss_fn = load_loss(cfg["loss_name"])

    check_and_make_dir(save_model_dir)

    best_loss = float("inf")
    num_epoch_no_improvement = 0

    for epoch in range(cfg["train_max_epoch"]):
        train_loss = _one_epoch(
            model, train_loader, processor, loss_fn, optimizer, device, desc=f"epoch {epoch + 1} train"
        )
        scheduler.step()
        print(f"Train epoch [{epoch + 1:<4d}/{cfg['train_max_epoch']:<4d}] Loss: {train_loss:.6f}")

        if cfg["save_model_step"] > 0 and epoch % cfg["save_model_step"] == 0:
            ckpt = os.path.join(save_model_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt)
            print("Saving checkpoint model:", ckpt)

        valid_loss = _one_epoch(
            model, valid_loader, processor, loss_fn, None, device, desc=f"epoch {epoch + 1} valid"
        )
        print(f"Validation loss: {valid_loss:.6f}")

        if valid_loss < best_loss:
            print(f"Validation loss decreases from {best_loss:.6f} to {valid_loss:.6f}")
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            best_path = os.path.join(save_model_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print("Saving best model:", best_path)
        else:
            num_epoch_no_improvement += 1
            print(
                f"Validation loss did not decrease from {best_loss:.6f} "
                f"(no improvement for {num_epoch_no_improvement} epochs)"
            )
        if num_epoch_no_improvement >= cfg["epoch_patience"]:
            print("Early stopping triggered.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the disc-landmark UNet.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json produced by `discslabel_prepare` / `scripts/prepare_datasets.sh`.",
    )
    parser.add_argument("--save-model-dir", required=True, help="Where to write checkpoints.")
    parser.add_argument("--config", default=None, help="Optional JSON config overriding defaults.")
    parser.add_argument("--cuda-id", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train(cfg, manifest_path=args.manifest, save_model_dir=args.save_model_dir, cuda_id=args.cuda_id)


if __name__ == "__main__":
    main()
