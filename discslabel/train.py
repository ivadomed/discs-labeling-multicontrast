"""Train the UNet heatmap-regression model on mid-sagittal disc CSVs."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import DiscLandmarkDataset
from .losses import load_loss
from .model import load_model
from .transforms import (
    RandomIntensityJitter,
    RandomVerticalFlip,
    Rescale,
    ToTensor,
)
from .utils.io import check_and_make_dir


DEFAULT_CONFIG = {
    "model_name": "UNet",
    "in_channels": 1,
    "image_height": 512,
    "image_width": 512,
    "heatmap_sigma": 8.0,
    "heatmap_radius": 20,
    "batch_size": 8,
    "batch_size_valid": 2,
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
    "intensity_jitter": 0.1,
    "vertical_flip_prob": 0.0,
}


def _load_config(path: str | None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r") as f:
            cfg.update(json.load(f))
    return cfg


def _build_transforms(cfg: dict, train: bool):
    steps = [Rescale(output_size=(cfg["image_height"], cfg["image_width"]))]
    if train:
        if cfg.get("vertical_flip_prob", 0.0) > 0.0:
            steps.append(RandomVerticalFlip(p=cfg["vertical_flip_prob"]))
        if cfg.get("intensity_jitter", 0.0) > 0.0:
            steps.append(RandomIntensityJitter(scale=cfg["intensity_jitter"], shift=cfg["intensity_jitter"]))
    steps.append(ToTensor(sigma=cfg["heatmap_sigma"], radius=cfg["heatmap_radius"]))
    return transforms.Compose(steps)


def train(cfg: dict, train_csv: str, valid_csv: str, save_model_dir: str, cuda_id: int = 0) -> None:
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    train_ds = DiscLandmarkDataset(train_csv, transform=_build_transforms(cfg, train=True))
    valid_ds = DiscLandmarkDataset(valid_csv, transform=_build_transforms(cfg, train=False))

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg["batch_size_valid"], shuffle=False, num_workers=cfg["num_workers"]
    )

    model = load_model(cfg["model_name"], in_channels=cfg["in_channels"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler_step_size"], gamma=cfg["scheduler_gamma"]
    )
    loss_fn = load_loss(cfg["loss_name"])

    check_and_make_dir(save_model_dir)

    best_loss = float("inf")
    num_epoch_no_improvement = 0

    for epoch in range(cfg["train_max_epoch"]):
        model.train()
        train_losses = []
        for image, heatmap in tqdm.tqdm(train_loader, desc=f"epoch {epoch + 1} train"):
            image = image.float().to(device)
            heatmap = heatmap.float().to(device)
            output = model(image)
            loss = loss_fn(output, heatmap)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        print(
            f"Train epoch [{epoch + 1:<4d}/{cfg['train_max_epoch']:<4d}] "
            f"Loss: {np.mean(train_losses):.6f}"
        )

        if cfg["save_model_step"] > 0 and epoch % cfg["save_model_step"] == 0:
            ckpt = os.path.join(save_model_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt)
            print("Saving checkpoint model:", ckpt)

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for image, heatmap in tqdm.tqdm(valid_loader, desc=f"epoch {epoch + 1} valid"):
                image = image.float().to(device)
                heatmap = heatmap.float().to(device)
                output = model(image)
                valid_losses.append(loss_fn(output, heatmap).item())
        valid_loss = float(np.mean(valid_losses)) if valid_losses else float("inf")
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
    parser.add_argument("--train-csv", required=True, help="Path to train.csv (from `discslabel_prepare`).")
    parser.add_argument("--valid-csv", required=True, help="Path to valid.csv (from `discslabel_prepare`).")
    parser.add_argument("--save-model-dir", required=True, help="Where to write checkpoints.")
    parser.add_argument("--config", default=None, help="Optional JSON config overriding defaults.")
    parser.add_argument("--cuda-id", type=int, default=0)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    train(cfg, args.train_csv, args.valid_csv, args.save_model_dir, cuda_id=args.cuda_id)


if __name__ == "__main__":
    main()
