"""Train the UNet heatmap-regression model with GPU-side 3D augmentation.

Originally based on https://github.com/haoyuuuuu/CL-Detection2024-Baseline-UNet

The training loop pulls 3D volumes from the dataset, moves the batch to the
GPU, and lets ``GpuBatchProcessor`` do:

    1. optional SmaugLab 3D augmentations on the paired (image, seg)
    2. mid-sagittal slice extraction (axis R-L)
    3. point-wise label projection + Gaussian heatmap synthesis
    4. optional 2D kornia random elastic transform on (image, heatmap)

The 2D model then sees ``(image_2d, heatmap_2d)`` for the loss.

Weights & Biases
----------------
When the config carries a ``wandb`` block, each epoch logs train/valid loss,
the current learning rate, and a per-epoch overlay figure that shows the
mid-sagittal input, the GT heatmap channels, and the predicted heatmap
channels for one validation sample — the same layout as the dry-run PNGs
with an extra row for the model output.

Debug / dry-run mode
--------------------
Pass ``--dry-run`` (or set ``debug: true`` in the pipeline config) to skip
training entirely and instead render 10 random training samples through the
full pipeline, saving one PNG per sample under
``<save-model-dir>/dry_run/`` that shows the mid-sagittal image fed to the
network with each disc-heatmap target channel overlaid. Use it to sanity-
check patch size, random crop, z-score normalisation and heatmap sigmas
before launching a long run.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from . import HEATMAP_CHANNELS
from .dataset import DiscVolumeDataset
from .gpu_pipeline import GpuBatchProcessor
from .losses import load_loss
from .model import load_model
from .utils.io import check_and_make_dir


def _load_config(path: str | None) -> dict:
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def _one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    processor: GpuBatchProcessor,
    loss_fn,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
    capture_first_batch: bool = False,
) -> tuple[float, dict | None]:
    """Run one pass over ``loader``. When ``optimizer`` is None, operates in eval mode.

    When ``capture_first_batch`` is set, the second return value holds a dict
    with the first batch's ``image_2d``, ``heatmap_2d`` (ground truth) and
    ``pred`` (model output) on CPU — handy for logging a per-epoch sample
    without a separate forward pass.
    """
    train = optimizer is not None
    model.train(train)
    losses = []
    first_sample: dict | None = None
    with torch.set_grad_enabled(train):
        for batch in tqdm.tqdm(loader, desc=desc):
            image_3d = batch["image"].to(device, non_blocking=True).float()
            label_3d = batch["label"].to(device, non_blocking=True).float()
            # Combined SPINE/CORD/CANAL anatomy mask — this is the "mask"
            # fed to AugTransformsGPU (not the disc label).
            seg_3d = batch.get("seg")
            if seg_3d is not None:
                seg_3d = seg_3d.to(device, non_blocking=True).float()

            image_2d, heatmap_2d, _label_2d, _seg_2d = processor(
                image_3d, label_3d, seg_3d=seg_3d, apply_augmentation=train
            )
            pred = model(image_2d)
            loss = loss_fn(pred, heatmap_2d)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())

            if capture_first_batch and first_sample is None:
                name = batch.get("name")
                if isinstance(name, (list, tuple)) and name:
                    name = name[0]
                first_sample = {
                    "image": image_2d.detach().cpu().numpy(),
                    "heatmap_gt": heatmap_2d.detach().cpu().numpy(),
                    "heatmap_pred": pred.detach().cpu().numpy(),
                    "name": str(name) if name is not None else "sample",
                }
    return float(np.mean(losses)) if losses else float("inf"), first_sample


def _display_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Reorient a mid-sagittal (P-A, I-S) slice for display: superior up, anterior left."""
    return np.ascontiguousarray(slice_2d.T[::-1, :])


def _intensity_window(img: np.ndarray) -> tuple[float, float]:
    """Robust 1st..99th-percentile window for the gray background."""
    vmin, vmax = float(np.percentile(img, 1.0)), float(np.percentile(img, 99.0))
    if vmax <= vmin:
        vmin, vmax = float(img.min()), float(img.max() + 1e-6)
    return vmin, vmax


def _draw_heatmap_row(axes, img, heatmap_2d, vmin, vmax, row_label):
    """Fill a row of axes with [input, image+ch0 overlay, image+ch1 overlay, ...]."""
    axes[0].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"image ({row_label})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for i, ch_name in enumerate(HEATMAP_CHANNELS):
        hm = _display_slice(heatmap_2d[i])
        ax = axes[i + 1]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        hm_masked = np.ma.masked_where(hm < 0.05, hm)
        ax.imshow(hm_masked, cmap="hot", alpha=0.65, vmin=0.0, vmax=1.0)
        ax.set_title(f"{row_label} ch{i}: {ch_name}")
        ax.set_xticks([])
        ax.set_yticks([])


def _build_overlay_figure(image_2d: np.ndarray, heatmap_2d: np.ndarray, title: str):
    """One-row figure: input + each heatmap channel overlaid. Returns a Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = _display_slice(image_2d)
    n_cols = len(HEATMAP_CHANNELS) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.6))
    vmin, vmax = _intensity_window(img)
    _draw_heatmap_row(axes, img, heatmap_2d, vmin, vmax, row_label="target")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _build_gt_pred_figure(
    image_2d: np.ndarray,
    heatmap_gt: np.ndarray,
    heatmap_pred: np.ndarray,
    title: str,
):
    """Two-row figure: GT row on top, Pred row below. Returns a Figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = _display_slice(image_2d)
    n_cols = len(HEATMAP_CHANNELS) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7.2))
    vmin, vmax = _intensity_window(img)
    _draw_heatmap_row(axes[0], img, heatmap_gt, vmin, vmax, row_label="GT")
    _draw_heatmap_row(axes[1], img, heatmap_pred, vmin, vmax, row_label="Pred")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _save_dry_run_png(path: str, image_2d: np.ndarray, heatmap_2d: np.ndarray, title: str) -> None:
    """Render one dry-run sample to disk."""
    import matplotlib.pyplot as plt

    fig = _build_overlay_figure(image_2d, heatmap_2d, title)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def dry_run(cfg: dict, manifest_path: str, save_model_dir: str, n_samples: int = 10) -> None:
    """Pull ``n_samples`` training batches through the data pipeline and the
    GPU processor and save one PNG per sample with the 2D mid-sagittal image
    (as it is fed to the network) and each disc-heatmap channel overlaid.

    No model is trained. Augmentations are applied so the PNG reflects exactly
    what the forward pass would see for a given sample.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dry-run] device: {device}, pulling {n_samples} samples")

    spine_disc_min = int(cfg.get("spine_disc_min_label", 60))
    dataset = DiscVolumeDataset(
        manifest_path,
        split="train",
        patch_size=tuple(cfg["patch_size"]),
        spine_disc_min_label=spine_disc_min,
        random_crop=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("Train split is empty — nothing to dry-run against.")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # keep the worker RNG trace reproducible-ish and errors in-process
        pin_memory=device.type == "cuda",
    )

    processor = GpuBatchProcessor(
        aug_config=cfg.get("aug_config"),
        heatmap_sigma=cfg["heatmap_sigma"],
        heatmap_radius=cfg["heatmap_radius"],
        elastic_config=cfg.get("elastic_config"),
    ).to(device)

    out_dir = os.path.join(save_model_dir, "dry_run")
    check_and_make_dir(out_dir)

    collected = 0
    iterator = iter(loader)
    while collected < n_samples:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        image_3d = batch["image"].to(device, non_blocking=True).float()
        label_3d = batch["label"].to(device, non_blocking=True).float()
        seg_3d = batch.get("seg")
        if seg_3d is not None:
            seg_3d = seg_3d.to(device, non_blocking=True).float()

        image_2d, heatmap_2d, _label_2d, _seg_2d = processor(
            image_3d, label_3d, seg_3d=seg_3d, apply_augmentation=True
        )

        name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else str(batch["name"])
        name_stem = name
        for ext in (".nii.gz", ".nii"):
            if name_stem.endswith(ext):
                name_stem = name_stem[: -len(ext)]
                break

        png_path = os.path.join(out_dir, f"sample_{collected:02d}_{name_stem}.png")
        try:
            _save_dry_run_png(
                png_path,
                image_2d[0, 0].detach().cpu().numpy(),
                heatmap_2d[0].detach().cpu().numpy(),
                title=f"[dry-run #{collected}] {name_stem}",
            )
        except ImportError:
            print("[dry-run] matplotlib unavailable; cannot render PNGs")
            return
        print(f"[dry-run] wrote {png_path}")
        collected += 1

    print(f"[dry-run] done. {collected} PNGs in {out_dir}")


def _init_wandb(cfg: dict, save_model_dir: str):
    """Initialize a Weights & Biases run from the ``wandb`` config block.

    Returns ``(run, wandb_module, log_images_every)``. All three are ``None``
    / ``0`` when logging is disabled (``cfg["wandb"]`` missing/falsy) or when
    the ``wandb`` package is not installed — training then runs with no
    external logging.
    """
    wandb_cfg = cfg.get("wandb")
    if not wandb_cfg:
        return None, None, 0
    
    import wandb
    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        dir=save_model_dir,
        config=cfg,
    )
    return run, wandb, int(wandb_cfg.get("log_images_every", 1))


def train(
    cfg: dict,
    manifest_path: str,
    save_model_dir: str,
) -> None:
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    spine_disc_min = int(cfg.get("spine_disc_min_label", 60))
    train_ds = DiscVolumeDataset(
        manifest_path,
        split="train",
        patch_size=tuple(cfg["patch_size"]),
        spine_disc_min_label=spine_disc_min,
        random_crop=True,
    )
    valid_ds = DiscVolumeDataset(
        manifest_path,
        split="valid",
        patch_size=tuple(cfg["patch_size"]),
        spine_disc_min_label=spine_disc_min,
        random_crop=False,
    )

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
        elastic_config=cfg.get("elastic_config"),
    ).to(device)

    model = load_model(cfg["model_name"], in_channels=cfg["in_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler_step_size"], gamma=cfg["scheduler_gamma"]
    )
    loss_fn = load_loss(cfg["loss_name"])

    check_and_make_dir(save_model_dir)

    wandb_run, wandb_mod, log_images_every = _init_wandb(cfg, save_model_dir)

    best_loss = float("inf")
    num_epoch_no_improvement = 0

    for epoch in range(cfg["train_max_epoch"]):
        train_loss, _ = _one_epoch(
            model, train_loader, processor, loss_fn, optimizer, device, desc=f"epoch {epoch + 1} train"
        )
        scheduler.step()
        print(f"Train epoch [{epoch + 1:<4d}/{cfg['train_max_epoch']:<4d}] Loss: {train_loss:.6f}")

        if cfg["save_model_step"] > 0 and epoch % cfg["save_model_step"] == 0:
            ckpt = os.path.join(save_model_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt)
            print("Saving checkpoint model:", ckpt)

        capture = wandb_run is not None and log_images_every > 0 and (epoch % log_images_every == 0)
        valid_loss, sample = _one_epoch(
            model,
            valid_loader,
            processor,
            loss_fn,
            None,
            device,
            desc=f"epoch {epoch + 1} valid",
            capture_first_batch=capture,
        )
        print(f"Validation loss: {valid_loss:.6f}")

        if wandb_run is not None:
            log = {
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/loss": valid_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if sample is not None:
                fig = _build_gt_pred_figure(
                    sample["image"][0, 0],
                    sample["heatmap_gt"][0],
                    sample["heatmap_pred"][0],
                    title=f"epoch {epoch + 1} valid — {sample['name']}",
                )
                log["valid/sample"] = wandb_mod.Image(fig)
                import matplotlib.pyplot as plt

                plt.close(fig)
            wandb_run.log(log, step=epoch)

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
    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the disc-landmark UNet.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json produced by `discslabel_prepare` / `scripts/prepare_datasets.sh`.",
    )
    parser.add_argument("--save-model-dir", required=True, help="Where to write checkpoints.")
    parser.add_argument("--config", required=True, help="Optional JSON config overriding defaults.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip training. Pull 10 random training samples through the data "
            "pipeline + GPU processor and save a PNG per sample under "
            "<save-model-dir>/dry_run/ showing the mid-sagittal image fed to "
            "the network with the disc-heatmap target channels overlaid. "
            "Can also be enabled by setting `debug: true` in the config."
        ),
    )
    parser.add_argument(
        "--dry-run-samples",
        type=int,
        default=10,
        help="How many random samples to render in dry-run mode (default: 10).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    is_dry = bool(args.dry_run) or bool(cfg.get("debug", False))
    if is_dry:
        dry_run(
            cfg,
            manifest_path=args.manifest,
            save_model_dir=args.save_model_dir,
            n_samples=int(args.dry_run_samples),
        )
        return
    train(cfg, manifest_path=args.manifest, save_model_dir=args.save_model_dir)


if __name__ == "__main__":
    main()
