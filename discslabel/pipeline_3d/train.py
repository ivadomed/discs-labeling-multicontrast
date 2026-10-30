"""Train the 3D UNet heatmap-regression model with GPU-side 3D augmentation.

Full-3D counterpart to ``discslabel.pipeline_2d.train``. The data loader still
yields 3D patches via ``DiscVolumeDataset``, but the GPU processor no longer
projects onto the mid-sagittal plane and the model consumes the whole volume.

Pipeline per batch:
    1. optional SmaugLab 3D augmentations on the paired (image, seg)
    2. 3D Gaussian heatmap synthesis from the point-wise disc label
    3. the 3D model sees ``(image_3d, heatmap_3d)`` for the loss

Weights & Biases
----------------
When the config carries a ``wandb`` block, each epoch logs train/valid loss,
the current learning rate, and a per-epoch overlay figure. Since we cannot
show a volume inline, previews display the mid-sagittal slice of the input
alongside max-intensity projections along R-L of each heatmap channel.

Debug / dry-run mode
--------------------
Pass ``--dry-run`` (or set ``debug: true`` in the pipeline config) to skip
training and render 10 random training samples through the full pipeline to
``<save-model-dir>/dry_run/``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader

from .. import HEATMAP_CHANNELS
from ..dataset import DiscVolumeDataset
from .gpu_pipeline import GpuBatchProcessor
from ..losses import load_loss
from .model import load_model
from ..utils.io import check_and_make_dir


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
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> tuple[float, dict | None]:
    """One pass over ``loader``. Mirrors the 2D version, but every tensor is 5D.

    When ``capture_first_batch`` is set, the second return value holds a dict
    with the first batch's ``image_3d``, ``heatmap_gt`` (ground truth) and
    ``heatmap_pred`` on CPU for per-epoch logging.
    """
    train = optimizer is not None
    model.train(train)
    losses = []
    first_sample: dict | None = None
    use_amp = scaler is not None
    with torch.set_grad_enabled(train):
        for batch in tqdm.tqdm(loader, desc=desc):
            image_3d = batch["image"].to(device, non_blocking=True).float()
            label_3d = batch["label"].to(device, non_blocking=True).float()
            seg_3d = batch.get("seg")
            if seg_3d is not None:
                seg_3d = seg_3d.to(device, non_blocking=True).float()

            image_3d, heatmap_3d, _label_3d, _seg_3d = processor(
                image_3d, label_3d, seg_3d=seg_3d, apply_augmentation=train
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred = model(image_3d)
                loss = loss_fn(pred, heatmap_3d)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            losses.append(loss.item())

            if capture_first_batch and first_sample is None:
                name = batch.get("name")
                if isinstance(name, (list, tuple)) and name:
                    name = name[0]
                first_sample = {
                    "image": image_3d.detach().float().cpu().numpy(),
                    "heatmap_gt": heatmap_3d.detach().float().cpu().numpy(),
                    "heatmap_pred": pred.detach().float().cpu().numpy(),
                    "name": str(name) if name is not None else "sample",
                }
    return float(np.mean(losses)) if losses else float("inf"), first_sample


def _display_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Reorient a (P-A, I-S) slice for display: superior up, anterior left."""
    return np.ascontiguousarray(slice_2d.T[::-1, :])


def _mid_sagittal(volume_3d: np.ndarray) -> np.ndarray:
    """Return the mid-sagittal (axis 0 = R-L) slice as (H, W)."""
    d = volume_3d.shape[0] // 2
    return volume_3d[d]


def _max_project_rl(volume_3d: np.ndarray) -> np.ndarray:
    """Max-intensity projection along R-L. All heatmap peaks become visible
    on one 2D (H, W) view regardless of their R-L position."""
    return volume_3d.max(axis=0)


def _intensity_window(img: np.ndarray) -> tuple[float, float]:
    vmin, vmax = float(np.percentile(img, 1.0)), float(np.percentile(img, 99.0))
    if vmax <= vmin:
        vmin, vmax = float(img.min()), float(img.max() + 1e-6)
    return vmin, vmax


def _draw_heatmap_row(axes, img, heatmap_3d, vmin, vmax, row_label):
    """[input, image+ch0 projection, image+ch1 projection, ...] for one row.

    ``heatmap_3d`` is (NUM_CLASSES, D, H, W) — each channel is max-projected
    along R-L before being overlaid on the mid-sagittal image.
    """
    axes[0].imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"image ({row_label})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for i, ch_name in enumerate(HEATMAP_CHANNELS):
        hm_proj = _max_project_rl(heatmap_3d[i])
        hm = _display_slice(hm_proj)
        ax = axes[i + 1]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        hm_masked = np.ma.masked_where(hm < 0.05, hm)
        ax.imshow(hm_masked, cmap="hot", alpha=0.65, vmin=0.0, vmax=1.0)
        ax.set_title(f"{row_label} ch{i}: {ch_name}")
        ax.set_xticks([])
        ax.set_yticks([])


def _apply_suptitle(fig, title: str) -> None:
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))


def _build_overlay_figure(image_3d: np.ndarray, heatmap_3d: np.ndarray, title: str):
    """One-row figure: mid-sagittal input + each heatmap channel max-projected."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = _display_slice(_mid_sagittal(image_3d))
    n_cols = len(HEATMAP_CHANNELS) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.6))
    vmin, vmax = _intensity_window(img)
    _draw_heatmap_row(axes, img, heatmap_3d, vmin, vmax, row_label="target")
    _apply_suptitle(fig, title)
    return fig


def _build_gt_pred_figure(
    image_3d: np.ndarray,
    heatmap_gt: np.ndarray,
    heatmap_pred: np.ndarray,
    title: str,
):
    """Two-row figure: GT row above, Pred row below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = _display_slice(_mid_sagittal(image_3d))
    n_cols = len(HEATMAP_CHANNELS) + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7.2))
    vmin, vmax = _intensity_window(img)
    _draw_heatmap_row(axes[0], img, heatmap_gt, vmin, vmax, row_label="GT")
    _draw_heatmap_row(axes[1], img, heatmap_pred, vmin, vmax, row_label="Pred")
    _apply_suptitle(fig, title)
    return fig


def _save_dry_run_png(path: str, image_3d: np.ndarray, heatmap_3d: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    fig = _build_overlay_figure(image_3d, heatmap_3d, title)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def dry_run(cfg: dict, manifest_path: str, save_model_dir: str, n_samples: int = 10) -> None:
    """Render ``n_samples`` training samples through the 3D pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dry-run 3D] device: {device}, pulling {n_samples} samples")

    spine_disc_min = int(cfg.get("spine_disc_min_label", 60))
    foreground_prob = float(cfg.get("foreground_prob", 0.7))
    dataset = DiscVolumeDataset(
        manifest_path,
        split="train",
        patch_size=tuple(cfg["patch_size"]),
        spine_disc_min_label=spine_disc_min,
        random_crop=True,
        foreground_prob=foreground_prob,
    )
    if len(dataset) == 0:
        raise RuntimeError("Train split is empty — nothing to dry-run against.")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    processor = GpuBatchProcessor(
        aug_config=cfg.get("aug_config"),
        heatmap_sigma=cfg["heatmap_sigma"],
        heatmap_radius=cfg["heatmap_radius"],
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

        image_3d, heatmap_3d, _label_3d, _seg_3d = processor(
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
                image_3d[0, 0].detach().cpu().numpy(),
                heatmap_3d[0].detach().cpu().numpy(),
                title=f"[dry-run 3D #{collected}] {name_stem}",
            )
        except ImportError:
            print("[dry-run 3D] matplotlib unavailable; cannot render PNGs")
            return
        print(f"[dry-run 3D] wrote {png_path}")
        collected += 1

    print(f"[dry-run 3D] done. {collected} PNGs in {out_dir}")


def _resolve_num_workers(value) -> int:
    """``"auto"`` / missing → ``min(cpu_count, 8)``; else honour the explicit int."""
    if isinstance(value, int) and value >= 0:
        return value
    cpu = os.cpu_count() or 1
    return min(cpu, 8)


def _resolve_batch_size(value, device: torch.device, minimum: int = 1) -> int:
    """Auto-size the batch. Full-3D is memory-heavy, so default to 1 even on big GPUs
    unless the config overrides it. An explicit positive int in the config always wins.
    """
    if isinstance(value, int) and value > 0:
        return value
    return minimum


def _init_wandb(cfg: dict, save_model_dir: str):
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    num_workers = _resolve_num_workers(cfg.get("num_workers"))
    batch_size = _resolve_batch_size(cfg.get("batch_size"), device)
    batch_size_valid = _resolve_batch_size(cfg.get("batch_size_valid"), device)
    use_amp = bool(cfg.get("use_amp", True)) and device.type == "cuda"
    prefetch_factor = int(cfg.get("prefetch_factor", 4)) if num_workers > 0 else None
    print(
        f"[train 3D] device={device} num_workers={num_workers} "
        f"batch_size={batch_size} batch_size_valid={batch_size_valid} "
        f"amp={use_amp} prefetch_factor={prefetch_factor}"
    )

    spine_disc_min = int(cfg.get("spine_disc_min_label", 60))
    foreground_prob = float(cfg.get("foreground_prob", 0.7))
    print(f"[train 3D] foreground_prob={foreground_prob}")
    train_ds = DiscVolumeDataset(
        manifest_path,
        split="train",
        patch_size=tuple(cfg["patch_size"]),
        spine_disc_min_label=spine_disc_min,
        random_crop=True,
        foreground_prob=foreground_prob,
    )
    valid_ds = DiscVolumeDataset(
        manifest_path,
        split="valid",
        patch_size=tuple(cfg["patch_size"]),
        spine_disc_min_label=spine_disc_min,
        random_crop=False,
    )

    loader_kwargs = {
        "pin_memory": device.type == "cuda",
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size_valid, shuffle=False, **loader_kwargs
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

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    wandb_run, wandb_mod, log_images_every = _init_wandb(cfg, save_model_dir)

    best_loss = float("inf")
    num_epoch_no_improvement = 0

    for epoch in range(cfg["train_max_epoch"]):
        train_loss, _ = _one_epoch(
            model,
            train_loader,
            processor,
            loss_fn,
            optimizer,
            device,
            desc=f"epoch {epoch + 1} train",
            scaler=scaler if use_amp else None,
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
            scaler=scaler if use_amp else None,
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
                epoch_label = f"Epoch {epoch + 1}"
                fig = _build_gt_pred_figure(
                    sample["image"][0, 0],
                    sample["heatmap_gt"][0],
                    sample["heatmap_pred"][0],
                    title=f"{epoch_label} — validation sample: {sample['name']}",
                )
                log["valid/sample"] = wandb_mod.Image(
                    fig, caption=f"{epoch_label} — {sample['name']}"
                )
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
    parser = argparse.ArgumentParser(description="Train the 3D disc-landmark UNet.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json produced by `discslabel_prepare` / `scripts/prepare_datasets.sh`.",
    )
    parser.add_argument("--save-model-dir", required=True, help="Where to write checkpoints.")
    parser.add_argument("--config", required=True, help="JSON config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip training. Pull 10 random training samples through the data "
            "pipeline + GPU processor and save a PNG per sample under "
            "<save-model-dir>/dry_run/."
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
