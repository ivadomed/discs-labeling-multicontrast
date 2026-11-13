"""Training-loop building blocks shared by the 2D and 3D pipelines.

Modelled on nnU-Net's ``nnUNetTrainer``:

* ``build_optimizer`` — SGD + Nesterov (nnU-Net default) or Adam / AdamW.
* ``PolyLRScheduler`` — ``lr = initial_lr * (1 - epoch / num_epochs) ** 0.9``.
* ``ForegroundBatchSampler`` — fixed number of batches per epoch, cases drawn
  with replacement, and the last ``round(bs * oversample_foreground_percent)``
  samples of every batch forced to contain a disc.
* ``EmaTracker`` — exponential moving average of the validation score, used to
  pick ``best_model.pt`` (nnU-Net's ``ema_fg_dice``).
* ``load_inference_weights`` — loads a checkpoint into a model built without
  deep-supervision heads.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler


def build_optimizer(cfg: dict, params) -> torch.optim.Optimizer:
    """Optimizer from ``cfg["optimizer"]`` (``"SGD"`` | ``"Adam"`` | ``"AdamW"``)."""
    name = str(cfg.get("optimizer", "Adam"))
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 3e-5))
    if name == "SGD":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(cfg.get("momentum", 0.99)),
            nesterov=True,
            weight_decay=weight_decay,
        )
    if name in ("Adam", "AdamW"):
        betas = (float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.999)))
        opt_cls = torch.optim.Adam if name == "Adam" else torch.optim.AdamW
        return opt_cls(params, lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}; expected SGD, Adam or AdamW.")


class PolyLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """nnU-Net's polynomial decay, stepped once per epoch."""

    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9):
        self.initial_lr = float(initial_lr)
        self.max_steps = int(max_steps)
        self.exponent = float(exponent)
        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        step = min(self.last_epoch, self.max_steps)
        lr = self.initial_lr * (1 - step / self.max_steps) ** self.exponent
        return [lr for _ in self.optimizer.param_groups]


def oversample_flags(
    batch_size: int, oversample_foreground_percent: float, rng: np.random.Generator
) -> List[bool]:
    """Which samples of a batch must contain foreground.

    nnU-Net rule: sample ``i`` is forced to foreground when
    ``i >= round(batch_size * (1 - p))``. With ``batch_size == 1`` that rule
    would never (p < 0.5) or always (p >= 0.5) force foreground, so we fall
    back to nnU-Net's probabilistic variant there.
    """
    p = float(oversample_foreground_percent)
    if batch_size == 1:
        return [bool(rng.random() < p)]
    n_random = round(batch_size * (1 - p))
    return [i >= n_random for i in range(batch_size)]


class ForegroundBatchSampler(Sampler):
    """Yield ``num_batches`` batches of ``(case_index, force_foreground)`` keys.

    Cases are drawn uniformly with replacement, so an "epoch" is a fixed
    number of iterations rather than a pass over the dataset.
    ``DiscVolumeDataset.__getitem__`` understands the tuple keys.
    """

    def __init__(
        self,
        n_cases: int,
        batch_size: int,
        num_batches: int,
        oversample_foreground_percent: float,
        seed: Optional[int] = None,
    ):
        if n_cases < 1:
            raise ValueError("ForegroundBatchSampler needs at least one case.")
        self.n_cases = int(n_cases)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.oversample_foreground_percent = float(oversample_foreground_percent)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[Tuple[int, bool]]]:
        for _ in range(self.num_batches):
            cases = self._rng.integers(0, self.n_cases, size=self.batch_size)
            flags = oversample_flags(
                self.batch_size, self.oversample_foreground_percent, self._rng
            )
            yield [(int(c), f) for c, f in zip(cases, flags)]


class EmaTracker:
    """``ema = alpha * ema + (1 - alpha) * value`` (nnU-Net uses alpha = 0.9)."""

    def __init__(self, alpha: float = 0.9):
        self.alpha = float(alpha)
        self.value: Optional[float] = None
        self.best: Optional[float] = None

    def update(self, value: float) -> bool:
        """Fold ``value`` in; return True when the EMA reaches a new best."""
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        if self.best is None or self.value > self.best:
            self.best = self.value
            return True
        return False


DEEP_SUPERVISION_PREFIX = "ds_heads."


def load_inference_weights(model: torch.nn.Module, weights_path: str, device) -> None:
    """Load a checkpoint, dropping deep-supervision heads.

    Inference always builds the model without auxiliary heads, so weights
    trained with or without deep supervision load the same way.
    """
    state = torch.load(weights_path, map_location=device)
    state = {k: v for k, v in state.items() if not k.startswith(DEEP_SUPERVISION_PREFIX)}
    model.load_state_dict(state)


# ---------------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------------

LATEST_CHECKPOINT = "checkpoint_latest.pt"


def save_training_state(
    save_model_dir: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: PolyLRScheduler,
    scaler,
    ema: EmaTracker,
    wandb_run_id: Optional[str] = None,
) -> None:
    """Write everything needed to resume after ``epoch`` to ``checkpoint_latest.pt``.

    Written to a temp file then renamed, so a job killed mid-save never
    leaves a truncated checkpoint behind.
    """
    path = os.path.join(save_model_dir, LATEST_CHECKPOINT)
    tmp = path + ".tmp"
    torch.save(
        {
            "epoch": int(epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": {"value": ema.value, "best": ema.best},
            "wandb_run_id": wandb_run_id,
        },
        tmp,
    )
    os.replace(tmp, path)


def _latest_epoch_checkpoint(save_model_dir: str) -> Optional[Tuple[int, str]]:
    """Highest ``checkpoint_epoch_<N>.pt`` (weights only) in ``save_model_dir``."""
    best = None
    for name in os.listdir(save_model_dir):
        m = re.fullmatch(r"checkpoint_epoch_(\d+)\.pt", name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), os.path.join(save_model_dir, name))
    return best


def resume_training_state(
    save_model_dir: str,
    num_epochs: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: PolyLRScheduler,
    scaler,
    ema: EmaTracker,
    device,
) -> Tuple[int, Optional[str]]:
    """Restore the last saved state if training stopped before ``num_epochs``.

    Returns ``(start_epoch, wandb_run_id)``. ``start_epoch == 0`` means a
    fresh run; ``start_epoch >= num_epochs`` means the run is already done.

    Prefers ``checkpoint_latest.pt`` (full state). Falls back to the newest
    weights-only ``checkpoint_epoch_<N>.pt`` from older runs: the model and
    LR schedule resume, but the optimizer moments restart from zero and the
    EMA best is recovered from ``best_model.json`` when present.
    """
    if not os.path.isdir(save_model_dir):
        return 0, None

    latest = os.path.join(save_model_dir, LATEST_CHECKPOINT)
    if os.path.isfile(latest):
        state = torch.load(latest, map_location=device)
        start_epoch = int(state["epoch"]) + 1
        if start_epoch >= num_epochs:
            return start_epoch, state.get("wandb_run_id")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if scaler.is_enabled() and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        ema.value = state["ema"]["value"]
        ema.best = state["ema"]["best"]
        wandb_run_id = state.get("wandb_run_id")
        print(f"[resume] {latest}: continuing at epoch {start_epoch + 1}/{num_epochs}")
    else:
        legacy = _latest_epoch_checkpoint(save_model_dir)
        if legacy is None:
            return 0, None
        epoch, path = legacy
        start_epoch = epoch + 1
        if start_epoch >= num_epochs:
            return start_epoch, None
        model.load_state_dict(torch.load(path, map_location=device))
        scheduler.last_epoch = epoch
        best_json = os.path.join(save_model_dir, "best_model.json")
        if os.path.isfile(best_json):
            with open(best_json) as f:
                ema.best = float(json.load(f)["ema_f1"])
        wandb_run_id = None
        print(
            f"[resume] {path} (weights only): continuing at epoch "
            f"{start_epoch + 1}/{num_epochs}; optimizer state starts fresh"
        )

    # The schedule length follows the current config, even if num_epochs was
    # edited since the checkpoint was written (state_dict restores max_steps).
    scheduler.max_steps = int(num_epochs)
    scheduler.last_epoch = start_epoch
    lr = scheduler.get_lr()
    for group, value in zip(optimizer.param_groups, lr):
        group["lr"] = value
    return start_epoch, wandb_run_id
