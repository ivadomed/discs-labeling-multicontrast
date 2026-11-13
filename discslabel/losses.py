"""Heatmap regression losses. All losses take logits (no sigmoid in the model)."""

import math
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


_LN2 = math.log(2.0)


def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pixel-wise focal loss (CornerNet) on heatmap logits.

    Positive pixels are those where the Gaussian target peak sits (>0.9).
    Negative pixels are down-weighted by (1 - target)^4 to suppress the
    massive background class.

    ``log p`` and ``log(1 - p)`` come from ``logsigmoid`` on the logits, so
    they stay finite without clamping and the gradient no longer blows up
    as ``p -> 1`` (it did as ``1 / (1 - p)`` with the old clamp at 1e-4).
    Base-2 logs keep values comparable with the previous implementation.

    The batch sum is normalised by the batch's number of positives, floored
    at 1, so a batch without any disc is no longer an un-normalised sum.
    Normalising per sample instead would let disc-free random crops (most
    of nnU-Net-style sampling) outweigh disc crops by ~``num_pos`` and
    push every prediction towards 0.
    """
    logits = logits.float()
    target = target.float()
    prob = torch.sigmoid(logits)
    log_p = F.logsigmoid(logits) / _LN2
    log_1mp = F.logsigmoid(-logits) / _LN2

    pos = target.gt(0.9).float()
    neg = target.lt(0.9).float()
    pos_loss = log_p * (1 - prob) ** 2 * pos
    neg_loss = log_1mp * prob ** 2 * (1 - target) ** 4 * neg

    num_pos = pos.sum().clamp_min(1.0)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


class _OnProbabilities(nn.Module):
    """Apply a regression loss to ``sigmoid(logits)``."""

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(torch.sigmoid(logits.float()), target.float())


def _single_loss(loss_name: str):
    """Every loss takes the model's logits (the model has no final sigmoid)."""
    if loss_name == "L1":
        return _OnProbabilities(nn.L1Loss())
    if loss_name == "MSE":
        return _OnProbabilities(nn.MSELoss())
    if loss_name == "focalLoss":
        return focal_loss
    raise ValueError(f"Please input valid loss name, {loss_name} not in loss zoo.")


class CombinedLoss(nn.Module):
    """Weighted sum of several heatmap losses: ``sum_i w_i * loss_i(pred, target)``."""

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        if not weights:
            raise ValueError("Combined loss needs at least one term.")
        self.weights = {name: float(w) for name, w in weights.items()}
        self.terms = {name: _single_loss(name) for name in self.weights}

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = predict.new_zeros((), dtype=torch.float32)
        for name, w in self.weights.items():
            total = total + w * self.terms[name](predict, target)
        return total


def load_loss(loss_name: Union[str, Dict[str, float]]):
    """Build the training loss from the config ``loss_name``.

    * ``"focalLoss"`` — a single loss (focalLoss | L1 | MSE);
    * ``{"focalLoss": 1.0, "MSE": 100.0}`` — weighted sum of losses;
    * ``"focalLoss+MSE"`` — sum with every weight equal to 1.
    """
    if isinstance(loss_name, dict):
        return CombinedLoss(loss_name)
    if "+" in loss_name:
        return CombinedLoss({name.strip(): 1.0 for name in loss_name.split("+")})
    return _single_loss(loss_name)


def _downsample_target(target: torch.Tensor) -> torch.Tensor:
    """Halve every spatial axis with max-pooling.

    Max (not average) pooling keeps each Gaussian peak at 1.0, so the focal
    loss still finds its ``> 0.9`` positive voxels at the coarse scales.
    """
    if target.dim() == 4:
        return F.max_pool2d(target, kernel_size=2)
    if target.dim() == 5:
        return F.max_pool3d(target, kernel_size=2)
    raise ValueError(f"Expected a 4D or 5D target, got {tuple(target.shape)}")


def deep_supervision_weights(n_outputs: int) -> List[float]:
    """nnU-Net weights: ``1 / 2**i`` per scale, normalised to sum to 1."""
    weights = [1.0 / (2 ** i) for i in range(n_outputs)]
    total = sum(weights)
    return [w / total for w in weights]


class DeepSupervisionLoss(nn.Module):
    """Apply ``loss_fn`` at every output scale and take the weighted sum.

    ``predict`` holds logits: either a single tensor (eval mode / no deep
    supervision) or a list ``[full, half, quarter, ...]``. Losses are
    computed in float32, outside the fp16 range limits of autocast.
    """

    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, predict, target: torch.Tensor) -> torch.Tensor:
        if not isinstance(predict, (list, tuple)):
            return self.loss_fn(predict.float(), target.float())
        weights: Sequence[float] = deep_supervision_weights(len(predict))
        total = predict[0].new_zeros((), dtype=torch.float32)
        scaled = target.float()
        for i, (pred, w) in enumerate(zip(predict, weights)):
            if i > 0:
                scaled = _downsample_target(scaled)
            total = total + w * self.loss_fn(pred.float(), scaled)
        return total
