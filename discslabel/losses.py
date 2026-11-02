"""Heatmap regression losses."""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pixel-wise focal loss on sigmoid heatmaps.

    Positive pixels are those where the Gaussian target peak sits (>0.9).
    Negative pixels are down-weighted by (1 - target)^4 to suppress the
    massive background class.
    """
    predict = torch.clamp(predict, min=1e-4, max=1 - 1e-4)

    pos_inds = target.gt(0.9)
    neg_inds = target.lt(0.9)
    neg_weights = torch.pow(1 - target[neg_inds], 4)

    pos_pred = predict[pos_inds]
    neg_pred = predict[neg_inds]

    pos_loss = torch.log2(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log2(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def load_loss(loss_name: str):
    if loss_name == "L1":
        return nn.L1Loss()
    if loss_name == "MSE":
        return nn.MSELoss()
    if loss_name == "focalLoss":
        return focal_loss
    raise ValueError(f"Please input valid loss name, {loss_name} not in loss zoo.")


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

    ``predict`` is either a single tensor (eval mode / no deep supervision)
    or a list ``[full, half, quarter, ...]``. Losses are computed in float32
    so the focal loss' ``log2(1 - p)`` cannot hit ``-inf`` under fp16 autocast.
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
