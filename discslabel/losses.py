"""Heatmap regression losses."""

import torch
import torch.nn as nn


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
