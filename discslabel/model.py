"""UNet backbone for 2D heatmap regression on mid-sagittal MRI slices."""

import torch
import torch.nn as nn

from . import NUM_CLASSES


class LUConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        return self.activation(self.bn1(self.conv1(x)))


def make_n_conv_layer(in_channels, depth, double_channel=False):
    if double_channel:
        layer1 = LUConv(in_channels, 32 * (2 ** (depth + 1)))
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)))
    else:
        layer1 = LUConv(in_channels, 32 * (2 ** depth))
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2)
    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channels, depth):
        super().__init__()
        self.ops = make_n_conv_layer(in_channels, depth)
        self.pool = nn.MaxPool2d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.pool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super().__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.ops = make_n_conv_layer(in_channels + out_channels // 2, depth, double_channel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        return self.ops(concat)


class OutputTransition(nn.Module):
    def __init__(self, in_channels, n_labels):
        super().__init__()
        self.final_conv = nn.Conv2d(in_channels, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.final_conv(x))


class UNet(nn.Module):
    def __init__(self, in_channels=1, n_class=NUM_CLASSES):
        super().__init__()

        self.down_tr64 = DownTransition(in_channels, 0)
        self.down_tr128 = DownTransition(64, 1)
        self.down_tr256 = DownTransition(128, 2)
        self.down_tr512 = DownTransition(256, 3)

        self.up_tr256 = UpTransition(512, 512, 2)
        self.up_tr128 = UpTransition(256, 256, 1)
        self.up_tr64 = UpTransition(128, 128, 0)
        self.out_tr = OutputTransition(64, n_class)

    def forward(self, x):
        out64, skip64 = self.down_tr64(x)
        out128, skip128 = self.down_tr128(out64)
        out256, skip256 = self.down_tr256(out128)
        out512, _ = self.down_tr512(out256)

        up256 = self.up_tr256(out512, skip256)
        up128 = self.up_tr128(up256, skip128)
        up64 = self.up_tr64(up128, skip64)
        return self.out_tr(up64)


def load_model(model_name: str = "UNet", in_channels: int = 1, n_class: int = NUM_CLASSES):
    if model_name == "UNet":
        return UNet(in_channels=in_channels, n_class=n_class)
    raise ValueError(f"Please input valid model name, {model_name} not in model zoo.")
