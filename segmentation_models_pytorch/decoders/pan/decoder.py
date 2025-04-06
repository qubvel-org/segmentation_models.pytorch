from collections.abc import Sequence
from typing import Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        add_relu: bool = True,
        interpolate: bool = False,
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

        self.add_relu = add_relu
        self.interpolate = interpolate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)

        if self.add_relu:
            x = self.activation(x)

        if self.interpolate:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)

        return x


class FPABlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, upscale_mode: str = "bilinear"
    ):
        super().__init__()

        self.upscale_mode = upscale_mode
        if self.upscale_mode == "bilinear":
            self.align_corners = True
        else:
            self.align_corners = False

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # middle branch
        self.mid = nn.Sequential(
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(
                in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2
            ),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(
                in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            ConvBnRelu(
                in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
        )
        self.conv2 = ConvBnRelu(
            in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2
        )
        self.conv1 = ConvBnRelu(
            in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape

        branch1_output = self.branch1(x)
        branch1_output = F.interpolate(
            branch1_output,
            size=(height, width),
            mode=self.upscale_mode,
            align_corners=self.align_corners,
        )

        middle_output = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(
            x3,
            size=(height // 4, width // 4),
            mode=self.upscale_mode,
            align_corners=self.align_corners,
        )

        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(
            x,
            size=(height // 2, width // 2),
            mode=self.upscale_mode,
            align_corners=self.align_corners,
        )

        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(
            x,
            size=(height, width),
            mode=self.upscale_mode,
            align_corners=self.align_corners,
        )

        x = torch.mul(x, middle_output)
        x = x + branch1_output

        return x


class GAUBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolation_mode: str = "bilinear",
    ):
        super(GAUBlock, self).__init__()

        self.interpolation_mode = interpolation_mode
        self.align_corners = True if interpolation_mode == "bilinear" else None

        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                add_relu=False,
            ),
            nn.Sigmoid(),
        )
        self.conv2 = ConvBnRelu(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: low level feature
            y: high level feature
        """
        height, width = x.shape[2:]
        y_up = F.interpolate(
            y,
            size=(height, width),
            mode=self.interpolation_mode,
            align_corners=self.align_corners,
        )
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)
        return y_up + z


class PANDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        encoder_depth: Literal[3, 4, 5],
        decoder_channels: int,
        interpolation_mode: str = "bilinear",
    ):
        super().__init__()

        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for PAN decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        encoder_channels = encoder_channels[2:]

        self.fpa = FPABlock(
            in_channels=encoder_channels[-1], out_channels=decoder_channels
        )

        if encoder_depth == 5:
            self.gau3 = GAUBlock(
                in_channels=encoder_channels[2],
                out_channels=decoder_channels,
                interpolation_mode=interpolation_mode,
            )
        if encoder_depth >= 4:
            self.gau2 = GAUBlock(
                in_channels=encoder_channels[1],
                out_channels=decoder_channels,
                interpolation_mode=interpolation_mode,
            )
        if encoder_depth >= 3:
            self.gau1 = GAUBlock(
                in_channels=encoder_channels[0],
                out_channels=decoder_channels,
                interpolation_mode=interpolation_mode,
            )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        features = features[2:]  # remove first and second skip

        out = self.fpa(features[-1])  # 1/16 or 1/32

        if hasattr(self, "gau3"):
            out = self.gau3(features[2], out)  # 1/16
        if hasattr(self, "gau2"):
            out = self.gau2(features[1], out)  # 1/8
        if hasattr(self, "gau1"):
            out = self.gau1(features[0], out)  # 1/4

        return out
