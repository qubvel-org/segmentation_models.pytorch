"""
BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from collections.abc import Iterable, Sequence
from typing import Literal, List

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["DeepLabV3Decoder", "DeepLabV3PlusDecoder"]


class DeepLabV3Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: Iterable[int],
        aspp_separable: bool,
        aspp_dropout: float,
    ):
        super().__init__()
        self.aspp = ASPP(
            in_channels,
            out_channels,
            atrous_rates,
            separable=aspp_separable,
            dropout=aspp_dropout,
        )
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        x = self.aspp(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DeepLabV3PlusDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        encoder_depth: Literal[3, 4, 5],
        out_channels: int,
        atrous_rates: Iterable[int],
        output_stride: Literal[8, 16],
        aspp_separable: bool,
        aspp_dropout: float,
    ):
        super().__init__()
        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for DeepLabV3Plus decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        self.aspp = nn.Sequential(
            ASPP(
                encoder_channels[-1],
                out_channels,
                atrous_rates,
                separable=aspp_separable,
                dropout=aspp_dropout,
            ),
            SeparableConv2d(
                out_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        scale_factor = 4 if output_stride == 16 and encoder_depth > 3 else 2
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

        highres_in_channels = encoder_channels[2]
        highres_out_channels = 48  # proposed by authors of paper
        self.block1 = nn.Sequential(
            nn.Conv2d(
                highres_in_channels, highres_out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(highres_out_channels),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            SeparableConv2d(
                highres_out_channels + out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        aspp_features = self.aspp(features[-1])
        aspp_features = self.up(aspp_features)
        high_res_features = self.block1(features[2])
        concat_features = torch.cat([aspp_features, high_res_features], dim=1)
        fused_features = self.block2(concat_features)
        return fused_features


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPSeparableConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        atrous_rates: Iterable[int],
        separable: bool,
        dropout: float,
    ):
        super(ASPP, self).__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        ]

        rate1, rate2, rate3 = tuple(atrous_rates)
        ASPPConvModule = ASPPConv if not separable else ASPPSeparableConv

        modules.append(ASPPConvModule(in_channels, out_channels, rate1))
        modules.append(ASPPConvModule(in_channels, out_channels, rate2))
        modules.append(ASPPConvModule(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ):
        dephtwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        super().__init__(dephtwise_conv, pointwise_conv)
