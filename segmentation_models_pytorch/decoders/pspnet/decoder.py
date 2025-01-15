import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple
from segmentation_models_pytorch.base import modules


class PSPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: int,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        if pool_size == 1:
            use_batchnorm = False  # PyTorch does not support BatchNorm for 1x1 shape

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            modules.Conv2dReLU(
                in_channels, out_channels, (1, 1), use_batchnorm=use_batchnorm
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2:]
        x = self.pool(x)
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=True)
        return x


class PSPModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        sizes: Tuple[int, ...] = (1, 2, 3, 6),
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                PSPBlock(
                    in_channels,
                    in_channels // len(sizes),
                    size,
                    use_batchnorm=use_batchnorm,
                )
                for size in sizes
            ]
        )

    def forward(self, x):
        xs = [block(x) for block in self.blocks] + [x]
        x = torch.cat(xs, dim=1)
        return x


class PSPDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        use_batchnorm: bool = True,
        out_channels: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.psp = PSPModule(
            in_channels=encoder_channels[-1],
            sizes=(1, 2, 3, 6),
            use_batchnorm=use_batchnorm,
        )

        self.conv = modules.Conv2dReLU(
            in_channels=encoder_channels[-1] * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        x = self.psp(x)
        x = self.conv(x)
        x = self.dropout(x)

        return x
