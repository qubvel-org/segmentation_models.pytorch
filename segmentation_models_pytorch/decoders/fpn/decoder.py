import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Literal


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: bool = False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(
        self,
        pyramid_channels: int,
        skip_channels: int,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
        self.interpolation_mode = interpolation_mode

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode=self.interpolation_mode)
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_upsamples: int = 0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy: Literal["add", "cat"]):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy)
            )
        self.policy = policy

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        if self.policy == "add":
            output = torch.stack(x).sum(dim=0)
        elif self.policy == "cat":
            output = torch.cat(x, dim=1)
        else:
            raise ValueError(
                "`merge_policy` must be one of: ['add', 'cat'], got {}".format(
                    self.policy
                )
            )
        return output


class FPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        encoder_depth: int = 5,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        dropout: float = 0.2,
        merge_policy: Literal["add", "cat"] = "add",
        interpolation_mode: str = "nearest",
    ):
        super().__init__()

        self.out_channels = (
            segmentation_channels
            if merge_policy == "add"
            else segmentation_channels * 4
        )
        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for FPN decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1], interpolation_mode)
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2], interpolation_mode)
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3], interpolation_mode)

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(
                    pyramid_channels, segmentation_channels, n_upsamples=n_upsamples
                )
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        c2, c3, c4, c5 = features[-4:]

        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        s5 = self.seg_blocks[0](p5)
        s4 = self.seg_blocks[1](p4)
        s3 = self.seg_blocks[2](p3)
        s2 = self.seg_blocks[3](p2)

        feature_pyramid = [s5, s4, s3, s2]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x
