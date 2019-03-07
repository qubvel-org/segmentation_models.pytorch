import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv2dReLU
from ..base.model import Model


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x += skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        return self.block(x)


class FPNDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            pyramid_channels=256,
            segmentation_channels=128,
            final_channels=1,
            dropout=0.2,
            use_batchnorm=True,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, use_batchnorm=use_batchnorm)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, use_batchnorm=use_batchnorm)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, use_batchnorm=use_batchnorm)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, use_batchnorm=use_batchnorm)

        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        self.final_conv = nn.Conv2d(4 * segmentation_channels, final_channels, kernel_size=3, padding=1)
        self.initialize()

    def forward(self, x):
        c5, c4, c3, c2, _ = x

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        x = torch.cat([
            F.interpolate(s5, scale_factor=8, mode='bilinear'),
            F.interpolate(s4, scale_factor=4, mode='bilinear'),
            F.interpolate(s3, scale_factor=2, mode='bilinear'),
            s2,
        ], dim=1)

        x = self.dropout(x)
        x = self.final_conv(x)

        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        return x
