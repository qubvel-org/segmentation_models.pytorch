import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.model import Model
from ..common.blocks import Conv2dReLU


def _upsample(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class PyramidStage(nn.Module):

    def __init__(self, in_channels, out_channels, pool_size, use_bathcnorm=True):
        super().__init__()
        if pool_size == 1:
            use_bathcnorm = False
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size)),
            Conv2dReLU(in_channels, out_channels, (1, 1), use_batchnorm=use_bathcnorm)
        )

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = self.pool(x)
        x = _upsample(x, size=(h, w))
        return x


class PSPModule(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6), use_bathcnorm=True):
        super().__init__()

        self.stages = nn.ModuleList([
            PyramidStage(in_channels, in_channels // len(sizes), size, use_bathcnorm=use_bathcnorm) for size in sizes
        ])

    def forward(self, x):
        xs = [stage(x) for stage in self.stages] + [x]
        x = torch.cat(xs, dim=1)
        return x


class AUXModule(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.adaptive_max_pool2d(x, output_size=(1, 1))
        x = x.view(-1, x.size(1))
        x = self.linear(x)
        return x


class PSPDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            downsample_factor=8,
            use_batchnorm=True,
            psp_out_channels=512,
            final_channels=21,
            aux_output=False,
            dropout=0.2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.out_channels = self._get(encoder_channels)
        self.aux_output = aux_output
        self.dropout_factor = dropout

        self.psp = PSPModule(
            self.out_channels,
            sizes=(1, 2, 3, 6),
            use_bathcnorm=use_batchnorm,
        )

        self.conv = Conv2dReLU(
            self.out_channels * 2,
            psp_out_channels,
            kernel_size=1,
            use_batchnorm=use_batchnorm,
        )

        if self.dropout_factor:
            self.dropout = nn.Dropout2d(p=dropout)

        self.final_conv = nn.Conv2d(psp_out_channels, final_channels,
                                    kernel_size=(3, 3), padding=1)

        if self.aux_output:
            self.aux = AUXModule(self.out_channels, final_channels)

        self.initialize()

    def _get(self, xs):
        if self.downsample_factor == 4:
            return xs[3]
        elif self.downsample_factor == 8:
            return xs[2]
        elif self.downsample_factor == 16:
            return xs[1]
        else:
            raise ValueError('Downsample factor should bi in [4, 8, 16], got {}'
                             .format(self.downsample_factor))

    def forward(self, x):

        features = self._get(x)
        x = self.psp(features)
        x = self.conv(x)
        if self.dropout_factor:
            x = self.dropout(x)
        x = self.final_conv(x)
        x = F.interpolate(
            x,
            scale_factor=self.downsample_factor,
            mode='bilinear',
            align_corners=True
        )

        if self.training and self.aux_output:
            aux = self.aux(features)
            x = [x, aux]

        return x