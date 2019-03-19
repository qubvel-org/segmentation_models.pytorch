import torch.nn as nn

from ..common.blocks import Conv2dReLU
from ..base.model import Model


class TransposeX2(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True, **batchnorm_params):
        super().__init__()
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, in_channels // 4, kernel_size=1, use_batchnorm=use_batchnorm),
            TransposeX2(in_channels // 4, in_channels // 4, use_batchnorm=use_batchnorm),
            Conv2dReLU(in_channels // 4, out_channels, kernel_size=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x


class LinknetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            prefinal_channels=32,
            final_channels=1,
            use_batchnorm=True,
    ):
        super().__init__()

        in_channels = encoder_channels

        self.layer1 = DecoderBlock(in_channels[0], in_channels[1], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], in_channels[2], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], in_channels[3], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], in_channels[4], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], prefinal_channels, use_batchnorm=use_batchnorm)
        self.final_conv = nn.Conv2d(prefinal_channels, final_channels, kernel_size=(1, 1))

        self.initialize()

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        return x
