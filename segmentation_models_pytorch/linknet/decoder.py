import torch.nn as nn

from ..base import modules


class TransposeX2(nn.Sequential):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels))

        super().__init__(*layers)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()

        self.block = nn.Sequential(
            modules.Conv2dReLU(in_channels, in_channels // 4, kernel_size=1, use_batchnorm=use_batchnorm),
            TransposeX2(in_channels // 4, in_channels // 4, use_batchnorm=use_batchnorm),
            modules.Conv2dReLU(in_channels // 4, out_channels, kernel_size=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x, skip=None):
        x = self.block(x)
        if skip is not None:
            x = x + skip
        return x


class LinknetDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            prefinal_channels=32,
            n_blocks=5,
            use_batchnorm=True,
    ):
        super().__init__()

        encoder_channels = encoder_channels[1:]  # remove first skip
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        channels = list(encoder_channels) + [prefinal_channels]

        self.blocks = nn.ModuleList([
            DecoderBlock(channels[i], channels[i + 1], use_batchnorm=use_batchnorm)
            for i in range(n_blocks)
        ])

    def forward(self, *features):
        features = features[1:]  # remove first skip
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
