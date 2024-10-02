import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class PSPModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sizes=(1, 2, 3, 6),
        use_batchnorm=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    md.Conv2dReLU(
                        in_channels,
                        in_channels // len(sizes),
                        kernel_size=1,
                        use_batchnorm=use_batchnorm,
                    ),
                )
                for size in sizes
            ]
        )
        self.out_conv = md.Conv2dReLU(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            use_batchnorm=True,
        )

    def forward(self, x):
        _, _, height, weight = x.shape
        out = [x] + [
            F.interpolate(
                block(x), size=(height, weight), mode="bilinear", align_corners=False
            )
            for block in self.blocks
        ]
        out = self.out_conv(torch.cat(out, dim=1))
        return out


class FPNBlock(nn.Module):
    def __init__(self, skip_channels, pyramid_channels, use_bathcnorm=True):
        super().__init__()
        self.skip_conv = (
            md.Conv2dReLU(
                skip_channels,
                pyramid_channels,
                kernel_size=1,
                use_batchnorm=use_bathcnorm,
            )
            if skip_channels != 0
            else nn.Identity()
        )

    def forward(self, x, skip):
        _, channels, height, weight = skip.shape
        x = F.interpolate(
            x, size=(height, weight), mode="bilinear", align_corners=False
        )
        if channels != 0:
            skip = self.skip_conv(skip)
            x = x + skip
        return x


class UPerNetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=64,
    ):
        super().__init__()

        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for UPerNet decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        encoder_channels = encoder_channels[::-1]

        # PSP Module
        self.psp = PSPModule(
            in_channels=encoder_channels[0],
            out_channels=pyramid_channels,
            sizes=(1, 2, 3, 6),
            use_batchnorm=True,
        )

        # FPN Module
        self.fpn_stages = nn.ModuleList(
            [FPNBlock(ch, pyramid_channels) for ch in encoder_channels[1:]]
        )

        self.fpn_bottleneck = md.Conv2dReLU(
            in_channels=(len(encoder_channels) - 1) * pyramid_channels,
            out_channels=segmentation_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def forward(self, *features):
        output_size = features[0].shape[2:]
        target_size = [size // 4 for size in output_size]

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        psp_out = self.psp(features[0])

        fpn_features = [psp_out]
        for feature, stage in zip(features[1:], self.fpn_stages):
            fpn_feature = stage(fpn_features[-1], feature)
            fpn_features.append(fpn_feature)

        # Resize all FPN features to 1/4 of the original resolution.
        resized_fpn_features = []
        for feature in fpn_features:
            resized_feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            resized_fpn_features.append(resized_feature)

        output = self.fpn_bottleneck(torch.cat(resized_fpn_features, dim=1))

        return output
