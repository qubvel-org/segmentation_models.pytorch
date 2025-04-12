from typing import Any, Dict, Union, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class PSPModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sizes: Sequence[int] = (1, 2, 3, 6),
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    md.Conv2dReLU(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        use_norm=use_norm,
                    ),
                )
                for size in sizes
            ]
        )
        self.out_conv = md.Conv2dReLU(
            in_channels=in_channels + len(sizes) * out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_norm="batchnorm",
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        _, _, height, width = feature.shape
        pyramid_features = [feature]
        for block in self.blocks:
            pooled_feature = block(feature)
            resized_feature = F.interpolate(
                pooled_feature,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            pyramid_features.append(resized_feature)
        fused_feature = self.out_conv(torch.cat(pyramid_features, dim=1))
        return fused_feature


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # to channels_last
        normed_x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        normed_x = normed_x.permute(0, 3, 1, 2)  # to channels_first
        return normed_x


class FPNLateralBlock(nn.Module):
    def __init__(
        self,
        lateral_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        super().__init__()
        self.conv_norm_relu = md.Conv2dReLU(
            lateral_channels,
            out_channels,
            kernel_size=1,
            use_norm=use_norm,
        )

    def forward(
        self, state_feature: torch.Tensor, lateral_feature: torch.Tensor
    ) -> torch.Tensor:
        # 1. Apply block to encoder feature
        lateral_feature = self.conv_norm_relu(lateral_feature)
        # 2. Upsample encoder feature to the "state" feature resolution
        _, _, height, width = lateral_feature.shape
        state_feature = F.interpolate(
            state_feature, size=(height, width), mode="bilinear", align_corners=False
        )
        # 3. Sum state and encoder features
        fused_feature = state_feature + lateral_feature
        return fused_feature


class UPerNetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        encoder_depth: int = 5,
        decoder_channels: int = 256,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        super().__init__()

        if encoder_depth < 3:
            raise ValueError(
                "Encoder depth for UPerNet decoder cannot be less than 3, got {}.".format(
                    encoder_depth
                )
            )

        # Encoder channels for input features starting from the highest resolution
        # [1, 1/2, 1/4, 1/8, 1/16, ...] for num_features = encoder_depth + 1,
        # but we use only [1/4, 1/8, 1/16, ...] for UPerNet
        encoder_channels = encoder_channels[2:]

        self.feature_norms = nn.ModuleList(
            [LayerNorm2d(channels, eps=1e-6) for channels in encoder_channels]
        )

        # PSP Module
        lowest_resolution_feature_channels = encoder_channels[-1]
        self.psp = PSPModule(
            in_channels=lowest_resolution_feature_channels,
            out_channels=decoder_channels,
            sizes=(1, 2, 3, 6),
            use_norm=use_norm,
        )

        # FPN Module
        # we skip lower resolution feature maps + reverse the order
        # [1/4, 1/8, 1/16, 1/32] -> [1/16, 1/8, 1/4]
        lateral_channels = encoder_channels[:-1][::-1]
        self.fpn_lateral_blocks = nn.ModuleList([])
        self.fpn_conv_blocks = nn.ModuleList([])
        for channels in lateral_channels:
            block = FPNLateralBlock(
                lateral_channels=channels,
                out_channels=decoder_channels,
                use_norm=use_norm,
            )
            self.fpn_lateral_blocks.append(block)
            conv_block = md.Conv2dReLU(
                in_channels=decoder_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                padding=1,
                use_norm=use_norm,
            )
            self.fpn_conv_blocks.append(conv_block)

        num_blocks_to_fuse = len(self.fpn_conv_blocks) + 1  # +1 for the PSP module
        self.fusion_block = md.Conv2dReLU(
            in_channels=num_blocks_to_fuse * decoder_channels,
            out_channels=decoder_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features (List[torch.Tensor]):
                features with: [1, 1/2, 1/4, 1/8, 1/16, ...] spatial resolutions,
                where the first feature is the highest resolution and the number
                of features is equal to encoder_depth + 1.
        """

        # skip 1/1 and 1/2 resolution features
        features = features[2:]

        # normalize feature maps
        for i, norm in enumerate(self.feature_norms):
            features[i] = norm(features[i])

        # pass lowest resolution feature to PSP module
        psp_out = self.psp(features[-1])

        # skip lowest features for FPN + reverse the order
        # [1/4, 1/8, 1/16, 1/32] -> [1/16, 1/8, 1/4]
        fpn_lateral_features = features[:-1][::-1]
        fpn_features = [psp_out]
        for i, block in enumerate(self.fpn_lateral_blocks):
            # 1. for each encoder (skip) feature we apply 1x1 ConvNormRelu,
            # 2. upsample latest fpn feature to it's resolution
            # 3. sum them together
            lateral_feature = fpn_lateral_features[i]
            state_feature = fpn_features[-1]
            fpn_feature = block(state_feature, lateral_feature)
            fpn_features.append(fpn_feature)

        # Apply FPN conv blocks, but skip PSP module
        for i, conv_block in enumerate(self.fpn_conv_blocks, start=1):
            fpn_features[i] = conv_block(fpn_features[i])

        # Resize all FPN features to 1/4 of the original resolution.
        resized_fpn_features = []
        target_size = fpn_features[-1].shape[2:]  # 1/4 of the original resolution
        for feature in fpn_features:
            resized_feature = F.interpolate(
                feature, size=target_size, mode="bilinear", align_corners=False
            )
            resized_fpn_features.append(resized_feature)

        # reverse and concatenate
        stacked_features = torch.cat(resized_fpn_features[::-1], dim=1)
        output = self.fusion_block(stacked_features)
        return output
