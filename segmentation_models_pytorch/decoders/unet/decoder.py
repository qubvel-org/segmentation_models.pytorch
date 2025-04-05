from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md


class UnetDecoderBlock(nn.Module):
    """A decoder block in the U-Net architecture that performs upsampling and feature fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(
        self,
        feature_map: torch.Tensor,
        target_height: int,
        target_width: int,
        skip_connection: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature_map = F.interpolate(
            feature_map,
            size=(target_height, target_width),
            mode=self.interpolation_mode,
        )
        if skip_connection is not None:
            feature_map = torch.cat([feature_map, skip_connection], dim=1)
            feature_map = self.attention1(feature_map)
        feature_map = self.conv1(feature_map)
        feature_map = self.conv2(feature_map)
        feature_map = self.attention2(feature_map)
        return feature_map


class UnetCenterBlock(nn.Sequential):
    """Center block of the Unet decoder. Applied to the last feature map of the encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_norm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    """The decoder part of the U-Net architecture.

    Takes encoded features from different stages of the encoder and progressively upsamples them while
    combining with skip connections. This helps preserve fine-grained details in the final segmentation.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_blocks: int = 5,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        add_center_block: bool = False,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if add_center_block:
            self.center = UnetCenterBlock(
                head_channels,
                head_channels,
                use_norm=use_norm,
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels in zip(
            in_channels, skip_channels, out_channels
        ):
            block = UnetDecoderBlock(
                block_in_channels,
                block_skip_channels,
                block_out_channels,
                use_norm=use_norm,
                attention_type=attention_type,
                interpolation_mode=interpolation_mode,
            )
            self.blocks.append(block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # spatial shapes of features: [hw, hw/2, hw/4, hw/8, ...]
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skip_connections = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            # upsample to the next spatial shape
            height, width = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip_connection)

        return x
