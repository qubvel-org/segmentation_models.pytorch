import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from typing import Optional, Sequence


class ProjectionBlock(nn.Module):
    """
    Concatenates the cls tokens with the features to make use of the global information aggregated in the cls token.
    Projects the combined feature map to the original embedding dimension using a MLP
    """

    def __init__(self, embed_dim: int, has_cls_token: bool):
        super().__init__()
        in_features = embed_dim * 2 if has_cls_token else embed_dim
        out_features = embed_dim
        self.project = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
        )

    def forward(
        self, features: torch.Tensor, cls_token: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, embed_dim, height, width = features.shape

        # Rearrange to (batch_size, height * width, embed_dim)
        features = features.view(batch_size, embed_dim, -1)
        features = features.transpose(1, 2).contiguous()

        # Add CLS token
        if cls_token is not None:
            cls_token = cls_token.expand_as(features)
            features = torch.cat([features, cls_token], dim=2)

        # Project to embedding dimension
        features = self.project(features)

        # Rearrange back to (batch_size, embed_dim, height, width)
        features = features.transpose(1, 2)
        features = features.view(batch_size, -1, height, width)

        return features


class ReassembleBlock(nn.Module):
    """
    Processes the features such that they have progressively increasing embedding size and progressively decreasing
    spatial dimension
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        upsample_factor: int,
    ):
        super().__init__()

        self.project_to_out_channel = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
        )

        if upsample_factor > 1.0:
            self.upsample = nn.ConvTranspose2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=int(upsample_factor),
                stride=int(upsample_factor),
            )
        elif upsample_factor == 1.0:
            self.upsample = nn.Identity()
        else:
            self.upsample = nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=int(1 / upsample_factor),
                padding=1,
            )

        self.project_to_feature_dim = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project_to_out_channel(x)
        x = self.upsample(x)
        x = self.project_to_feature_dim(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.batch_norm_1 = nn.BatchNorm2d(num_features=feature_dim)
        self.conv_2 = nn.Conv2d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.batch_norm_2 = nn.BatchNorm2d(num_features=feature_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Block 1
        x = self.activation(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x)

        # Block 2
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)

        # Add residual
        x = x + residual

        return x


class FusionBlock(nn.Module):
    """
    Fuses the processed encoder features in a residual manner and upsamples them
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.residual_conv_block1 = ResidualConvBlock(feature_dim)
        self.residual_conv_block2 = ResidualConvBlock(feature_dim)
        self.project = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.activation = nn.ReLU()

    def forward(
        self,
        feature: torch.Tensor,
        previous_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feature = self.residual_conv_block1(feature)
        if previous_feature is not None:
            feature = feature + previous_feature
        feature = self.residual_conv_block2(feature)
        feature = nn.functional.interpolate(
            feature, scale_factor=2, align_corners=True, mode="bilinear"
        )
        feature = self.project(feature)
        return feature


class DPTDecoder(nn.Module):
    """
    Decoder part for DPT

    Processes the encoder features and class tokens (if encoder has class_tokens) to have spatial downsampling ratios of
    [1/4, 1/8, 1/16, 1/32, ...] relative to the input image spatial dimension.

    The decoder then fuses these features in a residual manner and progressively upsamples them by a factor of 2 so that the
    output has a downsampling ratio of 1/2 relative to the input image spatial dimension

    """

    def __init__(
        self,
        encoder_out_channels: Sequence[int] = (756, 756, 756, 756),
        encoder_output_strides: Sequence[int] = (16, 16, 16, 16),
        intermediate_channels: Sequence[int] = (256, 512, 1024, 1024),
        fusion_channels: int = 256,
        has_cls_token: bool = False,
    ):
        super().__init__()

        num_blocks = len(encoder_output_strides)

        # If encoder has cls token, then concatenate it with the features along the embedding dimension and project it
        # back to the feature_dim dimension. Else, ignore the non-existent cls token
        blocks = [ProjectionBlock(in_channels, has_cls_token) for in_channels in encoder_out_channels]
        self.readout_blocks = nn.ModuleList(blocks)

        # Upsample factors to resize features to [1/4, 1/8, 1/16, 1/32, ...] scales
        scale_factors = [
            stride / 2 ** (i + 2) for i, stride in enumerate(encoder_output_strides)
        ]
        self.reassemble_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = ReassembleBlock(
                in_channels=encoder_out_channels[i],
                mid_channels=intermediate_channels[i],
                out_channels=fusion_channels,
                upsample_factor=scale_factors[i],
            )
            self.reassemble_blocks.append(block)

        # Fusion blocks to fuse the processed features in a sequential manner
        fusion_blocks = [FusionBlock(fusion_channels) for _ in range(num_blocks)]
        self.fusion_blocks = nn.ModuleList(fusion_blocks)

    def forward(
        self, features: list[torch.Tensor], cls_tokens: list[Optional[torch.Tensor]]
    ) -> torch.Tensor:
        # Process the encoder features to scale of [1/4, 1/8, 1/16, 1/32, ...]
        processed_features = []
        for i, (feature, cls_token) in enumerate(zip(features, cls_tokens)):
            readout_feature = self.readout_blocks[i](feature, cls_token)
            processed_feature = self.reassemble_blocks[i](readout_feature)
            processed_features.append(processed_feature)

        # Fusion and progressive upsampling starting from the last processed feature
        previous_feature = None
        processed_features = processed_features[::-1]
        for fusion_block, feature in zip(self.fusion_blocks, processed_features):
            fused_feature = fusion_block(feature, previous_feature)
            previous_feature = fused_feature

        return fused_feature


class DPTSegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = None,
        kernel_size: int = 3,
        upsampling: float = 2.0,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        self.activation = Activation(activation)
        self.upsampling_factor = upsampling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_output = self.head(x)
        resized_output = nn.functional.interpolate(
            head_output,
            scale_factor=self.upsampling_factor,
            mode="bilinear",
            align_corners=True,
        )
        activation_output = self.activation(resized_output)
        return activation_output
