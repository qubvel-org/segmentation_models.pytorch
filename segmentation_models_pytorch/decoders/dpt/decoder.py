import torch
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from typing import Optional


def _get_feature_processing_out_channels(encoder_name: str) -> list[int]:
    """
    Get the output embedding dimensions for the features after decoder processing
    """

    encoder_name = encoder_name.lower()
    # Output channels for hybrid ViT encoder after feature processing
    if "vit" in encoder_name and "resnet" in encoder_name:
        return [256, 512, 768, 768]

    # Output channels for ViT-large,ViT-huge,ViT-giant encoders after feature processing
    if "vit" in encoder_name and any(
        [variant in encoder_name for variant in ["huge", "large", "giant"]]
    ):
        return [256, 512, 1024, 1024]

    # Output channels for ViT-base and other encoders after feature processing
    return [96, 192, 384, 768]


class ProjectionReadout(nn.Module):
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
        self.has_cls_token = has_cls_token

    def forward(self, features: torch.Tensor, cls_token: Optional[torch.Tensor] = None):
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
        self, embed_dim: int, feature_dim: int, out_channel: int, upsample_factor: int
    ):
        super().__init__()

        self.project_to_out_channel = nn.Conv2d(
            in_channels=embed_dim, out_channels=out_channel, kernel_size=1
        )

        if upsample_factor > 1.0:
            self.upsample = nn.ConvTranspose2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=int(upsample_factor),
                stride=int(upsample_factor),
            )

        elif upsample_factor == 1.0:
            self.upsample = nn.Identity()

        else:
            self.upsample = nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=int(1 / upsample_factor),
                padding=1,
            )

        self.project_to_feature_dim = nn.Conv2d(
            in_channels=out_channel,
            out_channels=feature_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
        activated_x_1 = self.activation(x)
        conv_1_out = self.conv_1(activated_x_1)
        batch_norm_1_out = self.batch_norm_1(conv_1_out)
        activated_x_2 = self.activation(batch_norm_1_out)
        conv_2_out = self.conv_2(activated_x_2)
        batch_norm_2_out = self.batch_norm_2(conv_2_out)

        return x + batch_norm_2_out


class FusionBlock(nn.Module):
    """
    Fuses the processed encoder features in a residual manner and upsamples them
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.residual_conv_block1 = ResidualConvBlock(feature_dim=feature_dim)
        self.residual_conv_block2 = ResidualConvBlock(feature_dim=feature_dim)
        self.project = nn.Conv2d(
            in_channels=feature_dim, out_channels=feature_dim, kernel_size=1
        )
        self.activation = nn.ReLU()

    def forward(self, feature: torch.Tensor, preceding_layer_feature: torch.Tensor):
        feature = self.residual_conv_block1(feature)

        if preceding_layer_feature is not None:
            feature += preceding_layer_feature

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
    [1/32,1/16,1/8,1/4] relative to the input image spatial dimension.

    The decoder then fuses these features in a residual manner and progressively upsamples them by a factor of 2 so that the
    output has a downsampling ratio of 1/2 relative to the input image spatial dimension

    """

    def __init__(
        self,
        encoder_name: str,
        transformer_embed_dim: int,
        encoder_output_stride: int,
        feature_dim: int = 256,
        encoder_depth: int = 4,
        cls_token_supported: bool = False,
    ):
        super().__init__()

        self.cls_token_supported = cls_token_supported

        # If encoder has cls token, then concatenate it with the features along the embedding dimension and project it
        # back to the feature_dim dimension. Else, ignore the non-existent cls token
        self.readout_blocks = nn.ModuleList()
        for _ in range(encoder_depth):
            block = ProjectionReadout(
                embed_dim=transformer_embed_dim,
                has_cls_token=cls_token_supported,
            )
            self.readout_blocks.append(block)

        upsample_factors = [
            (encoder_output_stride / 2 ** (index + 2))
            for index in range(0, encoder_depth)
        ]
        feature_processing_out_channels = _get_feature_processing_out_channels(
            encoder_name
        )

        # slice in case encoder_depth < len(feature_processing_out_channels)
        feature_processing_out_channels = feature_processing_out_channels[
            :encoder_depth
        ]

        self.reassemble_blocks = nn.ModuleList(
            [
                ReassembleBlock(
                    transformer_embed_dim, feature_dim, out_channel, upsample_factor
                )
                for upsample_factor, out_channel in zip(
                    upsample_factors, feature_processing_out_channels
                )
            ]
        )

        self.fusion_blocks = nn.ModuleList(
            [FusionBlock(feature_dim=feature_dim) for _ in range(encoder_depth)]
        )

    def forward(
        self, features: list[torch.Tensor], cls_tokens: list[torch.Tensor]
    ) -> torch.Tensor:
        processed_features = []

        # Process the encoder features to scale of [1/32,1/16,1/8,1/4]
        for index, (feature, cls_token) in enumerate(zip(features, cls_tokens)):
            readout_feature = self.readout_blocks[index](feature, cls_token)
            processed_feature = self.reassemble_blocks[index](readout_feature)
            processed_features.append(processed_feature)

        preceding_layer_feature = None

        # Fusion and progressive upsampling starting from the last processed feature
        processed_features = processed_features[::-1]
        for fusion_block, feature in zip(self.fusion_blocks, processed_features):
            out = fusion_block(feature, preceding_layer_feature)
            preceding_layer_feature = out

        return out


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
