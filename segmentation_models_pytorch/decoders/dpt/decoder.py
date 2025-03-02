import torch
import torch.nn as nn


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


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        return torch.transpose(x, dim0=self.dim0, dim1=self.dim1)


class ProjectionReadout(nn.Module):
    """
    Concatenates the cls tokens with the features to make use of the global information aggregated in the cls token.
    Projects the combined feature map to the original embedding dimension using a MLP
    """

    def __init__(self, in_features: int, encoder_output_stride: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_features=2 * in_features, out_features=in_features), nn.GELU()
        )

        self.flatten = nn.Flatten(start_dim=2)
        self.transpose = Transpose(dim0=1, dim1=2)
        self.encoder_output_stride = encoder_output_stride

    def forward(self, feature: torch.Tensor, cls_token: torch.Tensor):
        batch_size, _, height_dim, width_dim = feature.shape
        feature = self.flatten(feature)
        feature = self.transpose(feature)

        cls_token = cls_token.expand_as(feature)

        features = torch.cat([feature, cls_token], dim=2)
        features = self.project(features)
        features = self.transpose(features)

        features = features.view(batch_size, -1, height_dim, width_dim)
        return features


class IgnoreReadout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature: torch.Tensor, cls_token: torch.Tensor):
        return feature


class FeatureProcessBlock(nn.Module):
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
            in_channels=out_channel, out_channels=feature_dim, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor):
        x = self.project_to_out_channel(x)
        x = self.upsample(x)
        x = self.project_to_feature_dim(x)

        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=feature_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=feature_dim,
                out_channels=feature_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=feature_dim),
        )

    def forward(self, x: torch.Tensor):
        return x + self.conv_block(x)


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
        feature = self.activation(feature)

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
        prefix_token_supported: bool = False,
    ):
        super().__init__()

        self.prefix_token_supported = prefix_token_supported

        # If encoder has cls token, then concatenate it with the features along the embedding dimension and project it
        # back to the feature_dim dimension. Else, ignore the non-existent cls token

        if prefix_token_supported:
            self.readout_blocks = nn.ModuleList(
                [
                    ProjectionReadout(
                        in_features=transformer_embed_dim,
                        encoder_output_stride=encoder_output_stride,
                    )
                    for _ in range(encoder_depth)
                ]
            )
        else:
            self.readout_blocks = [IgnoreReadout() for _ in range(encoder_depth)]

        upsample_factors = [
            (encoder_output_stride / 2 ** (index + 2))
            for index in range(0, encoder_depth)
        ]
        feature_processing_out_channels = _get_feature_processing_out_channels(
            encoder_name
        )
        if encoder_depth < len(feature_processing_out_channels):
            feature_processing_out_channels = feature_processing_out_channels[
                :encoder_depth
            ]

        self.feature_processing_blocks = nn.ModuleList(
            [
                FeatureProcessBlock(
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
        self, encoder_output: list[list[torch.Tensor], list[torch.Tensor]]
    ) -> torch.Tensor:
        features, cls_tokens = encoder_output
        processed_features = []

        # Process the encoder features to scale of [1/32,1/16,1/8,1/4]
        for index, (feature, cls_token) in enumerate(zip(features, cls_tokens)):
            readout_feature = self.readout_blocks[index](feature, cls_token)
            processed_feature = self.feature_processing_blocks[index](readout_feature)
            processed_features.append(processed_feature)

        preceding_layer_feature = None

        # Fusion and progressive upsampling starting from the last processed feature
        processed_features = processed_features[::-1]
        for fusion_block, feature in zip(self.fusion_blocks, processed_features):
            out = fusion_block(feature, preceding_layer_feature)
            preceding_layer_feature = out

        return out
