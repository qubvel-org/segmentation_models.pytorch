"""
TimmUniversalEncoder provides a unified feature extraction interface built on the
`timm` library, supporting various backbone architectures, including traditional
CNNs (e.g., ResNet) and models adopting a transformer-like feature hierarchy
(e.g., Swin Transformer, ConvNeXt).

This encoder produces standardized multi-level feature maps, facilitating integration
with semantic segmentation tasks. It allows configuring the number of feature extraction
stages (`depth`) and adjusting `output_stride` when supported.

Key Features:
- Flexible model selection through `timm.create_model`.
- A unified interface that outputs consistent, multi-level features even if the
  underlying model differs in its feature hierarchy.
- Automatic alignment: If a model lacks certain early-stage features (for example,
  modern architectures that start from a 1/4 scale rather than 1/2 scale), the encoder
  inserts dummy features to maintain consistency with traditional CNN structures.
- Easy access to channel information: Use the `out_channels` property to retrieve
  the number of channels at each feature stage.

Feature Scale Differences:
- Traditional CNNs (e.g., ResNet) typically provide features at 1/2, 1/4, 1/8, 1/16,
  and 1/32 scales.
- Transformer-style or next-generation models (e.g., Swin Transformer, ConvNeXt) often
  start from the 1/4 scale (then 1/8, 1/16, 1/32), omitting the initial 1/2 scale
  feature. TimmUniversalEncoder compensates for this omission to ensure a unified
  multi-stage output.

Notes:
- Not all models support modifying `output_stride` (especially transformer-based or
  transformer-like models).
- Certain models (e.g., TResNet, DLA) require special handling to ensure correct
  feature indexing.
- Most `timm` models output features in (B, C, H, W) format. However, some
  (e.g., MambaOut and certain Swin/SwinV2 variants) use (B, H, W, C) format, which is
  currently unsupported.
"""

from typing import Any

import timm
import torch
import torch.nn as nn


class TimmUniversalEncoder(nn.Module):
    """
    A universal encoder built on the `timm` library, designed to adapt to a wide variety of
    model architectures, including both traditional CNNs and those that follow a
    transformer-like hierarchy.

    Features:
    - Supports flexible depth and output stride for feature extraction.
    - Automatically adjusts to input/output channel structures based on the model type.
    - Compatible with both convolutional and transformer-like encoders.
    """

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the encoder.

        Args:
            name (str): Name of the model to be loaded from the `timm` library.
            pretrained (bool): If True, loads pretrained weights.
            in_channels (int): Number of input channels (default: 3 for RGB).
            depth (int): Number of feature extraction stages (default: 5).
            output_stride (int): Desired output stride (default: 32).
            **kwargs: Additional keyword arguments for `timm.create_model`.
        """
        super().__init__()

        common_kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        if output_stride == 32:
            common_kwargs.pop("output_stride")

        # Load a preliminary model to determine its feature hierarchy structure.
        self.model = timm.create_model(name, features_only=True)

        # Determine if this model uses a transformer-like hierarchy (i.e., starting at 1/4 scale)
        # rather than a traditional CNN hierarchy (starting at 1/2 scale).
        if len(self.model.feature_info.channels()) == 5:
            self._is_transformer_style = False
        else:
            self._is_transformer_style = True

        if self._is_transformer_style:
            if "tresnet" in name:
                # 'tresnet' models start feature extraction at stage 1,
                # so out_indices=(1, 2, 3, 4) for depth=5.
                common_kwargs["out_indices"] = tuple(range(1, depth))
            else:
                # Most transformer-like models use out_indices=(0, 1, 2, 3) for depth=5.
                common_kwargs["out_indices"] = tuple(range(depth - 1))

            self.model = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs, kwargs)
            )
            # Add a dummy output channel (0) to align with traditional encoder structures.
            self._out_channels = (
                [in_channels] + [0] + self.model.feature_info.channels()
            )
        else:
            if "dla" in name:
                # For 'dla' models, out_indices starts at 0 and matches the input size.
                kwargs["out_indices"] = tuple(range(1, depth + 1))

            self.model = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs, kwargs)
            )
            self._out_channels = [in_channels] + self.model.feature_info.channels()

        self._in_channels = in_channels
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Pass the input through the encoder and return extracted features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            list[torch.Tensor]: A list of feature maps extracted at various scales.
        """
        features = self.model(x)

        if self._is_transformer_style:
            # Models using a transformer-like hierarchy may not generate
            # all expected feature maps. Insert a dummy feature map to ensure
            # compatibility with decoders expecting a 5-level pyramid.
            B, _, H, W = x.shape
            dummy = torch.empty([B, 0, H // 2, W // 2], dtype=x.dtype, device=x.device)
            features = [x] + [dummy] + features
        else:
            features = [x] + features

        return features

    @property
    def out_channels(self) -> list[int]:
        """
        Returns:
            list[int]: A list of output channels for each stage of the encoder,
            including the input channels at the first stage.
        """
        return self._out_channels

    @property
    def output_stride(self) -> int:
        """
        Returns:
            int: The effective output stride of the encoder, considering the depth.
        """
        return min(self._output_stride, 2**self._depth)


def _merge_kwargs_no_duplicates(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    duplicates = a.keys() & b.keys()
    if duplicates:
        raise ValueError(f"'{duplicates}' already specified internally")

    return a | b
