"""
TimmUniversalEncoder provides a unified feature extraction interface built on the
`timm` library, supporting both traditional-style (e.g., ResNet) and transformer-style
models (e.g., Swin Transformer, ConvNeXt).

This encoder produces consistent multi-level feature maps for semantic segmentation tasks.
It allows configuring the number of feature extraction stages (`depth`) and adjusting
`output_stride` when supported.

Key Features:
- Flexible model selection using `timm.create_model`.
- Unified multi-level output across different model hierarchies.
- Automatic alignment for inconsistent feature scales:
  - Transformer-style models (start at 1/4 scale): Insert dummy features for 1/2 scale.
  - VGG-style models (include scale-1 features): Align outputs for compatibility.
- Easy access to feature scale information via the `reduction` property.

Feature Scale Differences:
- Traditional-style models (e.g., ResNet): Scales at 1/2, 1/4, 1/8, 1/16, 1/32.
- Transformer-style models (e.g., Swin Transformer): Start at 1/4 scale, skip 1/2 scale.
- VGG-style models: Include scale-1 features (input resolution).

Notes:
- `output_stride` is unsupported in some models, especially transformer-based architectures.
- Special handling for models like TResNet and DLA to ensure correct feature indexing.
- VGG-style models use `_is_vgg_style` to align scale-1 features with standard outputs.
"""

from typing import Any

import timm
import torch
import torch.nn as nn


class TimmUniversalEncoder(nn.Module):
    """
    A universal encoder leveraging the `timm` library for feature extraction from
    various model architectures, including traditional-style and transformer-style models.

    Features:
        - Supports configurable depth and output stride.
        - Ensures consistent multi-level feature extraction across diverse models.
        - Compatible with convolutional and transformer-like backbones.
    """

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

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
            name (str): Model name to load from `timm`.
            pretrained (bool): Load pretrained weights (default: True).
            in_channels (int): Number of input channels (default: 3 for RGB).
            depth (int): Number of feature stages to extract (default: 5).
            output_stride (int): Desired output stride (default: 32).
            **kwargs: Additional arguments passed to `timm.create_model`.
        """
        # At the moment we do not support models with more than 5 stages,
        # but can be reconfigured in the future.
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )

        super().__init__()
        self.name = name

        # Default model configuration for feature extraction
        common_kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # Ｎot all models support output stride argument, drop it by default
        if output_stride == 32:
            common_kwargs.pop("output_stride")

        # Load a temporary model to analyze its feature hierarchy
        try:
            with torch.device("meta"):
                tmp_model = timm.create_model(name, features_only=True)
        except Exception:
            tmp_model = timm.create_model(name, features_only=True)

        # Check if model output is in channel-last format (NHWC)
        self._is_channel_last = getattr(tmp_model, "output_fmt", None) == "NHWC"

        # Determine the model's downsampling pattern and set hierarchy flags
        encoder_stage = len(tmp_model.feature_info.reduction())
        reduction_scales = list(tmp_model.feature_info.reduction())

        if reduction_scales == [2 ** (i + 2) for i in range(encoder_stage)]:
            # Transformer-style downsampling: scales (4, 8, 16, 32)
            self._is_transformer_style = True
            self._is_vgg_style = False
        elif reduction_scales == [2 ** (i + 1) for i in range(encoder_stage)]:
            # Traditional-style downsampling: scales (2, 4, 8, 16, 32)
            self._is_transformer_style = False
            self._is_vgg_style = False
        elif reduction_scales == [2**i for i in range(encoder_stage)]:
            # Vgg-style models including scale 1: scales (1, 2, 4, 8, 16, 32)
            self._is_transformer_style = False
            self._is_vgg_style = True
        else:
            raise ValueError("Unsupported model downsampling pattern.")

        if self._is_transformer_style:
            # Transformer-like models (start at scale 4)
            if "tresnet" in name:
                # 'tresnet' models start feature extraction at stage 1,
                # so out_indices=(1, 2, 3, 4) for depth=5.
                common_kwargs["out_indices"] = tuple(range(1, depth))
            else:
                # Most transformer-like models use out_indices=(0, 1, 2, 3) for depth=5.
                common_kwargs["out_indices"] = tuple(range(depth - 1))

            timm_model_kwargs = _merge_kwargs_no_duplicates(common_kwargs, kwargs)
            self.model = timm.create_model(name, **timm_model_kwargs)

            # Add a dummy output channel (0) to align with traditional encoder structures.
            self._out_channels = (
                [in_channels] + [0] + self.model.feature_info.channels()
            )
        else:
            if "dla" in name:
                # For 'dla' models, out_indices starts at 0 and matches the input size.
                common_kwargs["out_indices"] = tuple(range(1, depth + 1))
            if self._is_vgg_style:
                common_kwargs["out_indices"] = tuple(range(depth + 1))

            self.model = timm.create_model(
                name, **_merge_kwargs_no_duplicates(common_kwargs, kwargs)
            )

            if self._is_vgg_style:
                self._out_channels = self.model.feature_info.channels()
            else:
                self._out_channels = [in_channels] + self.model.feature_info.channels()

        self._in_channels = in_channels
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass to extract multi-stage features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            list[torch.Tensor]: List of feature maps at different scales.
        """
        features = self.model(x)

        # Convert NHWC to NCHW if needed
        if self._is_channel_last:
            features = [
                feature.permute(0, 3, 1, 2).contiguous() for feature in features
            ]

        # Add dummy feature for scale 1/2 if missing (transformer-style models)
        if self._is_transformer_style:
            B, _, H, W = x.shape
            dummy = torch.empty([B, 0, H // 2, W // 2], dtype=x.dtype, device=x.device)
            features = [dummy] + features

        # Add input tensor as scale 1 feature if `self._is_vgg_style` is False
        if not self._is_vgg_style:
            features = [x] + features

        return features

    @property
    def out_channels(self) -> list[int]:
        """
        Returns the number of output channels for each feature stage.

        Returns:
            list[int]: A list of channel dimensions at each scale.
        """
        return self._out_channels

    @property
    def output_stride(self) -> int:
        """
        Returns the effective output stride based on the model depth.

        Returns:
            int: The effective output stride.
        """
        return int(min(self._output_stride, 2**self._depth))

    def load_state_dict(self, state_dict, **kwargs):
        # for compatibility of weights for
        # timm- ported encoders with TimmUniversalEncoder
        patterns = ["regnet", "res2", "resnest", "mobilenetv3", "gernet"]

        is_deprecated_encoder = any(
            self.name.startswith(pattern) for pattern in patterns
        )

        if is_deprecated_encoder:
            keys = list(state_dict.keys())
            for key in keys:
                new_key = key
                if not key.startswith("model."):
                    new_key = "model." + key
                if "gernet" in self.name:
                    new_key = new_key.replace(".stages.", ".stages_")
                state_dict[new_key] = state_dict.pop(key)

        return super().load_state_dict(state_dict, **kwargs)


def _merge_kwargs_no_duplicates(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two dictionaries, ensuring no duplicate keys exist.

    Args:
        a (dict): Base dictionary.
        b (dict): Additional parameters to merge.

    Returns:
        dict: A merged dictionary.
    """
    duplicates = a.keys() & b.keys()
    if duplicates:
        raise ValueError(f"'{duplicates}' already specified internally")

    return a | b
