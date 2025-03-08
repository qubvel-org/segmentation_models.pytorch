from typing import Any, Optional, Union

import timm
import torch
import torch.nn as nn


class TimmViTEncoder(nn.Module):
    """
    A universal encoder leveraging the `timm` library for feature extraction from
    ViT style models

    Features:
        - Supports configurable depth.
        - Ensures consistent multi-level feature extraction across all ViT models.
    """

    _is_torch_scriptable = False
    _is_torch_exportable = True
    _is_torch_compilable = False

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 4,
        output_indices: Optional[Union[list[int], int]] = None,
        **kwargs: dict[str, Any],
    ):
        """
        Initialize the encoder.

        Args:
            name (str): ViT model name to load from `timm`.
            pretrained (bool): Load pretrained weights (default: True).
            in_channels (int): Number of input channels (default: 3 for RGB).
            depth (int): Number of feature stages to extract (default: 4).
            output_indices (Optional[list[int] | int]): Indices of blocks in the model to be used for feature extraction.
            **kwargs: Additional arguments passed to `timm.create_model`.
        """
        # At the moment we do not support models with more than 4 stages,
        # but can be reconfigured in the future.
        if depth > 4 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 4], got {depth}"
            )

        super().__init__()
        self.name = name

        output_stride = kwargs.pop("output_stride", None)
        if output_stride is not None:
            raise ValueError("Dilated mode not supported, set output stride to None")

        # Default model configuration for feature extraction
        common_kwargs = dict(
            in_chans=in_channels,
            features_only=False,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # Load a temporary model to analyze its feature hierarchy
        try:
            with torch.device("meta"):
                tmp_model = timm.create_model(name)
        except Exception:
            tmp_model = timm.create_model(name)

        # Check if model output is in channel-last format (NHWC)
        self._is_channel_last = getattr(tmp_model, "output_fmt", None) == "NHWC"

        feature_info = tmp_model.feature_info
        model_num_blocks = len(feature_info)

        if output_indices is not None:
            if isinstance(output_indices, int):
                output_indices = list(output_indices)

            for output_index in output_indices:
                if output_indices < 0 or output_indices > model_num_blocks:
                    raise ValueError(
                        f"Output indices for feature extraction should be greater than 0 and less \
                                     than the number of blocks in the model ({model_num_blocks}), got {output_index}"
                    )

            if len(output_indices) != depth:
                raise ValueError(
                    f"Length of output indices for feature extraction should be equal to the depth of the encoder\
                                  architecture, got output indices length - {len(output_indices)}, encoder depth - {depth}"
                )

        if depth > model_num_blocks:
            raise ValueError(
                f"Depth of the encoder cannot exceed the number of blocks in the model \
                               got {depth} depth, model has {model_num_blocks} blocks"
            )

        if output_indices is None:
            output_indices = [
                int((model_num_blocks / 4) * index) - 1 for index in range(1, depth + 1)
            ]

        common_kwargs["out_indices"] = self.out_indices = output_indices
        feature_info_obj = timm.models.FeatureInfo(
            feature_info=feature_info, out_indices=output_indices
        )

        # Determine the model's downsampling pattern and set hierarchy flags
        reduction_scales = list(feature_info_obj.reduction())

        allow_downsampling = kwargs.pop("allow_downsampling", True)
        allow_output_stride_not_power_of_two = kwargs.pop(
            "allow_output_stride_not_power_of_two", True
        )
        # Raise an error if downsampling is not allowed and encoder outputs have progressive downsampling
        if len(set(reduction_scales)) > 1 and not allow_downsampling:
            raise ValueError("Unsupported model downsampling pattern.")

        self._output_stride = reduction_scales[0]

        if (
            bin(self._output_stride).count("1") != 1
            and not allow_output_stride_not_power_of_two
        ):
            raise ValueError(
                f"Models with stride which is not a power of 2 are not supported, \
                              got output stride {self._output_stride}"
            )

        self.cls_token_supported = getattr(tmp_model, "has_class_token", False)
        self.num_prefix_tokens = getattr(tmp_model, "num_prefix_tokens", 0)

        self.model = timm.create_model(
            name, **_merge_kwargs_no_duplicates(common_kwargs, kwargs)
        )

        self._out_channels = feature_info_obj.channels()
        self._in_channels = in_channels
        self._depth = depth
        self._embed_dim = tmp_model.embed_dim

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass to extract multi-stage features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Tuple of feature maps and cls tokens (if supported) at different scales.
        """
        intermediate_outputs = self.model.forward_intermediates(
            x,
            indices=self.out_indices,
            return_prefix_tokens=True,
            intermediates_only=True,
        )

        cls_tokens = [None] * len(self.out_indices)

        if self.num_prefix_tokens > 0:
            features, prefix_tokens = zip(*intermediate_outputs)
            if self.cls_token_supported:
                if self.num_prefix_tokens == 1:
                    cls_tokens = prefix_tokens

                elif self.num_prefix_tokens > 1:
                    cls_tokens = [
                        prefix_token[:, 0, :] for prefix_token in prefix_tokens
                    ]

        else:
            features = intermediate_outputs

        return features, cls_tokens

    @property
    def embed_dim(self) -> int:
        """
        Returns the embedding dimension for the ViT encoder.

        Returns:
            int: Embedding dimension.
        """
        return self._embed_dim

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
        return self._output_stride

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
