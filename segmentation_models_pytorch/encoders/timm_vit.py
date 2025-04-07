from typing import Any, Optional

import timm
import torch
import torch.nn as nn

from .timm_universal import _merge_kwargs_no_duplicates


def sample_block_indices_uniformly(n: int, total_num_blocks: int) -> list[int]:
    """
    Sample N block indices uniformly from the total number of blocks.
    """
    return [
        int(total_num_blocks / n * block_depth) - 1 for block_depth in range(1, n + 1)
    ]


def validate_output_indices(
    output_indices: list[int], model_num_blocks: int, depth: int
):
    """
    Validate the output indices are within the valid range of the model and the
    length of the output indices is equal to the depth of the encoder.
    """
    for output_index in output_indices:
        if output_index < -model_num_blocks or output_index >= model_num_blocks:
            raise ValueError(
                f"Output indices for feature extraction should be in range "
                f"[-{model_num_blocks}, {model_num_blocks}), because the model has {model_num_blocks} blocks, "
                f"got index = {output_index}."
            )


def preprocess_output_indices(
    output_indices: Optional[list[int]], model_num_blocks: int, depth: int
) -> list[int]:
    """
    Preprocess the output indices for the encoder.
    """

    # Refine encoder output indices
    if output_indices is None:
        output_indices = sample_block_indices_uniformly(depth, model_num_blocks)
    elif not isinstance(output_indices, (list, tuple)):
        raise ValueError(
            f"`output_indices` for encoder should be a list/tuple/None, got {type(output_indices)}"
        )
    validate_output_indices(output_indices, model_num_blocks, depth)

    return output_indices


class TimmViTEncoder(nn.Module):
    """
    A universal encoder leveraging the `timm` library for feature extraction from
    ViT style models

    Features:
        - Supports configurable depth.
        - Ensures consistent multi-level feature extraction across all ViT models.
    """

    # prefix tokens are not supported for scripting
    _is_torch_scriptable = False
    _is_torch_exportable = True
    _is_torch_compilable = True

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 4,
        output_indices: Optional[list[int]] = None,
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
        super().__init__()

        if depth < 1:
            raise ValueError(f"`encoder_depth` should be greater than 1, got {depth}.")

        # Output stride validation needed for smp encoder test consistency
        output_stride = kwargs.pop("output_stride", None)
        if output_stride is not None:
            raise ValueError("Dilated mode not supported, set output stride to None")

        if isinstance(output_indices, (list, tuple)) and len(output_indices) != depth:
            raise ValueError(
                f"Length of output indices for feature extraction should be equal to the depth of the encoder "
                f"architecture, got output indices length - {len(output_indices)}, encoder depth - {depth}"
            )

        self.name = name

        # Load a timm model
        encoder_kwargs = dict(in_chans=in_channels, pretrained=pretrained)
        encoder_kwargs = _merge_kwargs_no_duplicates(encoder_kwargs, kwargs)
        self.model = timm.create_model(name, **encoder_kwargs)

        if not hasattr(self.model, "forward_intermediates"):
            raise ValueError(
                f"Encoder `{name}` does not support `forward_intermediates` for feature extraction. "
                f"Please update `timm` or use another encoder."
            )

        # Get all the necessary information about the model
        feature_info = self.model.feature_info

        # Additional checks
        model_num_blocks = len(feature_info)
        if depth > model_num_blocks:
            raise ValueError(
                f"Depth of the encoder cannot exceed the number of blocks in the model "
                f"got {depth} depth, model has {model_num_blocks} blocks"
            )

        # Preprocess the output indices, uniformly sample from model_num_blocks if None
        output_indices = preprocess_output_indices(
            output_indices, model_num_blocks, depth
        )

        # Private attributes for model forward
        self._num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self._has_cls_token = getattr(self.model, "has_cls_token", False)
        self._output_indices = output_indices

        # Public attributes
        self.output_strides = [feature_info[i]["reduction"] for i in output_indices]
        self.output_stride = self.output_strides[-1]
        self.out_channels = [feature_info[i]["num_chs"] for i in output_indices]
        self.has_prefix_tokens = self._num_prefix_tokens > 0
        self.input_size = self.model.pretrained_cfg.get("input_size", None)
        self.is_fixed_input_size = self.model.pretrained_cfg.get(
            "fixed_input_size", False
        )

    def _forward_with_prefix_tokens(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        intermediate_outputs = self.model.forward_intermediates(
            x,
            indices=self._output_indices,
            intermediates_only=True,
            return_prefix_tokens=True,
        )

        features = [output[0] for output in intermediate_outputs]
        prefix_tokens = [output[1] for output in intermediate_outputs]

        return features, prefix_tokens

    def _forward_without_prefix_tokens(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.model.forward_intermediates(
            x,
            indices=self._output_indices,
            intermediates_only=True,
        )
        return features

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[Optional[torch.Tensor]]]:
        """
        Forward pass to extract multi-stage features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Tuple of feature maps and cls tokens (if supported) at different scales.
        """

        if self.has_prefix_tokens:
            features, prefix_tokens = self._forward_with_prefix_tokens(x)
        else:
            features = self._forward_without_prefix_tokens(x)
            prefix_tokens = [None] * len(features)

        return features, prefix_tokens
