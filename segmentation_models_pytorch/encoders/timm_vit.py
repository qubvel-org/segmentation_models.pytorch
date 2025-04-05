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

    if len(output_indices) != depth:
        raise ValueError(
            f"Length of output indices for feature extraction should be equal to the depth of the encoder "
            f"architecture, got output indices length - {len(output_indices)}, encoder depth - {depth}"
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

    _is_torch_scriptable = False
    _is_torch_exportable = True
    _is_torch_compilable = False

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
        # At the moment we do not support models with more than 4 stages,
        # but can be reconfigured in the future.
        if depth > 4 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 4], got {depth}"
            )

        super().__init__()
        self.name = name

        # Load a temporary model to analyze its feature hierarchy
        try:
            with torch.device("meta"):
                tmp_model = timm.create_model(name)
        except Exception:
            tmp_model = timm.create_model(name)

        # Get all the necessary information about the model, and delete the temporary model
        self._is_channel_last = getattr(tmp_model, "output_fmt", None) == "NHWC"
        feature_info = tmp_model.feature_info

        del tmp_model

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

        # Determine the model's downsampling pattern and set hierarchy flags
        reduction_scales = [feature_info[i]["reduction"] for i in output_indices]

        # We only support the same reduction scales for ViT encoder, e.g. [16, 16, 16], and not [16, 8, 4]
        if len(set(reduction_scales)) > 1:
            raise ValueError(
                f"We only support the same reduction scales for ViT encoder, e.g. [16, 16, 16], and not {reduction_scales}"
            )

        # Initiate timm model
        model_kwargs = dict(in_chans=in_channels, pretrained=pretrained)
        model_kwargs = _merge_kwargs_no_duplicates(model_kwargs, kwargs)
        self.model = timm.create_model(name, **model_kwargs)

        # Private attributes for model forward
        self._num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self._output_indices = output_indices

        # Public attributes
        self.output_stride = reduction_scales[-1]
        self.out_channels = [feature_info[i]["num_chs"] for i in output_indices]
        self.embed_dim = self.model.embed_dim
        self.has_class_token = getattr(self.model, "has_class_token", False)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[Optional[torch.Tensor]]]:
        """
        Forward pass to extract multi-stage features.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: Tuple of feature maps and cls tokens (if supported) at different scales.
        """
        intermediate_outputs = self.model.forward_intermediates(
            x,
            indices=self._output_indices,
            return_prefix_tokens=True,
            intermediates_only=True,
        )

        # Split to features and prefix tokens
        if self._num_prefix_tokens > 0:
            features = [output[0] for output in intermediate_outputs]
            prefix_tokens = [output[1] for output in intermediate_outputs]
        else:
            features = intermediate_outputs
            prefix_tokens = None

        # Get CLS token from prefix tokens
        if self.has_class_token and self._num_prefix_tokens == 1:
            cls_tokens = prefix_tokens
        elif self.has_class_token and self._num_prefix_tokens > 1:
            cls_tokens = [x[:, 0, :] for x in prefix_tokens]
        else:
            cls_tokens = [None] * len(self._output_indices)

        return features, cls_tokens
