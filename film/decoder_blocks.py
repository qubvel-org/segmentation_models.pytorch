"""Decoder-side FiLM: a forked/subclassed copy of SMP's ``UnetDecoder``, with
FiLM wired in after batchnorm and before activation — matching the placement
used in the original FiLM paper. Lives entirely in ``film/``; SMP's own
``decoders/unet/decoder.py`` is never imported for modification, only
generic, unmodified helpers (``get_norm_layer``, ``Attention``,
``UnetCenterBlock``) are reused as-is.

The decoder has no pretrained weights of its own (only the encoder does), so
swapping SMP's ``UnetDecoder`` for this FiLM-augmented equivalent is safe —
it trains from scratch either way, exactly like the vendor decoder would.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base.modules import Attention, get_norm_layer
from segmentation_models_pytorch.decoders.unet.decoder import UnetCenterBlock

from .generator import FiLMGenerator, apply_film
from .hooks import MetaBox


class FiLMConv2dReLU(nn.Module):
    """Fork of ``segmentation_models_pytorch.base.modules.Conv2dReLU``, split
    out of its ``nn.Sequential`` so FiLM can sit between norm and activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        meta_dim: int,
        meta_box: MetaBox,
        padding: int = 0,
        stride: int = 1,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    ):
        super().__init__()
        is_inplace_abn = use_norm == "inplace" or (
            isinstance(use_norm, dict) and use_norm.get("type") == "inplace"
        )
        if is_inplace_abn:
            raise NotImplementedError(
                "use_norm='inplace' (InPlaceABN) fuses its own activation into the norm layer, "
                "which leaves no post-norm/pre-activation seam for FiLM to attach to — "
                "use 'batchnorm', 'identity', 'layernorm', or 'instancenorm' with FiLM decoder blocks."
            )
        self.meta_box = meta_box
        norm = get_norm_layer(use_norm, out_channels)
        is_identity = isinstance(norm, nn.Identity)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=is_identity,
        )
        self.norm = norm
        self.film = FiLMGenerator(meta_dim, out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        m = self.meta_box.value
        if m is None:
            raise RuntimeError(
                "FiLM decoder block ran with no metadata set — call FiLMUnet.forward(x, metadata), "
                "not the wrapped Unet/decoder directly"
            )
        gamma, beta = self.film(m)
        x = apply_film(x, gamma, beta)
        x = self.activation(x)
        return x


class FiLMUnetDecoderBlock(nn.Module):
    """Fork of ``UnetDecoderBlock``: identical structure, `conv1`/`conv2` are
    ``FiLMConv2dReLU`` instead of ``Conv2dReLU`` — each is its own FiLM
    insertion point, independently auditable (see audit.py).
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        meta_dim: int,
        meta_box: MetaBox,
        use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
        attention_type: Optional[str] = None,
        interpolation_mode: str = "nearest",
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.conv1 = FiLMConv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            meta_dim=meta_dim,
            meta_box=meta_box,
            use_norm=use_norm,
        )
        self.attention1 = Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = FiLMConv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            meta_dim=meta_dim,
            meta_box=meta_box,
            use_norm=use_norm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

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


class FiLMUnetDecoder(nn.Module):
    """Fork of ``UnetDecoder``: same channel-plumbing logic, ``FiLMUnetDecoderBlock``
    in place of ``UnetDecoderBlock``. ``add_center_block`` still reuses the
    vendor ``UnetCenterBlock`` unmodified (vgg-style encoders only — no FiLM
    insertion point there, out of scope for this pass).
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        meta_dim: int,
        meta_box: MetaBox,
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

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if add_center_block:
            self.center = UnetCenterBlock(
                head_channels, head_channels, use_norm=use_norm
            )
        else:
            self.center = nn.Identity()

        self.blocks = nn.ModuleList()
        for block_in_channels, block_skip_channels, block_out_channels in zip(
            in_channels, skip_channels, out_channels
        ):
            block = FiLMUnetDecoderBlock(
                block_in_channels,
                block_skip_channels,
                block_out_channels,
                meta_dim=meta_dim,
                meta_box=meta_box,
                use_norm=use_norm,
                attention_type=attention_type,
                interpolation_mode=interpolation_mode,
            )
            self.blocks.append(block)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        spatial_shapes = [feature.shape[2:] for feature in features]
        spatial_shapes = spatial_shapes[::-1]

        features = features[1:]
        features = features[::-1]

        head = features[0]
        skip_connections = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            height, width = spatial_shapes[i + 1]
            skip_connection = skip_connections[i] if i < len(skip_connections) else None
            x = decoder_block(x, height, width, skip_connection=skip_connection)

        return x
