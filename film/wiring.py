"""Attaches FiLM to a ``segmentation_models_pytorch.Unet`` per an
encoder/decoder/both/none toggle — one wiring function, not four
hand-maintained model variants.

No vendor source file is edited: encoder-side FiLM is a forward hook
(hooks.py) attached after the vendor ``Unet`` is fully constructed (so
pretrained weights load exactly as ``smp.Unet(...)`` would, untouched);
decoder-side FiLM swaps ``unet.decoder`` for the forked ``FiLMUnetDecoder``
(decoder_blocks.py), built from the same channel config the vendor decoder
would have received. Both insertion points share one ``MetadataEncoder``/
``MetadataSchema`` and one ``MetaBox``, so metadata is encoded exactly once
per forward call regardless of how many places condition on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from segmentation_models_pytorch import Unet

from .decoder_blocks import FiLMUnetDecoder
from .hooks import EncoderFiLM, MetaBox
from .metadata_encoder import MetadataEncoder
from .metadata_schema import MetadataSchema

FiLMTarget = str  # "none" | "encoder" | "decoder" | "both"
_VALID_TARGETS = ("none", "encoder", "decoder", "both")


@dataclass
class FiLMConfig:
    target: FiLMTarget = "both"
    # encoder-side only: don't condition the raw-input passthrough feature (stage 0)
    skip_input_stage: bool = True

    def __post_init__(self) -> None:
        if self.target not in _VALID_TARGETS:
            raise ValueError(
                f"FiLMConfig.target must be one of {_VALID_TARGETS}, got {self.target!r}"
            )


class FiLMUnet(nn.Module):
    """Wraps a vendor ``Unet`` instance; adds a ``metadata`` argument to
    ``forward``. Construct via ``build_film_unet(...)``, not directly —
    that's what wires the encoder hook / decoder swap consistently with
    ``config.target``.
    """

    def __init__(
        self,
        unet: Unet,
        schema: MetadataSchema,
        config: FiLMConfig,
        meta_box: MetaBox,
        encoder_film: Optional[EncoderFiLM] = None,
    ):
        super().__init__()
        self.unet = unet
        self.schema = schema
        self.config = config
        self.meta_box = meta_box
        self.metadata_encoder = MetadataEncoder(schema)
        # setting an nn.Module attribute registers it as a submodule (params
        # visible via film_unet.parameters()/state_dict()); None is fine too,
        # it just stays a plain attribute when there's no encoder-side FiLM.
        self.encoder_film = encoder_film

    @property
    def uses_decoder_film(self) -> bool:
        return isinstance(self.unet.decoder, FiLMUnetDecoder)

    def forward(self, x: torch.Tensor, metadata: Sequence[Dict[str, Any]]):
        m = self.metadata_encoder.encode_batch(metadata).to(x.device)
        self.meta_box.value = m
        try:
            return self.unet(x)
        finally:
            self.meta_box.value = None


def build_film_unet(
    schema: MetadataSchema,
    film_config: Optional[FiLMConfig] = None,
    encoder_name: str = "resnet34",
    encoder_depth: int = 5,
    encoder_weights: Optional[str] = "imagenet",
    decoder_use_norm: Union[bool, str, Dict[str, Any]] = "batchnorm",
    decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    decoder_attention_type: Optional[str] = None,
    decoder_interpolation: str = "nearest",
    in_channels: int = 3,
    classes: int = 1,
    activation: Optional[Any] = None,
    aux_params: Optional[dict] = None,
    **encoder_kwargs: Any,
) -> FiLMUnet:
    """Build a ``smp.Unet`` exactly as ``smp.Unet(...)`` would — same
    arguments, same pretrained-weight loading — then attach FiLM per
    ``film_config.target`` without touching any vendor source file.
    """
    config = film_config or FiLMConfig()

    unet = Unet(
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        encoder_weights=encoder_weights,
        decoder_use_norm=decoder_use_norm,
        decoder_channels=decoder_channels,
        decoder_attention_type=decoder_attention_type,
        decoder_interpolation=decoder_interpolation,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        aux_params=aux_params,
        **encoder_kwargs,
    )

    meta_box = MetaBox()
    encoder_film = None

    if config.target in ("encoder", "both"):
        encoder_film = EncoderFiLM(
            out_channels=list(unet.encoder.out_channels),
            meta_dim=schema.dim,
            meta_box=meta_box,
            skip_input_stage=config.skip_input_stage,
        )
        encoder_film.attach(unet.encoder)

    if config.target in ("decoder", "both"):
        add_center_block = encoder_name.startswith("vgg")
        unet.decoder = FiLMUnetDecoder(
            encoder_channels=unet.encoder.out_channels,
            decoder_channels=decoder_channels,
            meta_dim=schema.dim,
            meta_box=meta_box,
            n_blocks=encoder_depth,
            use_norm=decoder_use_norm,
            add_center_block=add_center_block,
            attention_type=decoder_attention_type,
            interpolation_mode=decoder_interpolation,
        )

    return FiLMUnet(unet, schema, config, meta_box, encoder_film=encoder_film)
