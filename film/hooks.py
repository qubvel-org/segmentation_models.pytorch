"""Encoder-side FiLM: a forward hook on the encoder's feature-pyramid output.

Backbone-agnostic by construction — every SMP encoder implements the same
contract (`EncoderMixin`: ``forward(x) -> List[Tensor]``, ``out_channels``),
so a single hook on the whole ``encoder`` module modulates any backbone's
feature pyramid without knowing its internal layer names. This is what keeps
pretrained encoder weights untouched: the hook runs *after* the encoder's own
forward returns, rewriting the returned list — it never reaches into the
encoder's own conv/BN stack, so nothing about weight loading changes.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .generator import FiLMGenerator, apply_film


class MetaBox:
    """One-slot mutable container carrying the current batch's encoded
    metadata vector `m` from ``FiLMUnet.forward`` to the encoder hook and the
    FiLM decoder, without changing either's call signature — both are
    invoked positionally (``encoder(x)`` / ``decoder(features)``) by the
    untouched vendor ``SegmentationModel.forward``, so there's no argument
    slot to smuggle `m` through except a side channel like this one.

    Not thread-safe across concurrent forward calls on the same model
    instance — same limitation as any other stateful `nn.Module` buffer.
    """

    def __init__(self) -> None:
        self.value: Optional[torch.Tensor] = None


class EncoderFiLM(nn.Module):
    """Holds one ``FiLMGenerator`` per conditioned encoder stage plus the
    hook handle that applies them. Build and attach via ``EncoderFiLM(...).attach(encoder)``.
    """

    def __init__(
        self,
        out_channels: List[int],
        meta_dim: int,
        meta_box: MetaBox,
        skip_input_stage: bool = True,
    ):
        super().__init__()
        self.meta_box = meta_box
        self.skip_input_stage = skip_input_stage
        start = 1 if skip_input_stage else 0
        self.stage_indices = list(range(start, len(out_channels)))
        self.generators = nn.ModuleList(
            [FiLMGenerator(meta_dim, out_channels[i]) for i in self.stage_indices]
        )
        self._handle = None

    def _hook(
        self, module: nn.Module, inputs, output: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        m = self.meta_box.value
        if m is None:
            raise RuntimeError(
                "encoder FiLM hook fired with no metadata set — call FiLMUnet.forward(x, metadata), "
                "not the wrapped encoder directly"
            )
        features = list(output)
        for generator, stage_idx in zip(self.generators, self.stage_indices):
            gamma, beta = generator(m)
            features[stage_idx] = apply_film(features[stage_idx], gamma, beta)
        return features

    def attach(self, encoder: nn.Module) -> "EncoderFiLM":
        self._handle = encoder.register_forward_hook(self._hook)
        return self

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
