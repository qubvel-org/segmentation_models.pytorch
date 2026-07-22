"""Auditing layer: every transformation FiLM applies has a hook into here —
nothing modifies a tensor silently.

Attaches (non-invasively, via forward hooks on ``FiLMGenerator`` instances —
the same pattern as encoder-side conditioning itself) a recorder that keeps
each generator's weight matrix and a rolling history of the (gamma, beta) it
produced. Use ``AuditLog.summary()`` to check for the classic FiLM failure
mode: gamma collapsing to ~1 and beta to ~0 well into training means the
network has learned to ignore metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from .decoder_blocks import FiLMUnetDecoder
from .generator import FiLMGenerator


def collect_generators(film_unet: Any) -> Dict[str, FiLMGenerator]:
    """Gather every ``FiLMGenerator`` in a ``FiLMUnet``, named by insertion
    point, for handing to ``AuditLog.attach``.
    """
    generators: Dict[str, FiLMGenerator] = {}

    encoder_film = getattr(film_unet, "encoder_film", None)
    if encoder_film is not None:
        for stage_idx, generator in zip(
            encoder_film.stage_indices, encoder_film.generators
        ):
            generators[f"encoder.stage{stage_idx}"] = generator

    decoder = film_unet.unet.decoder
    if isinstance(decoder, FiLMUnetDecoder):
        for block_idx, block in enumerate(decoder.blocks):
            generators[f"decoder.block{block_idx}.conv1"] = block.conv1.film
            generators[f"decoder.block{block_idx}.conv2"] = block.conv2.film

    return generators


class AuditLog:
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._generators: Dict[str, FiLMGenerator] = {}
        self._handles: List[Any] = []

    def attach(self, named_generators: Dict[str, FiLMGenerator]) -> "AuditLog":
        for name, generator in named_generators.items():
            self._generators[name] = generator
            self._history.setdefault(name, [])
            handle = generator.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            gamma, beta = output
            history = self._history[name]
            history.append((gamma.detach().cpu(), beta.detach().cpu()))
            if len(history) > self.max_history:
                del history[: len(history) - self.max_history]

        return hook

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def weight_matrices(self) -> Dict[str, torch.Tensor]:
        """The learned W per generator — heatmap directly against
        (metadata field, channel) for interpretability, no forward pass needed.
        """
        return {
            name: g.weight_matrix.detach().cpu() for name, g in self._generators.items()
        }

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, history in self._history.items():
            if not history:
                continue
            gammas = torch.cat([g.flatten() for g, _ in history])
            betas = torch.cat([b.flatten() for _, b in history])
            out[name] = {
                "gamma_mean": gammas.mean().item(),
                "gamma_std": gammas.std().item(),
                "beta_mean": betas.mean().item(),
                "beta_std": betas.std().item(),
                "n_samples": len(history),
            }
        return out

    def reset(self) -> None:
        for name in self._history:
            self._history[name] = []
