"""The FiLM generator: a single tunable linear transform from metadata to
per-channel (gamma, beta), and the broadcast/apply step.

Deliberately a bare ``nn.Linear``, no nonlinearity — [gamma, beta] = W*m + b.
Every entry of W is directly interpretable ("this metadata field pushes this
channel's scale/shift by this much"). An MLP generator could capture
cross-field interactions, but at the cost of exactly that auditability.
Start linear; an MLP variant is a later ablation, not this one.

Identity-initialized: at construction, W and b are zero, so every generator
starts as gamma=1, beta=0 (a no-op) regardless of what metadata it's fed —
FiLM begins as the identity transform relative to pretrained/baseline
behavior and only deviates from it as training pushes W away from zero.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    def __init__(self, meta_dim: int, channels: int):
        super().__init__()
        self.meta_dim = meta_dim
        self.channels = channels
        self.linear = nn.Linear(meta_dim, 2 * channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    @property
    def weight_matrix(self) -> torch.Tensor:
        """The learned W, shape (2*channels, meta_dim) — first `channels` rows
        drive gamma's deviation from 1, the rest drive beta. Read directly for
        auditing (see audit.py), no need to run a forward pass.
        """
        return self.linear.weight

    def forward(self, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.linear(m)
        delta_gamma, beta = out.chunk(2, dim=-1)
        gamma = 1.0 + delta_gamma
        return gamma, beta


def apply_film(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Broadcast (B, C) gamma/beta onto a (B, C, H, W) feature map."""
    gamma = gamma[:, :, None, None]
    beta = beta[:, :, None, None]
    return x * gamma + beta
