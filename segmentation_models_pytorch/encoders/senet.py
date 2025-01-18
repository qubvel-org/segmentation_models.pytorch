"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""

import torch
from typing import List, Dict, Sequence

from ._base import EncoderMixin
from ._senet import (
    SENet,
)


class SENetEncoder(SENet, EncoderMixin):
    def __init__(
        self,
        out_channels: List[int],
        depth: int = 5,
        output_stride: int = 32,
        **kwargs,
    ):
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )
        super().__init__(**kwargs)

        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

        # for compatibility with torchscript
        self.layer0_pool = self.layer0.pool
        self.layer0.pool = torch.nn.Identity()

        del self.last_linear
        del self.avg_pool

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        return {
            16: [self.layer3],
            32: [self.layer4],
        }

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]

        if self._depth >= 1:
            x = self.layer0(x)
            features.append(x)

        if self._depth >= 2:
            x = self.layer0_pool(x)
            x = self.layer1(x)
            features.append(x)

        if self._depth >= 3:
            x = self.layer2(x)
            features.append(x)

        if self._depth >= 4:
            x = self.layer3(x)
            features.append(x)

        if self._depth >= 5:
            x = self.layer4(x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("last_linear.bias", None)
        state_dict.pop("last_linear.weight", None)
        super().load_state_dict(state_dict, **kwargs)
