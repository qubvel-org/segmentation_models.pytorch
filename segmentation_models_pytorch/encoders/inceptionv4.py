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
import torch.nn as nn

from typing import List

from ._base import EncoderMixin
from ._inceptionv4 import InceptionV4


class InceptionV4Encoder(InceptionV4, EncoderMixin):
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
        self._out_indexes = [2, 4, 8, 14, len(self.features) - 1]

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # remove linear layers
        del self.last_linear

    def make_dilated(self, *args, **kwargs):
        raise ValueError(
            "InceptionV4 encoder does not support dilated mode "
            "due to pooling operation for downsampling!"
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        depth = 0
        features = [x]

        for i, module in enumerate(self.features):
            x = module(x)

            if i in self._out_indexes:
                features.append(x)
                depth += 1

            # torchscript does not support break in cycle, so we just
            # go over all modules and then slice number of features
            if not torch.jit.is_scripting() and depth > self._depth:
                break

        features = features[: self._depth + 1]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("last_linear.bias", None)
        state_dict.pop("last_linear.weight", None)
        super().load_state_dict(state_dict, **kwargs)


inceptionv4_encoders = {
    "inceptionv4": {
        "encoder": InceptionV4Encoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/inceptionv4.imagenet",
                "revision": "918fb54f07811d82a4ecde3a51156041d0facba9",
            },
            "imagenet+background": {
                "repo_id": "smp-hub/inceptionv4.imagenet-background",
                "revision": "8c2a48e20d2709ee64f8421c61be309f05bfa536",
            },
        },
        "params": {
            "out_channels": [3, 64, 192, 384, 1024, 1536],
            "num_classes": 1001,
        },
    }
}
