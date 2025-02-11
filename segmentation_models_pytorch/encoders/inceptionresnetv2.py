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
from ._inceptionresnetv2 import InceptionResNetV2


class InceptionResNetV2Encoder(InceptionResNetV2, EncoderMixin):
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

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # for torchscript, block8 does not have relu defined
        self.block8.relu = nn.Identity()

        # remove linear layers
        del self.avgpool_1a
        del self.last_linear

    def make_dilated(self, *args, **kwargs):
        raise ValueError(
            "InceptionResnetV2 encoder does not support dilated mode "
            "due to pooling operation for downsampling!"
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]

        if self._depth >= 1:
            x = self.conv2d_1a(x)
            x = self.conv2d_2a(x)
            x = self.conv2d_2b(x)
            features.append(x)

        if self._depth >= 2:
            x = self.maxpool_3a(x)
            x = self.conv2d_3b(x)
            x = self.conv2d_4a(x)
            features.append(x)

        if self._depth >= 3:
            x = self.maxpool_5a(x)
            x = self.mixed_5b(x)
            x = self.repeat(x)
            features.append(x)

        if self._depth >= 4:
            x = self.mixed_6a(x)
            x = self.repeat_1(x)
            features.append(x)

        if self._depth >= 5:
            x = self.mixed_7a(x)
            x = self.repeat_2(x)
            x = self.block8(x)
            x = self.conv2d_7b(x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("last_linear.bias", None)
        state_dict.pop("last_linear.weight", None)
        super().load_state_dict(state_dict, **kwargs)


inceptionresnetv2_encoders = {
    "inceptionresnetv2": {
        "encoder": InceptionResNetV2Encoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/inceptionresnetv2.imagenet",
                "revision": "120c5afdbb80a1c989db0a7423ebb7a9db9b1e6c",
            },
            "imagenet+background": {
                "repo_id": "smp-hub/inceptionresnetv2.imagenet-background",
                "revision": "3ecf3491658dc0f6a76d69c9d1cb36511b1ee56c",
            },
        },
        "params": {"out_channels": [3, 64, 192, 320, 1088, 1536], "num_classes": 1000},
    }
}
