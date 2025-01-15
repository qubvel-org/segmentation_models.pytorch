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
import torchvision
from typing import Dict, Sequence, List

from ._base import EncoderMixin


class MobileNetV2Encoder(torchvision.models.MobileNetV2, EncoderMixin):
    def __init__(
        self, out_channels: List[int], depth: int = 5, output_stride: int = 32, **kwargs
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
        self._out_indexes = [1, 3, 6, 13, len(self.features) - 1]

        del self.classifier

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        return {
            16: [self.features[7:14]],
            32: [self.features[14:]],
        }

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]

        depth = 0
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
        state_dict.pop("classifier.1.bias", None)
        state_dict.pop("classifier.1.weight", None)
        super().load_state_dict(state_dict, **kwargs)


mobilenet_encoders = {
    "mobilenet_v2": {
        "encoder": MobileNetV2Encoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/mobilenet_v2.imagenet",
                "revision": "e67aa804e17f7b404b629127eabbd224c4e0690b",
            }
        },
        "params": {"out_channels": [3, 16, 24, 32, 96, 1280]},
    }
}
