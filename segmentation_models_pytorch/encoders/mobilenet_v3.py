""" Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

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

import torchvision
import torch.nn as nn
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf

from ._base import EncoderMixin


class MobileNetV3Encoder(torchvision.models.MobileNetV3, EncoderMixin):

    def __init__(self, out_channels, stage_idxs, model_name, depth=5, **kwargs):
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(model_name, kwargs)
        super().__init__(inverted_residual_setting, last_channel, **kwargs)
        
        self._depth = depth
        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._in_channels = 3
        
        del self.classifier

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:self._stage_idxs[0]],
            self.features[self._stage_idxs[0]:self._stage_idxs[1]],
            self.features[self._stage_idxs[1]:self._stage_idxs[2]],
            self.features[self._stage_idxs[2]:self._stage_idxs[3]],
            self.features[self._stage_idxs[3]:],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.0.bias")
        state_dict.pop("classifier.0.weight")
        state_dict.pop("classifier.3.bias")
        state_dict.pop("classifier.3.weight")
        super().load_state_dict(state_dict, **kwargs)


mobilenet_v3_encoders = {
    "mobilenet_v3_large": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "out_channels": (3, 16, 24, 40, 112, 960),
            "stage_idxs": (2, 4, 7, 13),
            "model_name": "mobilenet_v3_large",
        },
    },
    "mobilenet_v3_small": {
        "encoder": MobileNetV3Encoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "out_channels": (3, 16, 16, 24, 40, 576),
            "stage_idxs": (1, 2, 4, 7),
            "model_name": "mobilenet_v3_small",
        },
    },
}
