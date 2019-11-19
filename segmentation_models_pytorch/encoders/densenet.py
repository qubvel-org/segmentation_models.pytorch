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

import re
import torch.nn as nn

from pretrainedmodels.models.torchvision_models import pretrained_settings
from torchvision.models.densenet import DenseNet

from ._base import EncoderMixin


class DenseNetEncoder(DenseNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        del self.classifier

    @staticmethod
    def _transition(x, transition_block):
        for module in transition_block:
            x = module(x)
            if isinstance(module, nn.ReLU):
                skip = x
        return x, skip

    def forward(self, x):

        features = [x]

        if self._depth > 0:
            x = self.features.conv0(x)
            x = self.features.norm0(x)
            x = self.features.relu0(x)
            features.append(x)

        if self._depth > 1:
            x = self.features.pool0(x)
            x = self.features.denseblock1(x)
            x, x1 = self._transition(x, self.features.transition1)
            features.append(x1)

        if self._depth > 2:
            x = self.features.denseblock2(x)
            x, x2 = self._transition(x, self.features.transition2)
            features.append(x2)

        if self._depth > 3:
            x = self.features.denseblock3(x)
            x, x3 = self._transition(x, self.features.transition3)
            features.append(x3)

        if self._depth > 4:
            x = self.features.denseblock4(x)
            x4 = self.features.norm5(x)
            features.append(x4)

        return features

    def load_state_dict(self, state_dict):
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        # remove linear
        state_dict.pop("classifier.bias")
        state_dict.pop("classifier.weight")

        super().load_state_dict(state_dict)


densenet_encoders = {
    "densenet121": {
        "encoder": DenseNetEncoder,
        "pretrained_settings": pretrained_settings["densenet121"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 1024),
            "num_init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 24, 16),
        },
    },
    "densenet169": {
        "encoder": DenseNetEncoder,
        "pretrained_settings": pretrained_settings["densenet169"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1280, 1664),
            "num_init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 32, 32),
        },
    },
    "densenet201": {
        "encoder": DenseNetEncoder,
        "pretrained_settings": pretrained_settings["densenet201"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1792, 1920),
            "num_init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 48, 32),
        },
    },
    "densenet161": {
        "encoder": DenseNetEncoder,
        "pretrained_settings": pretrained_settings["densenet161"],
        "params": {
            "out_channels": (3, 96, 384, 768, 2112, 2208),
            "num_init_features": 96,
            "growth_rate": 48,
            "block_config": (6, 12, 36, 24),
        },
    },
}
