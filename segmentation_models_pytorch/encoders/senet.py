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

import torch.nn as nn

from pretrainedmodels.models.senet import (
    SENet,
    SEBottleneck,
    SEResNetBottleneck,
    SEResNeXtBottleneck,
    pretrained_settings,
)
from ._base import EncoderMixin


class SENetEncoder(SENet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        del self.last_linear
        del self.avg_pool

    def get_stages(self):
        return [
            nn.Identity(),
            self.layer0[:-1],
            nn.Sequential(self.layer0[-1], self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("last_linear.bias")
        state_dict.pop("last_linear.weight")
        super().load_state_dict(state_dict, **kwargs)


senet_encoders = {
    "senet154": {
        "encoder": SENetEncoder,
        "pretrained_settings": pretrained_settings["senet154"],
        "params": {
            "out_channels": (3, 128, 256, 512, 1024, 2048),
            "block": SEBottleneck,
            "dropout_p": 0.2,
            "groups": 64,
            "layers": [3, 8, 36, 3],
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "se_resnet50": {
        "encoder": SENetEncoder,
        "pretrained_settings": pretrained_settings["se_resnet50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": SEResNetBottleneck,
            "layers": [3, 4, 6, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "se_resnet101": {
        "encoder": SENetEncoder,
        "pretrained_settings": pretrained_settings["se_resnet101"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": SEResNetBottleneck,
            "layers": [3, 4, 23, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "se_resnet152": {
        "encoder": SENetEncoder,
        "pretrained_settings": pretrained_settings["se_resnet152"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": SEResNetBottleneck,
            "layers": [3, 8, 36, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "se_resnext50_32x4d": {
        "encoder": SENetEncoder,
        "pretrained_settings": pretrained_settings["se_resnext50_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": SEResNeXtBottleneck,
            "layers": [3, 4, 6, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 32,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
    "se_resnext101_32x4d": {
        "encoder": SENetEncoder,
        "pretrained_settings": pretrained_settings["se_resnext101_32x4d"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": SEResNeXtBottleneck,
            "layers": [3, 4, 23, 3],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 32,
            "inplanes": 64,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 16,
        },
    },
}
