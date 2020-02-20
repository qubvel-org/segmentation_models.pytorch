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
from torchvision.models.vgg import VGG
from torchvision.models.vgg import make_layers
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin

# fmt: off
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# fmt: on


class VGGEncoder(VGG, EncoderMixin):
    def __init__(self, out_channels, config, batch_norm=False, depth=5, **kwargs):
        super().__init__(make_layers(config, batch_norm=batch_norm), **kwargs)
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        del self.classifier

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("'VGG' models do not support dilated mode due to Max Pooling"
                         " operations for downsampling!")

    def get_stages(self):
        stages = []
        stage_modules = []
        for module in self.features:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith("classifier"):
                state_dict.pop(k)
        super().load_state_dict(state_dict, **kwargs)


vgg_encoders = {
    "vgg11": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": False,
        },
    },
    "vgg11_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": True,
        },
    },
    "vgg13": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": False,
        },
    },
    "vgg13_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": True,
        },
    },
    "vgg16": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": False,
        },
    },
    "vgg16_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": True,
        },
    },
    "vgg19": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": False,
        },
    },
    "vgg19_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": True,
        },
    },
}
