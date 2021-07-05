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
from pretrainedmodels.models.inceptionresnetv2 import InceptionResNetV2
from pretrainedmodels.models.inceptionresnetv2 import pretrained_settings

from ._base import EncoderMixin


class InceptionResNetV2Encoder(InceptionResNetV2, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # remove linear layers
        del self.avgpool_1a
        del self.last_linear

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("InceptionResnetV2 encoder does not support dilated mode "
                         "due to pooling operation for downsampling!")

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv2d_1a, self.conv2d_2a, self.conv2d_2b),
            nn.Sequential(self.maxpool_3a, self.conv2d_3b, self.conv2d_4a),
            nn.Sequential(self.maxpool_5a, self.mixed_5b, self.repeat),
            nn.Sequential(self.mixed_6a, self.repeat_1),
            nn.Sequential(self.mixed_7a, self.repeat_2, self.block8, self.conv2d_7b),
        ]

    def forward(self, x):

        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("last_linear.bias", None)
        state_dict.pop("last_linear.weight", None)
        super().load_state_dict(state_dict, **kwargs)


inceptionresnetv2_encoders = {
    "inceptionresnetv2": {
        "encoder": InceptionResNetV2Encoder,
        "pretrained_settings": pretrained_settings["inceptionresnetv2"],
        "params": {"out_channels": (3, 64, 192, 320, 1088, 1536), "num_classes": 1000},
    }
}
