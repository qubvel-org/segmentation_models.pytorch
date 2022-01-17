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

import torch.nn as nn
from pretrainedmodels.models.inceptionv4 import InceptionV4, BasicConv2d
from pretrainedmodels.models.inceptionv4 import pretrained_settings

from ._base import EncoderMixin


class InceptionV4Encoder(InceptionV4, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._stage_idxs = stage_idxs
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
        del self.last_linear

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError(
            "InceptionV4 encoder does not support dilated mode " "due to pooling operation for downsampling!"
        )

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[: self._stage_idxs[0]],
            self.features[self._stage_idxs[0] : self._stage_idxs[1]],
            self.features[self._stage_idxs[1] : self._stage_idxs[2]],
            self.features[self._stage_idxs[2] : self._stage_idxs[3]],
            self.features[self._stage_idxs[3] :],
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


inceptionv4_encoders = {
    "inceptionv4": {
        "encoder": InceptionV4Encoder,
        "pretrained_settings": pretrained_settings["inceptionv4"],
        "params": {
            "stage_idxs": (3, 5, 9, 15),
            "out_channels": (3, 64, 192, 384, 1024, 1536),
            "num_classes": 1001,
        },
    }
}
