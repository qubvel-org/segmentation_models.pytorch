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

import re

from torchvision.models.densenet import DenseNet

from ._base import EncoderMixin


class DenseNetEncoder(DenseNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, output_stride=32, **kwargs):
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )

        super().__init__(**kwargs)

        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride
        del self.classifier

    def make_dilated(self, *args, **kwargs):
        raise ValueError(
            "DenseNet encoders do not support dilated mode "
            "due to pooling operation for downsampling!"
        )

    def forward(self, x):
        features = [x]

        if self._depth >= 1:
            x = self.features.conv0(x)
            x = self.features.norm0(x)
            x = self.features.relu0(x)
            features.append(x)

        if self._depth >= 2:
            x = self.features.pool0(x)
            x = self.features.denseblock1(x)
            x = self.features.transition1.norm(x)
            x = self.features.transition1.relu(x)
            features.append(x)

        if self._depth >= 3:
            x = self.features.transition1.conv(x)
            x = self.features.transition1.pool(x)
            x = self.features.denseblock2(x)
            x = self.features.transition2.norm(x)
            x = self.features.transition2.relu(x)
            features.append(x)

        if self._depth >= 4:
            x = self.features.transition2.conv(x)
            x = self.features.transition2.pool(x)
            x = self.features.denseblock3(x)
            x = self.features.transition3.norm(x)
            x = self.features.transition3.relu(x)
            features.append(x)

        if self._depth >= 5:
            x = self.features.transition3.conv(x)
            x = self.features.transition3.pool(x)
            x = self.features.denseblock4(x)
            x = self.features.norm5(x)
            features.append(x)

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
        state_dict.pop("classifier.bias", None)
        state_dict.pop("classifier.weight", None)

        super().load_state_dict(state_dict)


densenet_encoders = {
    "densenet121": {
        "encoder": DenseNetEncoder,
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 1024],
            "num_init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 24, 16),
        },
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/densenet121.imagenet",
                "revision": "a17c96896a265b61338f66f61d3887b24f61995a",
            }
        },
    },
    "densenet169": {
        "encoder": DenseNetEncoder,
        "params": {
            "out_channels": [3, 64, 256, 512, 1280, 1664],
            "num_init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 32, 32),
        },
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/densenet169.imagenet",
                "revision": "8facfba9fc72f7750879dac9ac6ceb3ab990de8d",
            }
        },
    },
    "densenet201": {
        "encoder": DenseNetEncoder,
        "params": {
            "out_channels": [3, 64, 256, 512, 1792, 1920],
            "num_init_features": 64,
            "growth_rate": 32,
            "block_config": (6, 12, 48, 32),
        },
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/densenet201.imagenet",
                "revision": "ed5deb355d71659391d46fae5e7587460fbb5f84",
            }
        },
    },
    "densenet161": {
        "encoder": DenseNetEncoder,
        "params": {
            "out_channels": [3, 96, 384, 768, 2112, 2208],
            "num_init_features": 96,
            "growth_rate": 48,
            "block_config": (6, 12, 36, 24),
        },
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/densenet161.imagenet",
                "revision": "9afe0fec51ab2a627141769d97d6f83756d78446",
            }
        },
    },
}
