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
from typing import List, Dict, Sequence

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, url_map_advprop, get_model_params

from ._base import EncoderMixin


class EfficientNetEncoder(EfficientNet, EncoderMixin):
    _is_torch_scriptable = False

    def __init__(
        self,
        stage_idxs: List[int],
        out_channels: List[int],
        model_name: str,
        depth: int = 5,
        output_stride: int = 32,
    ):
        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._stage_idxs = stage_idxs
        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

        del self._fc

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        return {
            16: [self._blocks[self._stage_idxs[1] : self._stage_idxs[2]]],
            32: [self._blocks[self._stage_idxs[2] :]],
        }

    def apply_blocks(
        self, x: torch.Tensor, start_idx: int, end_idx: int
    ) -> torch.Tensor:
        drop_connect_rate = self._global_params.drop_connect_rate

        for block_number in range(start_idx, end_idx):
            drop_connect_prob = drop_connect_rate * block_number / len(self._blocks)
            x = self._blocks[block_number](x, drop_connect_prob)

        return x

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]

        if self._depth >= 1:
            x = self._conv_stem(x)
            x = self._bn0(x)
            x = self._swish(x)
            features.append(x)

        if self._depth >= 2:
            x = self.apply_blocks(x, 0, self._stage_idxs[0])
            features.append(x)

        if self._depth >= 3:
            x = self.apply_blocks(x, self._stage_idxs[0], self._stage_idxs[1])
            features.append(x)

        if self._depth >= 4:
            x = self.apply_blocks(x, self._stage_idxs[1], self._stage_idxs[2])
            features.append(x)

        if self._depth >= 5:
            x = self.apply_blocks(x, self._stage_idxs[2], len(self._blocks))
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias", None)
        state_dict.pop("_fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
        "advprop": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "url": url_map_advprop[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }
    return pretrained_settings


efficient_net_encoders = {
    "efficientnet-b0": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b0"),
        "params": {
            "out_channels": [3, 32, 24, 40, 112, 320],
            "stage_idxs": [3, 5, 9, 16],
            "model_name": "efficientnet-b0",
        },
    },
    "efficientnet-b1": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b1"),
        "params": {
            "out_channels": [3, 32, 24, 40, 112, 320],
            "stage_idxs": [5, 8, 16, 23],
            "model_name": "efficientnet-b1",
        },
    },
    "efficientnet-b2": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b2"),
        "params": {
            "out_channels": [3, 32, 24, 48, 120, 352],
            "stage_idxs": [5, 8, 16, 23],
            "model_name": "efficientnet-b2",
        },
    },
    "efficientnet-b3": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b3"),
        "params": {
            "out_channels": [3, 40, 32, 48, 136, 384],
            "stage_idxs": [5, 8, 18, 26],
            "model_name": "efficientnet-b3",
        },
    },
    "efficientnet-b4": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b4"),
        "params": {
            "out_channels": [3, 48, 32, 56, 160, 448],
            "stage_idxs": [6, 10, 22, 32],
            "model_name": "efficientnet-b4",
        },
    },
    "efficientnet-b5": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b5"),
        "params": {
            "out_channels": [3, 48, 40, 64, 176, 512],
            "stage_idxs": [8, 13, 27, 39],
            "model_name": "efficientnet-b5",
        },
    },
    "efficientnet-b6": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b6"),
        "params": {
            "out_channels": [3, 56, 40, 72, 200, 576],
            "stage_idxs": [9, 15, 31, 45],
            "model_name": "efficientnet-b6",
        },
    },
    "efficientnet-b7": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet-b7"),
        "params": {
            "out_channels": [3, 64, 48, 80, 224, 640],
            "stage_idxs": [11, 18, 38, 55],
            "model_name": "efficientnet-b7",
        },
    },
}
