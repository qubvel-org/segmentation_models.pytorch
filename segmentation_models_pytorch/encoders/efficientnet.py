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

from ._base import EncoderMixin
from ._efficientnet import EfficientNet, get_model_params


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
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )

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


efficient_net_encoders = {
    "efficientnet-b0": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/efficientnet-b0.imagenet",
                "revision": "1bbe7ecc1d5ea1d2058de1a2db063b8701aff314",
            },
            "advprop": {
                "repo_id": "smp-hub/efficientnet-b0.advprop",
                "revision": "29043c08140d9c6ee7de1468d55923f2b06bcec2",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 40, 112, 320],
            "stage_idxs": [3, 5, 9, 16],
            "model_name": "efficientnet-b0",
        },
    },
    "efficientnet-b1": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/efficientnet-b1.imagenet",
                "revision": "5d637466a5215de300a8ccb13a39357df2df2bf4",
            },
            "advprop": {
                "repo_id": "smp-hub/efficientnet-b1.advprop",
                "revision": "2e518b8b0955bbab467f50525578dab6b6086afc",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 40, 112, 320],
            "stage_idxs": [5, 8, 16, 23],
            "model_name": "efficientnet-b1",
        },
    },
    "efficientnet-b2": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/efficientnet-b2.imagenet",
                "revision": "a96d4f0295ffbae18ebba173bf7f3c0c8f21990e",
            },
            "advprop": {
                "repo_id": "smp-hub/efficientnet-b2.advprop",
                "revision": "be788c20dfb0bbe83b4c439f9cfe0dd937c0783e",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 48, 120, 352],
            "stage_idxs": [5, 8, 16, 23],
            "model_name": "efficientnet-b2",
        },
    },
    "efficientnet-b3": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/efficientnet-b3.imagenet",
                "revision": "074c54a6c473e0d294690d49cedb6cf463e7127d",
            },
            "advprop": {
                "repo_id": "smp-hub/efficientnet-b3.advprop",
                "revision": "9ccc166d87bd9c08d6bed4477638c7f4bb3eec78",
            },
        },
        "params": {
            "out_channels": [3, 40, 32, 48, 136, 384],
            "stage_idxs": [5, 8, 18, 26],
            "model_name": "efficientnet-b3",
        },
    },
    "efficientnet-b4": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/efficientnet-b4.imagenet",
                "revision": "05cd5dde5dab658f00c463f9b9aa0ced76784f40",
            },
            "advprop": {
                "repo_id": "smp-hub/efficientnet-b4.advprop",
                "revision": "f04caa809ea4eb08ee9e7fd555f5514ebe2a9ef5",
            },
        },
        "params": {
            "out_channels": [3, 48, 32, 56, 160, 448],
            "stage_idxs": [6, 10, 22, 32],
            "model_name": "efficientnet-b4",
        },
    },
    "efficientnet-b5": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/efficientnet-b5.imagenet",
                "revision": "69f4d28460a4e421b7860bc26ee7d832e03e01ca",
            },
            "advprop": {
                "repo_id": "smp-hub/efficientnet-b5.advprop",
                "revision": "dabe78fc8ab7ce93ddc2bb156b01db227caede88",
            },
        },
        "params": {
            "out_channels": [3, 48, 40, 64, 176, 512],
            "stage_idxs": [8, 13, 27, 39],
            "model_name": "efficientnet-b5",
        },
    },
    "efficientnet-b6": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/efficientnet-b6.imagenet",
                "revision": "8570752016f7c62ae149cffa058550fe44e21c8b",
            },
            "advprop": {
                "repo_id": "smp-hub/efficientnet-b6.advprop",
                "revision": "c2dbb4d1359151165ec7b96cfe54a9cac2142a31",
            },
        },
        "params": {
            "out_channels": [3, 56, 40, 72, 200, 576],
            "stage_idxs": [9, 15, 31, 45],
            "model_name": "efficientnet-b6",
        },
    },
    "efficientnet-b7": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/efficientnet-b7.imagenet",
                "revision": "5a5dbe687d612ebc3dca248274fd1191111deda6",
            },
            "advprop": {
                "repo_id": "smp-hub/efficientnet-b7.advprop",
                "revision": "ce33edb4e80c0cde268f098ae2299e23f615577d",
            },
        },
        "params": {
            "out_channels": [3, 64, 48, 80, 224, 640],
            "stage_idxs": [11, 18, 38, 55],
            "model_name": "efficientnet-b7",
        },
    },
}
