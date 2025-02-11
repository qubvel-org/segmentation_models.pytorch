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
import torch.nn.functional as F
from typing import List, Dict, Sequence

from ._base import EncoderMixin
from ._dpn import DPN


class DPNEncoder(DPN, EncoderMixin):
    _is_torch_scriptable = False
    _is_torch_exportable = True  # since torch 2.6.0

    def __init__(
        self,
        stage_idxs: List[int],
        out_channels: List[int],
        depth: int = 5,
        output_stride: int = 32,
        **kwargs,
    ):
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )

        super().__init__(**kwargs)
        self._stage_idxs = stage_idxs
        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

        del self.last_linear

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        return {
            16: [self.features[self._stage_idxs[1] : self._stage_idxs[2]]],
            32: [self.features[self._stage_idxs[2] : self._stage_idxs[3]]],
        }

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]

        if self._depth >= 1:
            x = self.features[0].conv(x)
            x = self.features[0].bn(x)
            x = self.features[0].act(x)
            features.append(x)

        if self._depth >= 2:
            x = self.features[0].pool(x)
            x = self.features[1 : self._stage_idxs[0]](x)
            skip = F.relu(torch.cat(x, dim=1), inplace=True)
            features.append(skip)

        if self._depth >= 3:
            x = self.features[self._stage_idxs[0] : self._stage_idxs[1]](x)
            skip = F.relu(torch.cat(x, dim=1), inplace=True)
            features.append(skip)

        if self._depth >= 4:
            x = self.features[self._stage_idxs[1] : self._stage_idxs[2]](x)
            skip = F.relu(torch.cat(x, dim=1), inplace=True)
            features.append(skip)

        if self._depth >= 5:
            x = self.features[self._stage_idxs[2] : self._stage_idxs[3]](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("last_linear.bias", None)
        state_dict.pop("last_linear.weight", None)
        super().load_state_dict(state_dict, **kwargs)


dpn_encoders = {
    "dpn68": {
        "encoder": DPNEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/dpn68.imagenet",
                "revision": "c209aefdeae6bc93937556629e974b44d4e58535",
            }
        },
        "params": {
            "stage_idxs": [4, 8, 20, 24],
            "out_channels": [3, 10, 144, 320, 704, 832],
            "groups": 32,
            "inc_sec": (16, 32, 32, 64),
            "k_r": 128,
            "k_sec": (3, 4, 12, 3),
            "num_classes": 1000,
            "num_init_features": 10,
            "small": True,
            "test_time_pool": True,
        },
    },
    "dpn68b": {
        "encoder": DPNEncoder,
        "pretrained_settings": {
            "imagenet+5k": {
                "repo_id": "smp-hub/dpn68b.imagenet-5k",
                "revision": "6c6615e77688e390ae0eaa81e26821fbd83cee4b",
            }
        },
        "params": {
            "stage_idxs": [4, 8, 20, 24],
            "out_channels": [3, 10, 144, 320, 704, 832],
            "b": True,
            "groups": 32,
            "inc_sec": (16, 32, 32, 64),
            "k_r": 128,
            "k_sec": (3, 4, 12, 3),
            "num_classes": 1000,
            "num_init_features": 10,
            "small": True,
            "test_time_pool": True,
        },
    },
    "dpn92": {
        "encoder": DPNEncoder,
        "pretrained_settings": {
            "imagenet+5k": {
                "repo_id": "smp-hub/dpn92.imagenet-5k",
                "revision": "d231f51ce4ad2c84ed5fcaf4ef0cfece6814a526",
            }
        },
        "params": {
            "stage_idxs": [4, 8, 28, 32],
            "out_channels": [3, 64, 336, 704, 1552, 2688],
            "groups": 32,
            "inc_sec": (16, 32, 24, 128),
            "k_r": 96,
            "k_sec": (3, 4, 20, 3),
            "num_classes": 1000,
            "num_init_features": 64,
            "test_time_pool": True,
        },
    },
    "dpn98": {
        "encoder": DPNEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/dpn98.imagenet",
                "revision": "b2836c86216c1ddce980d832f7deaa4ca22babd3",
            }
        },
        "params": {
            "stage_idxs": [4, 10, 30, 34],
            "out_channels": [3, 96, 336, 768, 1728, 2688],
            "groups": 40,
            "inc_sec": (16, 32, 32, 128),
            "k_r": 160,
            "k_sec": (3, 6, 20, 3),
            "num_classes": 1000,
            "num_init_features": 96,
            "test_time_pool": True,
        },
    },
    "dpn107": {
        "encoder": DPNEncoder,
        "pretrained_settings": {
            "imagenet+5k": {
                "repo_id": "smp-hub/dpn107.imagenet-5k",
                "revision": "dab4cd6b8b79de3db970f2dbff85359a8847db05",
            }
        },
        "params": {
            "stage_idxs": [5, 13, 33, 37],
            "out_channels": [3, 128, 376, 1152, 2432, 2688],
            "groups": 50,
            "inc_sec": (20, 64, 64, 128),
            "k_r": 200,
            "k_sec": (4, 8, 20, 3),
            "num_classes": 1000,
            "num_init_features": 128,
            "test_time_pool": True,
        },
    },
    "dpn131": {
        "encoder": DPNEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/dpn131.imagenet",
                "revision": "04bbb9f415ca2bb59f3d8227857967b74698515e",
            }
        },
        "params": {
            "stage_idxs": [5, 13, 41, 45],
            "out_channels": [3, 128, 352, 832, 1984, 2688],
            "groups": 40,
            "inc_sec": (16, 32, 32, 128),
            "k_r": 160,
            "k_sec": (4, 8, 28, 3),
            "num_classes": 1000,
            "num_init_features": 128,
            "test_time_pool": True,
        },
    },
}
