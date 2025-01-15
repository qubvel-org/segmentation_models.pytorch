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
from typing import Dict, Sequence, List
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck

from ._base import EncoderMixin


class ResNetEncoder(ResNet, EncoderMixin):
    """ResNet encoder implementation."""

    def __init__(
        self, out_channels: List[int], depth: int = 5, output_stride: int = 32, **kwargs
    ):
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )
        super().__init__(**kwargs)

        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

        del self.fc
        del self.avgpool

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        return {
            16: [self.layer3],
            32: [self.layer4],
        }

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [x]

        if self._depth >= 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            features.append(x)

        if self._depth >= 2:
            x = self.maxpool(x)
            x = self.layer1(x)
            features.append(x)

        if self._depth >= 3:
            x = self.layer2(x)
            features.append(x)

        if self._depth >= 4:
            x = self.layer3(x)
            features.append(x)

        if self._depth >= 5:
            x = self.layer4(x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


resnet_encoders = {
    "resnet18": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/resnet18.imagenet",
                "revision": "3f2325ff978283d47aa6a1d6878ca20565622683",
            },
            "ssl": {
                "repo_id": "smp-hub/resnet18.ssl",
                "revision": "d600d5116aac2e6e595f99f40612074c723c00b2",
            },
            "swsl": {
                "repo_id": "smp-hub/resnet18.swsl",
                "revision": "0e3a35d4d8e344088c14a96eee502a88ac70eae1",
            },
        },
        "params": {
            "out_channels": [3, 64, 64, 128, 256, 512],
            "block": BasicBlock,
            "layers": [2, 2, 2, 2],
        },
    },
    "resnet34": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/resnet34.imagenet",
                "revision": "7a57b34f723329ff020b3f8bc41771163c519d0c",
            },
        },
        "params": {
            "out_channels": [3, 64, 64, 128, 256, 512],
            "block": BasicBlock,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet50": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/resnet50.imagenet",
                "revision": "00cb74e366966d59cd9a35af57e618af9f88efe9",
            },
            "ssl": {
                "repo_id": "smp-hub/resnet50.ssl",
                "revision": "d07daf5b4377f3700c6ac61906b0aafbc4eca46b",
            },
            "swsl": {
                "repo_id": "smp-hub/resnet50.swsl",
                "revision": "b9520cce124f91c6fe7eee45721a2c7954f0d8c0",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
        },
    },
    "resnet101": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/resnet101.imagenet",
                "revision": "cd7c15e8c51da86ae6a084515fdb962d0c94e7d1",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
        },
    },
    "resnet152": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/resnet152.imagenet",
                "revision": "951dd835e9d086628e447b484584c8983f9e1dd0",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": Bottleneck,
            "layers": [3, 8, 36, 3],
        },
    },
    "resnext50_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/resnext50_32x4d.imagenet",
                "revision": "329793c85d62fd340ae42ae39fb905a63df872e7",
            },
            "ssl": {
                "repo_id": "smp-hub/resnext50_32x4d.ssl",
                "revision": "9b67cff77d060c7044493a58c24d1007c1eb06c3",
            },
            "swsl": {
                "repo_id": "smp-hub/resnext50_32x4d.swsl",
                "revision": "52e6e49da61b8e26ca691e1aef2cbb952884057d",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": Bottleneck,
            "layers": [3, 4, 6, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x4d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "ssl": {
                "repo_id": "smp-hub/resnext101_32x4d.ssl",
                "revision": "b39796c8459084d13523b7016c3ef13a2e9e472b",
            },
            "swsl": {
                "repo_id": "smp-hub/resnext101_32x4d.swsl",
                "revision": "3f8355b4892a31f001a832b49b2b01484d48516a",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 4,
        },
    },
    "resnext101_32x8d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/resnext101_32x8d.imagenet",
                "revision": "221af6198d03a4ee88992f78a1ee81b46a52d339",
            },
            "instagram": {
                "repo_id": "smp-hub/resnext101_32x8d.instagram",
                "revision": "44cd927aa6e64673ffe9d31230bad44abc18b823",
            },
            "ssl": {
                "repo_id": "smp-hub/resnext101_32x8d.ssl",
                "revision": "723a95ddeed335c9488c37c6cbef13d779ac8f97",
            },
            "swsl": {
                "repo_id": "smp-hub/resnext101_32x8d.swsl",
                "revision": "58cf0bb65f91365470398080d9588b187d1777c4",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 8,
        },
    },
    "resnext101_32x16d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "instagram": {
                "repo_id": "smp-hub/resnext101_32x16d.instagram",
                "revision": "64e8e320eeae6501185b0627b2429a68e52d050c",
            },
            "ssl": {
                "repo_id": "smp-hub/resnext101_32x16d.ssl",
                "revision": "1283fe03fbb6aa2599b2df24095255acb93c3d5c",
            },
            "swsl": {
                "repo_id": "smp-hub/resnext101_32x16d.swsl",
                "revision": "30ba61bbd4d6af0d955c513dbb4f557b84eb094f",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 16,
        },
    },
    "resnext101_32x32d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "instagram": {
                "repo_id": "smp-hub/resnext101_32x32d.instagram",
                "revision": "c9405de121fdaa275a89de470fb19409e3eeaa86",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 32,
        },
    },
    "resnext101_32x48d": {
        "encoder": ResNetEncoder,
        "pretrained_settings": {
            "instagram": {
                "repo_id": "smp-hub/resnext101_32x48d.instagram",
                "revision": "53e61a962b824ad7027409821f9ac3e3336dd024",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": Bottleneck,
            "layers": [3, 4, 23, 3],
            "groups": 32,
            "width_per_group": 48,
        },
    },
}
