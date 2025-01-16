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
import torch.nn as nn

from torchvision.models.vgg import VGG
from torchvision.models.vgg import make_layers

from typing import List, Union

from ._base import EncoderMixin

# fmt: off
cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
# fmt: on


class VGGEncoder(VGG, EncoderMixin):
    def __init__(
        self,
        out_channels: List[int],
        config: List[Union[int, str]],
        batch_norm: bool = False,
        depth: int = 5,
        output_stride: int = 32,
        **kwargs,
    ):
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )
        super().__init__(make_layers(config, batch_norm=batch_norm), **kwargs)

        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride
        self._out_indexes = [
            i - 1
            for i, module in enumerate(self.features)
            if isinstance(module, nn.MaxPool2d)
        ]
        self._out_indexes.append(len(self.features) - 1)

        del self.classifier

    def make_dilated(self, *args, **kwargs):
        raise ValueError(
            "'VGG' models do not support dilated mode due to Max Pooling"
            " operations for downsampling!"
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        depth = 0

        for i, module in enumerate(self.features):
            x = module(x)

            if i in self._out_indexes:
                features.append(x)
                depth += 1

            # torchscript does not support break in cycle, so we just
            # go over all modules and then slice number of features
            if not torch.jit.is_scripting() and depth > self._depth:
                break

        features = features[: self._depth + 1]

        return features

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith("classifier"):
                state_dict.pop(k, None)
        super().load_state_dict(state_dict, **kwargs)


pretrained_settings = {
    "vgg11": {
        "imagenet": {
            "url": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "vgg11_bn": {
        "imagenet": {
            "url": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "vgg13": {
        "imagenet": {
            "url": "https://download.pytorch.org/models/vgg13-c768596a.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "vgg13_bn": {
        "imagenet": {
            "url": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "vgg16": {
        "imagenet": {
            "url": "https://download.pytorch.org/models/vgg16-397923af.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "vgg16_bn": {
        "imagenet": {
            "url": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "vgg19": {
        "imagenet": {
            "url": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
    "vgg19_bn": {
        "imagenet": {
            "url": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
            "input_space": "RGB",
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }
    },
}

vgg_encoders = {
    "vgg11": {
        "encoder": VGGEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/vgg11.imagenet",
                "revision": "ad8b90e1051c38fdbf399cf5016886a1be357390",
            },
        },
        "params": {
            "out_channels": [64, 128, 256, 512, 512, 512],
            "config": cfg["A"],
            "batch_norm": False,
        },
    },
    "vgg11_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/vgg11_bn.imagenet",
                "revision": "59757f9215032c9f092977092d57d26a9df7fd9c",
            },
        },
        "params": {
            "out_channels": [64, 128, 256, 512, 512, 512],
            "config": cfg["A"],
            "batch_norm": True,
        },
    },
    "vgg13": {
        "encoder": VGGEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/vgg13.imagenet",
                "revision": "1b70ff2580f101a8007a48b51e2b5d1e5925dc42",
            },
        },
        "params": {
            "out_channels": [64, 128, 256, 512, 512, 512],
            "config": cfg["B"],
            "batch_norm": False,
        },
    },
    "vgg13_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/vgg13_bn.imagenet",
                "revision": "9be454515193af6612261b7614fe90607e27b143",
            },
        },
        "params": {
            "out_channels": [64, 128, 256, 512, 512, 512],
            "config": cfg["B"],
            "batch_norm": True,
        },
    },
    "vgg16": {
        "encoder": VGGEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/vgg16.imagenet",
                "revision": "49d74b799006ee252b86e25acd6f1fd8ac9a99c1",
            },
        },
        "params": {
            "out_channels": [64, 128, 256, 512, 512, 512],
            "config": cfg["D"],
            "batch_norm": False,
        },
    },
    "vgg16_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/vgg16_bn.imagenet",
                "revision": "2c186d02fb519e93219a99a1c2af6295aef0bf0d",
            },
        },
        "params": {
            "out_channels": [64, 128, 256, 512, 512, 512],
            "config": cfg["D"],
            "batch_norm": True,
        },
    },
    "vgg19": {
        "encoder": VGGEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/vgg19.imagenet",
                "revision": "2853d00d7bca364dbb98be4d6afa347e5aeec1f6",
            },
        },
        "params": {
            "out_channels": [64, 128, 256, 512, 512, 512],
            "config": cfg["E"],
            "batch_norm": False,
        },
    },
    "vgg19_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/vgg19_bn.imagenet",
                "revision": "f09a924cb0d201ea6f61601df9559141382271d7",
            },
        },
        "params": {
            "out_channels": [64, 128, 256, 512, 512, 512],
            "config": cfg["E"],
            "batch_norm": True,
        },
    },
}
