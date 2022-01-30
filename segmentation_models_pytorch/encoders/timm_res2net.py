from ._base import EncoderMixin
from timm.models.resnet import ResNet
from timm.models.res2net import Bottle2neck
import torch.nn as nn


class Res2NetEncoder(ResNet, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.global_pool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.act1),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def make_dilated(self, *args, **kwargs):
        raise ValueError("Res2Net encoders do not support dilated mode")

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


res2net_weights = {
    "timm-res2net50_26w_4s": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_4s-06e79181.pth",  # noqa
    },
    "timm-res2net50_48w_2s": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_48w_2s-afed724a.pth",  # noqa
    },
    "timm-res2net50_14w_8s": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_14w_8s-6527dddc.pth",  # noqa
    },
    "timm-res2net50_26w_6s": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_6s-19041792.pth",  # noqa
    },
    "timm-res2net50_26w_8s": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net50_26w_8s-2c7c9f12.pth",  # noqa
    },
    "timm-res2net101_26w_4s": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2net101_26w_4s-02a759a1.pth",  # noqa
    },
    "timm-res2next50": {
        "imagenet": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-res2net/res2next50_4s-6ef7e7bf.pth",  # noqa
    },
}

pretrained_settings = {}
for model_name, sources in res2net_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            "input_size": [3, 224, 224],
            "input_range": [0, 1],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "num_classes": 1000,
        }


timm_res2net_encoders = {
    "timm-res2net50_26w_4s": {
        "encoder": Res2NetEncoder,
        "pretrained_settings": pretrained_settings["timm-res2net50_26w_4s"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottle2neck,
            "layers": [3, 4, 6, 3],
            "base_width": 26,
            "block_args": {"scale": 4},
        },
    },
    "timm-res2net101_26w_4s": {
        "encoder": Res2NetEncoder,
        "pretrained_settings": pretrained_settings["timm-res2net101_26w_4s"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottle2neck,
            "layers": [3, 4, 23, 3],
            "base_width": 26,
            "block_args": {"scale": 4},
        },
    },
    "timm-res2net50_26w_6s": {
        "encoder": Res2NetEncoder,
        "pretrained_settings": pretrained_settings["timm-res2net50_26w_6s"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottle2neck,
            "layers": [3, 4, 6, 3],
            "base_width": 26,
            "block_args": {"scale": 6},
        },
    },
    "timm-res2net50_26w_8s": {
        "encoder": Res2NetEncoder,
        "pretrained_settings": pretrained_settings["timm-res2net50_26w_8s"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottle2neck,
            "layers": [3, 4, 6, 3],
            "base_width": 26,
            "block_args": {"scale": 8},
        },
    },
    "timm-res2net50_48w_2s": {
        "encoder": Res2NetEncoder,
        "pretrained_settings": pretrained_settings["timm-res2net50_48w_2s"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottle2neck,
            "layers": [3, 4, 6, 3],
            "base_width": 48,
            "block_args": {"scale": 2},
        },
    },
    "timm-res2net50_14w_8s": {
        "encoder": Res2NetEncoder,
        "pretrained_settings": pretrained_settings["timm-res2net50_14w_8s"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottle2neck,
            "layers": [3, 4, 6, 3],
            "base_width": 14,
            "block_args": {"scale": 8},
        },
    },
    "timm-res2next50": {
        "encoder": Res2NetEncoder,
        "pretrained_settings": pretrained_settings["timm-res2next50"],
        "params": {
            "out_channels": (3, 64, 256, 512, 1024, 2048),
            "block": Bottle2neck,
            "layers": [3, 4, 6, 3],
            "base_width": 4,
            "cardinality": 8,
            "block_args": {"scale": 4},
        },
    },
}
