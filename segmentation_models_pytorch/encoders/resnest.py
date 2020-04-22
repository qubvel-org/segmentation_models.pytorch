from torch import nn
from resnest.torch.resnet import ResNet, Bottleneck
from resnest.torch.resnest import resnest_model_urls

from ._base import EncoderMixin


class ResNestEncoder(ResNet, EncoderMixin):

    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.fc
        del self.avgpool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias")
        state_dict.pop("fc.weight")
        super().load_state_dict(state_dict, **kwargs)


def get_pretrained_settings(name):
    return {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": resnest_model_urls[name],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }


resnest_encoders = {
    "resnest50": {
        "encoder": ResNestEncoder,
        "pretrained_settings": get_pretrained_settings("resnest50"),
        "params": dict(block=Bottleneck, layers=[3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, out_channels=(3, 64, 256, 512, 1024, 2048)),
    },

    "resnest101": {
        "encoder": ResNestEncoder,
        "pretrained_settings": get_pretrained_settings("resnest101"),
        "params": dict(block=Bottleneck, layers=[3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, out_channels=(3, 64, 256, 512, 1024, 2048)),
    },

    "resnest200": {
        "encoder": ResNestEncoder,
        "pretrained_settings": get_pretrained_settings("resnest200"),
        "params": dict(block=Bottleneck, layers=[3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, out_channels=(3, 64, 256, 512, 1024, 2048)),
    },

    "resnest269": {
        "encoder": ResNestEncoder,
        "pretrained_settings": get_pretrained_settings("resnest269"),
        "params": dict(block=Bottleneck, layers=[3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, out_channels=(3, 64, 256, 512, 1024, 2048)),
    },
}