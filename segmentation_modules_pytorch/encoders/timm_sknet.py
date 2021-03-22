from ._base import EncoderMixin
from timm.models.resnet import ResNet
from timm.models.sknet import SelectiveKernelBottleneck, SelectiveKernelBasic
import torch.nn as nn


class SkNetEncoder(ResNet, EncoderMixin):
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


sknet_weights = {
    'timm-skresnet18': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnet18_ra-4eec2804.pth'
    },
    'timm-skresnet34': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnet34_ra-bdc0ccde.pth'
    },
    'timm-skresnext50_32x4d': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/skresnext50_ra-f40e40bf.pth',
    }
}

pretrained_settings = {}
for model_name, sources in sknet_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }

timm_sknet_encoders = {
    'timm-skresnet18': {
        'encoder': SkNetEncoder,
        "pretrained_settings": pretrained_settings["timm-skresnet18"],
        'params': {
            'out_channels': (3, 64, 64, 128, 256, 512),
            'block': SelectiveKernelBasic,
            'layers': [2, 2, 2, 2],
            'zero_init_last_bn': False,
            'block_args': {'sk_kwargs': {'min_attn_channels': 16, 'attn_reduction': 8, 'split_input': True}}
        }
    },
    'timm-skresnet34': {
        'encoder': SkNetEncoder,
        "pretrained_settings": pretrained_settings["timm-skresnet34"],
        'params': {
            'out_channels': (3, 64, 64, 128, 256, 512),
            'block': SelectiveKernelBasic,
            'layers': [3, 4, 6, 3],
            'zero_init_last_bn': False,
            'block_args': {'sk_kwargs': {'min_attn_channels': 16, 'attn_reduction': 8, 'split_input': True}}
        }
    },
    'timm-skresnext50_32x4d': {
        'encoder': SkNetEncoder,
        "pretrained_settings": pretrained_settings["timm-skresnext50_32x4d"],
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': SelectiveKernelBottleneck,
            'layers': [3, 4, 6, 3],
            'zero_init_last_bn': False,
            'cardinality': 32,
            'base_width': 4
        }
    }
}
