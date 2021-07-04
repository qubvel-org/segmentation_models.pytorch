from ._base import EncoderMixin
from timm.models.resnet import ResNet
from timm.models.resnest import ResNestBottleneck
import torch.nn as nn


class ResNestEncoder(ResNet, EncoderMixin):
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

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("ResNest encoders do not support dilated mode")

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


resnest_weights = {
    'timm-resnest14d': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest14-9c8fe254.pth'
    },
    'timm-resnest26d': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gluon_resnest26-50eb607c.pth'
    },
    'timm-resnest50d': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50-528c19ca.pth',
    },
    'timm-resnest101e': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest101-22405ba7.pth',
    },
    'timm-resnest200e': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest200-75117900.pth',
    },
    'timm-resnest269e': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest269-0cc87c48.pth',
    },
    'timm-resnest50d_4s2x40d': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_4s2x40d-41d14ed0.pth',
    },
    'timm-resnest50d_1s4x24d': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-resnest/resnest50_fast_1s4x24d-d4a4f76f.pth',
    }
}

pretrained_settings = {}
for model_name, sources in resnest_weights.items():
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


timm_resnest_encoders = {
    'timm-resnest14d': {
        'encoder': ResNestEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest14d"],
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': ResNestBottleneck,
            'layers': [1, 1, 1, 1],
            'stem_type': 'deep',
            'stem_width': 32,
            'avg_down': True,
            'base_width': 64,
            'cardinality': 1,
            'block_args': {'radix': 2, 'avd': True, 'avd_first': False}
        }
    },
    'timm-resnest26d': {
        'encoder': ResNestEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest26d"],
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': ResNestBottleneck,
            'layers': [2, 2, 2, 2],
            'stem_type': 'deep',
            'stem_width': 32,
            'avg_down': True,
            'base_width': 64,
            'cardinality': 1,
            'block_args': {'radix': 2, 'avd': True, 'avd_first': False}
        }
    },
    'timm-resnest50d': {
        'encoder': ResNestEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest50d"],
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': ResNestBottleneck,
            'layers': [3, 4, 6, 3],
            'stem_type': 'deep',
            'stem_width': 32,
            'avg_down': True,
            'base_width': 64,
            'cardinality': 1,
            'block_args': {'radix': 2, 'avd': True, 'avd_first': False}
        }
    },
    'timm-resnest101e': {
        'encoder': ResNestEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest101e"],
        'params': {
            'out_channels': (3, 128, 256, 512, 1024, 2048),
            'block': ResNestBottleneck,
            'layers': [3, 4, 23, 3],
            'stem_type': 'deep',
            'stem_width': 64,
            'avg_down': True,
            'base_width': 64,
            'cardinality': 1,
            'block_args': {'radix': 2, 'avd': True, 'avd_first': False}
        }
    },
    'timm-resnest200e': {
        'encoder': ResNestEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest200e"],
        'params': {
            'out_channels': (3, 128, 256, 512, 1024, 2048),
            'block': ResNestBottleneck,
            'layers': [3, 24, 36, 3],
            'stem_type': 'deep',
            'stem_width': 64,
            'avg_down': True,
            'base_width': 64,
            'cardinality': 1,
            'block_args': {'radix': 2, 'avd': True, 'avd_first': False}
        }
    },
    'timm-resnest269e': {
        'encoder': ResNestEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest269e"],
        'params': {
            'out_channels': (3, 128, 256, 512, 1024, 2048),
            'block': ResNestBottleneck,
            'layers': [3, 30, 48, 8],
            'stem_type': 'deep',
            'stem_width': 64,
            'avg_down': True,
            'base_width': 64,
            'cardinality': 1,
            'block_args': {'radix': 2, 'avd': True, 'avd_first': False}
        },
    },
    'timm-resnest50d_4s2x40d': {
        'encoder': ResNestEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest50d_4s2x40d"],
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': ResNestBottleneck,
            'layers': [3, 4, 6, 3],
            'stem_type': 'deep',
            'stem_width': 32,
            'avg_down': True,
            'base_width': 40,
            'cardinality': 2,
            'block_args': {'radix': 4, 'avd': True, 'avd_first': True}
        }
    },
    'timm-resnest50d_1s4x24d': {
        'encoder': ResNestEncoder,
        "pretrained_settings": pretrained_settings["timm-resnest50d_1s4x24d"],
        'params': {
            'out_channels': (3, 64, 256, 512, 1024, 2048),
            'block': ResNestBottleneck,
            'layers': [3, 4, 6, 3],
            'stem_type': 'deep',
            'stem_width': 32,
            'avg_down': True,
            'base_width': 24,
            'cardinality': 4,
            'block_args': {'radix': 1, 'avd': True, 'avd_first': True}
        }
    }
}
