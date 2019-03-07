import re
import torch.nn as nn

from pretrainedmodels.models.torchvision_models import pretrained_settings
from torchvision.models.densenet import DenseNet


class DenseNetEncoder(DenseNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.classifier
        self.initialize()

    @staticmethod
    def _transition(x, transition_block):
        for module in transition_block:
            x = module(x)
            if isinstance(module, nn.ReLU):
                skip = x
        return x, skip

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x0 = x

        x = self.features.pool0(x)
        x = self.features.denseblock1(x)
        x, x1 = self._transition(x, self.features.transition1)

        x = self.features.denseblock2(x)
        x, x2 = self._transition(x, self.features.transition2)

        x = self.features.denseblock3(x)
        x, x3 = self._transition(x, self.features.transition3)

        x = self.features.denseblock4(x)
        x4 = self.features.norm5(x)

        features = [x4, x3, x2, x1, x0]
        return features

    def load_state_dict(self, state_dict):
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        # remove linear
        state_dict.pop('classifier.bias')
        state_dict.pop('classifier.weight')

        super().load_state_dict(state_dict)


densenet_encoders = {
    'densenet121': {
        'encoder': DenseNetEncoder,
        'pretrained_settings': pretrained_settings['densenet121'],
        'out_shapes': (1024, 1024, 512, 256, 64),
        'params': {
            'num_init_features': 64,
            'growth_rate': 32,
            'block_config': (6, 12, 24, 16),
        }
    },

    'densenet169': {
        'encoder': DenseNetEncoder,
        'pretrained_settings': pretrained_settings['densenet169'],
        'out_shapes': (1664, 1280, 512, 256, 64),
        'params': {
            'num_init_features': 64,
            'growth_rate': 32,
            'block_config': (6, 12, 32, 32),
        }
    },

    'densenet201': {
        'encoder': DenseNetEncoder,
        'pretrained_settings': pretrained_settings['densenet201'],
        'out_shapes': (1920, 1792, 512, 256, 64),
        'params': {
            'num_init_features': 64,
            'growth_rate': 32,
            'block_config': (6, 12, 48, 32),
        }
    },

    'densenet161': {
        'encoder': DenseNetEncoder,
        'pretrained_settings': pretrained_settings['densenet161'],
        'out_shapes': (2208, 2112, 768, 384, 96),
        'params': {
            'num_init_features': 96,
            'growth_rate': 48,
            'block_config': (6, 12, 36, 24),
        }
    },

}
