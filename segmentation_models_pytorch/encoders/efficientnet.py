from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, get_model_params
import torch.nn as nn
import torch

from .base import EncoderMixin

class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, skip_connections, model_name, out_channels, depth=5):

        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)

        self._skip_connections = list(skip_connections) + [len(self._blocks)]
        self._out_channels = out_channels
        self._depth = depth

        del self._fc

    def forward(self, x):

        features = [x]

        if self._depth > 0:
            x = self._swish(self._bn0(self._conv_stem(x)))
            features.append(x)

        if self._depth > 1:
            skip_connection_idx = 0
            for idx, block in enumerate(self._blocks):
                drop_connect_rate = self._global_params.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self._blocks)
                x = block(x, drop_connect_rate=drop_connect_rate)
                if idx == self._skip_connections[skip_connection_idx] - 1:
                    skip_connection_idx += 1
                    features.append(x)
                    if skip_connection_idx + 1 == self._depth:
                        break

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('_fc.bias')
        state_dict.pop('_fc.weight')
        super().load_state_dict(state_dict, **kwargs)


def _get_pretrained_settings(encoder):
    pretrained_settings = {
        'imagenet': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'url': url_map[encoder],
            'input_space': 'RGB',
            'input_range': [0, 1]
        }
    }
    return pretrained_settings


efficient_net_encoders = {
    'efficientnet-b0': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (320, 112, 40, 24, 32),
        'out_channels': (3, 32, 24, 40, 112, 320),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b0'),
        'params': {
            'skip_connections': [3, 5, 9],
            'model_name': 'efficientnet-b0'
        }
    },
    'efficientnet-b1': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (320, 112, 40, 24, 32),
        'out_channels': (3, 32, 24, 40, 112, 320),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b1'),
        'params': {
            'skip_connections': [5, 8, 16],
            'model_name': 'efficientnet-b1'
        }
    },
    'efficientnet-b2': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (352, 120, 48, 24, 32),
        'out_channels': (3, 32, 24, 48, 120, 352),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b2'),
        'params': {
            'skip_connections': [5, 8, 16],
            'model_name': 'efficientnet-b2'
        }
    },
    'efficientnet-b3': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (384, 136, 48, 32, 40),
        'out_channels': (3, 40, 32, 48, 136, 384),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b3'),
        'params': {
            'skip_connections': [5, 8, 18],
            'model_name': 'efficientnet-b3'
        }
    },
    'efficientnet-b4': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (448, 160, 56, 32, 48),
        'out_channels': (3, 48, 32, 56, 160, 448),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b4'),
        'params': {
            'skip_connections': [6, 10, 22],
            'model_name': 'efficientnet-b4'
        }
    },
    'efficientnet-b5': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (512, 176, 64, 40, 48),
        'out_channels': (3, 48, 40, 64, 176, 512),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b5'),
        'params': {
            'skip_connections': [8, 13, 27],
            'model_name': 'efficientnet-b5'
        }
    },
    'efficientnet-b6': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (576, 200, 72, 40, 56),
        'out_channels': (3, 56, 40, 72, 200, 576),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b6'),
        'params': {
            'skip_connections': [9, 15, 31],
            'model_name': 'efficientnet-b6'
        }
    },
    'efficientnet-b7': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (640, 224, 80, 48, 64),
        'out_channels': (3, 64, 48, 80, 224, 640),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b7'),
        'params': {
            'skip_connections': [11, 18, 38],
            'model_name': 'efficientnet-b7'
        }
    }
}
