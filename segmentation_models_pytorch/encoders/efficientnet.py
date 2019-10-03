from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import relu_fn, url_map, get_model_params
import torch.nn as nn
import torch


class EfficientNetEncoder(EfficientNet):
    def __init__(self, skip_connections, model_name):
        blocks_args, global_params = get_model_params(model_name, override_params=None)

        super().__init__(blocks_args, global_params)
        self._skip_connections = list(skip_connections)
        self._skip_connections.append(len(self._blocks))
        
        del self._fc
        
    def forward(self, x):
        result = []
        x = relu_fn(self._bn0(self._conv_stem(x)))
        result.append(x)

        skip_connection_idx = 0
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self._skip_connections[skip_connection_idx] - 1:
                skip_connection_idx += 1
                result.append(x)

        return list(reversed(result))

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
        'pretrained_settings': _get_pretrained_settings('efficientnet-b0'),
        'params': {
            'skip_connections': [3, 5, 9],
            'model_name': 'efficientnet-b0'
        }
    },
    'efficientnet-b1': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (320, 112, 40, 24, 32),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b1'),
        'params': {
            'skip_connections': [5, 8, 16],
            'model_name': 'efficientnet-b1'
        }
    },
    'efficientnet-b2': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (352, 120, 48, 24, 32),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b2'),
        'params': {
            'skip_connections': [5, 8, 16],
            'model_name': 'efficientnet-b2'
        }
    },
    'efficientnet-b3': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (384, 136, 48, 32, 40),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b3'),
        'params': {
            'skip_connections': [5, 8, 18],
            'model_name': 'efficientnet-b3'
        }
    },
    'efficientnet-b4': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (448, 160, 56, 32, 48),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b4'),
        'params': {
            'skip_connections': [6, 10, 22],
            'model_name': 'efficientnet-b4'
        }
    },
    'efficientnet-b5': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (512, 176, 64, 40, 48),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b5'),
        'params': {
            'skip_connections': [8, 13, 27],
            'model_name': 'efficientnet-b5'
        }
    },
    'efficientnet-b6': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (576, 200, 72, 40, 56),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b6'),
        'params': {
            'skip_connections': [9, 15, 31],
            'model_name': 'efficientnet-b6'
        }
    },
    'efficientnet-b7': {
        'encoder': EfficientNetEncoder,
        'out_shapes': (640, 224, 80, 48, 64),
        'pretrained_settings': _get_pretrained_settings('efficientnet-b7'),
        'params': {
            'skip_connections': [11, 18, 38],
            'model_name': 'efficientnet-b7'
        }
    }
}
