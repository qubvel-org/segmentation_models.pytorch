from timm import create_model
import torch.nn as nn
from ._base import EncoderMixin


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileNetV3Encoder(nn.Module, EncoderMixin):
    def __init__(self, model, width_mult, depth=5, **kwargs):
        super().__init__()
        self._depth = depth
        if 'small' in str(model):
            self.mode = 'small'
            self._out_channels = (16*width_mult, 16*width_mult, 24*width_mult, 48*width_mult, 576*width_mult)
            self._out_channels = tuple(map(make_divisible, self._out_channels))
        elif 'large' in str(model):
            self.mode = 'large'
            self._out_channels = (16*width_mult, 24*width_mult, 40*width_mult, 112*width_mult, 960*width_mult)
            self._out_channels = tuple(map(make_divisible, self._out_channels))
        else:
            self.mode = 'None'
            raise ValueError(
                'MobileNetV3 mode should be small or large, got {}'.format(self.mode))
        self._out_channels = (3,) + self._out_channels
        self._in_channels = 3
        # minimal models replace hardswish with relu
        model = create_model(model_name=model,
                             scriptable=True,   # torch.jit scriptable
                             exportable=True,   # onnx export
                             features_only=True)
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.blocks = model.blocks

    def get_stages(self):
        if self.mode == 'small':
            return [
                nn.Identity(),
                nn.Sequential(self.conv_stem, self.bn1, self.act1),
                self.blocks[0],
                self.blocks[1],
                self.blocks[2:4],
                self.blocks[4:],
            ]
        elif self.mode == 'large':
            return [
                nn.Identity(),
                nn.Sequential(self.conv_stem, self.bn1, self.act1, self.blocks[0]),
                self.blocks[1],
                self.blocks[2],
                self.blocks[3:5],
                self.blocks[5:],
            ]
        else:
            ValueError('MobileNetV3 mode should be small or large, got {}'.format(self.mode))

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('conv_head.weight')
        state_dict.pop('conv_head.bias')
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')
        super().load_state_dict(state_dict, **kwargs)


mobilenetv3_weights = {
    'tf_mobilenetv3_large_075': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth'
    },
    'tf_mobilenetv3_large_100': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth'
    },
    'tf_mobilenetv3_large_minimal_100': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth'
    },
    'tf_mobilenetv3_small_075': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth'
    },
    'tf_mobilenetv3_small_100': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth'
    },
    'tf_mobilenetv3_small_minimal_100': {
        'imagenet': 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth'
    },


}

pretrained_settings = {}
for model_name, sources in mobilenetv3_weights.items():
    pretrained_settings[model_name] = {}
    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'input_space': 'RGB',
        }


timm_mobilenetv3_encoders = {
    'timm-mobilenetv3_large_075': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_large_075'],
        'params': {
            'model': 'tf_mobilenetv3_large_075',
            'width_mult': 0.75
        }
    },
    'timm-mobilenetv3_large_100': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_large_100'],
        'params': {
            'model': 'tf_mobilenetv3_large_100',
            'width_mult': 1.0
        }
    },
    'timm-mobilenetv3_large_minimal_100': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_large_minimal_100'],
        'params': {
            'model': 'tf_mobilenetv3_large_minimal_100',
            'width_mult': 1.0
        }
    },
    'timm-mobilenetv3_small_075': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_075'],
        'params': {
            'model': 'tf_mobilenetv3_small_075',
            'width_mult': 0.75
        }
    },
    'timm-mobilenetv3_small_100': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_100'],
        'params': {
            'model': 'tf_mobilenetv3_small_100',
            'width_mult': 1.0
        }
    },
    'timm-mobilenetv3_small_minimal_100': {
        'encoder': MobileNetV3Encoder,
        'pretrained_settings': pretrained_settings['tf_mobilenetv3_small_minimal_100'],
        'params': {
            'model': 'tf_mobilenetv3_small_minimal_100',
            'width_mult': 1.0
        }
    },
}
