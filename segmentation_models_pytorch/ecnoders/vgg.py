import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision.models.vgg import make_layers
from torchvision.models.vgg import cfg
from torchvision.models.vgg import model_urls


class VGGEncoder(VGG):

    def forward(self, x):

        features = []

        for module in self.features:
            if isinstance(module, nn.MaxPool2d):
                features.append(x)

            x = module(x)

        features.append(x)

        return features[::-1]


vgg_encoders = {

    'vgg11': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'url': model_urls['vgg11'],
        'params': {
            'features': make_layers(cfg['A'], batch_norm=False),
        },
    },

    'vgg11_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'url': model_urls['vgg11_bn'],
        'params': {
            'features': make_layers(cfg['A'], batch_norm=True),
        },
    },

    'vgg13': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'url': model_urls['vgg13'],
        'params': {
            'features': make_layers(cfg['B'], batch_norm=False),
        },
    },

    'vgg13_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'url': model_urls['vgg13_bn'],
        'params': {
            'features': make_layers(cfg['B'], batch_norm=True),
        },
    },

    'vgg16': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'url': model_urls['vgg16'],
        'params': {
            'features': make_layers(cfg['D'], batch_norm=False),
        },
    },

    'vgg16_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'url': model_urls['vgg16_bn'],
        'params': {
            'features': make_layers(cfg['D'], batch_norm=True),
        },
    },

    'vgg19': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'url': model_urls['vgg19'],
        'params': {
            'features': make_layers(cfg['E'], batch_norm=False),
        },
    },

    'vgg19_bn': {
        'encoder': VGGEncoder,
        'out_shapes': (512, 512, 512, 256, 128),
        'url': model_urls['vgg19_bn'],
        'params': {
            'features': make_layers(cfg['E'], batch_norm=True),
        },
    },
}
