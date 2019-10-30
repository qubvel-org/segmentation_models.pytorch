import torch.nn as nn
from pretrainedmodels.models.inceptionv4 import InceptionV4, BasicConv2d
from pretrainedmodels.models.inceptionv4 import pretrained_settings

from .base import EncoderMixin

class InceptionV4Encoder(InceptionV4, EncoderMixin):

    def __init__(self, out_channels, *args, depth=5, **kwargs):
        super().__init__(*args, **kwargs)

        # self.features[0] = BasicConv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1)
        # self.features[1] = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self._chunks = [3, 5, 9, 15]
        self._out_channels = out_channels
        self._depth = depth

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # remove linear layers
        del self.last_linear

    def forward(self, x):

        modules = [
            nn.Identity(),
            self.features[:self._chunks[0]],
            self.features[self._chunks[0]:self._chunks[1]],
            self.features[self._chunks[1]:self._chunks[2]],
            self.features[self._chunks[2]:self._chunks[3]],
            self.features[self._chunks[3]:],
        ]

        features = []
        for i in range(self._depth + 1):
            x = modules[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


inceptionv4_encoders = {
    'inceptionv4': {
        'encoder': InceptionV4Encoder,
        'pretrained_settings': pretrained_settings['inceptionv4'],
        'out_shapes': (1536, 1024, 384, 192, 64),
        'out_channels': (3, 64, 192, 384, 1024, 1536),
        'params': {
            'num_classes': 1001,
        }
    }
}
