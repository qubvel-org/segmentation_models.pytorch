import torch.nn as nn
from pretrainedmodels.models.inceptionv4 import InceptionV4, BasicConv2d
from pretrainedmodels.models.inceptionv4 import pretrained_settings


class InceptionV4Encoder(InceptionV4):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = 3
        self.features[0] = BasicConv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.features[1] = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.chunks = [3, 5, 9, 15]

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
        x0 = self.features[:self.chunks[0]](x)
        x1 = self.features[self.chunks[0]:self.chunks[1]](x0)
        x2 = self.features[self.chunks[1]:self.chunks[2]](x1)
        x3 = self.features[self.chunks[2]:self.chunks[3]](x2)
        x4 = self.features[self.chunks[3]:](x3)

        features = [x4, x3, x2, x1, x0]
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
        'params': {
            'num_classes': 1001,
        }
    }
}
