from pretrainedmodels.models.inceptionresnetv2 import InceptionResNetV2


class InceptionResNetV2Encoder(InceptionResNetV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # correct conv paddings
        self.conv2d_1a.conv.padding = (1, 1)
        self.conv2d_2a.conv.padding = (1, 1)
        self.conv2d_2b.conv.padding = (1, 1)
        self.conv2d_4a.conv.padding = (1, 1)

        # maxpool paddings
        self.maxpool_3a.padding = (1, 1)
        self.maxpool_5a.padding = (1, 1)

        # mixed 6a
        self.mixed_6a.branch0.conv.padding = (1, 1)
        self.mixed_6a.branch1[-1].conv.padding = (1, 1)
        self.mixed_6a.branch2.padding = (1, 1)

        # mixed 7a
        self.mixed_7a.branch0[-1].conv.padding = (1, 1)
        self.mixed_7a.branch1[-1].conv.padding = (1, 1)
        self.mixed_7a.branch2[-1].conv.padding = (1, 1)
        self.mixed_7a.branch3.padding = (1, 1)

        # remove linear layers
        self.avgpool_1a = None
        self.last_linear = None

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x0 = x

        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x1 = x

        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x2 = x

        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x3 = x

        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x4 = x

        features = [x4, x3, x2, x1, x0]

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


inception_encoders = {
    'inceptionresnetv2': {
        'encoder': InceptionResNetV2Encoder,
        'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
        'out_shapes': (1536, 1088, 320, 192, 64),
        'params': {
            'num_classes': 1000,
        }

    }
}
