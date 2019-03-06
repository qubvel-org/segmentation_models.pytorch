import numpy as np
import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.activation = activation
        
        self._input_size = None
        self._input_space = None
        self._input_range = None
        self._std = None
        self._mean = None

    @property
    def input_size(self):
        return self._input_size

    @property
    def input_space(self):
        return self._input_space

    @property
    def input_range(self):
        return self._input_range

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def set_preprocessing_params(
            self,
            input_size=(3, None, None),
            input_space='RGB',
            input_range=(0, 1),
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
    ):
        self._input_range = input_range
        self._input_size = input_size
        self._input_space = input_space
        self._mean = mean
        self._std = std

    def preprocess_input(self, x, x_space='RGB'):

        if x.ndim == 2:
            x = np.expand_dims(x, 0)
        if x.ndim == 3:
            x = np.expand_dims(x, 0)

        if x_space != self.input_space:
            x = x[:, ::-1].copy()

        if x.max() > self.input_range[1]:
            x /= 255.

        if self._mean is not None:
            mean = np.array(self.mean)[None, :, None, None]
            x -= mean

        if self._std is not None:
            std = np.array(self.std)[None, :, None, None]
            x /= std

        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            x = self.activation(x)

        return x
