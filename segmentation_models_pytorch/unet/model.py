import torch
import torch.nn as nn
from .decoder import UnetDecoder

from ..encoders import get_encoder


class Unet(nn.Module):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_pretrained=True,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
    ):

        super().__init__()

        self.encoder = get_encoder(encoder_name, pretrained=encoder_pretrained)
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
        )

        # define activation function
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid" or "softmax"')

        self.name = 'u-{}'.format(encoder_name)

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
