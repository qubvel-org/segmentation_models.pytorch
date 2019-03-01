import torch.nn as nn
from .decoder import UnetDecoder
from .decoder import CenterBlock

from ..ecnoders import get_encoder


class Unet(nn.Module):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_pretrained=True,
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False, # usefull for VGG models
    ):

        super().__init__()


        self.encoder = get_encoder(encoder_name, pretrained=encoder_pretrained)
        out_shapes = self.encoder.out_shapes

        if center:
            self.center = CenterBlock(out_shapes[0], out_shapes[0], out_shapes[0],
                                      use_batchnorm=decoder_use_batchnorm)
        else:
            self.center = center

        self.decoder = UnetDecoder(
            self.compute_channels(self.encoder.out_shapes, decoder_channels),
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
        )

        # define activation function
        if activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid" or "softmax"')

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        x = self.encoder(x)
        if self.center:
            head, features = x[0], x[1:]
            head = self.center(head)
            x = [head] + features
        x = self.decoder(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.activation(x)
        return x
