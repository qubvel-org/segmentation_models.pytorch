import torch
import torch.nn as nn
from .decoder import UnetDecoder

from ..base.model import EncoderDecoder
from ..encoders import get_encoder


class Unet(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
    ):

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
        )

        # define activation function
        if activation == 'softmax':
            activation_fn = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            activation_fn = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid" or "softmax"')

        super().__init__(encoder, decoder, activation_fn)

        if encoder_weights is not None:
            self.set_preprocessing_params(
                input_size=encoder.input_size,
                input_space=encoder.input_space,
                input_range=encoder.input_range,
                mean=encoder.mean,
                std=encoder.std,
            )

        self.name = 'u-{}'.format(encoder_name)
