from .decoder import FPNDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class FPN(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_pyramid_channels=256,
            decoder_segmenation_channels=128,
            decoder_use_batchnorm=True,
            classes=1,
            dropout=0.2,
            activation='sigmoid',
    ):

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = FPNDecoder(
            encoder_channels=encoder.out_shapes,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmenation_channels,
            final_channels=classes,
            dropout=dropout,
            use_batchnorm=decoder_use_batchnorm,
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'fpn-{}'.format(encoder_name)
