from .decoder import LinknetDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class Linknet(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            classes=1,
            activation='sigmoid',
    ):

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = LinknetDecoder(
            encoder_channels=encoder.out_shapes,
            prefinal_channels=32,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'link-{}'.format(encoder_name)
