from .decoder import PSPDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class PSPNet(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            psp_in_factor=8,
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_aux_output=False,
            classes=21,
            dropout=0.2,
            activation='sigmoid',
    ):

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = PSPDecoder(
            encoder_channels=encoder.out_shapes,
            downsample_factor=psp_in_factor,
            psp_out_channels=psp_out_channels,
            final_channels=classes,
            dropout=dropout,
            aux_output=psp_aux_output,
            use_batchnorm=psp_use_batchnorm,
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'psp-{}'.format(encoder_name)
