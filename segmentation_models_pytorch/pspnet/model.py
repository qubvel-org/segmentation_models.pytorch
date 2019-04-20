from .decoder import PSPDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class PSPNet(EncoderDecoder):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model used as feature
                extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        psp_in_factor: one of 4, 8 and 16. Downsampling rate or in other words backbone depth
            to construct PSP module on it.
        psp_out_channels: number of filters in PSP block.
        psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
                is used.
        psp_aux_output: if ``True`` add auxiliary classification output for encoder training
        psp_dropout: spatial dropout rate between 0 and 1.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: one of [``sigmoid``, ``softmax``, None]

    Returns:
        ``torch.nn.Module``: **PSPNet**

    .. _PSPNet:
        https://arxiv.org/pdf/1612.01105.pdf
    """

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
            activation='softmax',
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
