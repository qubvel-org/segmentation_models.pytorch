from .decoder import LinknetDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class Linknet(EncoderDecoder):
    """Linknet_ is a fully convolution neural network for fast image semantic segmentation

    Note:
        This implementation by default has 4 skip connections (original - 3).

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: one of [``sigmoid``, ``softmax``, None]

    Returns:
        ``torch.nn.Module``: **Linknet**

    .. _Linknet:
        https://arxiv.org/pdf/1707.03718.pdf
    """

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
