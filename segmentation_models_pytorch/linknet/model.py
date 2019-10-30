from .decoder import LinknetDecoder
from ..base import SegmentationHead, SegmentationModel, ClassificationHead
from ..encoders import get_encoder


class Linknet(SegmentationModel):
    """Linknet_ is a fully convolution neural network for fast image semantic segmentation

    Note:
        This implementation by default has 4 skip connections (original - 3).

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
    Returns:
        ``torch.nn.Module``: **Linknet**

    .. _Linknet:
        https://arxiv.org/pdf/1707.03718.pdf
    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            classes=1,
            activation=None,
            aux_params=None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            depth=encoder_depth,
            weights=encoder_weights
        )

        self.decoder = LinknetDecoder(
            encoder_channels=self.encoder.out_channels,
            n_blocks=encoder_depth,
            prefinal_channels=32,
            use_batchnorm=decoder_use_batchnorm,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=32,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = 'link-{}'.format(encoder_name)
