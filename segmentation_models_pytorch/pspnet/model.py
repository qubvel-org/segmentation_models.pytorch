from .decoder import PSPDecoder
from ..encoders import get_encoder

from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead


class PSPNet(SegmentationModel):
    """PSPNet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model used as feature
                extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        psp_in_factor: one of 4, 8 and 16. Downsampling rate or in other words backbone depth
            to construct PSP module on it.
        psp_out_channels: number of filters in PSP block.
        psp_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        psp_aux_output: if ``True`` add auxiliary classification output for encoder training
        psp_dropout: spatial dropout rate between 0 and 1.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
    Returns:
        ``torch.nn.Module``: **PSPNet**

    .. _PSPNet:
        https://arxiv.org/pdf/1612.01105.pdf
    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            encoder_depth=3,
            psp_out_channels=512,
            psp_use_batchnorm=True,
            psp_dropout=0.2,
            classes=1,
            activation=None,
            upsampling=8,
            aux_params=None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            depth=encoder_depth,
            weights=encoder_weights
        )

        self.decoder = PSPDecoder(
            encoder_channels=self.encoder.out_channels,
            use_batchnorm=psp_use_batchnorm,
            out_channels=psp_out_channels,
            dropout=psp_dropout,

        )

        self.segmentation_head = SegmentationHead(
            in_channels=psp_out_channels,
            out_channels=classes,
            kernel_size=3,
            activation=activation,
            upsampling=upsampling,
        )

        if aux_params:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                **aux_params,
            )
        else:
            self.classification_head = None

        self.name = 'psp-{}'.format(encoder_name)
