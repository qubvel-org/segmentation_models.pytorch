from .decoder import FPNDecoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class FPN(EncoderDecoder):
    """FPN_ is a fully convolution neural network for image semantic segmentation
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_pyramid_channels: a number of convolution filters in Feature Pyramid of FPN_.
        decoder_segmentation_channels: a number of convolution filters in segmentation head of FPN_.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        dropout: spatial dropout rate in range (0, 1).
        activation: activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
        final_upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)
        decoder_merge_policy: determines how to merge outputs inside FPN.
            One of [``add``, ``cat``]
    Returns:
        ``torch.nn.Module``: **FPN**

    .. _FPN:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf

    """

    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            classes=1,
            dropout=0.2,
            activation='sigmoid',
            final_upsampling=4,
            decoder_merge_policy='add'
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = FPNDecoder(
            encoder_channels=encoder.out_shapes,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            final_channels=classes,
            dropout=dropout,
            final_upsampling=final_upsampling,
            merge_policy=decoder_merge_policy
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'fpn-{}'.format(encoder_name)
