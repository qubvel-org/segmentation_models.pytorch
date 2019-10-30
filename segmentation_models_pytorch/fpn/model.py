from .decoder import FPNDecoder
from ..base import SegmentationModel, SegmentationHead, ClassificationHead
from ..encoders import get_encoder


class FPN(SegmentationModel):
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
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            decoder_merge_policy='add',
            decoder_dropout=0.2,
            classes=1,
            activation=None,
            upsampling=4,
            aux_params=None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = FPNDecoder(
            encoder_channels=self.encoder.out_channels,
            encoder_depth=encoder_depth,
            pyramid_channels=decoder_pyramid_channels,
            segmentation_channels=decoder_segmentation_channels,
            dropout=decoder_dropout,
            merge_policy=decoder_merge_policy
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = 'fpn-{}'.format(encoder_name)
