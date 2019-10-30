from .decoder import UnetDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead


class Unet(SegmentationModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation

    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: list of numbers of ``Conv2D`` layer filters in decoder blocks
        decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation`` layers
            is used. If 'inplace' InplaceABN will be used, allows to decrease memory consumption.
            One of [True, False, 'inplace']
        decoder_attention_type: attention module used in decoder of the model
            One of [``None``, ``scse``]
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        final_activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        center: if ``True`` add ``Conv2dReLU`` block on encoder head (useful for VGG models)

    Returns:
        ``torch.nn.Module``: **Unet**

    .. _Unet:
        https://arxiv.org/pdf/1505.04597

    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        encoder_depth=5,
        decoder_use_batchnorm=True,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_attention_type=None,
        classes=1,
        activation=None,
        aux_params=None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name, depth=encoder_depth, weights=encoder_weights
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
