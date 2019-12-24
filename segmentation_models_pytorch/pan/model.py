from typing import Optional, Union
from .decoder import PANDecoder
from ..encoders import get_encoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead


class PAN(SegmentationModel):
    """ Implementation of _PAN (Pyramid Attention Network).
    Currently works with shape of input tensor >= [B x C x 128 x 128] for pytorch <= 1.1.0
    and with shape of input tensor >= [B x C x 256 x 256] for pytorch == 1.3.1


    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_dilation: Flag to use dilation in encoder last layer.
            Doesn't work with [``*ception*``, ``vgg*``, ``densenet*``] backbones, default is True.
        decoder_channels: Number of ``Conv2D`` layer filters in decoder blocks
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation: activation function to apply after final convolution;
            One of [``sigmoid``, ``softmax``, ``logsoftmax``, ``identity``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 4 to preserve input -> output spatial shape identity)

        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)

    Returns:
        ``torch.nn.Module``: **PAN**

    .. _PAN:
        https://arxiv.org/abs/1805.10180

    """

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_weights: str = "imagenet",
            encoder_dilation: bool = True,
            decoder_channels: int = 32,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        if encoder_dilation:
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )

        self.decoder = PANDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "pan-{}".format(encoder_name)
        self.initialize()
