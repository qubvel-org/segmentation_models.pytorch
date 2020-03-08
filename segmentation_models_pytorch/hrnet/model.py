from typing import Optional, Union
import os

from .decoder import HRNetDecoder
from ..base import SegmentationModel
from ..base import SegmentationHead, ClassificationHead
from ..encoders.hrnet import hrnet_encoders

encoders = {}
encoders.update(hrnet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None):
    """
    temporary 'get_encoder' as weights are not on server,
    but located locally
    """
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        weights_path = os.path.join(os.getcwd(), settings["url"])
        encoder.init_weights(weights_path)

    encoder.set_in_channels(in_channels)

    return encoder


class HRNet(SegmentationModel):
    """HRNetv2_ implemetation from "Deep High-Resolution Representation Learning for Visual Recognition"
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_depth: number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        decoder_channels: a number of convolution filters in ASPP module (default 256).
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation (str, callable): activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax2d``, callable, None]
        decoder_dropout: spatial dropout rate in range (0, 1).
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)
    Returns:
        ``torch.nn.Module``: **HRNet**
    .. _HRNet:
        https://arxiv.org/abs/1908.07919
    """

    def __init__(
        self,
        encoder_name: str = "hrnetv2_32",
        encoder_depth: int = 4,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: int = 512,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        decoder_dropout: float = 0.0,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = HRNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            dropout=decoder_dropout,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            kernel_size=3,
            activation=activation,
        )

        if aux_params:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                **aux_params
            )
        else:
            self.classification_head = None

        self.name = encoder_name
        self.initialize()