import torch.nn as nn

from typing import Optional, Union
from torchvision.models.segmentation import deeplabv3

from ..base import SegmentationModel, SegmentationHead, ClassificationHead
from ..encoders import get_encoder


class DeepLabDecoder(nn.Sequential):
    def __init__(self, in_channels, out_channels=256, atrous_rates=(12, 24, 36)):
        super().__init__(
            deeplabv3.ASPP(in_channels, atrous_rates),
            nn.Conv2d(256, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.out_channels = out_channels


    def forward(self, *features):
        return super().forward(features[-1])


class DeepLabV3(SegmentationModel):

    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 8,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder.replace_strides_with_dilation(stages=[4, 5])

        self.decoder = DeepLabDecoder(
            in_channels=self.encoder.out_channels[-1],
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
