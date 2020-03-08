from typing import Optional, Union
import torch
import os

from ..base import SegmentationModel
# from ..base import SegmentationHead, ClassificationHead
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
    def __init__(
        self,
        encoder_name: str = "hrnetv2_32",
        encoder_depth: int = 4,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        # self.decoder = HRNetDecoder(
        #     encoder_channels=self.encoder.out_channels,
        #     use_batchnorm=hrnet_use_batchnorm,
        #     out_channels=hrnet_out_channels,
        # )

        # self.segmentation_head = SegmentationHead(
        #     in_channels=hrnet_out_channels,
        #     out_channels=classes,
        #     kernel_size=3,
        #     activation=activation,
        #     upsampling=upsampling,
        # )

        # if aux_params:
        #     self.classification_head = ClassificationHead(
        #         in_channels=self.encoder.out_channels[-1],
        #         **aux_params
        #     )
        # else:
        #     self.classification_head = None
        #
        # self.name = encoder_name
        # self.initialize()
