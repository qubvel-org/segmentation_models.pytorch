# some private imports for create_model function
from typing import Optional

import torch.nn as nn

from . import decoders, encoders, losses
from .decoders import (
    FPN,
    PAN,
    DeepLabV3,
    DeepLabV3Plus,
    Linknet,
    MAnet,
    PSPNet,
    Unet,
    UnetPlusPlus,
)
from .encoders import list_encoders

__all__ = ("encoders", "decoders", "losses", "list_encoders")


def create_model(
    arch: str,
    encoder_name: str = "resnet34",
    encoder_indices: Optional[tuple[int]] = None,
    encoder_depth: Optional[int] = 5,
    encoder_weights: Optional[str] = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    **kwargs,
) -> nn.Module:
    """
    Models entrypoint, allows to create any model architecture just with
    parameters, without using its class
    """

    archs = [
        Unet,
        UnetPlusPlus,
        MAnet,
        Linknet,
        FPN,
        PSPNet,
        DeepLabV3,
        DeepLabV3Plus,
        PAN,
    ]
    archs_dict = {a.__name__.lower(): a for a in archs}
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        err = "Wrong architecture type `{arch}`. Available options are: {list(archs_dict.keys())}"  # noqa: E501
        raise KeyError(err)

    return model_class(
        encoder_name=encoder_name,
        encoder_indices=encoder_indices,
        encoder_depth=encoder_depth,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )


__author__ = "Isaac Corley"
__version__ = "0.0.1"
