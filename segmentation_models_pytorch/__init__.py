from .unet import Unet
from .unetplusplus import UnetPlusPlus
from .linknet import Linknet
from .fpn import FPN
from .pspnet import PSPNet
from .deeplabv3 import DeepLabV3, DeepLabV3Plus
from .pan import PAN

from . import encoders
from . import utils

from .__version__ import __version__

from typing import Optional
import torch


def create_model(
    arch: str,
    encoder_name: str,
    encoder_weights: Optional[str],
    in_channels: int,
    classes: int,
    **kwargs,
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parametes"""
    
    archs = [Unet, UnetPlusPlus, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN]
    archs_dict = [a.__class__.__name__.lower() for a in archs]
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError("Wrong architecture type `{}`. Avalibale options are: {}".format(
            arch, list(archs_dict.keys()),
        ))
    return model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        **kwargs,
    )
