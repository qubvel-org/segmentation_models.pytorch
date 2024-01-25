import torch.utils.model_zoo as model_zoo

from .densenet import densenet_encoders
from .dpn import dpn_encoders
from .efficientnet import efficient_net_encoders
from .inceptionresnetv2 import inceptionresnetv2_encoders
from .inceptionv4 import inceptionv4_encoders
from .mix_transformer import mix_transformer_encoders
from .mobilenet import mobilenet_encoders
from .mobileone import mobileone_encoders
from .resnet import resnet_encoders
from .senet import senet_encoders
from .timm_efficientnet import timm_efficientnet_encoders
from .timm_gernet import timm_gernet_encoders
from .timm_mobilenetv3 import timm_mobilenetv3_encoders
from .timm_regnet import timm_regnet_encoders
from .timm_res2net import timm_res2net_encoders
from .timm_resnest import timm_resnest_encoders
from .timm_sknet import timm_sknet_encoders
from .timm_universal import TimmUniversalEncoder
from .vgg import vgg_encoders
from .xception import xception_encoders

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inceptionresnetv2_encoders)
encoders.update(inceptionv4_encoders)
encoders.update(efficient_net_encoders)
encoders.update(mobilenet_encoders)
encoders.update(xception_encoders)
encoders.update(timm_efficientnet_encoders)
encoders.update(timm_resnest_encoders)
encoders.update(timm_res2net_encoders)
encoders.update(timm_regnet_encoders)
encoders.update(timm_sknet_encoders)
encoders.update(timm_mobilenetv3_encoders)
encoders.update(timm_gernet_encoders)
encoders.update(mix_transformer_encoders)
encoders.update(mobileone_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    if name.startswith("tu-"):
        name = name[3:]
        encoder = TimmUniversalEncoder(
            name=name,
            in_channels=in_channels,
            depth=depth,
            output_stride=output_stride,
            pretrained=weights is not None,
            **kwargs,
        )
        return encoder

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        err = f"Wrong encoder name `{name}`, supported encoders: {list(encoders.keys())}"  # noqa: E501
        raise KeyError(err)

    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            err = f"""
            Wrong pretrained weights `{weights}` for encoder `{name}`.
            Available options are: {list(encoders[name]["pretrained_settings"].keys())}
            """
            raise KeyError(err)

        encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder


def get_encoder_names():
    return list(encoders.keys())
