import json
import timm
import copy
import warnings
import functools
from torch.utils.model_zoo import load_url
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


from .resnet import resnet_encoders
from .dpn import dpn_encoders
from .vgg import vgg_encoders
from .senet import senet_encoders
from .densenet import densenet_encoders
from .inceptionresnetv2 import inceptionresnetv2_encoders
from .inceptionv4 import inceptionv4_encoders
from .efficientnet import efficient_net_encoders
from .mobilenet import mobilenet_encoders
from .xception import xception_encoders
from .timm_efficientnet import timm_efficientnet_encoders
from .timm_sknet import timm_sknet_encoders
from .mix_transformer import mix_transformer_encoders
from .mobileone import mobileone_encoders

from .timm_universal import TimmUniversalEncoder
from .timm_vit import TimmViTEncoder  # noqa F401

from ._preprocessing import preprocess_input
from ._legacy_pretrained_settings import pretrained_settings

__all__ = [
    "encoders",
    "get_encoder",
    "get_encoder_names",
    "get_preprocessing_params",
    "get_preprocessing_fn",
]

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
encoders.update(timm_sknet_encoders)
encoders.update(mix_transformer_encoders)
encoders.update(mobileone_encoders)


def is_equivalent_to_timm_universal(name):
    patterns = [
        "timm-regnet",
        "timm-res2",
        "timm-resnest",
        "timm-mobilenetv3",
        "timm-gernet",
    ]
    for pattern in patterns:
        if name.startswith(pattern):
            return True
    return False


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    if name.startswith("timm-"):
        warnings.warn(
            "`timm-` encoders are deprecated and will be removed in the future. "
            "Please use `tu-` equivalent encoders instead (see 'Timm encoders' section in the documentation).",
            DeprecationWarning,
        )

    # convert timm- models to tu- models
    if is_equivalent_to_timm_universal(name):
        name = name.replace("timm-", "tu-")
        if "mobilenetv3" in name:
            name = name.replace("tu-", "tu-tf_")

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

    if name not in encoders:
        raise KeyError(
            f"Wrong encoder name `{name}`, supported encoders: {list(encoders.keys())}"
        )

    params = copy.deepcopy(encoders[name]["params"])
    params["depth"] = depth
    params["output_stride"] = output_stride

    EncoderClass = encoders[name]["encoder"]
    encoder = EncoderClass(**params)

    if weights is not None:
        if weights not in encoders[name]["pretrained_settings"]:
            available_weights = list(encoders[name]["pretrained_settings"].keys())
            raise KeyError(
                f"Wrong pretrained weights `{weights}` for encoder `{name}`. "
                f"Available options are: {available_weights}"
            )

        settings = encoders[name]["pretrained_settings"][weights]
        repo_id = settings["repo_id"]
        revision = settings["revision"]

        # First, try to load from  HF-Hub, but as far as I know not all countries have
        # access to the Hub (e.g. China), so we try to load from the original url if
        # the first attempt fails.
        weights_path = None
        try:
            hf_hub_download(repo_id, filename="config.json", revision=revision)
            weights_path = hf_hub_download(
                repo_id, filename="model.safetensors", revision=revision
            )
        except Exception as e:
            if name in pretrained_settings and weights in pretrained_settings[name]:
                message = (
                    f"Error loading {name} `{weights}` weights from Hugging Face Hub, "
                    "trying loading from original url..."
                )
                warnings.warn(message, UserWarning)
                url = pretrained_settings[name][weights]["url"]
                state_dict = load_url(url, map_location="cpu")
            else:
                raise e

        if weights_path is not None:
            state_dict = load_file(weights_path, device="cpu")

        # Load model weights
        encoder.load_state_dict(state_dict)

    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    if output_stride != 32:
        encoder.make_dilated(output_stride)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):
    if encoder_name.startswith("tu-"):
        encoder_name = encoder_name[3:]
        if not timm.models.is_model_pretrained(encoder_name):
            raise ValueError(
                f"{encoder_name} does not have pretrained weights and preprocessing parameters"
            )
        settings = timm.models.get_pretrained_cfg(encoder_name).__dict__
    else:
        all_settings = encoders[encoder_name]["pretrained_settings"]
        if pretrained not in all_settings.keys():
            raise ValueError(
                "Available pretrained options {}".format(all_settings.keys())
            )

        repo_id = all_settings[pretrained]["repo_id"]
        revision = all_settings[pretrained]["revision"]

        # Load config and model
        try:
            config_path = hf_hub_download(
                repo_id, filename="config.json", revision=revision
            )
            with open(config_path, "r") as f:
                settings = json.load(f)
        except Exception as e:
            if (
                encoder_name in pretrained_settings
                and pretrained in pretrained_settings[encoder_name]
            ):
                settings = pretrained_settings[encoder_name][pretrained]
            else:
                raise e

    formatted_settings = {}
    formatted_settings["input_space"] = settings.get("input_space", "RGB")
    formatted_settings["input_range"] = list(settings.get("input_range", [0, 1]))
    formatted_settings["mean"] = list(settings["mean"])
    formatted_settings["std"] = list(settings["std"])

    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
