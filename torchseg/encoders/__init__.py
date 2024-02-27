import torch.utils.model_zoo as model_zoo

from .mix_transformer import mix_transformer_encoders
from .supported import TIMM_ENCODERS, TIMM_VIT_ENCODERS, UNSUPPORTED_ENCODERS
from .timm import TimmEncoder, TimmViTEncoder


def list_unsupported_encoders():
    return UNSUPPORTED_ENCODERS


def list_encoders():
    return (
        list(TIMM_ENCODERS.keys())
        + TIMM_VIT_ENCODERS
        + list(mix_transformer_encoders.keys())
    )


def get_encoder(
    name,
    in_channels=3,
    depth=None,
    indices=None,
    weights=None,
    output_stride=32,
    scale_factors=None,
    **kwargs,
):
    assert (
        depth is not None or indices is not None
    ), "Either `depth` or `indices` should be specified"

    # MixTransformer encoder
    if name.startswith("mit_b"):
        encoders = mix_transformer_encoders
        params = encoders[name]["params"]
        params.update(depth=depth)

        try:
            Encoder = encoders[name]["encoder"]
        except KeyError:
            err = f"Wrong mit encoder name `{name}`, supported encoders: {list(encoders.keys())}"  # noqa: E501
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
                Available options are: {list(encoders[name]["pretrained_settings"].keys())}  # noqa: E501
                """
                raise KeyError(err)

            encoder.load_state_dict(model_zoo.load_url(settings["url"]))

    # Timm Encoders
    else:
        if name.split(".")[0] in TIMM_ENCODERS:
            encoder = TimmEncoder(
                name=name,
                in_channels=in_channels,
                depth=depth,
                indices=indices,
                output_stride=output_stride,
                pretrained=weights is not None,
                **kwargs,
            )
        elif name.split(".")[0] in TIMM_VIT_ENCODERS:
            encoder = TimmViTEncoder(
                name=name,
                in_channels=in_channels,
                depth=depth,
                indices=indices,
                pretrained=weights is not None,
                scale_factors=scale_factors,
                **kwargs,
            )
        elif name.split(".")[0] in UNSUPPORTED_ENCODERS:
            err = f"""
            {name} is an unsupported timm encoder that does not support
            `features_only=True` or does not have a `get_intermediate_layers` method.
            """
            raise ValueError(err)
        else:
            err = f"""
            {name} is an unknown encoder. Check available encoders using
            `torchseg.list_encoders()`
            """
            raise ValueError(err)
    return encoder
