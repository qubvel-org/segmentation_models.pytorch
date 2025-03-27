from typing import Any, Dict, Union

import torch
import torch.nn as nn

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None

def get_norm_layer(use_norm: Union[bool, str, Dict[str, Any]], out_channels: int) -> nn.Module:
    supported_norms = ("inplace", "batchnorm", "identity", "layernorm", "instancenorm")
    if use_norm is True:
        norm_params = {"type": "batchnorm"}
    elif use_norm is False:
        norm_params = {"type": "identity"}
    elif use_norm == "inplace":
        norm_params = {"type": "inplace", "activation": "leaky_relu", "activation_param": 0.0}
    elif isinstance(use_norm, str):
        norm_str = use_norm.lower()
        if norm_str == "inplace":
            norm_params = {
                "type": "inplace",
                "activation": "leaky_relu",
                "activation_param": 0.0,
            }
        elif norm_str in (
                "batchnorm",
                "identity",
                "layernorm",
                "instancenorm",
        ):
            norm_params = {"type": norm_str}
        else:
            raise ValueError(f"Unrecognized normalization type string provided: {use_norm}. Should be in {supported_norms}")
    elif isinstance(use_norm, dict):
        norm_params = use_norm
    else:
        raise ValueError("use_norm must be a dictionary, boolean, or string. Please refer to the documentation.")


    if not "type" in norm_params:
        raise ValueError(f"Malformed dictionary given in use_norm: {use_norm}. Should contain key 'type'.")
    if norm_params["type"] not in supported_norms:
        raise ValueError(f"Unrecognized normalization type string provided: {use_norm}. Should be in {supported_norms}")

    norm_type = norm_params["type"]
    extra_kwargs = {k: v for k, v in norm_params.items() if k != "type"}

    if norm_type == "inplace" and InPlaceABN is None:
        raise RuntimeError(
            "In order to use `use_batchnorm='inplace'` or `use_norm='inplace'` the inplace_abn package must be installed. "
            "To install see: https://github.com/mapillary/inplace_abn"
        )
    elif norm_type == "inplace":
        norm = InPlaceABN(out_channels, **extra_kwargs)
    elif norm_type == "batchnorm":
        norm = nn.BatchNorm2d(out_channels, **extra_kwargs)
    elif norm_type == "identity":
        norm = nn.Identity()
    elif norm_type == "layernorm":
        norm = nn.LayerNorm(out_channels, **extra_kwargs)
    elif norm_type == "instancenorm":
        norm = nn.InstanceNorm2d(out_channels, **extra_kwargs)
    else:
        raise ValueError(f"Unrecognized normalization type: {norm_type}")

    return norm

class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_norm="batchnorm",
    ):
        norm = get_norm_layer(use_norm, out_channels)
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=norm._get_name() != "BatchNorm2d",
        )

        if norm._get_name() == "Inplace":
            relu = nn.Identity()
        else:
            relu = nn.ReLU(inplace=True)

        super(Conv2dReLU, self).__init__(conv, norm, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    def __init__(self, min=0, max=1):
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)
