import warnings

import torch
import torch.nn as nn

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
        use_norm="batchnorm",
    ):
        if use_batchnorm is not None:
            warnings.warn(
                "The usage of use_batchnorm is deprecated. Please modify your code for use_norm",
                DeprecationWarning,
            )
            if use_batchnorm is True:
                use_norm = {"type": "batchnorm"}
            elif use_batchnorm is False:
                use_norm = {"type": "identity"}
            elif use_batchnorm == "inplace":
                use_norm = {
                    "type": "inplace",
                    "activation": "leaky_relu",
                    "activation_param": 0.0,
                }
            else:
                raise ValueError("Unrecognized value for use_batchnorm")

        if isinstance(use_norm, str):
            norm_str = use_norm.lower()
            if norm_str == "inplace":
                use_norm = {
                    "type": "inplace",
                    "activation": "leaky_relu",
                    "activation_param": 0.0,
                }
            elif norm_str in (
                "batchnorm",
                "identity",
                "layernorm",
                "groupnorm",
                "instancenorm",
            ):
                use_norm = {"type": norm_str}
            else:
                raise ValueError("Unrecognized normalization type string provided")
        elif isinstance(use_norm, bool):
            use_norm = {"type": "batchnorm" if use_norm else "identity"}
        elif not isinstance(use_norm, dict):
            raise ValueError("use_norm must be a dictionary, boolean, or string")

        if use_norm["type"] == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` or `use_norm='inplace'` the inplace_abn package must be installed. "
                "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=use_norm["type"] != "inplace",
        )
        relu = nn.ReLU(inplace=True)

        norm_type = use_norm["type"]
        extra_kwargs = {k: v for k, v in use_norm.items() if k != "type"}

        if norm_type == "inplace":
            norm = InPlaceABN(out_channels, **extra_kwargs)
            relu = nn.Identity()
        elif norm_type == "batchnorm":
            norm = nn.BatchNorm2d(out_channels, **extra_kwargs)
        elif norm_type == "identity":
            norm = nn.Identity()
        elif norm_type == "layernorm":
            norm = nn.LayerNorm(out_channels, **extra_kwargs)
        elif norm_type == "groupnorm":
            norm = nn.GroupNorm(out_channels, **extra_kwargs)
        elif norm_type == "instancenorm":
            norm = nn.InstanceNorm2d(out_channels, **extra_kwargs)
        else:
            raise ValueError(f"Unrecognized normalization type: {norm_type}")

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


if __name__ == "__main__":
    print(Conv2dReLU(3, 12, 4))
    print(Conv2dReLU(3, 12, 4, use_norm={"type": "batchnorm"}))
    print(Conv2dReLU(3, 12, 4, use_norm={"type": "layernorm", "eps": 1e-3}))
