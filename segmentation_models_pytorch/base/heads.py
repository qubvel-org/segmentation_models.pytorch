from segmentation_models_pytorch.decoders import ActivationType
import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation


class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: ActivationType = None,
        upsampling: int = 1,
    ):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        classes: int,
        pooling: str = "avg",
        dropout: float = 0.2,
        activation: ActivationType = None,
    ):
        if pooling not in ("max", "avg"):
            raise ValueError(f"Pooling should be one of ('max', 'avg'), got {pooling}.")
        pool = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)
