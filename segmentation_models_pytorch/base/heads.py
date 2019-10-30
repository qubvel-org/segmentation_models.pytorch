import torch.nn as nn
from .modules import Flatten, Activation


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = Activation(activation)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, activation=None):
        pool = nn.AdaptiveAvgPool2d(output_size=1)
        flatten = Flatten()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, linear, activation)
