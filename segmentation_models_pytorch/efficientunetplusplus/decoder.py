import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F

from ..base import modules as md

class InvertedResidual(nn.Module):
    """
    Inverted bottleneck residual block with an scSE block embedded into the residual layer, after the 
    depthwise convolution. By default, uses batch normalization and Hardswish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, expansion_ratio = 1, squeeze_ratio = 1, \
        activation = nn.Hardswish(True), normalization = nn.BatchNorm2d):
        super().__init__()
        self.same_shape = in_channels == out_channels
        self.mid_channels = expansion_ratio*in_channels
        self.block = nn.Sequential(
            md.PointWiseConv2d(in_channels, self.mid_channels),
            normalization(self.mid_channels),
            activation,
            md.DepthWiseConv2d(self.mid_channels, kernel_size=kernel_size, stride=stride),
            normalization(self.mid_channels),
            activation,
            #md.sSEModule(self.mid_channels),
            md.SCSEModule(self.mid_channels, reduction = squeeze_ratio),
            #md.SEModule(self.mid_channels, reduction = squeeze_ratio),
            md.PointWiseConv2d(self.mid_channels, out_channels),
            normalization(out_channels)
        )
        
        if not self.same_shape:
            # 1x1 convolution used to match the number of channels in the skip feature maps with that 
            # of the residual feature maps
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                normalization(out_channels)
            )
          
    def forward(self, x):
        residual = self.block(x)
        
        if not self.same_shape:
            x = self.skip_conv(x)
        return x + residual
        
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            squeeze_ratio=1,
            expansion_ratio=1
    ):
        super().__init__()

        # Inverted Residual block convolutions
        self.conv1 = InvertedResidual(
            in_channels=in_channels+skip_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )
        self.conv2 = InvertedResidual(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class EfficientUnetPlusPlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            squeeze_ratio=1,
            expansion_ratio=1
    ):
        super().__init__()
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels

        # combine decoder keyword arguments
        kwargs = dict(squeeze_ratio=squeeze_ratio, expansion_ratio=expansion_ratio)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx+1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx+1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx+1-depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f'x_{depth_idx}_{layer_idx}'] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f'x_{0}_{len(self.in_channels)-1}'] =\
            DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], **kwargs)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels)-1):
            for depth_idx in range(self.depth-layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f'x_{depth_idx}_{depth_idx}'](features[depth_idx], features[depth_idx+1])
                    dense_x[f'x_{depth_idx}_{depth_idx}'] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f'x_{idx}_{dense_l_i}'] for idx in range(depth_idx+1, dense_l_i+1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i+1]], dim=1)
                    dense_x[f'x_{depth_idx}_{dense_l_i}'] =\
                        self.blocks[f'x_{depth_idx}_{dense_l_i}'](dense_x[f'x_{depth_idx}_{dense_l_i-1}'], cat_features)
        dense_x[f'x_{0}_{self.depth}'] = self.blocks[f'x_{0}_{self.depth}'](dense_x[f'x_{0}_{self.depth-1}'])
        return dense_x[f'x_{0}_{self.depth}']