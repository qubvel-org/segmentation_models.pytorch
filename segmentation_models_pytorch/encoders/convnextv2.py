import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from ._base import EncoderMixin


class LayerNorm(nn.Module):
    """ 
        LayerNorm that supports two data formats: channels_last (default) or channels_first. 
        The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
        shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
        with shape (batch_size, channels, height, width).
        
        Args:
            normalized_shape (int): Input shape from an expected input of size.
            eps (float): A value added to the denominator for numerical stability. Default: 1e-6
            data_format (str): channels_last (default) or channels_first. Default: channels_last
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ 
        Global Response Normalization
        
        Args:
            dim (int): Number of input channels.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ 
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """
        
    Args:
        in_channels (int): Number of input image channels.
        depth (int): Number of stages.
        depths (tuple(int)): Number of blocks at each stage.
        dims (int): Feature dimension at each stage.
        stem_kernel_size (int): Kernel size of the stem conv. Default: 4
        stem_stride (int): Stride of the stem conv. Default: 4
    """

    def __init__(self, in_channels, depth, depths, dims, stem_kernel_size=4, stem_stride=4):
        super().__init__()

        self.depth = depth
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=stem_kernel_size, stride=stem_stride),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(depth - 1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        drop_path_rate = 0.
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(depth):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def get_stages(self):
        stages = [nn.Identity(), 
                  *[nn.Sequential(self.downsample_layers[i], self.stages[i]) for i in range(self.depth)]]
        return stages

    def forward_features(self, x):
        outs = [x]
        for i in range(self.depth):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)

        return outs

    def forward(self, x):
        return self.forward_features(x)
        

class ConvNeXtV2Encoder(torch.nn.Module, EncoderMixin):
    def __init__(self, model_name, in_channels, out_channels, depth, depths):
        super().__init__()
        self._depth = depth
        self._out_channels = [in_channels] + out_channels
        self._encoder = ConvNeXtV2(depth=depth,
                                   depths=depths,
                                   in_channels=in_channels,
                                   dims=out_channels)

    def forward(self, x):
        return self._encoder.forward_features(x)


convnextv2_encoders = {
    "convnextv2_atto": {
        "encoder": ConvNeXtV2Encoder,
        "pretrained_settings": {
            'imagenet': {
                'url': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt',
                'input_space': 'RGB',
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000,
                'input_size': [3, 224, 224],
            }
        },
        "params": {
            "out_channels": [40, 80, 160, 320],
            "depth": 4,
            "depths": [2, 2, 6, 2],
            "in_channels": 3,
            "model_name": "convnext2_atto"
        },
    },
    "convnextv2_femto": {
        "encoder": ConvNeXtV2Encoder,
        "pretrained_settings": {
            'imagenet': {
                'url': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt',
                'input_space': 'RGB',
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000,
                'input_size': [3, 224, 224],
            }    
        },
        "params": {
            "out_channels": [48, 96, 192, 384],
            "depth": 4,
            "depths": [2, 2, 6, 2],
            "in_channels": 3,
            "model_name": "convnextv2_femto"
        },
    },
    "convnextv2_pico": {
        "encoder": ConvNeXtV2Encoder,
        "pretrained_settings": {
            'imagenet': {
                'url': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt',
                'input_space': 'RGB',
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000,
                'input_size': [3, 224, 224],
            }    
        },
        "params": {
            "out_channels": [64, 128, 256, 512],
            "depth": 4,
            "depths": [2, 2, 6, 2],
            "in_channels": 3,
            "model_name": "convnextv2_pico"
        },
    },
    "convnextv2_nano": {
        "encoder": ConvNeXtV2Encoder,
        "pretrained_settings": {
            'imagenet': {
                'url': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt',
                'input_space': 'RGB',
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000,
                'input_size': [3, 224, 224],
            }
        },
        "params": {
            "out_channels": [80, 160, 320, 640],
            "depth": 4,
            "depths": [2, 2, 8, 2],
            "in_channels": 3,
            "model_name": "convnextv2_nano",
        },
    },
    "convnextv2_tiny": {
        "encoder": ConvNeXtV2Encoder,
        "pretrained_settings": {
            'imagenet': {
                'url': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt',
                'input_space': 'RGB',
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000,
                'input_size': [3, 224, 224],
            }
        },
        "params": {
            "out_channels": [96, 192, 384, 768],
            "depth": 4,
            "depths": [3, 3, 9, 3],
            "in_channels": 3,
            "model_name": "convnextv2_tiny"
        },
    },
    "convnextv2_base": {
        "encoder": ConvNeXtV2Encoder,
        "pretrained_settings": {
            'imagenet': {
                'url': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt',
                'input_space': 'RGB',
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000,
                'input_size': [3, 224, 224],
            }
        },
        "params": {
            "out_channels": [128, 256, 512, 1024],
            "depth": 4,
            "depths": [3, 3, 27, 3],
            "in_channels": 3,
            "model_name": "convnextv2_base"
        },
    },
    "convnextv2_large": {
        "encoder": ConvNeXtV2Encoder,
        "pretrained_settings": {
            'imagenet': {
                'url': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt',
                'input_space': 'RGB',
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000,
                'input_size': [3, 224, 224],
            }
        },
        "params": {
            "out_channels": [192, 384, 768, 1536],
            "depth": 4,
            "depths": [3, 3, 27, 3],
            "in_channels": 3,
            "model_name": "convnextv2_large"
        },
    },
    "convnextv2_huge": {
        "encoder": ConvNeXtV2Encoder,
        "pretrained_settings": {
            'imagenet': {
                'url': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_384_ema.pt',  
                'input_space': 'RGB',
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000,
                'input_size': [3, 384, 384], 
            }
        },
        "params": {
            "out_channels": [352, 704, 1408, 2816],
            "depth": 4,
            "depths": [3, 3, 27, 3],
            "in_channels": 3,
            "model_name": "convnextv2_huge"
        },
    },
}
