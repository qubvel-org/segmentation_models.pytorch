import torch.nn as nn


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size,
                stride=stride, 
                padding=padding, 
                bias=not (use_batchnorm)
            )
        ]

        if use_batchnorm == 'inplace':
            try:
                from inplace_abn import InPlaceABN
            except ImportError:
                raise RuntimeError("In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. To install see: https://github.com/mapillary/inplace_abn")
            
            layers.append(InPlaceABN(out_channels, activation='leaky_relu', activation_param=0.0, **batchnorm_params))
        elif use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels, **batchnorm_params))
            layers.append(nn.ReLU(inplace=True))  
        else:
            layers.append(nn.ReLU(inplace=True))
                
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re, ch, 1),
                                 nn.Sigmoid()
                                )
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
