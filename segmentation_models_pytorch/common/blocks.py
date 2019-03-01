import torch.nn as nn


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not (use_batchnorm))
        if use_batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_channels, **batchnorm_params)

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'batchnorm'):
            x = self.batchnorm(x)
        x = self.activation(x)
        return x
