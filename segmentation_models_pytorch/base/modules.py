import torch.nn as nn
import torch

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
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        super().__init__()

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


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
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class EncoderDecoderAttention(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, out_channels):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.out_channels = out_channels
        self.attention_weight = nn.Parameter(torch.Tensor(encoder_channels, out_channels, decoder_channels))
        nn.init.xavier_uniform_(self.attention_weight)

    def forward(self, encoder_inp, decoder_inp):
        x1 = encoder_inp.permute(0, 2, 3, 1)
        batch_size, height, width = x1.shape[0:-1]
        x2 = decoder_inp.permute(0, 2, 3, 1)
        result = torch.mm(x1.reshape(-1, self.encoder_channels), self.attention_weight.view(self.encoder_channels, -1))
        result = result.view(-1, self.out_channels, self.decoder_channels)
        result = torch.bmm(result, x2.reshape(-1, self.decoder_channels, 1)).\
            reshape(batch_size, height, width, self.out_channels).permute(0, 3, 1, 2)
        return result


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
