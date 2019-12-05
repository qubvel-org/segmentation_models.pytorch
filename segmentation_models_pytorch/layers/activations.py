import torch
from torch import nn


activations = {'sigmoid': torch.nn.Sigmoid, 'softmax2d': torch.nn.Softmax2d, None: 'linear',
               'softmax': torch.nn.Softmax}


class ActivationMaker(nn.Module):
    def __init__(self, activation_name):
        super().__init__()
        self.activation_name = activation_name

    def forward(self, x):
        if self.activation_name:
            activation_fn = activations[self.activation_name]()
            x = activation_fn(x)
        return x


class ThresholdTaker(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        if self.threshold:
            x = (x > self.threshold).float()
        return x


class ArgmaxTaker(nn.Module):
    """
    input: size (b, c, h, w)
    output: size (b, 1, h, w)
    """
    def __init__(self, take_argmax_flag):
        super().__init__()
        self.take_argmax_flag = take_argmax_flag

    def forward(self, x):
        if self.take_argmax_flag:
            x = torch.argmax(x, dim=1).unsqueeze(1).type(torch.cuda.FloatTensor)
        return x



