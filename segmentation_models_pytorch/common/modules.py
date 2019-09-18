import torch.nn as nn


class Activation(nn.Module):

    def __init__(self, name, **params):

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softamx/logsoftmax/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)
