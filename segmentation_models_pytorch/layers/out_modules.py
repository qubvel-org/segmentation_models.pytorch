from ..base.module import Module
from . import activations


class OutLayer(Module):
    def __init__(self):
        super().__init__()


class OutSigmoid(OutLayer):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.activation_maker = activations.ActivationMaker('sigmoid')
        self.threshold_taker = activations.ThresholdTaker(threshold)

    def forward(self, x):
        x = self.activation_maker(x)
        x = self.threshold_taker(x)
        return x


class OutSoftmax(OutLayer):
    def __init__(self, take_argmax_flag=True):
        super().__init__()
        self.activation_maker = activations.ActivationMaker('softmax2d')
        self.argmax_taker = activations.ArgmaxTaker(take_argmax_flag)

    def forward(self, x):
        x = self.activation_maker(x)
        x = self.argmax_taker(x)
        return x


class PredictionLayer(OutLayer):
    def __init__(self, activation_name):
        super().__init__()
        if activation_name == 'sigmoid':
            self.layer = OutSigmoid()
        elif activation_name == 'softmax2d':
            self.layer = OutSoftmax()
        else:
            self.layer = None

    def forward(self, x):
        if self.layer:
            x = self.layer(x)
        return x




