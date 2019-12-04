import torch
from .module import Module
from . import initialization as init
from ..layers import out_modules


class Model(Module):
    def __init__(self):
        super().__init__()

    def predict(self, x):
        raise NotImplementedError


class ANNModel(Model):
    def __init__(self):
        super().__init__()

    def predict(self, x):
        raise NotImplementedError

    def predict_prob(self, x):
        raise NotImplementedError


class EnsembleModel(Model):
    def __init__(self):
        super().__init__()

    def predict(self, x):
        raise NotImplementedError


class SegmentationModel(ANNModel):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.out_channels = num_classes

        if self.out_channels == 1:
            self.prediction_layer = out_modules.OutSigmoid(threshold=0.5)
            self.prediction_prob_layer = out_modules.OutSigmoid(threshold=None)
        else:
            self.prediction_layer = out_modules.OutSoftmax(take_argmax_flag=True)
            self.prediction_prob_layer = out_modules.OutSoftmax(take_argmax_flag=False)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self(x)

        return self.prediction_layer(x)

    def predict_prob(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            # x = self.forward(x)
            x = self(x)

        return self.prediction_prob_layer(x)


class TwoHeadsSegmentationModel(ANNModel):

    def __init__(self, num_classes1, num_classes2):
        super().__init__()

        self.num_classes_lst = []
        self.out_channels_lst = []
        self.prediction_layer_lst = []
        self.prediction_prob_layer_lst = []

        for num_classes in [num_classes1, num_classes2]:
            self.num_classes_lst.append(num_classes)
            self.out_channels_lst.append(num_classes)

            if self.out_channels_lst[-1] == 1:
                self.prediction_layer_lst.append(out_modules.OutSigmoid(threshold=0.5))
                self.prediction_prob_layer_lst.append(out_modules.OutSigmoid(threshold=None))
            else:
                self.prediction_layer_lst.append(out_modules.OutSoftmax(take_argmax_flag=True))
                self.prediction_prob_layer_lst.append(out_modules.OutSoftmax(take_argmax_flag=False))

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x0, x1 = self(x)

        x0, x1 = self.prediction_layer_lst[0](x0), self.prediction_layer_lst[1](x1)
        return x0, x1

    def predict_prob(self, x):

        if self.training:
            self.eval()

        with torch.no_grad():
            x0, x1 = self(x)

        x0, x1 = self.prediction_prob_layer_lst[0](x0), self.prediction_prob_layer_lst[1](x1)
        return x0, x1
