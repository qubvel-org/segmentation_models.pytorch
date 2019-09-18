import torch.nn as nn

from . import base
from . import functions as F
from .. import common as cmn





class JaccardLoss(base.Named):

    def __init__(self, eps=1e-7, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = cmn.Activation(activation, dim=1)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(y_pr, y_gt, eps=self.eps, threshold=None)


class DiceLoss(base.Named):

    def __init__(self, eps=1e-7, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = cmn.Activation(activation, dim=1)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None)


class BCEJaccardLoss(JaccardLoss):

    def __init__(self, eps=1e-7, activation=None, logits=True, **kwargs):
        super().__init__(eps, activation, **kwargs)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean') if logits else nn.BCELoss(reduction='mean')
        self.activation = cmn.Activation(activation)

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):

    def __init__(self, eps=1e-7, activation=None, logits=True, **kwargs):
        super().__init__(eps, activation, **kwargs)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean') if logits else nn.BCELoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce
