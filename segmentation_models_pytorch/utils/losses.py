import torch.nn as nn
from . import functions as F


class JaccardLoss(nn.Module):
    __name__ = 'jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - F.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - F.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class BCEJaccardLoss(JaccardLoss):
    __name__ = 'bce_jaccard_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        jaccard = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return jaccard + bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce
