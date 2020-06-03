import torch.nn as nn
import torch.nn.functional as func

from . import base
from . import functional as F
from .base import Activation


class JaccardLoss(base.Loss):
    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):
    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr,
            y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass


class SoftJaccardLoss(nn.Module):
    def __init__(self, num_classes, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, pred_logits, targets):
        preds = pred_logits.softmax(dim=1)
        loss = 0
        for cls in range(self.num_classes):
            target = (targets - 1 == cls).float()
            pred = preds[:, cls]
            intersection = (pred * target).sum()
            iou = intersection / (pred.sum() + target.sum() - intersection + self.eps) + self.eps
            loss = loss - iou.log()
        return loss / self.num_classes


class FocalLoss(base.Loss):
    def __init__(self, num_classes, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, pred_logits, target):
        pred = func.softmax(pred_logits, dim=1)
        pred_log = func.log_softmax(pred_logits, dim=1)
        target_one_hot = target.float()
        cross_entropy = -target_one_hot * pred_log
        focal = (1.0 - pred).pow(self.gamma)
        loss = focal * cross_entropy
        return loss.sum() / pred_logits.size()[-2:].numel() / pred_logits.size()[0]


class SemanticSegmentationLoss(nn.Module):
    def __init__(self, num_classes, jaccard_alpha=0.9):
        super().__init__()
        self.jaccard_alpha = jaccard_alpha
        self.jaccard = SoftJaccardLoss(num_classes)
        self.focal = FocalLoss(num_classes)

    def forward(self, pred_logits, target):
        loss = self.jaccard_alpha * self.jaccard(pred_logits, target)
        loss = loss + self.focal(pred_logits, target)
        return loss
