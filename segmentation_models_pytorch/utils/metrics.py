from . import base
from . import functions as F
from .. import common as cmn


class IoUMetric(base.Named):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = cmn.Activation(activation, dim=1)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(y_pr, y_gt, self.eps, self.threshold)


class FscoreMetric(base.Named):

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = cmn.Activation(activation, dim=1)

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(y_pr, y_gt, self.beta, self.eps, self.threshold)
