from typing import List

import torch
from ._functional import soft_tversky_score
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from .dice import DiceLoss

__all__ = ["TverskyLoss", "TverskyLossFocal"]


class TverskyLoss(DiceLoss):
    """
    Implementation of Tversky loss for image segmentation task. Where TP and FP is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(
            self,
            mode: str,
            classes: List[int] = None,
            log_loss=False,
            from_logits=True,
            smooth: float = 0.0,
            ignore_index=None,
            eps=1e-7,
            alpha=0.5,
            beta=0.5
    ):
        """
        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        :param alpha: Weight constant that penalize model for FPs (False Positives)
        :param beta: Weight constant that penalize model for FNs (False Positives)
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(mode, classes, log_loss, from_logits, smooth, ignore_index, eps)
        self.alpha = alpha
        self.beta = beta

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)


class TverskyLossFocal(TverskyLoss):
    """
    A variant on the Tversky loss that also includes the gamma modifier from Focal Loss https://arxiv.org/abs/1708.02002
    It supports binary, multiclass and multilabel cases
    """

    def __init__(
            self,
            mode: str,
            classes: List[int] = None,
            log_loss=False,
            from_logits=True,
            smooth: float = 0.0,
            ignore_index=None,
            eps=1e-7,
            alpha=0.5,
            beta=0.5,
            gamma=1
    ):
        """
        :param mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        :param alpha: Weight constant that penalize model for FPs (False Positives)
        :param beta: Weight constant that penalize model for FNs (False Positives)
        :param gamma: Constant that squares the error function
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(mode, classes, log_loss, from_logits, smooth, ignore_index, eps, alpha, beta)
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma
