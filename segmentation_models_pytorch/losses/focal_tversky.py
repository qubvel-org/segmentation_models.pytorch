from typing import List, Optional

import torch
from ._functional import soft_tversky_score
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from .dice import DiceLoss

__all__ = ["TverskyLoss"]


class FocalTverskyLoss(DiceLoss):
    """Focal Tversky loss for image segmentation tasks.
    FP and FN are weighted by alpha and beta parameters, respectively.
    With alpha == beta == 0.5, this loss becomes equivalent to DiceLoss.
    The gamma parameter focuses the loss on hard-to-classify examples.
    If gamma > 1, the function focuses more on misclassified examples.
    If gamma = 1, it is equivalent to Tversky Loss.
    This loss supports binary, multiclass, and multilabel cases.

    Args:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        classes: Optional list of classes that contribute to loss computation;
        By default, all channels are included.
        log_loss: If True, computes loss as ``-log(tversky)``; otherwise as ``1 - tversky``
        from_logits: If True, assumes input is raw logits
        smooth: Smoothing factor to avoid division by zero
        ignore_index: Label indicating ignored pixels (not contributing to loss)
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalizes False Positives (FPs)
        beta: Weight constant that penalizes False Negatives (FNs)
        gamma: Focal factor to adjust the focus on harder examples (defaults to 1.0)


    Return:
        loss: torch.Tensor

    """

    def __init__(
        self,
        mode: str,
        classes: List[int] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(
            mode, classes, log_loss, from_logits, smooth, ignore_index, eps
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        tversky_score = soft_tversky_score(
            output, target, self.alpha, self.beta, smooth, eps, dims
        )
        return (1 - tversky_score) ** self.gamma