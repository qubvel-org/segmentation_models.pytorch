from typing import List, Optional

import torch
from ._functional import soft_tversky_score
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from .dice import DiceLoss

__all__ = ["TverskyLoss"]


class TverskyLoss(DiceLoss):
    """Tversky loss for image segmentation task.
    Where FP and FN is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases

    Args:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        classes: Optional list of classes that contribute in loss computation;
            By default, all channels are included.
        log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
        from_logits: If True assumes input is raw logits
        smooth:
        ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Negatives)
        gamma: Constant that squares the error function. Defaults to ``1.0``
        class_weights: List of weights for each class. If not ``None``, the loss for each class
            is multiplied by the corresponding weight. Only supported for multiclass and
            multilabel modes. Weights do not need to be normalized.

    Return:
        torch.Tensor: loss
    """

    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        class_weights: Optional[List[float]] = None,
    ):
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__(
            mode,
            classes,
            log_loss,
            from_logits,
            smooth,
            ignore_index,
            eps,
            class_weights,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Aggregate per-class losses into a single scalar, raised to the power of gamma.

        Args:
            loss: Per-class loss tensor of shape (C,)

        Returns:
            Scalar loss value
        """
        if self.class_weights is not None:
            weights = self.class_weights.to(loss.device)
            if self.classes is not None:
                weights = weights[self.classes]
            return ((loss * weights).sum() / weights.sum()) ** self.gamma
        return loss.mean() ** self.gamma

    def compute_score(
        self, output, target, smooth=0.0, eps=1e-7, dims=None
    ) -> torch.Tensor:
        return soft_tversky_score(
            output, target, self.alpha, self.beta, smooth, eps, dims
        )
