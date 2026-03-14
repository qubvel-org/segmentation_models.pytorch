from typing import Optional
from functools import partial

import torch
from torch.nn.modules.loss import _Loss
from ._functional import focal_loss_with_logits
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["FocalLoss"]


class FocalLoss(_Loss):
    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()

        self.mode = mode
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:
            num_classes = y_pred.size(1)

            # If ignore_index parameter is passed, treat it as an extra class during one-hot encoding and remove it later
            # One-hot encoding the labels allows us to vectorise the focal loss computation
            if self.ignore_index is not None:
                y_true[y_true == self.ignore_index] = num_classes
                y_true_one_hot = torch.nn.functional.one_hot(y_true,num_classes = num_classes + 1)
                y_true_one_hot = y_true_one_hot[ ... , : -1]

            else:     
                y_true_one_hot = torch.nn.functional.one_hot(y_true,num_classes = num_classes)

            y_true_one_hot = torch.permute(y_true_one_hot,(0,3,1,2))

            # Multiplying the loss by num_classes in order to stay consistent with the earlier loss computation which did not
            # take a classwise average of the loss
            loss = num_classes * self.focal_loss_fn(y_pred, y_true_one_hot)

        return loss
