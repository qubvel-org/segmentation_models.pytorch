"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

__all__ = ["LovaszLoss"]


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(*_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Logits at each prediction (between -infinity and +infinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------


def _lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [B, C, H, W] Class probabilities at each prediction (between 0 and 1).
        Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        @param labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        @param per_image: compute the loss per image instead of per batch
        @param ignore: void class labels
    """
    if per_image:
        loss = mean(
            _lovasz_softmax_flat(*_flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas, labels)
        )
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def _lovasz_softmax_flat(probas, labels, classes="present"):
    """Multi-class Lovasz-Softmax loss
    Args:
        @param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        @param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        @param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).type_as(probas)  # foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)


def _flatten_probas(probas, labels, ignore=None):
    """Flattens predictions in the batch"""
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)

    C = probas.size(1)
    probas = torch.movedim(probas, 1, -1)  # [B, C, Di, Dj, ...] -> [B, Di, Dj, ..., C]
    probas = probas.contiguous().view(-1, C)  # [P, C]

    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid]
    vlabels = labels[valid]
    return vprobas, vlabels


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators."""
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszLoss(_Loss):
    def __init__(
        self,
        mode: str,
        per_image: bool = False,
        ignore_index: Optional[int] = None,
        from_logits: bool = True,
    ):
        """Lovasz loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            per_image: If True loss computed per each image and then averaged, else computed per whole batch

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
        self.per_image = per_image

    def forward(self, y_pred, y_true):

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            loss = _lovasz_hinge(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        elif self.mode == MULTICLASS_MODE:
            y_pred = y_pred.softmax(dim=1)
            loss = _lovasz_softmax(y_pred, y_true, per_image=self.per_image, ignore=self.ignore_index)
        else:
            raise ValueError("Wrong mode {}.".format(self.mode))
        return loss
