import torch.nn as nn
import torch.nn.functional as F


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


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma

    def forward(self, pred_logits, target):
        pred = F.softmax(pred_logits, dim=1)
        pred_log = F.log_softmax(pred_logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes + 1).permute(0, 3, 1, 2).float()
        target_one_hot = target_one_hot[:, 1:]
        cross_entropy = -target_one_hot * pred_log
        focal = (1.0 - pred).pow(self.gamma)
        loss = focal * cross_entropy
        return loss.sum() / pred_logits.size()[-2:].numel()


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
