import torch


def iou(outputs, targets, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Dice is only implemented for sigmoid and softmax2d"
        )

    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs) - intersection + eps
    return (intersection + eps) / union

jaccard = iou