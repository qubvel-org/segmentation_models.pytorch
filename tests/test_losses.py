import pytest
import torch
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.losses._functional as F
from segmentation_models_pytorch.losses import (
    DiceLoss,
    JaccardLoss,
    SoftBCEWithLogitsLoss,
    SoftCrossEntropyLoss,
    TverskyLoss,
    MCCLoss,
)


def test_focal_loss_with_logits():
    input_good = torch.tensor([10, -10, 10]).float()
    input_bad = torch.tensor([-1, 2, 0]).float()
    target = torch.tensor([1, 0, 1])

    loss_good = F.focal_loss_with_logits(input_good, target)
    loss_bad = F.focal_loss_with_logits(input_bad, target)
    assert loss_good < loss_bad


def test_softmax_focal_loss_with_logits():
    input_good = torch.tensor([[0, 10, 0], [10, 0, 0], [0, 0, 10]]).float()
    input_bad = torch.tensor([[0, -10, 0], [0, 10, 0], [0, 0, 10]]).float()
    target = torch.tensor([1, 0, 2]).long()

    loss_good = F.softmax_focal_loss_with_logits(input_good, target)
    loss_bad = F.softmax_focal_loss_with_logits(input_bad, target)
    assert loss_good < loss_bad


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps"],
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], 1.0, 1e-5],
        [[0, 1, 1, 0], [0, 1, 1, 0], 1.0, 1e-5],
        [[1, 1, 1, 1], [1, 1, 0, 0], 0.5, 1e-5],
    ],
)
def test_soft_jaccard_score(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true, eps=eps)
    assert float(actual) == pytest.approx(expected, eps)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps"],
    [
        [[[1, 1, 0, 0], [0, 0, 1, 1]], [[1, 1, 0, 0], [0, 0, 1, 1]], 1.0, 1e-5],
        [[[1, 1, 0, 0], [0, 0, 1, 1]], [[0, 0, 1, 0], [0, 1, 0, 0]], 0.0, 1e-5],
        [[[1, 1, 0, 0], [0, 0, 0, 1]], [[1, 1, 0, 0], [0, 0, 0, 0]], 0.5, 1e-5],
    ],
)
def test_soft_jaccard_score_2(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_jaccard_score(y_pred, y_true, dims=[1], eps=eps)
    actual = actual.mean()
    assert float(actual) == pytest.approx(expected, eps)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps"],
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], 1.0, 1e-5],
        [[0, 1, 1, 0], [0, 1, 1, 0], 1.0, 1e-5],
        [[1, 1, 1, 1], [1, 1, 0, 0], 2.0 / 3.0, 1e-5],
    ],
)
def test_soft_dice_score(y_true, y_pred, expected, eps):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_dice_score(y_pred, y_true, eps=eps)
    assert float(actual) == pytest.approx(expected, eps)


@pytest.mark.parametrize(
    ["y_true", "y_pred", "expected", "eps", "alpha", "beta"],
    [
        [[1, 1, 1, 1], [1, 1, 1, 1], 1.0, 1e-5, 0.5, 0.5],
        [[0, 1, 1, 0], [0, 1, 1, 0], 1.0, 1e-5, 0.5, 0.5],
        [[1, 1, 1, 1], [1, 1, 0, 0], 2.0 / 3.0, 1e-5, 0.5, 0.5],
    ],
)
def test_soft_tversky_score(y_true, y_pred, expected, eps, alpha, beta):
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    actual = F.soft_tversky_score(y_pred, y_true, eps=eps, alpha=alpha, beta=beta)
    assert float(actual) == pytest.approx(expected, eps)


@torch.no_grad()
def test_dice_loss_binary():
    eps = 1e-5
    criterion = DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 1, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)


@torch.no_grad()
def test_tversky_loss_binary():
    eps = 1e-5
    # with alpha=0.5; beta=0.5 it is equal to DiceLoss
    criterion = TverskyLoss(mode=smp.losses.BINARY_MODE, from_logits=False, alpha=0.5, beta=0.5)

    # Ideal case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 1, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)


@torch.no_grad()
def test_binary_jaccard_loss():
    eps = 1e-5
    criterion = JaccardLoss(mode=smp.losses.BINARY_MODE, from_logits=False)

    # Ideal case
    y_pred = torch.tensor([1.0]).view(1, 1, 1, 1)
    y_true = torch.tensor(([1])).view(1, 1, 1, 1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([1, 0, 1])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, 1, -1)
    y_true = torch.tensor(([0, 0, 0])).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([1.0, 1.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 0, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    y_pred = torch.tensor([1.0, 0.0, 1.0]).view(1, 1, -1)
    y_true = torch.tensor([0, 1, 0]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)

    y_pred = torch.tensor([0.0, 0.0, 0.0]).view(1, 1, -1)
    y_true = torch.tensor([1, 1, 1]).view(1, 1, 1, -1)
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, eps)


@torch.no_grad()
def test_multiclass_jaccard_loss():
    eps = 1e-5
    criterion = JaccardLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=False)

    # Ideal case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[0, 0, 1, 1]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    # 1 - 1/3 case
    y_pred = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])
    y_true = torch.tensor([[1, 1, 0, 0]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0 - 1.0 / 3.0, abs=eps)


@torch.no_grad()
def test_multilabel_jaccard_loss():
    eps = 1e-5
    criterion = JaccardLoss(mode=smp.losses.MULTILABEL_MODE, from_logits=False)

    # Ideal case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = 1 - y_pred
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    # 1 - 1/3 case
    y_pred = torch.tensor([[[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0 - 1.0 / 3.0, abs=eps)


@torch.no_grad()
def test_soft_ce_loss():
    criterion = SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=-100)

    y_pred = torch.tensor([[+9, -9, -9, -9], [-9, +9, -9, -9], [-9, -9, +9, -9], [-9, -9, -9, +9]]).float()
    y_true = torch.tensor([0, 1, -100, 3]).long()

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0125, abs=0.0001)


@torch.no_grad()
def test_soft_bce_loss():
    criterion = SoftBCEWithLogitsLoss(smooth_factor=0.1, ignore_index=-100)

    y_pred = torch.tensor([-9, 9, 1, 9, -9]).float()
    y_true = torch.tensor([0, 1, -100, 1, 0]).long()

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.7201, abs=0.0001)


@torch.no_grad()
def test_binary_mcc_loss():
    eps = 1e-5
    criterion = MCCLoss(eps=eps)

    # Ideal case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.0, abs=eps)

    # Worst case
    y_pred = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]])
    y_true = 1 - y_pred
    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(2.0, abs=eps)

    # 1 - 1/3 case
    y_pred = torch.tensor([[[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    # Trivial classifier case #1.
    y_pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0, abs=eps)

    # Trivial classifier case #2.
    y_pred = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1 + 0.5, abs=eps)

    # Trivial classifier case #3.
    y_pred = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])
    y_true = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]])

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.5, abs=eps)
