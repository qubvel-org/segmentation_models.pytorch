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
    FocalLoss,
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


@torch.inference_mode()
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


@torch.inference_mode()
def test_tversky_loss_binary():
    eps = 1e-5
    # with alpha=0.5; beta=0.5 it is equal to DiceLoss
    criterion = TverskyLoss(
        mode=smp.losses.BINARY_MODE, from_logits=False, alpha=0.5, beta=0.5
    )

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


@torch.inference_mode()
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


@torch.inference_mode()
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


@torch.inference_mode()
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


@torch.inference_mode()
def test_soft_ce_loss():
    criterion = SoftCrossEntropyLoss(smooth_factor=0.1, ignore_index=-100)

    y_pred = torch.tensor(
        [[+9, -9, -9, -9], [-9, +9, -9, -9], [-9, -9, +9, -9], [-9, -9, -9, +9]]
    ).float()
    y_true = torch.tensor([0, 1, -100, 3]).long()

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(1.0125, abs=0.0001)


@torch.inference_mode()
def test_soft_bce_loss():
    criterion = SoftBCEWithLogitsLoss(smooth_factor=0.1, ignore_index=-100)

    y_pred = torch.tensor([-9, 9, 1, 9, -9]).float()
    y_true = torch.tensor([0, 1, -100, 1, 0]).long()

    loss = criterion(y_pred, y_true)
    assert float(loss) == pytest.approx(0.7201, abs=0.0001)


@torch.inference_mode()
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


@torch.inference_mode()
def test_class_weights_uniform_equivalent_to_no_weights_multiclass():
    """Uniform class_weights should produce the same loss as no weights (multiclass)."""
    eps = 1e-5
    torch.manual_seed(0)
    y_pred = torch.randn(2, 3, 4, 4)
    y_true = torch.randint(0, 3, (2, 4, 4))

    for loss_cls in [DiceLoss, JaccardLoss, TverskyLoss]:
        loss_no_w = loss_cls(mode=smp.losses.MULTICLASS_MODE)(y_pred, y_true)
        loss_uniform = loss_cls(
            mode=smp.losses.MULTICLASS_MODE, class_weights=[1.0, 1.0, 1.0]
        )(y_pred, y_true)
        assert torch.allclose(loss_no_w, loss_uniform, atol=eps), (
            f"Uniform weights should be equivalent to no weights for {loss_cls.__name__}"
        )


@torch.inference_mode()
def test_class_weights_uniform_equivalent_to_no_weights_multilabel():
    """Uniform class_weights should produce the same loss as no weights (multilabel)."""
    eps = 1e-5
    torch.manual_seed(0)
    y_pred = torch.randn(2, 3, 4, 4)
    y_true = torch.randint(0, 2, (2, 3, 4, 4)).float()

    for loss_cls in [DiceLoss, JaccardLoss, TverskyLoss]:
        loss_no_w = loss_cls(mode=smp.losses.MULTILABEL_MODE)(y_pred, y_true)
        loss_uniform = loss_cls(
            mode=smp.losses.MULTILABEL_MODE, class_weights=[1.0, 1.0, 1.0]
        )(y_pred, y_true)
        assert torch.allclose(loss_no_w, loss_uniform, atol=eps), (
            f"Uniform weights should be equivalent to no weights for {loss_cls.__name__}"
        )


@torch.inference_mode()
def test_class_weights_nonuniform_changes_loss_multiclass():
    """Non-uniform class_weights should change the loss value (multiclass)."""
    torch.manual_seed(0)
    y_pred = torch.randn(2, 3, 4, 4)
    y_true = torch.randint(0, 3, (2, 4, 4))

    for loss_cls in [DiceLoss, JaccardLoss, TverskyLoss]:
        loss_no_w = loss_cls(mode=smp.losses.MULTICLASS_MODE)(y_pred, y_true)
        loss_weighted = loss_cls(
            mode=smp.losses.MULTICLASS_MODE, class_weights=[1.0, 2.0, 0.5]
        )(y_pred, y_true)
        assert not torch.allclose(loss_no_w, loss_weighted, atol=1e-6), (
            f"Non-uniform weights should change the loss for {loss_cls.__name__}"
        )


@torch.inference_mode()
def test_class_weights_scale_invariant_multiclass():
    """Scaling all weights by a constant should not change the loss (multiclass)."""
    eps = 1e-5
    torch.manual_seed(0)
    y_pred = torch.randn(2, 3, 4, 4)
    y_true = torch.randint(0, 3, (2, 4, 4))

    for loss_cls in [DiceLoss, JaccardLoss, TverskyLoss]:
        loss_w = loss_cls(
            mode=smp.losses.MULTICLASS_MODE, class_weights=[1.0, 2.0, 0.5]
        )(y_pred, y_true)
        loss_w_scaled = loss_cls(
            mode=smp.losses.MULTICLASS_MODE, class_weights=[10.0, 20.0, 5.0]
        )(y_pred, y_true)
        assert torch.allclose(loss_w, loss_w_scaled, atol=eps), (
            f"Loss should be scale-invariant w.r.t. class_weights for {loss_cls.__name__}"
        )


@torch.inference_mode()
def test_class_weights_binary_mode_raises():
    """class_weights should raise an error when used with binary mode."""
    for loss_cls in [DiceLoss, JaccardLoss, TverskyLoss]:
        with pytest.raises(ValueError):
            loss_cls(mode=smp.losses.BINARY_MODE, class_weights=[1.0, 2.0])


@torch.inference_mode()
def test_focal_class_weights_uniform_equivalent_to_no_weights():
    """Uniform class_weights should produce a loss equivalent to no-weights loss."""
    eps = 1e-5
    torch.manual_seed(0)
    y_pred = torch.randn(2, 3, 4, 4)
    y_true = torch.randint(0, 3, (2, 4, 4))

    loss_no_w = FocalLoss(mode=smp.losses.MULTICLASS_MODE)(y_pred, y_true)
    loss_uniform = FocalLoss(
        mode=smp.losses.MULTICLASS_MODE, class_weights=[1.0, 1.0, 1.0]
    )(y_pred, y_true)
    assert torch.allclose(loss_no_w, loss_uniform, atol=eps)


@torch.inference_mode()
def test_focal_class_weights_scale_invariant():
    """Scaling all weights by a constant should not change FocalLoss."""
    eps = 1e-5
    torch.manual_seed(0)
    y_pred = torch.randn(2, 3, 4, 4)
    y_true = torch.randint(0, 3, (2, 4, 4))

    loss_w = FocalLoss(mode=smp.losses.MULTICLASS_MODE, class_weights=[1.0, 2.0, 0.5])(
        y_pred, y_true
    )
    loss_w_scaled = FocalLoss(
        mode=smp.losses.MULTICLASS_MODE, class_weights=[10.0, 20.0, 5.0]
    )(y_pred, y_true)
    assert torch.allclose(loss_w, loss_w_scaled, atol=eps)
