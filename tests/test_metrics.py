import pytest
import torch
import segmentation_models_pytorch as smp


@pytest.mark.parametrize(
    "reduction", ["micro", "macro", "micro-imagewise", "macro-imagewise"]
)
@pytest.mark.parametrize("zero_division", [0.0, 1.0])
@pytest.mark.parametrize(
    "metric_fn",
    [smp.metrics.iou_score, smp.metrics.f1_score, smp.metrics.precision],
)
def test_zero_division_with_empty_masks(metric_fn, zero_division, reduction):
    # no foreground in either the prediction or the target: 0 / 0 for these metrics
    output = torch.zeros(2, 1, 4, 4, dtype=torch.long)
    target = torch.zeros(2, 1, 4, 4, dtype=torch.long)
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode="binary")

    score = metric_fn(tp, fp, fn, tn, reduction=reduction, zero_division=zero_division)

    assert score.item() == pytest.approx(zero_division)


def test_zero_division_warn_micro():
    output = torch.zeros(2, 1, 4, 4, dtype=torch.long)
    target = torch.zeros(2, 1, 4, 4, dtype=torch.long)
    tp, fp, fn, tn = smp.metrics.get_stats(output, target, mode="binary")

    with pytest.warns(UserWarning, match="Zero division"):
        score = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro", zero_division="warn"
        )

    assert score.item() == 0.0
