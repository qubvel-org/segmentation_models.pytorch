import pytest
import torch
import segmentation_models_pytorch as smp


@pytest.mark.parametrize("reduction", ["weighted", "weighted-imagewise"])
def test_weighted_reductions_give_one_for_perfect_prediction(reduction):
    torch.manual_seed(0)
    target = torch.randint(0, 3, (4, 16, 16))

    tp, fp, fn, tn = smp.metrics.get_stats(
        target, target, mode="multiclass", num_classes=3
    )
    score = smp.metrics.iou_score(
        tp, fp, fn, tn, reduction=reduction, class_weights=[1.0, 2.0, 3.0]
    )

    assert score.item() == pytest.approx(1.0)


def test_weighted_imagewise_is_weighted_average_of_class_scores():
    torch.manual_seed(0)
    output = torch.randint(0, 3, (4, 16, 16))
    target = torch.randint(0, 3, (4, 16, 16))
    class_weights = [0.2, 0.3, 0.5]

    tp, fp, fn, tn = smp.metrics.get_stats(
        output, target, mode="multiclass", num_classes=3
    )
    score = smp.metrics.iou_score(
        tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=class_weights
    )

    # score of each class on each image, averaged over images, then weighted over classes
    per_image = smp.metrics.iou_score(tp, fp, fn, tn, reduction="none")
    expected = (per_image.mean(0) * torch.tensor(class_weights)).sum()
    assert score.item() == pytest.approx(expected.item())


def test_weighted_imagewise_with_equal_weights_matches_macro_imagewise():
    torch.manual_seed(0)
    output = torch.randint(0, 3, (4, 16, 16))
    target = torch.randint(0, 3, (4, 16, 16))

    tp, fp, fn, tn = smp.metrics.get_stats(
        output, target, mode="multiclass", num_classes=3
    )
    weighted = smp.metrics.f1_score(
        tp, fp, fn, tn, reduction="weighted-imagewise", class_weights=[1.0, 1.0, 1.0]
    )
    macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")

    assert weighted.item() == pytest.approx(macro.item())
