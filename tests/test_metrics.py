import pytest
import torch
import segmentation_models_pytorch as smp


def test_get_stats_threshold_float():
    output = torch.tensor([[[[0.2, 0.7], [0.8, 0.3]]]])
    target = torch.tensor([[[[0, 1], [1, 0]]]])
    tp, fp, fn, tn = smp.metrics.get_stats(
        output, target, mode="binary", threshold=0.5
    )
    assert tp.item() == 2
    assert fp.item() == 0
    assert fn.item() == 0
    assert tn.item() == 2


def test_get_stats_threshold_list_multilabel():
    # 2 classes, 2x2 spatial dims
    output = torch.tensor(
        [
            [
                [[0.7, 0.2], [0.3, 0.8]],  # class 0
                [[0.4, 0.9], [0.6, 0.1]],  # class 1
            ]
        ]
    )
    target = torch.tensor(
        [
            [
                [[1, 0], [0, 1]],  # class 0
                [[0, 1], [1, 0]],  # class 1
            ]
        ]
    )
    # class 0 threshold 0.8, class 1 threshold 0.3
    tp, fp, fn, tn = smp.metrics.get_stats(
        output, target, mode="multilabel", threshold=[0.8, 0.3]
    )
    assert torch.equal(tp, torch.tensor([[1, 2]]))
    assert torch.equal(fp, torch.tensor([[0, 1]]))
    assert torch.equal(fn, torch.tensor([[1, 0]]))
    assert torch.equal(tn, torch.tensor([[2, 1]]))


def test_get_stats_threshold_list_mismatched_channels():
    output = torch.rand(2, 3, 16, 16)
    target = torch.randint(0, 2, (2, 3, 16, 16)).long()
    with pytest.raises(ValueError, match="Number of thresholds"):
        smp.metrics.get_stats(
            output, target, mode="multilabel", threshold=[0.5, 0.5]
        )
