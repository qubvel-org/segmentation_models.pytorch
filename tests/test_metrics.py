import pytest
import torch
import segmentation_models_pytorch as smp


class TestGetStatsMulticlass:
    """Tests for get_stats in multiclass mode."""

    def test_correct_input(self):
        """Class index tensors of shape (N, ...) should work correctly."""
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=torch.tensor([[0, 1, 2, 1]]),
            target=torch.tensor([[0, 1, 2, 2]]),
            mode="multiclass",
            num_classes=3,
        )
        assert tp.shape == (1, 3)
        assert fp.tolist() == [[0, 1, 0]]
        assert fn.tolist() == [[0, 0, 1]]

    def test_onehot_output_raises(self):
        """Passing a one-hot encoded output (N, C, ...) should raise ValueError with hint."""
        with pytest.raises(ValueError, match="output.argmax"):
            smp.metrics.get_stats(
                output=torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
                target=torch.tensor([[0, 1, 2]]),
                mode="multiclass",
                num_classes=3,
            )

    def test_onehot_target_raises(self):
        """Passing a one-hot encoded target (N, C, ...) should raise ValueError with hint."""
        with pytest.raises(ValueError, match="target.argmax"):
            smp.metrics.get_stats(
                output=torch.tensor([[0, 1, 2]]),
                target=torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
                mode="multiclass",
                num_classes=3,
            )


    def test_argmax_fix_gives_perfect_iou(self):
        """Correcting a one-hot tensor with argmax(dim=1) should yield IoU=1.0."""
        output_onehot = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        target_onehot = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        tp, fp, fn, tn = smp.metrics.get_stats(
            output=output_onehot.argmax(dim=1),
            target=target_onehot.argmax(dim=1),
            mode="multiclass",
            num_classes=3,
        )
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        assert iou.item() == pytest.approx(1.0)
