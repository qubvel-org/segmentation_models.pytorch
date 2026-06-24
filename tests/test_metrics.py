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
        with pytest.raises(ValueError, match="Dimensions should match"):
            smp.metrics.get_stats(
                output=torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]),
                target=torch.tensor([[0, 1, 2]]),
                mode="multiclass",
                num_classes=3,
            )

    def test_onehot_target_raises(self):
        """Passing a one-hot encoded target (N, C, ...) should raise ValueError with hint."""
        with pytest.raises(ValueError, match="Dimensions should match"):
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

    def test_valid_3d_volumetric(self):
        """4D tensors (N, H, W, D) for 3D segmentation should work with num_classes=3."""
        output = torch.randint(0, 3, (2, 4, 4, 4))
        target = torch.randint(0, 3, (2, 4, 4, 4))
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=output,
            target=target,
            mode="multiclass",
            num_classes=3,
        )
        assert tp.shape == (2, 3)
        assert fp.shape == (2, 3)
        assert fn.shape == (2, 3)
        assert tn.shape == (2, 3)

    def test_valid_shape1_equals_numclasses(self):
        """(N, C) where C=num_classes but values are class indices should work."""
        output = torch.tensor([[0, 1, 2], [2, 1, 0]])
        target = torch.tensor([[0, 1, 2], [2, 2, 0]])
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=output,
            target=target,
            mode="multiclass",
            num_classes=3,
        )
        assert tp.shape == (2, 3)
        # First sample: all match -> tp=[1,1,1], fp=[0,0,0], fn=[0,0,0]
        assert tp[0].tolist() == [1, 1, 1]
        assert fp[0].tolist() == [0, 0, 0]
        assert fn[0].tolist() == [0, 0, 0]
        # Second sample: pixel 1 is output=1, target=2 -> mismatch
        # tp=[1,0,1], fp=[0,1,0], fn=[0,0,1]
        assert tp[1].tolist() == [1, 0, 1]
        assert fp[1].tolist() == [0, 1, 0]
        assert fn[1].tolist() == [0, 0, 1]

    def test_range_violation_output(self):
        """Output with value >= num_classes should raise ValueError."""
        with pytest.raises(ValueError, match="output values should be in range"):
            smp.metrics.get_stats(
                output=torch.tensor([[0, 1, 3]]),
                target=torch.tensor([[0, 1, 2]]),
                mode="multiclass",
                num_classes=3,
            )

    def test_range_violation_target(self):
        """Target with value >= num_classes should raise ValueError."""
        with pytest.raises(ValueError, match="target values should be in range"):
            smp.metrics.get_stats(
                output=torch.tensor([[0, 1, 2]]),
                target=torch.tensor([[0, 1, 3]]),
                mode="multiclass",
                num_classes=3,
            )

    def test_negative_values_output(self):
        """Output with negative values should raise ValueError."""
        with pytest.raises(ValueError, match="output values should be in range"):
            smp.metrics.get_stats(
                output=torch.tensor([[0, -1, 2]]),
                target=torch.tensor([[0, 1, 2]]),
                mode="multiclass",
                num_classes=3,
            )

    def test_ignore_index_masking(self):
        """Target with ignore_index=255 values should succeed and mask correctly."""
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=torch.tensor([[0, 1, 2, 1]]),
            target=torch.tensor([[0, 1, 255, 2]]),
            mode="multiclass",
            num_classes=3,
            ignore_index=255,
        )
        assert tp.shape == (1, 3)
        # Pixel 0: match class 0 -> tp[0]=1
        # Pixel 1: match class 1 -> tp[1]=1
        # Pixel 2: ignored (255)
        # Pixel 3: output=1, target=2 -> fp[1]=1, fn[2]=1
        assert tp.tolist() == [[1, 1, 0]]
        assert fp.tolist() == [[0, 1, 0]]
        assert fn.tolist() == [[0, 0, 1]]
        # Total=4, ignored=1, effective=3
        # tn[0] = 3 - 1 - 0 - 0 = 2
        # tn[1] = 3 - 1 - 0 - 1 = 1
        # tn[2] = 3 - 0 - 1 - 0 = 2
        assert tn.tolist() == [[2, 1, 2]]

    def test_strict_detects_onehot_output(self):
        """strict=True should detect one-hot encoded output with matching shapes."""
        output = torch.tensor([[[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]])
        target = torch.tensor([[[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]])
        with pytest.raises(ValueError, match="output appears to be one-hot"):
            smp.metrics.get_stats(
                output=output,
                target=target,
                mode="multiclass",
                num_classes=3,
                strict=True,
            )

    def test_strict_detects_onehot_target(self):
        """strict=True should detect one-hot encoded target with matching shapes."""
        output = torch.tensor([[[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]]])
        target = torch.tensor([[[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]])
        with pytest.raises(ValueError, match="target appears to be one-hot"):
            smp.metrics.get_stats(
                output=output,
                target=target,
                mode="multiclass",
                num_classes=3,
                strict=True,
            )

    def test_strict_false_allows_onehot_like(self):
        """strict=False (default) should not reject valid class indices."""
        output = torch.tensor([[[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]]])
        target = torch.tensor([[[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]]])
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=output,
            target=target,
            mode="multiclass",
            num_classes=3,
            strict=False,
        )
        assert tp.shape == (1, 3)
        # All match -> perfect prediction
        assert tp.tolist() == [[4, 4, 4]]
        assert fp.tolist() == [[0, 0, 0]]
        assert fn.tolist() == [[0, 0, 0]]

    def test_strict_true_passes_valid_indices(self):
        """strict=True with valid class indices (not one-hot) should work."""
        output = torch.tensor([[[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]]])
        target = torch.tensor([[[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1, 2]]])
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=output,
            target=target,
            mode="multiclass",
            num_classes=3,
            strict=True,
        )
        assert tp.shape == (1, 3)
        assert tp.tolist() == [[4, 4, 4]]
        assert fp.tolist() == [[0, 0, 0]]
        assert fn.tolist() == [[0, 0, 0]]

    def test_strict_with_3d_volumetric(self):
        """strict=True should work with 3D volumetric input that is valid class indices."""
        output = torch.randint(0, 3, (2, 4, 4, 4))
        target = torch.randint(0, 3, (2, 4, 4, 4))
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=output,
            target=target,
            mode="multiclass",
            num_classes=3,
            strict=True,
        )
        assert tp.shape == (2, 3)

    def test_valid_5d_input(self):
        """4D input (N, D, H, W) for 3D segmentation after argmax should work."""
        output = torch.randint(0, 3, (1, 2, 4, 4))
        target = torch.randint(0, 3, (1, 2, 4, 4))
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=output,
            target=target,
            mode="multiclass",
            num_classes=3,
        )
        assert tp.shape == (1, 3)
        assert fp.shape == (1, 3)
        assert fn.shape == (1, 3)
        assert tn.shape == (1, 3)

    def test_binary_mode_untouched(self):
        """Binary mode should still work without validation changes."""
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=torch.tensor([[[0.2, 0.8]], [[0.7, 0.3]]]),
            target=torch.tensor([[[0, 1]], [[1, 0]]]),
            mode="binary",
            threshold=0.5,
        )
        assert tp.shape == (2, 1)
        # Sample 0: output=[0,1], target=[0,1] -> tp=1, fp=0, fn=0, tn=1
        assert tp[0].item() == 1
        assert fp[0].item() == 0
        assert fn[0].item() == 0
        assert tn[0].item() == 1
        # Sample 1: output=[1,0], target=[1,0] -> tp=1, fp=0, fn=0, tn=1
        assert tp[1].item() == 1
        assert fp[1].item() == 0
        assert fn[1].item() == 0
        assert tn[1].item() == 1

    def test_multilabel_mode_untouched(self):
        """Multilabel mode should still work without validation changes."""
        tp, fp, fn, tn = smp.metrics.get_stats(
            output=torch.tensor([[[0.2, 0.8], [0.7, 0.3]], [[0.9, 0.1], [0.4, 0.6]]]),
            target=torch.tensor([[[0, 1], [1, 0]], [[1, 0], [0, 1]]]),
            mode="multilabel",
            threshold=0.5,
        )
        assert tp.shape == (2, 2)
        # Sample 0: output=[[0,1],[1,0]], target=[[0,1],[1,0]] -> tp=[1,1], fp=[0,0], fn=[0,0], tn=[1,1]
        assert tp[0].tolist() == [1, 1]
        assert fp[0].tolist() == [0, 0]
        assert fn[0].tolist() == [0, 0]
        assert tn[0].tolist() == [1, 1]
        # Sample 1: output=[[1,0],[0,1]], target=[[1,0],[0,1]] -> tp=[1,1], fp=[0,0], fn=[0,0], tn=[1,1]
        assert tp[1].tolist() == [1, 1]
        assert fp[1].tolist() == [0, 0]
        assert fn[1].tolist() == [0, 0]
        assert tn[1].tolist() == [1, 1]
