import pytest
import torch

from segmentation_models_pytorch.metrics import get_stats


def test_get_stats_explains_float_output_requires_threshold() -> None:
    output = torch.tensor([[[0.1, 0.8], [0.3, 0.7]]], dtype=torch.float32)
    target = torch.tensor([[[0, 1], [0, 1]]], dtype=torch.long)

    with pytest.raises(ValueError) as error:
        get_stats(output, target, mode="binary", threshold=None)

    assert str(error.value) == (
        "Output should be one of the integer types if ``threshold`` is None, "
        "got torch.float32."
    )
