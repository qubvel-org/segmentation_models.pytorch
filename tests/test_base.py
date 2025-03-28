import torch
import tempfile
import segmentation_models_pytorch as smp

import pytest


def test_from_pretrained_with_mismatched_keys():
    original_model = smp.Unet(classes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        original_model.save_pretrained(temp_dir)

        # we should catch warning here and check if there specific keys there
        with pytest.warns(UserWarning):
            restored_model = smp.from_pretrained(temp_dir, classes=2, strict=False)

    assert restored_model.segmentation_head[0].out_channels == 2

    # verify all the weight are the same expect mismatched ones
    original_state_dict = original_model.state_dict()
    restored_state_dict = restored_model.state_dict()

    expected_mismatched_keys = [
        "segmentation_head.0.weight",
        "segmentation_head.0.bias",
    ]
    mismatched_keys = []
    for key in original_state_dict:
        if key not in expected_mismatched_keys:
            assert torch.allclose(original_state_dict[key], restored_state_dict[key])
        else:
            mismatched_keys.append(key)

    assert len(mismatched_keys) == 2
    assert sorted(mismatched_keys) == sorted(expected_mismatched_keys)
