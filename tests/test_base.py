import os
import json
import torch
import tempfile
from unittest import mock
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


def test_from_pretrained_does_not_download_encoder_weights():
    # encoder weights are stored in the checkpoint, so reloading a saved model
    # should not re-download the encoder's pretrained weights (see issue #957)
    model = smp.Unet("resnet18", encoder_weights=None, classes=1)

    with tempfile.TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir)

        # emulate a checkpoint trained from imagenet-pretrained encoder
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        config["encoder_weights"] = "imagenet"
        with open(config_path, "w") as f:
            json.dump(config, f)

        with (
            mock.patch(
                "segmentation_models_pytorch.encoders.hf_hub_download",
                side_effect=AssertionError("encoder weights should not be downloaded"),
            ) as mock_hf_hub_download,
            mock.patch(
                "segmentation_models_pytorch.encoders.load_url",
                side_effect=AssertionError("encoder weights should not be downloaded"),
            ) as mock_load_url,
        ):
            restored_model = smp.from_pretrained(temp_dir)

        mock_hf_hub_download.assert_not_called()
        mock_load_url.assert_not_called()

    # weights must still match the saved checkpoint
    original_state_dict = model.state_dict()
    restored_state_dict = restored_model.state_dict()
    for key in original_state_dict:
        assert torch.allclose(original_state_dict[key], restored_state_dict[key])
