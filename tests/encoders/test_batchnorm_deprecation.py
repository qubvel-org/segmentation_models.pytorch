import pytest

import torch

import segmentation_models_pytorch as smp
from tests.utils import check_two_models_strictly_equal


@pytest.mark.parametrize("model_name", ["unet", "unetplusplus", "linknet", "manet"])
@pytest.mark.parametrize("decoder_option", [True, False, "inplace"])
def test_seg_models_before_after_use_norm(model_name, decoder_option):
    torch.manual_seed(42)
    with pytest.warns(DeprecationWarning):
        model_decoder_batchnorm = smp.create_model(
            model_name,
            "mobilenet_v2",
            encoder_weights=None,
            decoder_use_batchnorm=decoder_option,
        )
    model_decoder_norm = smp.create_model(
        model_name,
        "mobilenet_v2",
        encoder_weights=None,
        decoder_use_norm=decoder_option,
    )

    model_decoder_norm.load_state_dict(model_decoder_batchnorm.state_dict())

    check_two_models_strictly_equal(
        model_decoder_batchnorm, model_decoder_norm, torch.rand(1, 3, 224, 224)
    )


@pytest.mark.parametrize("decoder_option", [True, False, "inplace"])
def test_pspnet_before_after_use_norm(decoder_option):
    torch.manual_seed(42)
    with pytest.warns(DeprecationWarning):
        model_decoder_batchnorm = smp.create_model(
            "pspnet",
            "mobilenet_v2",
            encoder_weights=None,
            psp_use_batchnorm=decoder_option,
        )
    model_decoder_norm = smp.create_model(
        "pspnet",
        "mobilenet_v2",
        encoder_weights=None,
        decoder_use_norm=decoder_option,
    )
    model_decoder_norm.load_state_dict(model_decoder_batchnorm.state_dict())

    check_two_models_strictly_equal(
        model_decoder_batchnorm, model_decoder_norm, torch.rand(1, 3, 224, 224)
    )
