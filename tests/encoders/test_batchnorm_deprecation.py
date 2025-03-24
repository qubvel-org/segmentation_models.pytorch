import pytest

import torch

from segmentation_models_pytorch import create_model
from tests.utils import check_two_models_strictly_equal


@pytest.mark.parametrize("model_name", ["unet", "unetplusplus", "linknet", "manet"])
@pytest.mark.parametrize("decoder_option", [True, False, "inplace"])
def test_seg_models_before_after_use_norm(model_name, decoder_option):
    torch.manual_seed(42)
    model_decoder_batchnorm = create_model(model_name, "mobilenet_v2", None, decoder_use_batchnorm=decoder_option)
    torch.manual_seed(42)
    model_decoder_norm = create_model(model_name, "mobilenet_v2", None, decoder_use_batchnorm=None, decoder_use_norm=decoder_option)

    check_two_models_strictly_equal(model_decoder_batchnorm, model_decoder_norm)



@pytest.mark.parametrize("decoder_option", [True, False, "inplace"])
def test_pspnet_before_after_use_norm(decoder_option):
    torch.manual_seed(42)
    model_decoder_batchnorm = create_model("pspnet", "mobilenet_v2", None, psp_use_batchnorm=decoder_option)
    torch.manual_seed(42)
    model_decoder_norm = create_model("pspnet", "mobilenet_v2", None, psp_use_batchnorm=None, decoder_use_norm=decoder_option)

    check_two_models_strictly_equal(model_decoder_batchnorm, model_decoder_norm)
