import pytest
import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from tests.test_models import get_sample, _test_forward, _test_forward_backward


@pytest.mark.parametrize("encoder_name", ["sam-vit_b", "sam-vit_l"])
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("patch_size", [8, 16])
def test_sam_encoder(encoder_name, img_size, patch_size):
    encoder = get_encoder(encoder_name, img_size=img_size, patch_size=patch_size)
    assert encoder._name == encoder_name[4:]
    assert encoder.output_stride == 32

    sample = torch.ones(1, 3, img_size, img_size)
    with torch.no_grad():
        out = encoder(sample)

    expected_patches = img_size // patch_size
    assert out.size() == torch.Size([1, 256, expected_patches, expected_patches])


@pytest.mark.parametrize("encoder_name", ["sam-vit_b"])
@pytest.mark.parametrize("image_size", [64])
def test_sam(encoder_name, image_size):
    model_class = smp.SAM
    model = model_class(encoder_name, encoder_weights=None, image_size=image_size)
    sample = get_sample(model_class)
    model.eval()

    _test_forward(model, sample, test_shape=True)
    _test_forward_backward(model, sample, test_shape=True)
