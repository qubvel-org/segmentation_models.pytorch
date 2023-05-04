import pytest
import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from tests.test_models import get_sample, _test_forward, _test_forward_backward


@pytest.mark.parametrize("encoder_name", ["sam-vit_b", "sam-vit_l"])
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("patch_size", [8, 16])
@pytest.mark.parametrize("depth", [6, 24, None])
def test_sam_encoder(encoder_name, img_size, patch_size, depth):
    encoder = get_encoder(encoder_name, img_size=img_size, patch_size=patch_size, depth=depth)
    assert encoder._name == encoder_name[4:]
    assert encoder.output_stride == 32

    sample = torch.ones(1, 3, img_size, img_size)
    with torch.no_grad():
        out = encoder(sample)

    expected_patches = img_size // patch_size
    assert out.size() == torch.Size([1, 256, expected_patches, expected_patches])


@pytest.mark.parametrize("decoder_multiclass_output", [True, False])
@pytest.mark.parametrize("n_classes", [1, 3])
def test_sam(decoder_multiclass_output, n_classes):
    model = smp.SAM(
        "sam-vit_b",
        encoder_weights=None,
        weights=None,
        image_size=64,
        decoder_multimask_output=decoder_multiclass_output,
        classes=n_classes,
    )
    sample = get_sample(smp.SAM)
    model.eval()

    _test_forward(model, sample, test_shape=True)
    _test_forward_backward(model, sample, test_shape=True)


@pytest.mark.skip(reason="Run this test manually as it needs to download weights")
def test_sam_weights():
    smp.create_model("sam", encoder_name="sam-vit_b", encoder_weights=None, weights="sa-1b")
