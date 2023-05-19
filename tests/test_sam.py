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
    assert out[-1].size() == torch.Size([1, 256, expected_patches, expected_patches])


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


@pytest.mark.parametrize("model_class", [smp.Unet])
@pytest.mark.parametrize("decoder_channels,patch_size", [([64, 32, 16, 8], 16), ([64, 32, 16], 8)])
def test_sam_as_encoder_only(model_class, decoder_channels, patch_size):
    img_size = 64
    model = model_class(
        "sam-vit_b",
        encoder_weights=None,
        encoder_depth=3,
        encoder_kwargs=dict(img_size=img_size, out_chans=decoder_channels[0], patch_size=patch_size),
        decoder_channels=decoder_channels,
    )
    smp = torch.ones(1, 3, img_size, img_size)
    _test_forward_backward(model, smp, test_shape=True)


@pytest.mark.skip(reason="Run this test manually as it needs to download weights")
def test_sam_weights():
    smp.create_model("sam", encoder_name="sam-vit_b", encoder_weights=None, weights="sa-1b")


# @pytest.mark.skip(reason="Run this test manually as it needs to download weights")
def test_sam_encoder_weights():
    smp.create_model(
        "unet", encoder_name="sam-vit_b", encoder_weights="sa-1b", encoder_depth=12, decoder_channels=[64, 32, 16, 8]
    )
