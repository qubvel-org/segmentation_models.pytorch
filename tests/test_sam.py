import pytest
import torch

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from tests.test_models import get_sample, _test_forward, _test_forward_backward


@pytest.mark.parametrize("encoder_name", ["sam-vit_b", "sam-vit_l"])
@pytest.mark.parametrize("img_size", [64, 128])
@pytest.mark.parametrize("patch_size,depth", [(8, 3), (16, 4)])
@pytest.mark.parametrize("vit_depth", [12, 24])
def test_sam_encoder(encoder_name, img_size, patch_size, depth, vit_depth):
    encoder = get_encoder(encoder_name, img_size=img_size, patch_size=patch_size, depth=depth, vit_depth=vit_depth)
    assert encoder.output_stride == 32

    sample = torch.ones(1, 3, img_size, img_size)
    with torch.no_grad():
        out = encoder(sample)

    expected_patches = img_size // patch_size
    assert out[-1].size() == torch.Size([1, 256, expected_patches, expected_patches])


def test_sam_encoder_validation_error():
    with pytest.raises(ValueError):
        get_encoder("sam-vit_b", img_size=64, patch_size=16, depth=5, vit_depth=12)
        get_encoder("sam-vit_b", img_size=64, patch_size=16, depth=4, vit_depth=None)
        get_encoder("sam-vit_b", img_size=64, patch_size=16, depth=4, vit_depth=6)


@pytest.mark.skip(reason="Decoder has been removed, keeping this for future integration")
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
@pytest.mark.parametrize("decoder_channels,encoder_depth", [([64, 32, 16, 8], 4), ([64, 32, 16, 8], 4)])
def test_sam_encoder_arch(model_class, decoder_channels, encoder_depth):
    img_size = 1024
    model = model_class(
        "sam-vit_b",
        encoder_weights=None,
        encoder_depth=encoder_depth,
        decoder_channels=decoder_channels,
    )
    smp = torch.ones(1, 3, img_size, img_size)
    _test_forward_backward(model, smp, test_shape=True)


@pytest.mark.skip(reason="Run this test manually as it needs to download weights")
def test_sam_weights():
    smp.create_model("sam", encoder_name="sam-vit_b", encoder_weights=None, weights="sa-1b")


@pytest.mark.skip(reason="Run this test manually as it needs to download weights")
def test_sam_encoder_weights():
    smp.create_model(
        "unet", encoder_name="sam-vit_b", encoder_depth=4, encoder_weights="sa-1b", decoder_channels=[64, 32, 16, 8]
    )
