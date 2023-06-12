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
    assert encoder.out_channels == [256 // (2**i) for i in range(depth + 1)][::-1]

    sample = torch.ones(1, 3, img_size, img_size)
    with torch.no_grad():
        out = encoder(sample)

    assert len(out) == depth + 1

    expected_spatial_size = img_size // patch_size
    expected_chans = 256
    for i in range(1, len(out)):
        assert out[-i].size() == torch.Size([1, expected_chans, expected_spatial_size, expected_spatial_size])
        expected_spatial_size *= 2
        expected_chans //= 2


def test_sam_encoder_trainable():
    encoder = get_encoder("sam-vit_b", depth=4)

    encoder.requires_grad_(False)
    for name, param in encoder.named_parameters():
        if name.startswith("intermediate_necks"):
            assert param.requires_grad
        else:
            assert not param.requires_grad

    encoder.requires_grad_(True)
    for param in encoder.parameters():
        assert param.requires_grad


def test_sam_encoder_validation_error():
    with pytest.raises(ValueError):
        get_encoder("sam-vit_b", img_size=64, patch_size=16, depth=5, vit_depth=12)
        get_encoder("sam-vit_b", img_size=64, patch_size=16, depth=4, vit_depth=None)
        get_encoder("sam-vit_b", img_size=64, patch_size=16, depth=4, vit_depth=6)


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
def test_sam_encoder_weights():
    smp.create_model(
        "unet", encoder_name="sam-vit_b", encoder_depth=4, encoder_weights="sa-1b", decoder_channels=[64, 32, 16, 8]
    )
