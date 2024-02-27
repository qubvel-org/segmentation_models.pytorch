import pytest
import torch

import torchseg
from torchseg.encoders import mix_transformer_encoders

ENCODERS = [
    "seresnet18",
    "senet154",
    "mobilenetv2_035",
    "mobileone_s0",
    "inception_v4",
    "inception_resnet_v2",
    "dpn68",
    "densenet121",
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "mobilenetv3_small_075",
    "resnext50_32x4d",
]
VIT_ENCODERS = [
    "vit_base_patch8_224",
    "vit_small_patch16_224",
    "vit_small_patch16_384",
    "vit_small_patch32_224",
    "vit_small_patch16_384",
]
DEEPLABV3_ENCODERS = [
    "seresnet18",
    "senet154",
    "mobilenetv2_035",
    "mobileone_s0",
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "mobilenetv3_small_075",
    "resnext50_32x4d",
]
MIT_ENCODERS = list(mix_transformer_encoders.keys())
DEFAULT_ENCODER = "resnet18"
SCALE_FACTORS = {8: (4, 2, 1, 0.5, 0.25), 16: (8, 4, 2, 1, 0.5), 32: (16, 8, 4, 2, 1)}


def get_sample(model_class):
    if model_class in (
        torchseg.Unet,
        torchseg.Linknet,
        torchseg.FPN,
        torchseg.PSPNet,
        torchseg.UnetPlusPlus,
        torchseg.MAnet,
    ):
        sample = torch.ones([1, 3, 64, 64])
    elif model_class == torchseg.PAN:
        sample = torch.ones([2, 3, 256, 256])
    elif model_class in (torchseg.DeepLabV3, torchseg.DeepLabV3Plus):
        sample = torch.ones([2, 3, 128, 128])
    else:
        raise ValueError(f"Not supported model class {model_class}")
    return sample


@torch.inference_mode()
@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("encoder_depth", [3, 5])
@pytest.mark.parametrize(
    "model_class",
    [
        torchseg.Unet,
        torchseg.FPN,
        torchseg.PSPNet,
        torchseg.Linknet,
        torchseg.UnetPlusPlus,
        torchseg.MAnet,
    ],
)
def test_timm_models(model_class, encoder_name, encoder_depth, **kwargs):
    if (
        model_class is torchseg.Unet
        or model_class is torchseg.UnetPlusPlus
        or model_class is torchseg.MAnet
    ):
        kwargs["decoder_channels"] = (16, 16, 16, 16, 16)[-encoder_depth:]

    model = model_class(
        encoder_name, encoder_depth=encoder_depth, encoder_weights=None, **kwargs
    )
    model.eval()
    sample = get_sample(model_class)
    if encoder_depth == 5 and model_class != torchseg.PSPNet:
        test_shape = True
    else:
        test_shape = False

    out = model(sample)

    if test_shape:
        assert out.shape[2:] == sample.shape[2:]


@torch.inference_mode()
@pytest.mark.parametrize("encoder_name", VIT_ENCODERS)
@pytest.mark.parametrize(
    "model_class",
    [
        torchseg.Unet,
        torchseg.FPN,
        torchseg.PSPNet,
        torchseg.Linknet,
        torchseg.UnetPlusPlus,
        torchseg.MAnet,
    ],
)
def test_timm_vit_models(model_class, encoder_name, **kwargs):
    if (
        model_class is torchseg.Unet
        or model_class is torchseg.UnetPlusPlus
        or model_class is torchseg.MAnet
    ):
        kwargs["decoder_channels"] = (16, 16, 16, 16, 16)[-5:]

    image_size = int(encoder_name.split("_")[-1])
    patch_size = int(encoder_name.split("patch")[1].split("_")[0])
    scales = SCALE_FACTORS[patch_size]
    kwargs["encoder_params"] = {"scale_factors": scales, "img_size": image_size}
    model = model_class(encoder_name, encoder_depth=5, encoder_weights=None, **kwargs)
    model.eval()
    sample = torch.ones([2, 3, image_size, image_size])
    if model_class != torchseg.PSPNet:
        test_shape = True
    else:
        test_shape = False

    out = model(sample)

    if test_shape:
        assert out.shape[2:] == sample.shape[2:]


@torch.inference_mode()
@pytest.mark.parametrize("encoder_name", DEEPLABV3_ENCODERS)
@pytest.mark.parametrize("model_class", [torchseg.DeepLabV3, torchseg.DeepLabV3Plus])
def test_deeplabv3(model_class, encoder_name, **kwargs):
    """
    DeepLabV3 requires output_stride=8. Some timm models don't support
    output_strides other than 32. So, we skip these tests
    """
    model = model_class(encoder_name, encoder_depth=5, encoder_weights=None, **kwargs)
    model.eval()
    sample = get_sample(model_class)
    out = model(sample)
    assert out.shape[2:] == sample.shape[2:]


@torch.inference_mode()
@pytest.mark.parametrize("encoder_name", MIT_ENCODERS)
@pytest.mark.parametrize("encoder_depth", [3, 5])
@pytest.mark.parametrize(
    "model_class",
    [
        torchseg.Unet,
        torchseg.FPN,
        torchseg.PSPNet,
        torchseg.Linknet,
        torchseg.UnetPlusPlus,
    ],
)
def test_mix_transformer(model_class, encoder_name, encoder_depth, **kwargs):
    if model_class in [torchseg.UnetPlusPlus, torchseg.Linknet]:
        return  # skip mit_b*
    if model_class is torchseg.FPN and encoder_depth != 5:
        return  # skip mit_b*

    if (
        model_class is torchseg.Unet
        or model_class is torchseg.UnetPlusPlus
        or model_class is torchseg.MAnet
    ):
        kwargs["decoder_channels"] = (16, 16, 16, 16, 16)[-encoder_depth:]
    model = model_class(
        encoder_name, encoder_depth=encoder_depth, encoder_weights=None, **kwargs
    )
    model.eval()
    sample = get_sample(model_class)

    if encoder_depth == 5 and model_class != torchseg.PSPNet:
        test_shape = True
    else:
        test_shape = False

    out = model(sample)

    if test_shape:
        assert out.shape[2:] == sample.shape[2:]


@torch.inference_mode()
@pytest.mark.parametrize(
    "model_class",
    [
        torchseg.PAN,
        torchseg.FPN,
        torchseg.PSPNet,
        torchseg.Linknet,
        torchseg.Unet,
        torchseg.UnetPlusPlus,
        torchseg.MAnet,
    ],
)
def test_aux_output(model_class):
    model = model_class(
        DEFAULT_ENCODER, encoder_weights=None, aux_params=dict(classes=2)
    )
    sample = get_sample(model_class)
    label_size = (sample.shape[0], 2)
    label = model(sample)[1]
    assert label.size() == label_size


@torch.inference_mode()
@pytest.mark.parametrize("upsampling", [2, 4, 8])
@pytest.mark.parametrize("model_class", [torchseg.FPN, torchseg.PSPNet])
def test_upsample(model_class, upsampling):
    default_upsampling = 4 if model_class is torchseg.FPN else 8
    model = model_class(DEFAULT_ENCODER, encoder_weights=None, upsampling=upsampling)
    sample = get_sample(model_class)
    mask = model(sample)
    assert mask.size()[-1] / 64 == upsampling / default_upsampling
