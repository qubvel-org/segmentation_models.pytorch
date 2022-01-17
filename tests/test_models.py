import sys
import mock
import pytest
import torch

# mock detection module
sys.modules["torchvision._C"] = mock.Mock()
import segmentation_models_pytorch as smp  # noqa


def get_encoders():
    exclude_encoders = [
        "senet154",
        "resnext101_32x16d",
        "resnext101_32x32d",
        "resnext101_32x48d",
    ]
    encoders = smp.encoders.get_encoder_names()
    encoders = [e for e in encoders if e not in exclude_encoders]
    encoders.append("tu-resnet34")  # for timm universal encoder
    return encoders


ENCODERS = get_encoders()
DEFAULT_ENCODER = "resnet18"


def get_sample(model_class):
    if model_class in [smp.Unet, smp.Linknet, smp.FPN, smp.PSPNet, smp.UnetPlusPlus, smp.MAnet]:
        sample = torch.ones([1, 3, 64, 64])
    elif model_class == smp.PAN:
        sample = torch.ones([2, 3, 256, 256])
    elif model_class == smp.DeepLabV3:
        sample = torch.ones([2, 3, 128, 128])
    else:
        raise ValueError("Not supported model class {}".format(model_class))
    return sample


def _test_forward(model, sample, test_shape=False):
    with torch.no_grad():
        out = model(sample)
    if test_shape:
        assert out.shape[2:] == sample.shape[2:]


def _test_forward_backward(model, sample, test_shape=False):
    out = model(sample)
    out.mean().backward()
    if test_shape:
        assert out.shape[2:] == sample.shape[2:]


@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("encoder_depth", [3, 5])
@pytest.mark.parametrize("model_class", [smp.FPN, smp.PSPNet, smp.Linknet, smp.Unet, smp.UnetPlusPlus])
def test_forward(model_class, encoder_name, encoder_depth, **kwargs):
    if model_class is smp.Unet or model_class is smp.UnetPlusPlus or model_class is smp.MAnet:
        kwargs["decoder_channels"] = (16, 16, 16, 16, 16)[-encoder_depth:]
    model = model_class(encoder_name, encoder_depth=encoder_depth, encoder_weights=None, **kwargs)
    sample = get_sample(model_class)
    model.eval()
    if encoder_depth == 5 and model_class != smp.PSPNet:
        test_shape = True
    else:
        test_shape = False

    _test_forward(model, sample, test_shape)


@pytest.mark.parametrize(
    "model_class", [smp.PAN, smp.FPN, smp.PSPNet, smp.Linknet, smp.Unet, smp.UnetPlusPlus, smp.MAnet, smp.DeepLabV3]
)
def test_forward_backward(model_class):
    sample = get_sample(model_class)
    model = model_class(DEFAULT_ENCODER, encoder_weights=None)
    _test_forward_backward(model, sample)


@pytest.mark.parametrize(
    "model_class", [smp.PAN, smp.FPN, smp.PSPNet, smp.Linknet, smp.Unet, smp.UnetPlusPlus, smp.MAnet]
)
def test_aux_output(model_class):
    model = model_class(DEFAULT_ENCODER, encoder_weights=None, aux_params=dict(classes=2))
    sample = get_sample(model_class)
    label_size = (sample.shape[0], 2)
    mask, label = model(sample)
    assert label.size() == label_size


@pytest.mark.parametrize("upsampling", [2, 4, 8])
@pytest.mark.parametrize("model_class", [smp.FPN, smp.PSPNet])
def test_upsample(model_class, upsampling):
    default_upsampling = 4 if model_class is smp.FPN else 8
    model = model_class(DEFAULT_ENCODER, encoder_weights=None, upsampling=upsampling)
    sample = get_sample(model_class)
    mask = model(sample)
    assert mask.size()[-1] / 64 == upsampling / default_upsampling


@pytest.mark.parametrize("model_class", [smp.FPN])
@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("in_channels", [1, 2, 4])
def test_in_channels(model_class, encoder_name, in_channels):
    sample = torch.ones([1, in_channels, 64, 64])
    model = model_class(DEFAULT_ENCODER, encoder_weights=None, in_channels=in_channels)
    model.eval()
    with torch.no_grad():
        model(sample)

    assert model.encoder._in_channels == in_channels


@pytest.mark.parametrize("encoder_name", ENCODERS)
def test_dilation(encoder_name):
    if (
        encoder_name in ["inceptionresnetv2", "xception", "inceptionv4"]
        or encoder_name.startswith("vgg")
        or encoder_name.startswith("densenet")
        or encoder_name.startswith("timm-res")
    ):
        return

    encoder = smp.encoders.get_encoder(encoder_name, output_stride=16)

    encoder.eval()
    with torch.no_grad():
        sample = torch.ones([1, 3, 64, 64])
        output = encoder(sample)

    shapes = [out.shape[-1] for out in output]
    assert shapes == [64, 32, 16, 8, 4, 4]  # last downsampling replaced with dilation


if __name__ == "__main__":
    pytest.main([__file__])
