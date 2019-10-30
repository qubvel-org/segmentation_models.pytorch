import os
import sys
import mock
import pytest
import torch
import random
import importlib

# mock detection module
sys.modules["torchvision._C"] = mock.Mock()

import segmentation_models_pytorch as smp


def get_encoders():
    is_travis = True  # os.environ.get("TRAVIS", False)
    travis_exclude_encoders = [
        "senet154",
        "resnext101_32x16d",
        "resnext101_32x32d",
        "resnext101_32x48d",
    ]

    encoders = smp.encoders.get_encoder_names()
    if is_travis:
        encoders = [e for e in encoders if e not in travis_exclude_encoders]

    return encoders


def get_pretrained_weights_name(encoder_name):
    return list(smp.encoders.encoders[encoder_name]["pretrained_settings"].keys())[0]


ENCODERS = get_encoders()
DEFAULT_ENCODER = 'resnet18'
DEFAULT_SAMPLE = torch.ones([1, 3, 64, 64])


def _test_forward(model):
    with torch.no_grad():
        model(DEFAULT_SAMPLE)


def _test_forward_backward(model):
    out = model(DEFAULT_SAMPLE)
    out.mean().backward()


@pytest.mark.parametrize('encoder_name', ENCODERS)
@pytest.mark.parametrize('encoder_depth', [3, 5])
@pytest.mark.parametrize('model_class', [smp.FPN, smp.PSPNet, smp.Linknet, smp.Unet])
def test_forward(model_class, encoder_name, encoder_depth, **kwargs):
    if model_class is smp.Unet:
        kwargs['decoder_channels'] = (16, 16, 16, 16, 16)[-encoder_depth:]
    model = model_class(encoder_name, encoder_depth=encoder_depth, encoder_weights=None, **kwargs)
    _test_forward(model)


@pytest.mark.parametrize('model_class', [smp.FPN, smp.PSPNet, smp.Linknet, smp.Unet])
def test_forward_backward(model_class):
    model = model_class(DEFAULT_ENCODER)
    _test_forward_backward(model)


@pytest.mark.parametrize('model_class', [smp.FPN, smp.PSPNet, smp.Linknet, smp.Unet])
def test_aux_output(model_class):
    model = model_class(DEFAULT_ENCODER, aux_params=dict(classes=2))
    mask, label = model(DEFAULT_SAMPLE)
    assert label.size() == (1, 2)


# @pytest.mark.parametrize('upsampling', [2, 4, 8])
# @pytest.mark.parametrize('model_class', [smp.FPN, smp.PSPNet])
# def test_upsample(model_class):
#     model = model_class(DEFAULT_ENCODER, aux_params=dict(classes=2))
#     mask, label = model(DEFAULT_SAMPLE)
#     assert label.size() == (1, 2)


if __name__ == "__main__":
    pytest.main([__file__])
