import os
import sys
import mock
import pytest
import torch
import random
import importlib

# mock detection module 
sys.modules['torchvision._C'] = mock.Mock()

import segmentation_models_pytorch as smp


def get_encoder():
    is_travis = os.environ.get('TRAVIS', False)
    exclude = ['senet154']

    encoders = smp.encoders.get_encoder_names()
    if is_travis:
        encoders = [e for e in encoders if e not in exclude]

    return encoders

def get_pretrained_weights_name(encoder_name):
    return list(smp.encoders.encoders[encoder_name]['pretrained_settings'].keys())[0]

ENCODERS = get_encoder()


def _select_names(names, k=2):
    is_full = os.environ.get('FULL_TEST', False)
    if not is_full:
        return random.sample(names, k)
    else:
        return names


def _test_forward_backward(model_fn, encoder_name, **model_params):
    model = model_fn(encoder_name, encoder_weights=None, **model_params)

    x = torch.ones((1, 3, 64, 64))
    y = model.forward(x)
    l = y.mean()
    l.backward()


def _test_pretrained_model(model_fn, encoder_name, encoder_weights, **model_params):
    model = model_fn(encoder_name, encoder_weights=encoder_weights, **model_params)

    x = torch.ones((1, 3, 64, 64))
    y = model.predict(x)

    assert x.shape[2:] == y.shape[2:]


@pytest.mark.parametrize('encoder_name', _select_names(ENCODERS, k=1))
def test_unet(encoder_name):
    _test_forward_backward(smp.Unet, encoder_name)
    _test_pretrained_model(smp.Unet, encoder_name, get_pretrained_weights_name(encoder_name))


@pytest.mark.parametrize('encoder_name', _select_names(ENCODERS, k=1))
def test_fpn(encoder_name):
    _test_forward_backward(smp.FPN, encoder_name)
    _test_pretrained_model(smp.FPN, encoder_name, get_pretrained_weights_name(encoder_name))

    from functools import partial
    _test_forward_backward(partial(smp.FPN, decoder_merge_policy='cat'), encoder_name)
    _test_pretrained_model(partial(smp.FPN, decoder_merge_policy='cat'), encoder_name, get_pretrained_weights_name(encoder_name))


@pytest.mark.parametrize('encoder_name', _select_names(ENCODERS, k=1))
def test_linknet(encoder_name):
    _test_forward_backward(smp.Linknet, encoder_name)
    _test_pretrained_model(smp.Linknet, encoder_name, get_pretrained_weights_name(encoder_name))


@pytest.mark.parametrize('encoder_name', _select_names(ENCODERS, k=1))
def test_pspnet(encoder_name):
    _test_forward_backward(smp.PSPNet, encoder_name)
    _test_pretrained_model(smp.PSPNet, encoder_name, get_pretrained_weights_name(encoder_name))


@pytest.mark.skipif(importlib.util.find_spec('inplace_abn') is None, reason='')
def test_inplace_abn():
    _test_forward_backward(smp.Unet, 'resnet18', decoder_use_batchnorm='inplace')
    _test_pretrained_model(smp.Unet, 'resnet18', get_pretrained_weights_name('resnet18'), decoder_use_batchnorm='inplace')


if __name__ == '__main__':
    pytest.main([__file__])
