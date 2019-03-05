import os
import pytest
import torch
import random

import segmentation_models_pytorch as smp


def get_encoder():
    is_travis = os.environ.get('TRAVIS', False)
    exclude = ['senet154']

    encoders = smp.encoders.get_encoder_names()
    if is_travis:
        encoders = [e for e in encoders if e not in exclude]

    return encoders


ENCODERS = get_encoder()


def _select_names(names, k=2):
    is_full = os.environ.get('FULL_TEST', False)
    if not is_full:
        return random.sample(names, k)
    else:
        return names


def _test_model(model_fn, encoder_name):
    model = model_fn(encoder_name)

    x = torch.ones((1, 3, 224, 224))
    y = model.predict(x)

    assert x.shape[2:] == y.shape[2:]


@pytest.mark.parametrize('encoder_name', _select_names(ENCODERS, k=5))
def test_unet(encoder_name):
    _test_model(smp.Unet, encoder_name)


if __name__ == '__main__':
    pytest.main([__file__])
