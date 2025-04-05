import pytest
from torch import nn
from segmentation_models_pytorch.base.modules import Conv2dReLU


def test_conv2drelu_batchnorm():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="batchnorm")

    assert isinstance(module[0], nn.Conv2d)
    assert isinstance(module[1], nn.BatchNorm2d)
    assert isinstance(module[2], nn.ReLU)


def test_conv2drelu_batchnorm_with_keywords():
    module = Conv2dReLU(
        3,
        16,
        kernel_size=3,
        padding=1,
        use_norm={"type": "batchnorm", "momentum": 1e-4, "affine": False},
    )

    assert isinstance(module[0], nn.Conv2d)
    assert isinstance(module[1], nn.BatchNorm2d)
    assert module[1].momentum == 1e-4 and module[1].affine is False
    assert isinstance(module[2], nn.ReLU)


def test_conv2drelu_identity():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="identity")

    assert isinstance(module[0], nn.Conv2d)
    assert isinstance(module[1], nn.Identity)
    assert isinstance(module[2], nn.ReLU)


def test_conv2drelu_layernorm():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="layernorm")

    assert isinstance(module[0], nn.Conv2d)
    assert isinstance(module[1], nn.LayerNorm)
    assert isinstance(module[2], nn.ReLU)


def test_conv2drelu_instancenorm():
    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="instancenorm")

    assert isinstance(module[0], nn.Conv2d)
    assert isinstance(module[1], nn.InstanceNorm2d)
    assert isinstance(module[2], nn.ReLU)


def test_conv2drelu_inplace():
    try:
        from inplace_abn import InPlaceABN
    except ImportError:
        pytest.skip("InPlaceABN is not installed")

    module = Conv2dReLU(3, 16, kernel_size=3, padding=1, use_norm="inplace")

    assert len(module) == 3
    assert isinstance(module[0], nn.Conv2d)
    assert isinstance(module[1], InPlaceABN)
    assert isinstance(module[2], nn.Identity)
