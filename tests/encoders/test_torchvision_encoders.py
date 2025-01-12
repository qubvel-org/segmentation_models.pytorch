from tests.encoders import base
from tests.utils import RUN_ALL_ENCODERS


class TestResNetEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["resnet18"]
        if not RUN_ALL_ENCODERS
        else [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x4d",
            "resnext101_32x8d",
            "resnext101_32x16d",
            "resnext101_32x32d",
            "resnext101_32x48d",
        ]
    )


class TestDenseNetEncoder(base.BaseEncoderTester):
    supports_dilated = False
    encoder_names = (
        ["densenet121"]
        if not RUN_ALL_ENCODERS
        else ["densenet121", "densenet169", "densenet161"]
    )


class TestMobileNetEncoder(base.BaseEncoderTester):
    encoder_names = ["mobilenet_v2"] if not RUN_ALL_ENCODERS else ["mobilenet_v2"]


class TestVggEncoder(base.BaseEncoderTester):
    supports_dilated = False
    encoder_names = (
        ["vgg11"]
        if not RUN_ALL_ENCODERS
        else [
            "vgg11",
            "vgg11_bn",
            "vgg13",
            "vgg13_bn",
            "vgg16",
            "vgg16_bn",
            "vgg19",
            "vgg19_bn",
        ]
    )
