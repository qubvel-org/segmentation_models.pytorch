from tests.encoders import base
from tests.utils import RUN_ALL_ENCODERS


class TestMobileoneEncoder(base.BaseEncoderTester):
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
