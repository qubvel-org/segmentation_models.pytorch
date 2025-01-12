from tests.encoders import base
from tests.utils import RUN_ALL_ENCODERS


class TestDPNEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["dpn68"]
        if not RUN_ALL_ENCODERS
        else ["dpn68", "dpn68b", "dpn92", "dpn98", "dpn107", "dpn131"]
    )


class TestInceptionResNetV2Encoder(base.BaseEncoderTester):
    supports_dilated = False
    encoder_names = (
        ["inceptionresnetv2"] if not RUN_ALL_ENCODERS else ["inceptionresnetv2"]
    )


class TestInceptionV4Encoder(base.BaseEncoderTester):
    supports_dilated = False
    encoder_names = ["inceptionv4"] if not RUN_ALL_ENCODERS else ["inceptionv4"]


class TestSeNetEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["se_resnet50"]
        if not RUN_ALL_ENCODERS
        else [
            "se_resnet50",
            "se_resnet101",
            "se_resnet152",
            "se_resnext50_32x4d",
            "se_resnext101_32x4d",
            # "senet154",  # extra large model
        ]
    )


class TestXceptionEncoder(base.BaseEncoderTester):
    supports_dilated = False
    encoder_names = ["xception"] if not RUN_ALL_ENCODERS else ["xception"]
