from tests.encoders import base


class TestTimmUniversalEncoder(base.BaseEncoderTester):
    encoder_names = [
        "tu-test_resnet.r160_in1k",
        "tu-resnet18",  # for timm universal traditional-like encoder
        "tu-convnext_atto",  # for timm universal transformer-like encoder
        "tu-darknet17",  # for timm universal vgg-like encoder
    ]
