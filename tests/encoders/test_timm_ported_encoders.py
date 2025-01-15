from tests.encoders import base
from tests.utils import RUN_ALL_ENCODERS


class TestTimmEfficientNetEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["timm-efficientnet-b0"]
        if not RUN_ALL_ENCODERS
        else [
            "timm-efficientnet-b0",
            "timm-efficientnet-b1",
            "timm-efficientnet-b2",
            "timm-efficientnet-b3",
            "timm-efficientnet-b4",
            "timm-efficientnet-b5",
            "timm-efficientnet-b6",
            "timm-efficientnet-b7",
            "timm-efficientnet-b8",
            "timm-efficientnet-l2",
            "timm-tf_efficientnet_lite0",
            "timm-tf_efficientnet_lite1",
            "timm-tf_efficientnet_lite2",
            "timm-tf_efficientnet_lite3",
            "timm-tf_efficientnet_lite4",
        ]
    )
    files_for_diff = ["encoders/timm_efficientnet.py"]


class TestTimmGERNetEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["timm-gernet_s"]
        if not RUN_ALL_ENCODERS
        else ["timm-gernet_s", "timm-gernet_m", "timm-gernet_l"]
    )

    def test_compile(self):
        self.skipTest("Test to be removed")


class TestTimmMobileNetV3Encoder(base.BaseEncoderTester):
    encoder_names = (
        ["timm-mobilenetv3_small_100"]
        if not RUN_ALL_ENCODERS
        else [
            "timm-mobilenetv3_large_075",
            "timm-mobilenetv3_large_100",
            "timm-mobilenetv3_large_minimal_100",
            "timm-mobilenetv3_small_075",
            "timm-mobilenetv3_small_100",
            "timm-mobilenetv3_small_minimal_100",
        ]
    )

    def test_compile(self):
        self.skipTest("Test to be removed")


class TestTimmRegNetEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["timm-regnetx_002", "timm-regnety_002"]
        if not RUN_ALL_ENCODERS
        else [
            "timm-regnetx_002",
            "timm-regnetx_004",
            "timm-regnetx_006",
            "timm-regnetx_008",
            "timm-regnetx_016",
            "timm-regnetx_032",
            "timm-regnetx_040",
            "timm-regnetx_064",
            "timm-regnetx_080",
            "timm-regnetx_120",
            "timm-regnetx_160",
            "timm-regnetx_320",
            "timm-regnety_002",
            "timm-regnety_004",
            "timm-regnety_006",
            "timm-regnety_008",
            "timm-regnety_016",
            "timm-regnety_032",
            "timm-regnety_040",
            "timm-regnety_064",
            "timm-regnety_080",
            "timm-regnety_120",
            "timm-regnety_160",
            "timm-regnety_320",
        ]
    )

    def test_compile(self):
        self.skipTest("Test to be removed")


class TestTimmRes2NetEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["timm-res2net50_26w_4s"]
        if not RUN_ALL_ENCODERS
        else [
            "timm-res2net50_26w_4s",
            "timm-res2net101_26w_4s",
            "timm-res2net50_26w_6s",
            "timm-res2net50_26w_8s",
            "timm-res2net50_48w_2s",
            "timm-res2net50_14w_8s",
            "timm-res2next50",
        ]
    )

    def test_compile(self):
        self.skipTest("Test to be removed")


class TestTimmResnestEncoder(base.BaseEncoderTester):
    default_batch_size = 2
    encoder_names = (
        ["timm-resnest14d"]
        if not RUN_ALL_ENCODERS
        else [
            "timm-resnest14d",
            "timm-resnest26d",
            "timm-resnest50d",
            "timm-resnest101e",
            "timm-resnest200e",
            "timm-resnest269e",
            "timm-resnest50d_4s2x40d",
            "timm-resnest50d_1s4x24d",
        ]
    )

    def test_compile(self):
        self.skipTest("Test to be removed")


class TestTimmSkNetEncoder(base.BaseEncoderTester):
    default_batch_size = 2
    encoder_names = (
        ["timm-skresnet18"]
        if not RUN_ALL_ENCODERS
        else [
            "timm-skresnet18",
            "timm-skresnet34",
            "timm-skresnext50_32x4d",
        ]
    )
    files_for_diff = ["encoders/timm_sknet.py"]
