import segmentation_models_pytorch as smp

from tests.encoders import base
from tests.utils import RUN_ALL_ENCODERS


class TestDPNEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["dpn68"]
        if not RUN_ALL_ENCODERS
        else ["dpn68", "dpn68b", "dpn92", "dpn98", "dpn107", "dpn131"]
    )
    files_for_diff = ["encoders/dpn.py"]

    def get_tiny_encoder(self):
        params = {
            "stage_idxs": [2, 3, 4, 6],
            "out_channels": [3, 2, 70, 134, 262, 518],
            "groups": 2,
            "inc_sec": (2, 2, 2, 2),
            "k_r": 2,
            "k_sec": (1, 1, 1, 1),
            "num_classes": 1000,
            "num_init_features": 2,
            "small": True,
            "test_time_pool": True,
        }
        return smp.encoders.dpn.DPNEncoder(**params)


class TestInceptionResNetV2Encoder(base.BaseEncoderTester):
    encoder_names = (
        ["inceptionresnetv2"] if not RUN_ALL_ENCODERS else ["inceptionresnetv2"]
    )
    files_for_diff = ["encoders/inceptionresnetv2.py"]
    supports_dilated = False


class TestInceptionV4Encoder(base.BaseEncoderTester):
    encoder_names = ["inceptionv4"] if not RUN_ALL_ENCODERS else ["inceptionv4"]
    files_for_diff = ["encoders/inceptionv4.py"]
    supports_dilated = False


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
    files_for_diff = ["encoders/senet.py"]

    def get_tiny_encoder(self):
        params = {
            "out_channels": [3, 2, 256, 512, 1024, 2048],
            "block": smp.encoders.senet.SEResNetBottleneck,
            "layers": [1, 1, 1, 1],
            "downsample_kernel_size": 1,
            "downsample_padding": 0,
            "dropout_p": None,
            "groups": 1,
            "inplanes": 2,
            "input_3x3": False,
            "num_classes": 1000,
            "reduction": 2,
        }
        return smp.encoders.senet.SENetEncoder(**params)


class TestXceptionEncoder(base.BaseEncoderTester):
    supports_dilated = False
    encoder_names = ["xception"] if not RUN_ALL_ENCODERS else ["xception"]
    files_for_diff = ["encoders/xception.py"]
