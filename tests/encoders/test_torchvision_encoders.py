import segmentation_models_pytorch as smp

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
    files_for_diff = ["encoders/resnet.py"]

    def get_tiny_encoder(self):
        params = {
            "out_channels": [3, 64, 64, 128, 256, 512],
            "block": smp.encoders.resnet.BasicBlock,
            "layers": [1, 1, 1, 1],
        }
        return smp.encoders.resnet.ResNetEncoder(**params)


class TestDenseNetEncoder(base.BaseEncoderTester):
    supports_dilated = False
    encoder_names = (
        ["densenet121"]
        if not RUN_ALL_ENCODERS
        else ["densenet121", "densenet169", "densenet161"]
    )
    files_for_diff = ["encoders/densenet.py"]

    def get_tiny_encoder(self):
        params = {
            "out_channels": [3, 2, 3, 2, 2, 2],
            "num_init_features": 2,
            "growth_rate": 1,
            "block_config": (1, 1, 1, 1),
        }
        return smp.encoders.densenet.DenseNetEncoder(**params)


class TestMobileNetEncoder(base.BaseEncoderTester):
    encoder_names = ["mobilenet_v2"] if not RUN_ALL_ENCODERS else ["mobilenet_v2"]
    files_for_diff = ["encoders/mobilenet.py"]


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
    files_for_diff = ["encoders/vgg.py"]

    def get_tiny_encoder(self):
        params = {
            "out_channels": [4, 4, 4, 4, 4, 4],
            "config": [4, "M", 4, "M", 4, "M", 4, "M", 4, "M"],
            "batch_norm": False,
        }
        return smp.encoders.vgg.VGGEncoder(**params)
