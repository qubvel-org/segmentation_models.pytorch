import timm
from tests.encoders import base
from packaging.version import Version

# check if timm >= 1.0.12
timm_encoders = [
    "tu-resnet18",  # for timm universal traditional-like encoder
    "tu-convnext_atto",  # for timm universal transformer-like encoder
    "tu-darknet17",  # for timm universal vgg-like encoder
]

if Version(timm.__version__) >= Version("1.0.12"):
    timm_encoders.append("tu-test_resnet.r160_in1k")


class TestTimmUniversalEncoder(base.BaseEncoderTester):
    encoder_names = timm_encoders
