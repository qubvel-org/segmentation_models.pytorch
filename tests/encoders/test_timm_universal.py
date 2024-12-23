from tests.encoders import base
from tests.config import has_timm_test_models

# check if timm >= 1.0.12
timm_encoders = [
    "tu-resnet18",  # for timm universal traditional-like encoder
    "tu-convnext_atto",  # for timm universal transformer-like encoder
    "tu-darknet17",  # for timm universal vgg-like encoder
]

if has_timm_test_models:
    timm_encoders.append("tu-test_resnet.r160_in1k")


class TestTimmUniversalEncoder(base.BaseEncoderTester):
    encoder_names = timm_encoders
