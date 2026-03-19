import unittest
import warnings

import segmentation_models_pytorch as smp

from tests.encoders import base
from tests.utils import has_timm_test_models

# check if timm >= 1.0.12
timm_encoders = [
    "tu-resnet18",  # for timm universal traditional-like encoder
    "tu-convnext_atto",  # for timm universal transformer-like encoder
    "tu-darknet17",  # for timm universal vgg-like encoder
]

if has_timm_test_models:
    timm_encoders.insert(0, "tu-test_resnet.r160_in1k")


class TestTimmUniversalEncoder(base.BaseEncoderTester):
    encoder_names = timm_encoders
    files_for_diff = ["encoders/timm_universal.py"]


class TestTimmUniversalEncoderWeightsWarning(unittest.TestCase):
    """Test that tu- encoders emit appropriate warnings for encoder_weights."""

    def test_string_weights_emits_warning(self):
        with self.assertWarns(UserWarning):
            smp.encoders.get_encoder("tu-resnet18", weights="imagenet")

    def test_true_weights_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            smp.encoders.get_encoder("tu-resnet18", weights=True)

    def test_none_weights_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            smp.encoders.get_encoder("tu-resnet18", weights=None)
