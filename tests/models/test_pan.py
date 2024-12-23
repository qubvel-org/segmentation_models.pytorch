import pytest
from tests.models import base


@pytest.mark.pan
class TestPanModel(base.BaseModelTester):
    test_model_type = "pan"
    test_encoder_name = "resnet-18"

    default_batch_size = 2
    default_height = 128
    default_width = 128
