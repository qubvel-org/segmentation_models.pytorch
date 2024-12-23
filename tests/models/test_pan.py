import pytest
from tests.models import base


@pytest.mark.pan
class TestPanModel(base.BaseModelTester):
    test_model_type = "pan"

    default_batch_size = 2
    default_height = 128
    default_width = 128
