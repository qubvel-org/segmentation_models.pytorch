import pytest
from tests.models import base


@pytest.mark.unet
class TestUnetModel(base.BaseModelTester):
    test_model_type = "unet"
