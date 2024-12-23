import pytest
from tests.models import base


@pytest.mark.upernet
class TestUnetModel(base.BaseModelTester):
    test_model_type = "upernet"
    default_batch_size = 2
