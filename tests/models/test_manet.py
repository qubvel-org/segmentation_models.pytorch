import pytest
from tests.models import base


@pytest.mark.manet
class TestManetModel(base.BaseModelTester):
    test_model_type = "manet"
