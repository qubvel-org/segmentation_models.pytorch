import pytest
from tests.models import base


@pytest.mark.fpn
class TestFpnModel(base.BaseModelTester):
    test_model_type = "fpn"
