import pytest
from tests.models import base


@pytest.mark.unetplusplus
class TestUnetPlusPlusModel(base.BaseModelTester):
    test_model_type = "unetplusplus"
