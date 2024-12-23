import pytest
from tests.models import base


@pytest.mark.deeplabv3
class TestDeeplabV3Model(base.BaseModelTester):
    test_model_type = "deeplabv3"

    default_batch_size = 2


@pytest.mark.deeplabv3plus
class TestDeeplabV3PlusModel(base.BaseModelTester):
    test_model_type = "deeplabv3plus"

    default_batch_size = 2
