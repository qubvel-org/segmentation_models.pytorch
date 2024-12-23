import pytest
from tests.models import base


@pytest.mark.linknet
class TestLinknetModel(base.BaseModelTester):
    test_model_type = "linknet"
