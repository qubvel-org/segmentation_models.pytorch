import pytest
from tests.models import base


@pytest.mark.segformer
class TestSegformerModel(base.BaseModelTester):
    test_model_type = "segformer"
