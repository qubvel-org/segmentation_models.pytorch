from tests.models import base


class TestPspModel(base.BaseModelTester):
    test_model_type = "pspnet"

    default_batch_size = 2
