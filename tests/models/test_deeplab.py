from tests.models import base


class TestDeeplabV3Model(base.BaseModelTester):
    test_model_type = "deeplabv3"

    default_batch_size = 2


class TestDeeplabV3PlusModel(base.BaseModelTester):
    test_model_type = "deeplabv3plus"

    default_batch_size = 2
