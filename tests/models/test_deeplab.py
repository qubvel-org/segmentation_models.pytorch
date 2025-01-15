from tests.models import base


class TestDeeplabV3Model(base.BaseModelTester):
    test_model_type = "deeplabv3"
    files_for_diff = [r"decoders/deeplabv3/", r"base/"]

    default_batch_size = 2


class TestDeeplabV3PlusModel(base.BaseModelTester):
    test_model_type = "deeplabv3plus"
    files_for_diff = [r"decoders/deeplabv3plus/", r"base/"]

    default_batch_size = 2
