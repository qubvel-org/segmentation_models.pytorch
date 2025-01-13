from tests.models import base


class TestUnetPlusPlusModel(base.BaseModelTester):
    test_model_type = "unetplusplus"
    files_for_diff = [r"decoders/unetplusplus/", r"base/"]
