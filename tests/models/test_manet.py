from tests.models import base


class TestManetModel(base.BaseModelTester):
    test_model_type = "manet"
    files_for_diff = [r"decoders/manet/", r"base/"]
