from tests.models import base


class TestFpnModel(base.BaseModelTester):
    test_model_type = "fpn"
    files_for_diff = [r"decoders/fpn/", r"base/"]
