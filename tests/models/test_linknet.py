from tests.models import base


class TestLinknetModel(base.BaseModelTester):
    test_model_type = "linknet"
    files_for_diff = [r"decoders/linknet/", r"base/"]
