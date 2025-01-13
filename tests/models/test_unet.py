from tests.models import base


class TestUnetModel(base.BaseModelTester):
    test_model_type = "unet"
    files_for_diff = [r"decoders/unet/", r"base/"]
