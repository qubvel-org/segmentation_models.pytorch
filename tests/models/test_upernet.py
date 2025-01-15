from tests.models import base


class TestUnetModel(base.BaseModelTester):
    test_model_type = "upernet"
    files_for_diff = [r"decoders/upernet/", r"base/"]

    default_batch_size = 2
