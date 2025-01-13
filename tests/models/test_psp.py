from tests.models import base


class TestPspModel(base.BaseModelTester):
    test_model_type = "pspnet"
    files_for_diff = [r"decoders/pspnet/", r"base/"]

    default_batch_size = 2
