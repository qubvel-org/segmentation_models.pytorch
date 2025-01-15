from tests.models import base


class TestPanModel(base.BaseModelTester):
    test_model_type = "pan"
    files_for_diff = [r"decoders/pan/", r"base/"]

    default_batch_size = 2
    default_height = 128
    default_width = 128
