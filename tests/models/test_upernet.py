import pytest

from tests.models import base


class TestUnetModel(base.BaseModelTester):
    test_model_type = "upernet"
    files_for_diff = [r"decoders/upernet/", r"base/"]

    default_batch_size = 2

    @pytest.mark.torch_export
    def test_torch_export(self):
        super().test_torch_export(eps=1e-3)
