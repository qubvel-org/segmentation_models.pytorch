import pytest
import segmentation_models_pytorch as smp

from tests.models import base


class TestPanModel(base.BaseModelTester):
    test_model_type = "pan"
    files_for_diff = [r"decoders/pan/", r"base/"]

    default_batch_size = 2
    default_height = 128
    default_width = 128

    def test_interpolation(self):
        # test bilinear
        model_1 = smp.create_model(
            self.test_model_type,
            self.test_encoder_name,
            decoder_interpolation="bilinear",
        )
        assert model_1.decoder.gau1.interpolation_mode == "bilinear"
        assert model_1.decoder.gau1.align_corners is True
        assert model_1.decoder.gau2.interpolation_mode == "bilinear"
        assert model_1.decoder.gau2.align_corners is True
        assert model_1.decoder.gau3.interpolation_mode == "bilinear"
        assert model_1.decoder.gau3.align_corners is True

        # test bicubic
        model_2 = smp.create_model(
            self.test_model_type,
            self.test_encoder_name,
            decoder_interpolation="bicubic",
        )
        assert model_2.decoder.gau1.interpolation_mode == "bicubic"
        assert model_2.decoder.gau1.align_corners is None
        assert model_2.decoder.gau2.interpolation_mode == "bicubic"
        assert model_2.decoder.gau2.align_corners is None
        assert model_2.decoder.gau3.interpolation_mode == "bicubic"
        assert model_2.decoder.gau3.align_corners is None

        with pytest.warns(DeprecationWarning):
            smp.create_model(
                self.test_model_type,
                self.test_encoder_name,
                upscale_mode="bicubic",
            )
            assert model_2.decoder.gau1.interpolation_mode == "bicubic"
