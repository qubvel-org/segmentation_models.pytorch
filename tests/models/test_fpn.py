import segmentation_models_pytorch as smp

from tests.models import base


class TestFpnModel(base.BaseModelTester):
    test_model_type = "fpn"
    files_for_diff = [r"decoders/fpn/", r"base/"]

    def test_interpolation(self):
        # test bilinear
        model_1 = smp.create_model(
            self.test_model_type,
            self.test_encoder_name,
            decoder_interpolation="bilinear",
        )
        assert model_1.decoder.p2.interpolation_mode == "bilinear"
        assert model_1.decoder.p3.interpolation_mode == "bilinear"
        assert model_1.decoder.p4.interpolation_mode == "bilinear"

        # test bicubic
        model_2 = smp.create_model(
            self.test_model_type,
            self.test_encoder_name,
            decoder_interpolation="bicubic",
        )
        assert model_2.decoder.p2.interpolation_mode == "bicubic"
        assert model_2.decoder.p3.interpolation_mode == "bicubic"
        assert model_2.decoder.p4.interpolation_mode == "bicubic"
