import segmentation_models_pytorch as smp

from tests.models import base


class TestManetModel(base.BaseModelTester):
    test_model_type = "manet"
    files_for_diff = [r"decoders/manet/", r"base/"]

    def test_interpolation(self):
        # test bilinear
        model_1 = smp.create_model(
            self.test_model_type,
            self.test_encoder_name,
            decoder_interpolation="bilinear",
        )
        for block in model_1.decoder.blocks:
            assert block.interpolation_mode == "bilinear"

        # test bicubic
        model_2 = smp.create_model(
            self.test_model_type,
            self.test_encoder_name,
            decoder_interpolation="bicubic",
        )
        for block in model_2.decoder.blocks:
            assert block.interpolation_mode == "bicubic"
