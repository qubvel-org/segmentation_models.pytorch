import segmentation_models_pytorch as smp

from tests.models import base


class TestUnetPlusPlusModel(base.BaseModelTester):
    test_model_type = "unetplusplus"
    files_for_diff = [r"decoders/unetplusplus/", r"base/"]

    def test_interpolation(self):
        # test bilinear
        model_1 = smp.create_model(
            self.test_model_type,
            self.test_encoder_name,
            decoder_interpolation="bilinear",
        )
        is_tested = False
        for module in model_1.decoder.modules():
            if module.__class__.__name__ == "DecoderBlock":
                assert module.interpolation_mode == "bilinear"
                is_tested = True
        assert is_tested

        # test bicubic
        model_2 = smp.create_model(
            self.test_model_type,
            self.test_encoder_name,
            decoder_interpolation="bicubic",
        )
        is_tested = False
        for module in model_2.decoder.modules():
            if module.__class__.__name__ == "DecoderBlock":
                assert module.interpolation_mode == "bicubic"
                is_tested = True
        assert is_tested
