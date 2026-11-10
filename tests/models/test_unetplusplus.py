import pytest
import segmentation_models_pytorch as smp
import torch

from tests.models import base


class TestUnetPlusPlusModel(base.BaseModelTester):
    test_model_type = "unetplusplus"
    files_for_diff = [r"decoders/unetplusplus/", r"base/"]

    def test_timm_transformer_style_encoder(self):
        model = smp.UnetPlusPlus("tu-convnext_atto", encoder_weights=None).eval()

        with torch.inference_mode():
            output = model(torch.rand(1, 3, 256, 256))

        assert output.shape == (1, 1, 256, 256)

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

    @pytest.mark.torch_script
    def test_torch_script_decoder_configs(self):
        configs = [
            dict(encoder_name=self.test_encoder_name),
            dict(
                encoder_name=self.test_encoder_name,
                encoder_depth=3,
                decoder_channels=(32, 16, 8),
            ),
            dict(
                encoder_name=self.test_encoder_name,
                encoder_depth=4,
                decoder_channels=(32, 16, 8, 4),
            ),
            dict(encoder_name=self.test_encoder_name, decoder_attention_type="scse"),
            dict(encoder_name=self.test_encoder_name, decoder_interpolation="bilinear"),
            dict(encoder_name="vgg11"),  # decoder with center block
        ]
        sample = torch.rand(1, 3, 64, 64)

        for config in configs:
            with self.subTest(**config):
                model = smp.UnetPlusPlus(encoder_weights=None, classes=2, **config)
                model.eval()
                scripted_model = torch.jit.script(model)

                with torch.inference_mode():
                    eager_output = model(sample)
                    scripted_output = scripted_model(sample)

                torch.testing.assert_close(
                    scripted_output, eager_output, rtol=1e-3, atol=1e-3
                )

    @pytest.mark.torch_script
    def test_torch_script_with_aux_head(self):
        model = smp.UnetPlusPlus(
            self.test_encoder_name,
            encoder_weights=None,
            classes=2,
            aux_params=dict(classes=3),
        ).eval()
        sample = torch.rand(1, 3, 64, 64)

        scripted_model = torch.jit.script(model)

        with torch.inference_mode():
            eager_mask, eager_label = model(sample)
            scripted_mask, scripted_label = scripted_model(sample)

        torch.testing.assert_close(scripted_mask, eager_mask, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(scripted_label, eager_label, rtol=1e-3, atol=1e-3)

    def test_decoder_state_dict_keys_unchanged(self):
        model = smp.UnetPlusPlus(self.test_encoder_name, encoder_weights=None)
        depth = model.decoder.depth
        expected = {f"x_{i}_{j}" for j in range(depth) for i in range(j + 1)}
        expected.add(f"x_0_{depth}")
        names = {
            k.split(".")[2]
            for k in model.state_dict()
            if k.startswith("decoder.blocks.")
        }
        assert names == expected
