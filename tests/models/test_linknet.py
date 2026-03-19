import torch

import segmentation_models_pytorch as smp

from tests.models import base


class TestLinknetModel(base.BaseModelTester):
    test_model_type = "linknet"
    files_for_diff = [r"decoders/linknet/", r"base/"]

    def test_timm_transformer_style_encoder(self):
        model = smp.Linknet("tu-convnext_atto", encoder_weights=None).eval()

        with torch.inference_mode():
            output = model(torch.rand(1, 3, 256, 256))

        assert output.shape == (1, 1, 256, 256)
