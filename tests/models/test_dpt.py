import pytest
import inspect
import torch
import segmentation_models_pytorch as smp

from tests.models import base
from tests.utils import (
    slow_test,
    default_device,
    requires_torch_greater_or_equal,
)


class TestDPTModel(base.BaseModelTester):
    test_encoder_name = "tu-vit_tiny_patch16_224"
    files_for_diff = [r"decoders/dpt/", r"base/"]

    default_height = 224
    default_width = 224

    # should be overriden
    test_model_type = "dpt"

    compile_dynamic = False

    @property
    def decoder_channels(self):
        signature = inspect.signature(self.model_class)
        return signature.parameters["decoder_intermediate_channels"].default

    @property
    def hub_checkpoint(self):
        return "smp-test-models/dpt-tu-test_vit"

    @slow_test
    @requires_torch_greater_or_equal("2.0.1")
    @pytest.mark.logits_match
    def test_load_pretrained(self):
        hub_checkpoint = "smp-hub/dpt-large-ade20k"

        model = smp.from_pretrained(hub_checkpoint)
        model = model.eval().to(default_device)

        input_tensor = torch.ones((1, 3, 384, 384))
        input_tensor = input_tensor.to(default_device)

        expected_logits_slice = torch.tensor(
            [3.4166, 3.4422, 3.4677, 3.2784, 3.0880, 2.9497]
        )
        with torch.inference_mode():
            output = model(input_tensor)

        resulted_logits_slice = output[0, 0, 0, 0:6].cpu()

        self.assertEqual(expected_logits_slice.shape, resulted_logits_slice.shape)
        is_close = torch.allclose(
            expected_logits_slice, resulted_logits_slice, atol=5e-2
        )
        max_diff = torch.max(torch.abs(expected_logits_slice - resulted_logits_slice))
        self.assertTrue(is_close, f"Max diff: {max_diff}")
