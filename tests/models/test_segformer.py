import torch
import pytest
import segmentation_models_pytorch as smp

from tests.models import base
from tests.utils import slow_test, default_device, requires_torch_greater_or_equal


class TestSegformerModel(base.BaseModelTester):
    test_model_type = "segformer"
    files_for_diff = [r"decoders/segformer/", r"base/"]

    @slow_test
    @requires_torch_greater_or_equal("2.0.1")
    @pytest.mark.logits_match
    def test_load_pretrained(self):
        hub_checkpoint = "smp-hub/segformer-b0-512x512-ade-160k"

        model = smp.from_pretrained(hub_checkpoint)
        model = model.eval().to(default_device)

        sample = torch.ones([1, 3, 512, 512]).to(default_device)

        with torch.inference_mode():
            output = model(sample)

        self.assertEqual(output.shape, (1, 150, 512, 512))

        expected_logits_slice = torch.tensor(
            [-4.4172, -4.4723, -4.5273, -4.5824, -4.6375, -4.7157]
        )
        resulted_logits_slice = output[0, 0, 256, :6].cpu()
        is_equal = torch.allclose(
            expected_logits_slice, resulted_logits_slice, atol=1e-2
        )
        max_diff = torch.max(torch.abs(expected_logits_slice - resulted_logits_slice))
        self.assertTrue(
            is_equal,
            f"Expected logits slice and resulted logits slice are not equal.\n"
            f"Max diff: {max_diff}\n"
            f"Expected: {expected_logits_slice}\n"
            f"Resulted: {resulted_logits_slice}\n",
        )
