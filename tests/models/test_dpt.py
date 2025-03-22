import os
import pytest
import inspect
import tempfile
from functools import lru_cache
from huggingface_hub import hf_hub_download
import torch
import segmentation_models_pytorch as smp

from tests.models import base
from tests.utils import (
    slow_test,
    default_device,
    requires_torch_greater_or_equal,
    check_run_test_on_diff_or_main,
)


class TestDPTModel(base.BaseModelTester):
    test_encoder_name = "tu-vit_large_patch16_384"
    files_for_diff = [r"decoders/dpt/", r"base/"]

    default_height = 384
    default_width = 384

    # should be overriden
    test_model_type = "dpt"

    @property
    def hub_checkpoint(self):
        return f"vedantdalimkar/DPT"

    @pytest.mark.compile
    def test_compile(self):
        if not check_run_test_on_diff_or_main(self.files_for_diff):
            self.skipTest("No diff and not on `main`.")

        sample = self._get_sample().to(default_device)
        model = self.get_default_model()
        model = model.eval().to(default_device)

        if not model._is_torch_compilable:
            with self.assertRaises(RuntimeError):
                torch.compiler.reset()
                compiled_model = torch.compile(
                    model, fullgraph=True, dynamic=True, backend="eager"
                )
            return

        with torch.inference_mode():
            compiled_model(sample)

    @pytest.mark.torch_script
    def test_torch_script(self):
        if not check_run_test_on_diff_or_main(self.files_for_diff):
            self.skipTest("No diff and not on `main`.")

        sample = self._get_sample().to(default_device)
        model = self.get_default_model()
        model.eval()

        if not model._is_torch_scriptable:
            with self.assertRaises(RuntimeError):
                scripted_model = torch.jit.script(model)
            return

        scripted_model = torch.jit.script(model)

        with torch.inference_mode():
            scripted_output = scripted_model(sample)
            eager_output = model(sample)

        self.assertEqual(scripted_output.shape, eager_output.shape)
        torch.testing.assert_close(scripted_output, eager_output)

    @slow_test
    @requires_torch_greater_or_equal("2.0.1")
    @pytest.mark.logits_match
    def test_preserve_forward_output(self):
        model = smp.from_pretrained(self.hub_checkpoint).eval().to(default_device)

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
