import os
import pytest
import inspect
import tempfile
import unittest
from functools import lru_cache
from huggingface_hub import hf_hub_download

import torch
import segmentation_models_pytorch as smp

from tests.utils import (
    has_timm_test_models,
    default_device,
    slow_test,
    requires_torch_greater_or_equal,
    check_run_test_on_diff_or_main,
)


class BaseModelTester(unittest.TestCase):
    test_encoder_name = (
        "tu-test_resnet.r160_in1k" if has_timm_test_models else "resnet18"
    )
    files_for_diff = [r".*"]

    # should be overriden
    test_model_type = None

    # test sample configuration
    default_batch_size = 1
    default_num_channels = 3
    default_height = 64
    default_width = 64

    compile_dynamic = True

    @property
    def model_type(self):
        if self.test_model_type is None:
            raise ValueError("test_model_type is not set")
        return self.test_model_type

    @property
    def hub_checkpoint(self):
        return f"smp-test-models/{self.model_type}-tu-resnet18"

    @property
    def model_class(self):
        return smp.MODEL_ARCHITECTURES_MAPPING[self.model_type]

    @property
    def decoder_channels(self):
        signature = inspect.signature(self.model_class)
        # check if decoder_channels is in the signature
        if "decoder_channels" in signature.parameters:
            return signature.parameters["decoder_channels"].default
        return None

    @lru_cache
    def _get_sample(self, batch_size=None, num_channels=None, height=None, width=None):
        batch_size = batch_size or self.default_batch_size
        num_channels = num_channels or self.default_num_channels
        height = height or self.default_height
        width = width or self.default_width
        return torch.rand(batch_size, num_channels, height, width)

    @lru_cache
    def get_default_model(self):
        model = smp.create_model(self.model_type, self.test_encoder_name)
        model = model.to(default_device)
        return model

    def test_forward_backward(self):
        sample = self._get_sample().to(default_device)

        model = self.get_default_model()

        # check default in_channels=3
        output = model(sample)

        # check default output number of classes = 1
        expected_number_of_classes = 1
        result_number_of_classes = output.shape[1]
        self.assertEqual(
            result_number_of_classes,
            expected_number_of_classes,
            f"Default output number of classes should be {expected_number_of_classes}, but got {result_number_of_classes}",
        )

        # check backward pass
        output.mean().backward()

    def test_in_channels_and_depth_and_out_classes(
        self, in_channels=1, depth=3, classes=7
    ):
        kwargs = {}

        if self.model_type in ["unet", "unetplusplus", "manet"]:
            kwargs = {"decoder_channels": self.decoder_channels[:depth]}

        if self.model_type == "dpt":
            kwargs = {"decoder_intermediate_channels": self.decoder_channels[:depth]}

        model = (
            smp.create_model(
                arch=self.model_type,
                encoder_name=self.test_encoder_name,
                encoder_depth=depth,
                in_channels=in_channels,
                classes=classes,
                **kwargs,
            )
            .to(default_device)
            .eval()
        )

        sample = self._get_sample(num_channels=in_channels).to(default_device)

        # check in channels correctly set
        with torch.inference_mode():
            output = model(sample)

        self.assertEqual(output.shape[1], classes)

    def test_classification_head(self):
        model = smp.create_model(
            arch=self.model_type,
            encoder_name=self.test_encoder_name,
            aux_params={
                "pooling": "avg",
                "classes": 10,
                "dropout": 0.5,
                "activation": "sigmoid",
            },
        )
        model = model.to(default_device).eval()

        self.assertIsNotNone(model.classification_head)
        self.assertIsInstance(model.classification_head[0], torch.nn.AdaptiveAvgPool2d)
        self.assertIsInstance(model.classification_head[1], torch.nn.Flatten)
        self.assertIsInstance(model.classification_head[2], torch.nn.Dropout)
        self.assertEqual(model.classification_head[2].p, 0.5)
        self.assertIsInstance(model.classification_head[3], torch.nn.Linear)
        self.assertIsInstance(model.classification_head[4].activation, torch.nn.Sigmoid)

        sample = self._get_sample().to(default_device)

        with torch.inference_mode():
            _, cls_probs = model(sample)

        self.assertEqual(cls_probs.shape[1], 10)

    def test_any_resolution(self):
        model = self.get_default_model()

        sample = self._get_sample(
            height=self.default_height + 3,
            width=self.default_width + 7,
        ).to(default_device)

        if model.requires_divisible_input_shape:
            with self.assertRaises(RuntimeError, msg="Wrong input shape"):
                output = model(sample)
            return

        with torch.inference_mode():
            output = model(sample)

        self.assertEqual(output.shape[2], self.default_height + 3)
        self.assertEqual(output.shape[3], self.default_width + 7)

    @requires_torch_greater_or_equal("2.0.1")
    def test_save_load_with_hub_mixin(self):
        # instantiate model
        model = self.get_default_model()
        model.eval()

        # save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(
                tmpdir, dataset="test_dataset", metrics={"my_awesome_metric": 0.99}
            )
            restored_model = smp.from_pretrained(tmpdir).to(default_device)
            restored_model.eval()

            with open(os.path.join(tmpdir, "README.md"), "r") as f:
                readme = f.read()

        # check inference is correct
        sample = self._get_sample().to(default_device)

        with torch.inference_mode():
            output = model(sample)
            restored_output = restored_model(sample)

        self.assertEqual(output.shape, restored_output.shape)
        self.assertEqual(output.shape[1], 1)

        # check dataset and metrics are saved in readme
        self.assertIn("test_dataset", readme)
        self.assertIn("my_awesome_metric", readme)

    @slow_test
    @requires_torch_greater_or_equal("2.0.1")
    @pytest.mark.logits_match
    def test_preserve_forward_output(self):
        model = smp.from_pretrained(self.hub_checkpoint).eval().to(default_device)

        input_tensor_path = hf_hub_download(
            repo_id=self.hub_checkpoint, filename="input-tensor.pth"
        )
        output_tensor_path = hf_hub_download(
            repo_id=self.hub_checkpoint, filename="output-tensor.pth"
        )

        input_tensor = torch.load(input_tensor_path, weights_only=True)
        input_tensor = input_tensor.to(default_device)
        output_tensor = torch.load(output_tensor_path, weights_only=True)
        output_tensor = output_tensor.to(default_device)

        with torch.inference_mode():
            output = model(input_tensor)

        self.assertEqual(output.shape, output_tensor.shape)
        is_close = torch.allclose(output, output_tensor, atol=5e-2)
        max_diff = torch.max(torch.abs(output - output_tensor))
        self.assertTrue(is_close, f"Max diff: {max_diff}")

    @pytest.mark.compile
    def test_compile(self):
        if not check_run_test_on_diff_or_main(self.files_for_diff):
            self.skipTest("No diff and not on `main`.")

        sample = self._get_sample().to(default_device)
        model = self.get_default_model()
        model = model.eval().to(default_device)

        if not model._is_torch_compilable:
            with self.assertRaises((RuntimeError)):
                torch.compiler.reset()
                compiled_model = torch.compile(
                    model, fullgraph=True, dynamic=self.compile_dynamic, backend="eager"
                )
            return

        torch.compiler.reset()
        compiled_model = torch.compile(
            model, fullgraph=True, dynamic=self.compile_dynamic, backend="eager"
        )
        with torch.inference_mode():
            compiled_model(sample)

    @pytest.mark.torch_export
    def test_torch_export(self, eps=1e-5):
        if not check_run_test_on_diff_or_main(self.files_for_diff):
            self.skipTest("No diff and not on `main`.")

        torch.manual_seed(42)
        sample = self._get_sample().to(default_device)
        model = self.get_default_model()
        model.eval()

        exported_model = torch.export.export(
            model,
            args=(sample,),
            strict=True,
        )

        with torch.inference_mode():
            eager_output = model(sample)
            exported_output = exported_model.module().forward(sample)

        self.assertEqual(eager_output.shape, exported_output.shape)
        torch.testing.assert_close(eager_output, exported_output, rtol=eps, atol=eps)

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
        torch.testing.assert_close(scripted_output, eager_output, rtol=1e-3, atol=1e-3)
