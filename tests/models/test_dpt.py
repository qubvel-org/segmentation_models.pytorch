import pytest
import inspect
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.dpt.decoder import (
    DPTDecoder,
    DPTSegmentationHead,
)
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


def test_patch14_encoder():
    model = smp.DPT(
        encoder_name="tu-eva02_tiny_patch14_224",
        encoder_weights=None,
        decoder_intermediate_channels=(16, 16, 16, 16),
        decoder_fusion_channels=16,
        classes=2,
    ).eval()

    with torch.inference_mode():
        masks = model(torch.randn(1, 3, 224, 224))

    assert model.encoder.output_strides == [14, 14, 14, 14]
    assert masks.shape == (1, 2, 224, 224)


def test_non_power_of_two_patch_stride_with_odd_feature_sizes():
    input_size = 518
    patch_size = 14
    feature_size = input_size // patch_size
    encoder_output_strides = (patch_size,) * 4

    decoder = DPTDecoder(
        encoder_out_channels=(8, 8, 8, 8),
        encoder_output_strides=encoder_output_strides,
        encoder_has_prefix_tokens=False,
        readout="ignore",
        intermediate_channels=(4, 4, 4, 4),
        fusion_channels=4,
    ).eval()
    head = DPTSegmentationHead(in_channels=4, out_channels=2).eval()
    features = [torch.randn(1, 8, feature_size, feature_size) for _ in range(4)]

    with torch.inference_mode():
        reassembled_features = [
            block(feature)
            for block, feature in zip(decoder.reassemble_blocks, features)
        ]
        decoder_output = decoder(features, [None] * 4)
        masks = head(decoder_output, output_size=(input_size, input_size))

    expected_sizes = [input_size // stride for stride in (4, 8, 16, 32)]
    assert [feature.shape[-1] for feature in reassembled_features] == expected_sizes
    assert masks.shape == (1, 2, input_size, input_size)
