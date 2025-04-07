import timm
import torch
import pytest

from segmentation_models_pytorch.encoders import TimmViTEncoder
from segmentation_models_pytorch.encoders.timm_vit import sample_block_indices_uniformly

from tests.encoders import base
from tests.utils import (
    default_device,
    check_run_test_on_diff_or_main,
    requires_torch_greater_or_equal,
    requires_timm_greater_or_equal,
)

timm_vit_encoders = ["vit_tiny_patch16_224"]


@requires_timm_greater_or_equal("1.0.0")
class TestTimmViTEncoders(base.BaseEncoderTester):
    encoder_names = timm_vit_encoders
    tiny_encoder_patch_size = 224
    default_height = 224
    default_width = 224

    files_for_diff = ["encoders/dpt.py"]

    num_output_features = 4
    default_depth = 4
    output_strides = None
    supports_dilated = False

    depth_to_test = [2, 3, 4]

    def get_tiny_encoder(self) -> TimmViTEncoder:
        return TimmViTEncoder(
            name=self.encoder_names[0],
            pretrained=False,
            depth=self.default_depth,
            in_channels=3,
        )

    def get_encoder(self, encoder_name: str, **kwargs) -> TimmViTEncoder:
        default_kwargs = {
            "name": encoder_name,
            "pretrained": False,
            "depth": self.default_depth,
            "in_channels": 3,
        }
        default_kwargs.update(kwargs)
        return TimmViTEncoder(**default_kwargs)

    def test_forward_backward(self):
        for encoder_name in self.encoder_names:
            sample = self._get_sample().to(default_device)
            with self.subTest(encoder_name=encoder_name):
                # init encoder
                encoder = self.get_encoder(encoder_name).to(default_device)

                # forward
                features, prefix_tokens = encoder.forward(sample)
                self.assertEqual(
                    len(features),
                    self.num_output_features,
                    f"Encoder `{encoder_name}` should have {self.num_output_features} output feature maps, but has {len(features)}",
                )
                if encoder.has_prefix_tokens:
                    self.assertEqual(
                        len(prefix_tokens),
                        self.num_output_features,
                        f"Encoder `{encoder_name}` should have {self.num_output_features} prefix tokens, but has {len(prefix_tokens)}",
                    )

                # backward
                features[-1].mean().backward()

    def test_in_channels(self):
        cases = [
            (encoder_name, in_channels)
            for encoder_name in self.encoder_names
            for in_channels in self.in_channels_to_test
        ]

        for encoder_name, in_channels in cases:
            sample = self._get_sample(num_channels=in_channels).to(default_device)

            with self.subTest(encoder_name=encoder_name, in_channels=in_channels):
                encoder = self.get_encoder(encoder_name, in_channels=in_channels).to(
                    default_device
                )
                encoder.eval()

                # forward
                with torch.inference_mode():
                    encoder.forward(sample)

    def test_depth(self):
        cases = [
            (encoder_name, depth)
            for encoder_name in self.encoder_names
            for depth in self.depth_to_test
        ]

        for encoder_name, depth in cases:
            sample = self._get_sample().to(default_device)
            with self.subTest(encoder_name=encoder_name, depth=depth):
                encoder = self.get_encoder(encoder_name, depth=depth).to(default_device)
                encoder.eval()

                # forward
                with torch.inference_mode():
                    features, _ = encoder.forward(sample)

                # check number of features
                self.assertEqual(
                    len(features),
                    depth,
                    f"Encoder `{encoder_name}` should have {depth} output feature maps, but has {len(features)}",
                )

                # check feature strides
                height_strides, width_strides = self.get_features_output_strides(
                    sample, features
                )

                encoder_out_indices = sample_block_indices_uniformly(depth, 12)
                feature_info = timm.create_model(model_name=encoder_name).feature_info
                output_strides = [
                    feature_info[i]["reduction"] for i in encoder_out_indices
                ]

                self.assertEqual(
                    height_strides,
                    output_strides,
                    f"Encoder `{encoder_name}` should have output strides {output_strides}, but has {height_strides}",
                )
                self.assertEqual(
                    width_strides,
                    output_strides,
                    f"Encoder `{encoder_name}` should have output strides {output_strides}, but has {width_strides}",
                )

                # check encoder output stride property
                self.assertEqual(
                    encoder.output_strides,
                    output_strides,
                    f"Encoder `{encoder_name}` last feature map should have output stride {output_strides[depth - 1]}, but has {encoder.output_stride}",
                )

                # check out channels also have proper length
                self.assertEqual(
                    len(encoder.out_channels),
                    depth,
                    f"Encoder `{encoder_name}` should have {depth} out_channels, but has {len(encoder.out_channels)}",
                )

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            self.get_encoder(self.encoder_names[0], depth=0)
        with self.assertRaises(ValueError):
            self.get_encoder(self.encoder_names[0], depth=25)

    def test_invalid_out_indices(self):
        # out of range
        with self.assertRaises(ValueError):
            self.get_encoder(self.encoder_names[0], depth=1, output_indices=-25)
        with self.assertRaises(ValueError):
            self.get_encoder(self.encoder_names[0], depth=3, output_indices=[1, 2, 25])

        # invalid length
        with self.assertRaises(ValueError):
            self.get_encoder(
                self.encoder_names[0],
                depth=2,
                output_indices=[
                    2,
                ],
            )

    def test_dilated(self):
        pytest.skip("Dilation is not supported for ViT encoders")

    @pytest.mark.compile
    def test_compile(self):
        if not check_run_test_on_diff_or_main(self.files_for_diff):
            self.skipTest("No diff and not on `main`.")

        encoder = self.get_tiny_encoder()
        encoder = encoder.eval().to(default_device)

        sample = self._get_sample(
            height=self.tiny_encoder_patch_size, width=self.tiny_encoder_patch_size
        ).to(default_device)

        torch.compiler.reset()
        compiled_encoder = torch.compile(
            encoder, fullgraph=True, dynamic=True, backend="eager"
        )

        if encoder._is_torch_compilable:
            compiled_encoder(sample)
        else:
            with self.assertRaises(Exception):
                compiled_encoder(sample)

    @pytest.mark.torch_export
    @requires_torch_greater_or_equal("2.4.0")
    def test_torch_export(self):
        if not check_run_test_on_diff_or_main(self.files_for_diff):
            self.skipTest("No diff and not on `main`.")

        sample = self._get_sample(
            height=self.tiny_encoder_patch_size, width=self.tiny_encoder_patch_size
        ).to(default_device)

        encoder = self.get_tiny_encoder()
        encoder = encoder.eval().to(default_device)

        exported_encoder = torch.export.export(
            encoder,
            args=(sample,),
            strict=True,
        )

        with torch.inference_mode():
            eager_output = encoder(sample)
            exported_output = exported_encoder.module().forward(sample)

        for eager_feature, exported_feature in zip(eager_output, exported_output):
            torch.testing.assert_close(eager_feature, exported_feature)

    @pytest.mark.torch_script
    def test_torch_script(self):
        pytest.skip(
            "Encoder with prefix tokens are not supported for scripting, due to poor type handling"
        )
