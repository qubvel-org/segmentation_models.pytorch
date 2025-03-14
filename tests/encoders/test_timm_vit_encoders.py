from tests.encoders import base
import timm
import torch
import segmentation_models_pytorch as smp
import pytest

from tests.utils import (
    default_device,
    check_run_test_on_diff_or_main,
    requires_torch_greater_or_equal,
    requires_timm_greater_or_equal,
)

timm_vit_encoders = [
    "tu-vit_tiny_patch16_224",
    "tu-vit_small_patch32_224",
    "tu-vit_base_patch32_384",
    "tu-vit_base_patch16_gap_224",
    "tu-vit_medium_patch16_reg4_gap_256",
    "tu-vit_so150m2_patch16_reg1_gap_256",
    "tu-vit_medium_patch16_gap_240",
]


class TestTimmViTEncoders(base.BaseEncoderTester):
    encoder_names = timm_vit_encoders
    tiny_encoder_patch_size = 224

    files_for_diff = ["encoders/dpt.py"]

    num_output_features = 4
    default_depth = 4
    output_strides = None
    supports_dilated = False

    depth_to_test = [2, 3, 4]

    default_encoder_kwargs = {"use_vit_encoder": True}

    def _get_model_expected_input_shape(self, encoder_name: str) -> int:
        patch_size_str = encoder_name[-3:]
        return int(patch_size_str)

    def get_tiny_encoder(self):
        return smp.encoders.get_encoder(
            self.encoder_names[0],
            encoder_weights=None,
            output_stride=None,
            depth=self.default_depth,
            **self.default_encoder_kwargs,
        )

    @requires_timm_greater_or_equal("1.0.15")
    def test_forward_backward(self):
        for encoder_name in self.encoder_names:
            patch_size = self._get_model_expected_input_shape(encoder_name)
            sample = self._get_sample(height=patch_size, width=patch_size).to(
                default_device
            )
            with self.subTest(encoder_name=encoder_name):
                # init encoder
                encoder = smp.encoders.get_encoder(
                    encoder_name,
                    in_channels=3,
                    encoder_weights=None,
                    depth=self.default_depth,
                    output_stride=None,
                    **self.default_encoder_kwargs,
                ).to(default_device)

                # forward
                features, cls_tokens = encoder.forward(sample)
                self.assertEqual(
                    len(features),
                    self.num_output_features,
                    f"Encoder `{encoder_name}` should have {self.num_output_features} output feature maps, but has {len(features)}",
                )

                # backward
                features[-1].mean().backward()

    @requires_timm_greater_or_equal("1.0.15")
    def test_in_channels(self):
        cases = [
            (encoder_name, in_channels)
            for encoder_name in self.encoder_names
            for in_channels in self.in_channels_to_test
        ]

        for encoder_name, in_channels in cases:
            patch_size = self._get_model_expected_input_shape(encoder_name)
            sample = self._get_sample(
                height=patch_size, width=patch_size, num_channels=in_channels
            ).to(default_device)

            with self.subTest(encoder_name=encoder_name, in_channels=in_channels):
                encoder = smp.encoders.get_encoder(
                    encoder_name,
                    in_channels=in_channels,
                    encoder_weights=None,
                    depth=4,
                    output_stride=None,
                    **self.default_encoder_kwargs,
                ).to(default_device)
                encoder.eval()

                # forward
                with torch.inference_mode():
                    encoder.forward(sample)

    @requires_timm_greater_or_equal("1.0.15")
    def test_depth(self):
        cases = [
            (encoder_name, depth)
            for encoder_name in self.encoder_names
            for depth in self.depth_to_test
        ]

        for encoder_name, depth in cases:
            patch_size = self._get_model_expected_input_shape(encoder_name)
            sample = self._get_sample(height=patch_size, width=patch_size).to(
                default_device
            )
            with self.subTest(encoder_name=encoder_name, depth=depth):
                encoder = smp.encoders.get_encoder(
                    encoder_name,
                    in_channels=self.default_num_channels,
                    encoder_weights=None,
                    depth=depth,
                    output_stride=None,
                    **self.default_encoder_kwargs,
                ).to(default_device)
                encoder.eval()

                # forward
                with torch.inference_mode():
                    features, cls_tokens = encoder.forward(sample)

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

                timm_encoder_name = encoder_name[3:]
                encoder_out_indices = encoder.out_indices
                timm_model_feature_info = timm.create_model(
                    model_name=timm_encoder_name
                ).feature_info
                feature_info_obj = timm.models.FeatureInfo(
                    feature_info=timm_model_feature_info,
                    out_indices=encoder_out_indices,
                )
                self.output_strides = feature_info_obj.reduction()

                self.assertEqual(
                    height_strides,
                    self.output_strides[:depth],
                    f"Encoder `{encoder_name}` should have output strides {self.output_strides[:depth]}, but has {height_strides}",
                )
                self.assertEqual(
                    width_strides,
                    self.output_strides[:depth],
                    f"Encoder `{encoder_name}` should have output strides {self.output_strides[:depth]}, but has {width_strides}",
                )

                # check encoder output stride property
                self.assertEqual(
                    encoder.output_stride,
                    self.output_strides[depth - 1],
                    f"Encoder `{encoder_name}` last feature map should have output stride {self.output_strides[depth - 1]}, but has {encoder.output_stride}",
                )

                # check out channels also have proper length
                self.assertEqual(
                    len(encoder.out_channels),
                    depth,
                    f"Encoder `{encoder_name}` should have {depth} out_channels, but has {len(encoder.out_channels)}",
                )

    @requires_timm_greater_or_equal("1.0.15")
    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            smp.encoders.get_encoder(self.encoder_names[0], depth=5, output_stride=None)
        with self.assertRaises(ValueError):
            smp.encoders.get_encoder(self.encoder_names[0], depth=0, output_stride=None)

    def test_invalid_out_indices(self):
        with self.assertRaises(ValueError):
            smp.encoders.get_encoder(
                self.encoder_names[0], output_stride=None, out_indices=-1
            )

        with self.assertRaises(ValueError):
            smp.encoders.get_encoder(
                self.encoder_names[0], output_stride=None, out_indices=[1, 2, 25]
            )

    def test_invalid_out_indices_length(self):
        with self.assertRaises(ValueError):
            smp.encoders.get_encoder(
                self.encoder_names[0], output_stride=None, out_indices=2, depth=2
            )

        with self.assertRaises(ValueError):
            smp.encoders.get_encoder(
                self.encoder_names[0],
                output_stride=None,
                out_indices=[0, 1, 2, 3, 4],
                depth=4,
            )

    @requires_timm_greater_or_equal("1.0.15")
    def test_dilated(self):
        cases = [
            (encoder_name, stride)
            for encoder_name in self.encoder_names
            for stride in self.strides_to_test
        ]

        # special case for encoders that do not support dilated model
        # just check proper error is raised
        if not self.supports_dilated:
            with self.assertRaises(
                ValueError, msg="Dilated mode not supported, set output stride to None"
            ):
                encoder_name, stride = cases[0]
                patch_size = self._get_model_expected_input_shape(encoder_name)
                sample = self._get_sample(height=patch_size, width=patch_size).to(
                    default_device
                )
                encoder = smp.encoders.get_encoder(
                    encoder_name,
                    in_channels=self.default_num_channels,
                    encoder_weights=None,
                    output_stride=stride,
                    depth=self.default_depth,
                    **self.default_encoder_kwargs,
                ).to(default_device)
            return

        for encoder_name, stride in cases:
            with self.subTest(encoder_name=encoder_name, stride=stride):
                encoder = smp.encoders.get_encoder(
                    encoder_name,
                    in_channels=self.default_num_channels,
                    encoder_weights=None,
                    output_stride=stride,
                    depth=self.default_depth,
                    **self.default_encoder_kwargs,
                ).to(default_device)
                encoder.eval()

                # forward
                with torch.inference_mode():
                    features, cls_tokens = encoder.forward(sample)

                height_strides, width_strides = self.get_features_output_strides(
                    encoder, sample, features
                )
                expected_height_strides = [min(stride, s) for s in height_strides]
                expected_width_strides = [min(stride, s) for s in width_strides]

                self.assertEqual(
                    height_strides,
                    expected_height_strides,
                    f"Encoder `{encoder_name}` should have height output strides {expected_height_strides}, but has {height_strides}",
                )
                self.assertEqual(
                    width_strides,
                    expected_width_strides,
                    f"Encoder `{encoder_name}` should have width output strides {expected_width_strides}, but has {width_strides}",
                )

    @requires_timm_greater_or_equal("1.0.15")
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

    @requires_timm_greater_or_equal("1.0.15")
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

        if not encoder._is_torch_exportable:
            with self.assertRaises(Exception):
                exported_encoder = torch.export.export(
                    encoder,
                    args=(sample,),
                    strict=True,
                )
            return

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

    @requires_timm_greater_or_equal("1.0.15")
    @pytest.mark.torch_script
    def test_torch_script(self):
        sample = self._get_sample(
            height=self.tiny_encoder_patch_size, width=self.tiny_encoder_patch_size
        ).to(default_device)

        encoder = self.get_tiny_encoder()
        encoder = encoder.eval().to(default_device)

        if not encoder._is_torch_scriptable:
            with self.assertRaises(RuntimeError, msg="not torch scriptable"):
                scripted_encoder = torch.jit.script(encoder)
            return

        scripted_encoder = torch.jit.script(encoder)

        with torch.inference_mode():
            eager_output = encoder(sample)
            scripted_output = scripted_encoder(sample)

        for eager_feature, scripted_feature in zip(eager_output, scripted_output):
            torch.testing.assert_close(eager_feature, scripted_feature)
