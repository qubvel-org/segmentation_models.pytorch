import pytest
import unittest
import torch
import segmentation_models_pytorch as smp

from functools import lru_cache
from tests.utils import default_device, check_run_test_on_diff_or_main


class BaseEncoderTester(unittest.TestCase):
    encoder_names = []

    # some tests might be slow, running them only on diff
    files_for_diff = []

    # standard encoder configuration
    num_output_features = 6
    output_strides = [1, 2, 4, 8, 16, 32]
    supports_dilated = True

    # test sample configuration
    default_batch_size = 1
    default_num_channels = 3
    default_height = 64
    default_width = 64

    # test configurations
    in_channels_to_test = [1, 3, 4]
    depth_to_test = [3, 4, 5]
    strides_to_test = [8, 16]  # 32 is a default one

    def get_tiny_encoder(self):
        return smp.encoders.get_encoder(self.encoder_names[0], encoder_weights=None)

    @lru_cache
    def _get_sample(self, batch_size=1, num_channels=3, height=32, width=32):
        return torch.rand(batch_size, num_channels, height, width)

    def get_features_output_strides(self, sample, features):
        height, width = sample.shape[2:]
        height_strides = [height // f.shape[2] for f in features]
        width_strides = [width // f.shape[3] for f in features]
        return height_strides, width_strides

    def test_forward_backward(self):
        sample = self._get_sample(
            batch_size=self.default_batch_size,
            num_channels=self.default_num_channels,
            height=self.default_height,
            width=self.default_width,
        ).to(default_device)
        for encoder_name in self.encoder_names:
            with self.subTest(encoder_name=encoder_name):
                # init encoder
                encoder = smp.encoders.get_encoder(
                    encoder_name, in_channels=3, encoder_weights=None
                ).to(default_device)

                # forward
                features = encoder.forward(sample)
                self.assertEqual(
                    len(features),
                    self.num_output_features,
                    f"Encoder `{encoder_name}` should have {self.num_output_features} output feature maps, but has {len(features)}",
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
            sample = self._get_sample(
                batch_size=self.default_batch_size,
                num_channels=in_channels,
                height=self.default_height,
                width=self.default_width,
            ).to(default_device)

            with self.subTest(encoder_name=encoder_name, in_channels=in_channels):
                encoder = smp.encoders.get_encoder(
                    encoder_name, in_channels=in_channels, encoder_weights=None
                ).to(default_device)
                encoder.eval()

                # forward
                with torch.inference_mode():
                    encoder.forward(sample)

    def test_depth(self):
        sample = self._get_sample(
            batch_size=self.default_batch_size,
            num_channels=self.default_num_channels,
            height=self.default_height,
            width=self.default_width,
        ).to(default_device)

        cases = [
            (encoder_name, depth)
            for encoder_name in self.encoder_names
            for depth in self.depth_to_test
        ]

        for encoder_name, depth in cases:
            with self.subTest(encoder_name=encoder_name, depth=depth):
                encoder = smp.encoders.get_encoder(
                    encoder_name,
                    in_channels=self.default_num_channels,
                    encoder_weights=None,
                    depth=depth,
                ).to(default_device)
                encoder.eval()

                # forward
                with torch.inference_mode():
                    features = encoder.forward(sample)

                # check number of features
                self.assertEqual(
                    len(features),
                    depth + 1,
                    f"Encoder `{encoder_name}` should have {depth + 1} output feature maps, but has {len(features)}",
                )

                # check feature strides
                height_strides, width_strides = self.get_features_output_strides(
                    sample, features
                )
                self.assertEqual(
                    height_strides,
                    self.output_strides[: depth + 1],
                    f"Encoder `{encoder_name}` should have output strides {self.output_strides[: depth + 1]}, but has {height_strides}",
                )
                self.assertEqual(
                    width_strides,
                    self.output_strides[: depth + 1],
                    f"Encoder `{encoder_name}` should have output strides {self.output_strides[: depth + 1]}, but has {width_strides}",
                )

                # check encoder output stride property
                self.assertEqual(
                    encoder.output_stride,
                    self.output_strides[depth],
                    f"Encoder `{encoder_name}` last feature map should have output stride {self.output_strides[depth]}, but has {encoder.output_stride}",
                )

                # check out channels also have proper length
                self.assertEqual(
                    len(encoder.out_channels),
                    depth + 1,
                    f"Encoder `{encoder_name}` should have {depth + 1} out_channels, but has {len(encoder.out_channels)}",
                )

    def test_dilated(self):
        sample = self._get_sample(
            batch_size=self.default_batch_size,
            num_channels=self.default_num_channels,
            height=self.default_height,
            width=self.default_width,
        ).to(default_device)

        cases = [
            (encoder_name, stride)
            for encoder_name in self.encoder_names
            for stride in self.strides_to_test
        ]

        # special case for encoders that do not support dilated model
        # just check proper error is raised
        if not self.supports_dilated:
            with self.assertRaises(ValueError, msg="not support dilated mode"):
                encoder_name, stride = cases[0]
                encoder = smp.encoders.get_encoder(
                    encoder_name,
                    in_channels=self.default_num_channels,
                    encoder_weights=None,
                    output_stride=stride,
                ).to(default_device)
            return

        for encoder_name, stride in cases:
            with self.subTest(encoder_name=encoder_name, stride=stride):
                encoder = smp.encoders.get_encoder(
                    encoder_name,
                    in_channels=self.default_num_channels,
                    encoder_weights=None,
                    output_stride=stride,
                ).to(default_device)
                encoder.eval()

                # forward
                with torch.inference_mode():
                    features = encoder.forward(sample)

                height_strides, width_strides = self.get_features_output_strides(
                    sample, features
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

    @pytest.mark.compile
    def test_compile(self):
        if not check_run_test_on_diff_or_main(self.files_for_diff):
            self.skipTest("No diff and not on `main`.")

        sample = self._get_sample(
            batch_size=self.default_batch_size,
            num_channels=self.default_num_channels,
            height=self.default_height,
            width=self.default_width,
        ).to(default_device)

        encoder = self.get_tiny_encoder().eval().to(default_device)
        compiled_encoder = torch.compile(encoder, fullgraph=True, dynamic=True)

        with torch.inference_mode():
            compiled_encoder(sample)
