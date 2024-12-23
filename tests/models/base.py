import unittest
from functools import lru_cache

import torch
import segmentation_models_pytorch as smp


class BaseModelTester(unittest.TestCase):
    test_encoder_name = "tu-test_resnet.r160_in1k"

    # should be overriden
    test_model_type = None

    # test sample configuration
    default_batch_size = 1
    default_num_channels = 3
    default_height = 64
    default_width = 64

    @property
    def model_type(self):
        if self.test_model_type is None:
            raise ValueError("test_model_type is not set")
        return self.test_model_type

    @lru_cache
    def _get_sample(self, batch_size=1, num_channels=3, height=32, width=32):
        return torch.rand(batch_size, num_channels, height, width)

    def test_forward_backward(self):
        sample = self._get_sample(
            batch_size=self.default_batch_size,
            num_channels=self.default_num_channels,
            height=self.default_height,
            width=self.default_width,
        )
        model = smp.create_model(arch=self.model_type)

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
