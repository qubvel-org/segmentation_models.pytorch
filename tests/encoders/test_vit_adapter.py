from tests.encoders import base
from tests.utils import has_timm_test_models

class TestViTAdapterEncoder(base.BaseEncoderTester):
    encoder_names = ["tu-vit_base_patch16_224", "tu-vit_tiny_patch16_224", "tu-vit_large_patch16_224"]

    default_height = 224
    default_width = 224

    supports_dilated = False

    depth_to_test = [3, 4, 5]
    in_channels_to_test = [1, 3, 4]
