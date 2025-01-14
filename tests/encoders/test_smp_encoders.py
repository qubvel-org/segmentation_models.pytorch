import segmentation_models_pytorch as smp
from functools import partial

from tests.encoders import base
from tests.utils import RUN_ALL_ENCODERS


class TestMobileoneEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["mobileone_s0"]
        if not RUN_ALL_ENCODERS
        else [
            "mobileone_s0",
            "mobileone_s1",
            "mobileone_s2",
            "mobileone_s3",
            "mobileone_s4",
        ]
    )
    files_for_diff = ["encoders/mobileone.py"]


class TestMixTransformerEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["mit_b0"]
        if not RUN_ALL_ENCODERS
        else ["mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"]
    )
    files_for_diff = ["encoders/mix_transformer.py"]

    def get_tiny_encoder(self):
        params = {
            "out_channels": [3, 0, 4, 4, 4, 4],
            "patch_size": 4,
            "embed_dims": [4, 4, 4, 4],
            "num_heads": [1, 1, 1, 1],
            "mlp_ratios": [1, 1, 1, 1],
            "qkv_bias": True,
            "norm_layer": partial(smp.encoders.mix_transformer.LayerNorm, eps=1e-6),
            "depths": [1, 1, 1, 1],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        }

        return smp.encoders.mix_transformer.MixVisionTransformerEncoder(**params)


class TestEfficientNetEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["efficientnet-b0"]
        if not RUN_ALL_ENCODERS
        else [
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
            "efficientnet-b5",
            "efficientnet-b6",
            # "efficientnet-b7",  # extra large model
        ]
    )
    files_for_diff = ["encoders/efficientnet.py"]
