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


class TestMixTransformerEncoder(base.BaseEncoderTester):
    encoder_names = (
        ["mit_b0"]
        if not RUN_ALL_ENCODERS
        else ["mit_b0", "mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"]
    )
