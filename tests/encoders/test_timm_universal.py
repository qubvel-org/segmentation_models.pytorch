from tests.encoders import base
from tests.config import RUN_ALL_ENCODERS


class TestTimmUniversalEncoder(base.BaseEncoderTester):
    encoder_names = ["tu-resnet18"] if not RUN_ALL_ENCODERS else ["tu-resnet18"]
