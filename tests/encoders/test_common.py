import pytest
import segmentation_models_pytorch as smp
from tests.utils import slow_test


@pytest.mark.parametrize(
    "encoder_name_and_weights",
    [
        ("resnet18", "imagenet"),
    ],
)
@slow_test
def test_load_encoder_from_hub(encoder_name_and_weights):
    encoder_name, weights = encoder_name_and_weights
    smp.encoders.get_encoder(encoder_name, weights=weights)
