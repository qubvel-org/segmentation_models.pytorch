from segmentation_models_pytorch.base import (
    SegmentationModel,
)

from .decoder import Mask2FormerPixelModule, Mask2FormerTransformerModule


class Mask2Former(SegmentationModel):
    def __init__(self):
        super().__init__()
        pixel_module = Mask2FormerPixelModule()
        transformer_module = Mask2FormerTransformerModule()

    def forward(self, x):
        return x
