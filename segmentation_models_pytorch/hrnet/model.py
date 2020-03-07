from ..base import SegmentationModel
from .decoder import HighResolutionNet
from .config import hrnet_config
from typing import Optional


BN_MOMENTUM = 0.01


class HRNet(SegmentationModel):
    def __init__(
            self,
            pretrained_weights: Optional[str] = '',
            num_classes: int = 1,
    ):
        super().__init__()
        self.decoder = HighResolutionNet(hrnet_config, num_classes)


