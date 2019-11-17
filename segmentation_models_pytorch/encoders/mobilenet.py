import torchvision
import torch.nn as nn
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin


class MobileNetV2Encoder(torchvision.MobileNetV2, EncoderMixin):

    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.classifier

    def forward(self):
        stages = [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.1.bias")
        state_dict.pop("classifier.1.weight")
        super().load_state_dict(state_dict, **kwargs)


mobilenet_encoders = {
    "mobilenet_v2": {
        "encoder": MobileNetV2Encoder,
        "pretrained_settings": pretrained_settings["mobilenet_v2"],
        "params": {
            "out_channels": (3, 16, 24, 32, 96, 1280),
        },
    },
}