import torch.nn as nn
from pretrainedmodels.models.inceptionresnetv2 import (
    InceptionResNetV2,
    pretrained_settings,
)

from ._base import EncoderMixin


class InceptionResNetV2Encoder(InceptionResNetV2, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # remove linear layers
        del self.avgpool_1a
        del self.last_linear

    def make_dilated(self, *args, **kwargs):
        raise ValueError(
            "InceptionResnetV2 encoder does not support dilated mode "
            "due to pooling operation for downsampling!"
        )

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv2d_1a, self.conv2d_2a, self.conv2d_2b),
            nn.Sequential(self.maxpool_3a, self.conv2d_3b, self.conv2d_4a),
            nn.Sequential(self.maxpool_5a, self.mixed_5b, self.repeat),
            nn.Sequential(self.mixed_6a, self.repeat_1),
            nn.Sequential(self.mixed_7a, self.repeat_2, self.block8, self.conv2d_7b),
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("last_linear.bias", None)
        state_dict.pop("last_linear.weight", None)
        super().load_state_dict(state_dict, **kwargs)


inceptionresnetv2_encoders = {
    "inceptionresnetv2": {
        "encoder": InceptionResNetV2Encoder,
        "pretrained_settings": pretrained_settings["inceptionresnetv2"],
        "params": {"out_channels": (3, 64, 192, 320, 1088, 1536), "num_classes": 1000},
    }
}
