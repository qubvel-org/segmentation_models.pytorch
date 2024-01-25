import torch.nn as nn
from pretrainedmodels.models.inceptionv4 import InceptionV4, pretrained_settings

from ._base import EncoderMixin


class InceptionV4Encoder(InceptionV4, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._stage_idxs = stage_idxs
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
        del self.last_linear

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError(
            "InceptionV4 encoder does not support dilated mode "
            "due to pooling operation for downsampling!"
        )

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[: self._stage_idxs[0]],
            self.features[self._stage_idxs[0] : self._stage_idxs[1]],
            self.features[self._stage_idxs[1] : self._stage_idxs[2]],
            self.features[self._stage_idxs[2] : self._stage_idxs[3]],
            self.features[self._stage_idxs[3] :],
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


inceptionv4_encoders = {
    "inceptionv4": {
        "encoder": InceptionV4Encoder,
        "pretrained_settings": pretrained_settings["inceptionv4"],
        "params": {
            "stage_idxs": (3, 5, 9, 15),
            "out_channels": (3, 64, 192, 384, 1024, 1536),
            "num_classes": 1001,
        },
    }
}
