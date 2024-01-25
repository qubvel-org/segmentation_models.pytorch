import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrainedmodels.models.dpn import DPN, pretrained_settings

from ._base import EncoderMixin


class DPNEncoder(DPN, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._stage_idxs = stage_idxs
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.last_linear

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(
                self.features[0].conv, self.features[0].bn, self.features[0].act
            ),
            nn.Sequential(
                self.features[0].pool, self.features[1 : self._stage_idxs[0]]
            ),
            self.features[self._stage_idxs[0] : self._stage_idxs[1]],
            self.features[self._stage_idxs[1] : self._stage_idxs[2]],
            self.features[self._stage_idxs[2] : self._stage_idxs[3]],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            if isinstance(x, (list, tuple)):
                features.append(F.relu(torch.cat(x, dim=1), inplace=True))
            else:
                features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("last_linear.bias", None)
        state_dict.pop("last_linear.weight", None)
        super().load_state_dict(state_dict, **kwargs)


dpn_encoders = {
    "dpn68": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn68"],
        "params": {
            "stage_idxs": (4, 8, 20, 24),
            "out_channels": (3, 10, 144, 320, 704, 832),
            "groups": 32,
            "inc_sec": (16, 32, 32, 64),
            "k_r": 128,
            "k_sec": (3, 4, 12, 3),
            "num_classes": 1000,
            "num_init_features": 10,
            "small": True,
            "test_time_pool": True,
        },
    },
    "dpn68b": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn68b"],
        "params": {
            "stage_idxs": (4, 8, 20, 24),
            "out_channels": (3, 10, 144, 320, 704, 832),
            "b": True,
            "groups": 32,
            "inc_sec": (16, 32, 32, 64),
            "k_r": 128,
            "k_sec": (3, 4, 12, 3),
            "num_classes": 1000,
            "num_init_features": 10,
            "small": True,
            "test_time_pool": True,
        },
    },
    "dpn92": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn92"],
        "params": {
            "stage_idxs": (4, 8, 28, 32),
            "out_channels": (3, 64, 336, 704, 1552, 2688),
            "groups": 32,
            "inc_sec": (16, 32, 24, 128),
            "k_r": 96,
            "k_sec": (3, 4, 20, 3),
            "num_classes": 1000,
            "num_init_features": 64,
            "test_time_pool": True,
        },
    },
    "dpn98": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn98"],
        "params": {
            "stage_idxs": (4, 10, 30, 34),
            "out_channels": (3, 96, 336, 768, 1728, 2688),
            "groups": 40,
            "inc_sec": (16, 32, 32, 128),
            "k_r": 160,
            "k_sec": (3, 6, 20, 3),
            "num_classes": 1000,
            "num_init_features": 96,
            "test_time_pool": True,
        },
    },
    "dpn107": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn107"],
        "params": {
            "stage_idxs": (5, 13, 33, 37),
            "out_channels": (3, 128, 376, 1152, 2432, 2688),
            "groups": 50,
            "inc_sec": (20, 64, 64, 128),
            "k_r": 200,
            "k_sec": (4, 8, 20, 3),
            "num_classes": 1000,
            "num_init_features": 128,
            "test_time_pool": True,
        },
    },
    "dpn131": {
        "encoder": DPNEncoder,
        "pretrained_settings": pretrained_settings["dpn131"],
        "params": {
            "stage_idxs": (5, 13, 41, 45),
            "out_channels": (3, 128, 352, 832, 1984, 2688),
            "groups": 40,
            "inc_sec": (16, 32, 32, 128),
            "k_r": 160,
            "k_sec": (4, 8, 28, 3),
            "num_classes": 1000,
            "num_init_features": 128,
            "test_time_pool": True,
        },
    },
}
