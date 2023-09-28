"""Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""
from functools import partial
import torch.nn as nn
from torchvision.models.efficientnet import EfficientNet, _efficientnet_conf
from torchvision.models.efficientnet import (
    EfficientNet_V2_S_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_L_Weights,
    WeightsEnum,
)
from ._base import EncoderMixin


class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5, **kwargs):
        if depth > len(stage_idxs) - 1:
            raise Exception(f"depth value over {len(stage_idxs)} not allowed for encoder '{model_name}'")
        inverted_residual_setting, last_channel = _efficientnet_conf(model_name)
        super().__init__(inverted_residual_setting, last_channel=last_channel, **kwargs)

        self._stage_idxs = stage_idxs[: depth + 1]
        self._out_channels = [out_channels[c_idx] for c_idx in self._stage_idxs]
        self._depth = depth
        self._in_channels = 3

        del self.features[self._stage_idxs[-1] :]
        del self.classifier

    def get_stages(self):

        stages = [nn.Identity()] + [
            self.features[self._stage_idxs[c_stage_idx] : self._stage_idxs[c_stage_idx + 1]]
            for c_stage_idx in range(len(self._stage_idxs) - 1)
        ]
        return stages

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for c_stage in stages:
            x = c_stage(x)
            features.append(x)
        return features

    def load_state_dict(self, state_dict, **kwargs):
        keys_to_delete = ["classifier.1.weight", "classifier.1.bias"]
        for k in state_dict.keys():
            feature_layer_idx = int(k.split(".")[1])
            if feature_layer_idx >= self._stage_idxs[-1]:
                keys_to_delete.append(k)
        for key_td in keys_to_delete:
            state_dict.pop(key_td, None)
        super().load_state_dict(state_dict, **kwargs)


def _get_pretrained_settings(encoder):
    weights_map: WeightsEnum = {
        "efficientnet_v2_s": EfficientNet_V2_S_Weights,
        "efficientnet_v2_m": EfficientNet_V2_M_Weights,
        "efficientnet_v2_l": EfficientNet_V2_L_Weights,
    }
    transform_values_map = {
        "efficientnet_v2_s": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "efficientnet_v2_m": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "efficientnet_v2_l": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    }
    pretrained_settings = {
        "IMAGENET1K_V1": {
            "mean": transform_values_map[encoder.lower()]["mean"],
            "std": transform_values_map[encoder.lower()]["std"],
            "url": weights_map[encoder.lower()].IMAGENET1K_V1.url,
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }
    return pretrained_settings


efficientv2_net_encoders = {
    "efficientnet_v2_s": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet_v2_s"),
        "params": {
            "out_channels": (3, 24, 24, 48, 64, 128, 160, 256),
            "stage_idxs": (0, 2, 3, 4, 5, 7),
            "model_name": "efficientnet_v2_s",
            "dropout": 0.2,
            "stochastic_depth_prob": 0.2,
            "norm_layer": partial(nn.BatchNorm2d, eps=1e-03),
        },
    },
    "efficientnet_v2_m": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet_v2_m"),
        "params": {
            "out_channels": (3, 24, 24, 48, 80, 160, 176, 304, 512),
            "stage_idxs": (0, 2, 3, 4, 5, 8),
            "model_name": "efficientnet_v2_m",
            "dropout": 0.2,
            "stochastic_depth_prob": 0.2,
            "norm_layer": partial(nn.BatchNorm2d, eps=1e-03),
        },
    },
    "efficientnet_v2_l": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": _get_pretrained_settings("efficientnet_v2_l"),
        "params": {
            "out_channels": (3, 32, 32, 64, 96, 192, 224, 384, 640),
            "stage_idxs": (0, 2, 3, 4, 5, 8),
            "model_name": "efficientnet_v2_l",
            "dropout": 0.2,
            "stochastic_depth_prob": 0.2,
            "norm_layer": partial(nn.BatchNorm2d, eps=1e-03),
        },
    },
}
