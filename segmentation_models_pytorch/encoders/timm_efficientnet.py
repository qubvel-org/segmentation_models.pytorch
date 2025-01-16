import torch
import torch.nn as nn

from typing import List, Dict, Sequence
from functools import partial

from timm.models.efficientnet import EfficientNet
from timm.models.efficientnet import decode_arch_def, round_channels
from timm.layers.activations import Swish

from ._base import EncoderMixin


def get_efficientnet_kwargs(
    channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2
):
    """Create EfficientNet model.
    Ref impl: https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py
    Paper: https://arxiv.org/abs/1905.11946
    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    'efficientnet-b8': (2.2, 3.6, 672, 0.5),
    'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16_se0.25"],
        ["ir_r2_k3_s2_e6_c24_se0.25"],
        ["ir_r2_k5_s2_e6_c40_se0.25"],
        ["ir_r3_k3_s2_e6_c80_se0.25"],
        ["ir_r3_k5_s1_e6_c112_se0.25"],
        ["ir_r4_k5_s2_e6_c192_se0.25"],
        ["ir_r1_k3_s1_e6_c320_se0.25"],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        stem_size=32,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=Swish,
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )
    return model_kwargs


def gen_efficientnet_lite_kwargs(
    channel_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2
):
    """EfficientNet-Lite model.

    Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
    Paper: https://arxiv.org/abs/1905.11946

    EfficientNet params
    name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
      'efficientnet-lite0': (1.0, 1.0, 224, 0.2),
      'efficientnet-lite1': (1.0, 1.1, 240, 0.2),
      'efficientnet-lite2': (1.1, 1.2, 260, 0.3),
      'efficientnet-lite3': (1.2, 1.4, 280, 0.3),
      'efficientnet-lite4': (1.4, 1.8, 300, 0.3),

    Args:
      channel_multiplier: multiplier to number of channels per layer
      depth_multiplier: multiplier to number of repeats per stage
    """
    arch_def = [
        ["ds_r1_k3_s1_e1_c16"],
        ["ir_r2_k3_s2_e6_c24"],
        ["ir_r2_k5_s2_e6_c40"],
        ["ir_r3_k3_s2_e6_c80"],
        ["ir_r3_k5_s1_e6_c112"],
        ["ir_r4_k5_s2_e6_c192"],
        ["ir_r1_k3_s1_e6_c320"],
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier, fix_first_last=True),
        num_features=1280,
        stem_size=32,
        fix_stem=True,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        act_layer=nn.ReLU6,
        drop_rate=drop_rate,
        drop_path_rate=0.2,
    )
    return model_kwargs


class EfficientNetBaseEncoder(EfficientNet, EncoderMixin):
    def __init__(
        self,
        stage_idxs: List[int],
        out_channels: List[int],
        depth: int = 5,
        output_stride: int = 32,
        **kwargs,
    ):
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )
        super().__init__(**kwargs)

        self._stage_idxs = stage_idxs
        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

        del self.classifier

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        return {
            16: [self.blocks[self._stage_idxs[1] : self._stage_idxs[2]]],
            32: [self.blocks[self._stage_idxs[2] :]],
        }

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]

        if self._depth >= 1:
            x = self.conv_stem(x)
            x = self.bn1(x)
            features.append(x)

        if self._depth >= 2:
            x = self.blocks[0](x)
            x = self.blocks[1](x)
            features.append(x)

        if self._depth >= 3:
            x = self.blocks[2](x)
            features.append(x)

        if self._depth >= 4:
            x = self.blocks[3](x)
            x = self.blocks[4](x)
            features.append(x)

        if self._depth >= 5:
            x = self.blocks[5](x)
            x = self.blocks[6](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.bias", None)
        state_dict.pop("classifier.weight", None)
        super().load_state_dict(state_dict, **kwargs)


class EfficientNetEncoder(EfficientNetBaseEncoder):
    def __init__(
        self,
        stage_idxs: List[int],
        out_channels: List[int],
        depth: int = 5,
        channel_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        drop_rate: float = 0.2,
        output_stride: int = 32,
    ):
        kwargs = get_efficientnet_kwargs(
            channel_multiplier, depth_multiplier, drop_rate
        )
        super().__init__(
            stage_idxs=stage_idxs,
            depth=depth,
            out_channels=out_channels,
            output_stride=output_stride,
            **kwargs,
        )


class EfficientNetLiteEncoder(EfficientNetBaseEncoder):
    def __init__(
        self,
        stage_idxs: List[int],
        out_channels: List[int],
        depth: int = 5,
        channel_multiplier: float = 1.0,
        depth_multiplier: float = 1.0,
        drop_rate: float = 0.2,
        output_stride: int = 32,
    ):
        kwargs = gen_efficientnet_lite_kwargs(
            channel_multiplier, depth_multiplier, drop_rate
        )
        super().__init__(
            stage_idxs=stage_idxs,
            depth=depth,
            out_channels=out_channels,
            output_stride=output_stride,
            **kwargs,
        )


def prepare_settings(settings):
    return {
        "mean": settings.mean,
        "std": settings.std,
        "url": settings.url,
        "input_range": (0, 1),
        "input_space": "RGB",
    }


timm_efficientnet_encoders = {
    "timm-efficientnet-b0": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b0.imagenet",
                "revision": "8419e9cc19da0b68dcd7bb12f19b7c92407ad7c4",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b0.advprop",
                "revision": "a5870af2d24ce79e0cc7fae2bbd8e0a21fcfa6d8",
            },
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-b0.noisy-student",
                "revision": "bea8b0ff726a50e48774d2d360c5fb1ac4815836",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 40, 112, 320],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "drop_rate": 0.2,
        },
    },
    "timm-efficientnet-b1": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b1.imagenet",
                "revision": "63bdd65ef6596ef24f1cadc7dd4f46b624442349",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b1.advprop",
                "revision": "79b3d102080ef679b16c2748e608a871112233d0",
            },
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-b1.noisy-student",
                "revision": "36856124a699f6032574ceeefab02040daa90a9a",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 40, 112, 320],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.1,
            "drop_rate": 0.2,
        },
    },
    "timm-efficientnet-b2": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b2.imagenet",
                "revision": "e693adb39d3cb3847e71e3700a0c2aa58072cff1",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b2.advprop",
                "revision": "b58479bf78007cfbb365091d64eeee369bddfa21",
            },
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-b2.noisy-student",
                "revision": "67c558827c6d3e0975ff9b4bce8557bc2ca80931",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 48, 120, 352],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.1,
            "depth_multiplier": 1.2,
            "drop_rate": 0.3,
        },
    },
    "timm-efficientnet-b3": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b3.imagenet",
                "revision": "1666b835b5151d6bb2067c7cd67e67ada6c39edf",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b3.advprop",
                "revision": "70474cdb9f1ff4fcbd7434e66560ead1ab8e506b",
            },
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-b3.noisy-student",
                "revision": "2367bc9f61e79ee97684169a71a87db280bcf4db",
            },
        },
        "params": {
            "out_channels": [3, 40, 32, 48, 136, 384],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.2,
            "depth_multiplier": 1.4,
            "drop_rate": 0.3,
        },
    },
    "timm-efficientnet-b4": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b4.imagenet",
                "revision": "07868c28ab308f4de4cf1e7ec54b33b8b002ccdb",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b4.advprop",
                "revision": "8ea1772ee9a2a0d18c1b56dce0dfac8dd33d537d",
            },
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-b4.noisy-student",
                "revision": "faeb77b6e8292a700380c840d39442d7ce4d6443",
            },
        },
        "params": {
            "out_channels": [3, 48, 32, 56, 160, 448],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.4,
            "depth_multiplier": 1.8,
            "drop_rate": 0.4,
        },
    },
    "timm-efficientnet-b5": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b5.imagenet",
                "revision": "004153b4ddd93d30afd9bbf34329d7f57396d413",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b5.advprop",
                "revision": "1d1c5f05aab5ed9a1d5052847ddd4024c06a464d",
            },
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-b5.noisy-student",
                "revision": "9bc3a1e5490de92b1af061d5c2c474ab3129e38c",
            },
        },
        "params": {
            "out_channels": [3, 48, 40, 64, 176, 512],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.6,
            "depth_multiplier": 2.2,
            "drop_rate": 0.4,
        },
    },
    "timm-efficientnet-b6": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b6.imagenet",
                "revision": "dbbf28a5c33f021486db4070de693caad6b56c3d",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b6.advprop",
                "revision": "3b5d3412047f7711c56ffde997911cfefe79f835",
            },
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-b6.noisy-student",
                "revision": "9b899ea9e8e0ce2ccada0f34a8cb8b5028e9bb36",
            },
        },
        "params": {
            "out_channels": [3, 56, 40, 72, 200, 576],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.8,
            "depth_multiplier": 2.6,
            "drop_rate": 0.5,
        },
    },
    "timm-efficientnet-b7": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b7.imagenet",
                "revision": "8ef7ffccf54dad9baceb21d05b7ef86b6b70f4cc",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b7.advprop",
                "revision": "fcbc576ffb939c12d5cd8dad523fdae6eb0177ca",
            },
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-b7.noisy-student",
                "revision": "6b1dd73e61bf934d485d7bd4381dc3e2ab374664",
            },
        },
        "params": {
            "out_channels": [3, 64, 48, 80, 224, 640],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 2.0,
            "depth_multiplier": 3.1,
            "drop_rate": 0.5,
        },
    },
    "timm-efficientnet-b8": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-efficientnet-b8.imagenet",
                "revision": "b5e9dde35605a3a6d17ea2a727382625f9066a37",
            },
            "advprop": {
                "repo_id": "smp-hub/timm-efficientnet-b8.advprop",
                "revision": "e43f381de72e7467383c2c80bacbb7fcb9572866",
            },
        },
        "params": {
            "out_channels": [3, 72, 56, 88, 248, 704],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 2.2,
            "depth_multiplier": 3.6,
            "drop_rate": 0.5,
        },
    },
    "timm-efficientnet-l2": {
        "encoder": EfficientNetEncoder,
        "pretrained_settings": {
            "noisy-student": {
                "repo_id": "smp-hub/timm-efficientnet-l2.noisy-student",
                "revision": "cdc711e76d1becdd9197169f1a8bb1b2094e980c",
            },
            "noisy-student-475": {
                "repo_id": "smp-hub/timm-efficientnet-l2.noisy-student-475",
                "revision": "35f5ba667a64bf4f3f0689daf84fc6d0f8e1311b",
            },
        },
        "params": {
            "out_channels": [3, 136, 104, 176, 480, 1376],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 4.3,
            "depth_multiplier": 5.3,
            "drop_rate": 0.5,
        },
    },
    "timm-tf_efficientnet_lite0": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-tf_efficientnet_lite0.imagenet",
                "revision": "f5729249af07e5d923fb8b16922256ce2865d108",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 40, 112, 320],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.0,
            "drop_rate": 0.2,
        },
    },
    "timm-tf_efficientnet_lite1": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-tf_efficientnet_lite1.imagenet",
                "revision": "7b5e3f8dbb0c13b74101773584bba7523721be72",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 40, 112, 320],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.0,
            "depth_multiplier": 1.1,
            "drop_rate": 0.2,
        },
    },
    "timm-tf_efficientnet_lite2": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-tf_efficientnet_lite2.imagenet",
                "revision": "cc5f6cd4c7409ebacc13292f09d369ae88547f6a",
            },
        },
        "params": {
            "out_channels": [3, 32, 24, 48, 120, 352],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.1,
            "depth_multiplier": 1.2,
            "drop_rate": 0.3,
        },
    },
    "timm-tf_efficientnet_lite3": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-tf_efficientnet_lite3.imagenet",
                "revision": "ab29c8402991591d66f813bbb1f061565d9b0cd0",
            },
        },
        "params": {
            "out_channels": [3, 32, 32, 48, 136, 384],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.2,
            "depth_multiplier": 1.4,
            "drop_rate": 0.3,
        },
    },
    "timm-tf_efficientnet_lite4": {
        "encoder": EfficientNetLiteEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-tf_efficientnet_lite4.imagenet",
                "revision": "91a822e0f03c255b34dfb7846d3858397e50ba39",
            },
        },
        "params": {
            "out_channels": [3, 32, 32, 56, 160, 448],
            "stage_idxs": [2, 3, 5],
            "channel_multiplier": 1.4,
            "depth_multiplier": 1.8,
            "drop_rate": 0.4,
        },
    },
}
