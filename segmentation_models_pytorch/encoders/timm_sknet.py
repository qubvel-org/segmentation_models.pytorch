import torch
from typing import Dict, List, Sequence
from timm.models.resnet import ResNet
from timm.models.sknet import SelectiveKernelBottleneck, SelectiveKernelBasic

from ._base import EncoderMixin


class SkNetEncoder(ResNet, EncoderMixin):
    def __init__(
        self,
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

        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

        del self.fc
        del self.global_pool

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        return {
            16: [self.layer3],
            32: [self.layer4],
        }

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [x]

        if self._depth >= 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            features.append(x)

        if self._depth >= 2:
            x = self.maxpool(x)
            x = self.layer1(x)
            features.append(x)

        if self._depth >= 3:
            x = self.layer2(x)
            features.append(x)

        if self._depth >= 4:
            x = self.layer3(x)
            features.append(x)

        if self._depth >= 5:
            x = self.layer4(x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)


timm_sknet_encoders = {
    "timm-skresnet18": {
        "encoder": SkNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-skresnet18.imagenet",
                "revision": "6c97652bb744d89177b68274d2fda3923a7d1f95",
            },
        },
        "params": {
            "out_channels": [3, 64, 64, 128, 256, 512],
            "block": SelectiveKernelBasic,
            "layers": [2, 2, 2, 2],
            "zero_init_last": False,
            "block_args": {"sk_kwargs": {"rd_ratio": 1 / 8, "split_input": True}},
        },
    },
    "timm-skresnet34": {
        "encoder": SkNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-skresnet34.imagenet",
                "revision": "2367796924a8182cc835ef6b5dc303917f923f99",
            },
        },
        "params": {
            "out_channels": [3, 64, 64, 128, 256, 512],
            "block": SelectiveKernelBasic,
            "layers": [3, 4, 6, 3],
            "zero_init_last": False,
            "block_args": {"sk_kwargs": {"rd_ratio": 1 / 8, "split_input": True}},
        },
    },
    "timm-skresnext50_32x4d": {
        "encoder": SkNetEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/timm-skresnext50_32x4d.imagenet",
                "revision": "50207e407cc4c6ea9e6872963db6844ca7b7b9de",
            },
        },
        "params": {
            "out_channels": [3, 64, 256, 512, 1024, 2048],
            "block": SelectiveKernelBottleneck,
            "layers": [3, 4, 6, 3],
            "zero_init_last": False,
            "cardinality": 32,
            "base_width": 4,
        },
    },
}
