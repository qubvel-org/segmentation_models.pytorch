from typing import Any

import timm
import torch.nn as nn


class TimmUniversalEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        **kwargs: dict[str, Any],
    ):
        super().__init__()
        common_kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # not all models support output stride argument, drop it by default
        if output_stride == 32:
            common_kwargs.pop("output_stride")

        self.model = timm.create_model(
            name, **_merge_kwargs_no_duplicates(common_kwargs, kwargs)
        )

        self._in_channels = in_channels
        self._out_channels = [in_channels] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x):
        features = self.model(x)
        features = [x] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)


def _merge_kwargs_no_duplicates(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    duplicates = a.keys() & b.keys()
    if duplicates:
        raise ValueError(f"'{duplicates}' already specified internally")

    return a | b
