import timm
import torch.nn as nn


class TimmUniversalEncoder(nn.Module):

    def __init__(self, name, pretrained=True, in_channels=3, depth=5, output_stride=32): 
        super().__init__()
        self.model = timm.create_model(
            name,
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        self._in_channels = in_channels
        self._out_channels = [3, ] + self.model.feature_info.channels()
        self._depth = depth

    def forward(self, x):
        features = self.model(x)
        features = [x,] + features
        return features
