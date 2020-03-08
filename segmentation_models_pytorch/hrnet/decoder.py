import torch
from torch import nn, Tensor
from typing import List
from collections import OrderedDict
import torch.nn.functional as F


class HRNetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: int,
        dropout=0.0,
    ):
        super().__init__()

        self.interpolation_mode = "nearest"
        self.align_corners = None

        features = sum(encoder_channels)
        self.embedding = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(in_channels=features, out_channels=features,
                                  kernel_size=3, padding=1, bias=False),
                    ),
                    ("bn1", nn.BatchNorm2d(features)),
                    ("relu", nn.ReLU(inplace=True)),
                ]
            )
        )

        self.logits = nn.Sequential(
            OrderedDict(
                [
                    ("drop", nn.Dropout2d(dropout)),
                    ("final", nn.Conv2d(in_channels=features, out_channels=decoder_channels,
                                        kernel_size=1)),
                ]
            )
        )

    def forward(self, features: List[Tensor]):
        x_size = features[0].size()[2:]

        resized_feature_maps = [features[0]]
        for feature_map in features[1:]:
            feature_map = F.interpolate(
                feature_map, size=x_size, mode=self.interpolation_mode,
                align_corners=self.align_corners
            )
            resized_feature_maps.append(feature_map)

        feature_map = torch.cat(resized_feature_maps, dim=1)
        embedding = self.embedding(feature_map)
        return self.logits(embedding)