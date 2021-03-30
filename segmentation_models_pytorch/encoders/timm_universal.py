from ._base import EncoderMixin
from timm import create_model, list_models
import torch.nn as nn


class TimmUniversalEncoder(nn.Module, EncoderMixin):
    def __init__(self, model, in_channels, depth=5, pretrained=True, **kwargs):
        super().__init__()
        self._depth = depth
        self._in_channels = in_channels

        model = create_model(model_name=model,
                             in_chans=in_channels,
                             exportable=True,   # onnx export
                             features_only=True,
                             pretrained=pretrained,
                             out_indices=tuple(range(depth)))

        channels = model.feature_info.channels()
        self._out_channels = (in_channels, 2*channels[0]) + tuple(channels[1:])

        self._stage_idxs = model.feature_info.get('stage')[1:]

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1
        self.blocks = model.blocks

        self.formatted_settings = {}
        self.formatted_settings["input_space"] = "RGB"
        self.formatted_settings["input_range"] = (0, 1)
        self.formatted_settings["mean"] = model.default_cfg['mean']
        self.formatted_settings["std"] = model.default_cfg['std']

    def get_stages(self):
        stages = [
            nn.Identity(),
            nn.Sequential(self.conv_stem, self.bn1, self.act1),
            self.blocks[:self._stage_idxs[0]],
            self.blocks[self._stage_idxs[0]:self._stage_idxs[1]],
        ]
        if self._depth > 3:
            stages.append(self.blocks[self._stage_idxs[1]:self._stage_idxs[2]])
        if self._depth > 4:
            stages.append(self.blocks[self._stage_idxs[2]:])
        return stages


    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features


timm_universal_encoders = list_models()