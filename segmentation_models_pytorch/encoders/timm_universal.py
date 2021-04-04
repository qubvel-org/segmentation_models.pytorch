import torch.nn as nn
from timm import create_model, list_models

from ._base import EncoderMixin


class TimmUniversalEncoder(nn.Module, EncoderMixin):
    def __init__(self, encoder_name, in_channels, depth=5, pretrained=True, **kwargs):
        super().__init__()
        self._depth = depth
        self._in_channels = in_channels

        self.encoder = create_model(model_name=encoder_name[len(timm_universal_prefix):],
                                    in_chans=in_channels,
                                    exportable=True,  # onnx export
                                    features_only=True,
                                    pretrained=pretrained,
                                    out_indices=tuple(range(depth)) # FIXME need to handle a few special cases for specific models
                                    )

        channels = self.encoder.feature_info.channels()
        self._out_channels = (in_channels,) + tuple(channels)

        self.formatted_settings = {}
        self.formatted_settings["input_space"] = "RGB"
        self.formatted_settings["input_range"] = (0, 1)
        self.formatted_settings["mean"] = self.encoder.default_cfg['mean']
        self.formatted_settings["std"] = self.encoder.default_cfg['std']

    def forward(self, x):
        features = self.encoder(x)
        return [x] + features


timm_universal_prefix = 'timm-u-'
def timm_universal_encoders(**kwargs):
    return [f'{timm_universal_prefix}{i}' for i in list_models(**kwargs)]
