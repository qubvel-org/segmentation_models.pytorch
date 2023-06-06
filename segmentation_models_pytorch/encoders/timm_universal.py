import timm
import torch.nn as nn
import torch.nn.functional as F


class TimmUniversalEncoder(nn.Module):
    def __init__(self, name, pretrained=True, in_channels=3, depth=5, output_stride=32):
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # not all models support output stride argument, drop it by default
        if output_stride == 32:
            kwargs.pop("output_stride")

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x):
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)


class TimmUniversalViTEncoder(nn.Module):
    def __init__(self, name, pretrained=True, in_channels=3, norm=True):
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            pretrained=pretrained,
        )
        self.model = timm.create_model(name, **kwargs)
        self.norm = norm
        self.indexes = [i for i in range(-1, len(self.model.blocks), len(self.model.blocks) // 4)][1:]
        self._depth = 4
        self._in_channels = in_channels
        self._out_channels = [in_channels] + [self.model.num_features] * len(self.indexes)
        self.upsample = nn.ModuleList([nn.UpsamplingBilinear2d(scale_factor=s) for s in self.scale_factors])

    def forward(self, x):
        features = self.model.get_intermediate_layers(
            x, n=self.indexes, reshape=True, return_class_token=False, norm=self.norm
        )
        features = [up(f) for up, f in zip(self.upsample, features)]
        features = [x] + list(features)
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return self.model.patch_embed.patch_size[0]

    @property
    def image_size(self):
        return self.model.patch_embed.img_size[0]

    @property
    def num_tokens(self):
        return self.image_size // self.output_stride

    @property
    def scale_factors(self):
        sizes = [self.image_size // i for i in [2 ** (i + 1) for i in range(4)]]
        return [s // self.num_tokens for s in sizes]
