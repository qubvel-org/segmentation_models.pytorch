import timm
import torch.nn as nn
from einops import rearrange

from .supported import TIMM_ENCODERS


class TimmEncoder(nn.Module):
    def __init__(
        self,
        name,
        pretrained=True,
        in_channels=3,
        depth=None,
        indices=None,
        output_stride=None,
        **kwargs,
    ):
        super().__init__()

        assert (
            depth is not None or indices is not None
        ), "Either `depth` or `indices` should be specified"

        if indices is not None:
            depth = len(indices)
        else:
            indices = tuple(range(depth))

        total_depth = TIMM_ENCODERS[name.split(".")[0]]["indices"]
        if len(total_depth) < depth:
            err = f"""
            The specified depth={depth}or indices={indices} is greater than
            the maximum available depth={total_depth} for the {name} encoder.
            """
            raise ValueError(err)

        params = dict(
            in_chans=in_channels,
            features_only=True,
            pretrained=pretrained,
            out_indices=indices,
        )

        if output_stride is not None:
            params["output_stride"] = output_stride

        if "densenet" in name and "output_stride" in params:
            params.pop("output_stride")

        params.update(kwargs)

        self.model = timm.create_model(name, **params)
        self.in_channels = in_channels
        self.indices = indices
        self.depth = depth
        self.output_stride = 32 if output_stride is None else output_stride
        self.out_channels = [self.in_channels] + self.model.feature_info.channels()
        self.reductions = [1] + self.model.feature_info.reduction()
        self.fix_padding()

    def fix_padding(self):
        """
        Some models like inceptionv4 or inceptionresnetv2 3x3 kernels with no padding
        resulting in odd numbered feature height/width dimensions. Update padding=1
        """
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3) and m.padding == (0, 0):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d) and m.padding == 0:
                m.padding = 1

    def forward(self, x):
        features = self.model(x)
        features = [x] + features

        # Check for swin-like models returning features as channels last
        for i in range(len(features)):
            if (
                not features[i].shape[1] == self.out_channels[i]
                and features[i].shape[-1] == self.out_channels[i]
            ):
                features[i] = rearrange(features[i], "b h w c -> b c h w")

        return features


class TimmViTEncoder(nn.Module):
    def __init__(
        self,
        name,
        pretrained=True,
        in_channels=3,
        depth=None,
        indices=None,
        norm=True,
        scale_factors=None,
        **kwargs,
    ):
        super().__init__()

        assert (
            depth is not None or indices is not None
        ), "Either `depth` or `indices` should be specified"

        if indices is not None:
            depth = len(indices)
        else:
            indices = tuple(range(depth))

        params = dict(in_chans=in_channels, pretrained=pretrained)
        params.update(kwargs)

        self.model = timm.create_model(name, **params)
        self.in_channels = in_channels
        self.indices = indices
        self.depth = depth
        self.out_channels = [in_channels] + [self.model.num_features] * self.depth
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.image_size = self.model.patch_embed.img_size[0]
        self.num_tokens = (self.image_size // self.patch_size) ** 2
        self.output_stride = self.patch_size
        self.reductions = [1] + [self.patch_size] * self.depth
        self.norm = norm

        if scale_factors is not None:
            err = "`scale_factors` must be the same length as `depth`. Got {len(scale_factors)} != {self.depth}"  # noqa: E501
            assert len(scale_factors) == self.depth, err
            self.scale_factors = scale_factors
        else:
            self.scale_factors = [1] * self.depth

        self.upsample = nn.ModuleList(
            [
                nn.UpsamplingBilinear2d(scale_factor=scale)
                for scale in self.scale_factors
            ]
        )

    def forward(self, x):
        features = self.model.get_intermediate_layers(
            x, n=self.indices, reshape=True, return_prefix_tokens=False, norm=self.norm
        )
        features = [up(feat) for up, feat in zip(self.upsample, features)]
        features = [x] + features
        return features
