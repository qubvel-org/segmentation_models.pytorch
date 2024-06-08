import timm
import torch
import torch.nn as nn
from typing import List, Optional
from loguru import logger


def select_feature_indices(
    actual_indices: List[int],
    encoder_depth: int,
    encoder_indices: Optional[List[int]] = None,
) -> List[int]:
    # Case when indices are provided
    if encoder_indices is not None and len(encoder_indices) != encoder_depth:
        raise ValueError(
            f"Invalid `encoder_indices={encoder_indices}` for encoder with `encoder_depth={encoder_depth}`. "
            f"Expected `encoder_indices` length is equal to `encoder_depth`."
        )
    if encoder_indices is not None:
        selected_indices = []
        for i in encoder_indices:
            try:
                selected_indices.append(actual_indices[i])
            except IndexError:
                raise IndexError(
                    f"Invalid selection on `encoder_indices={encoder_indices}` for encoder "
                    f"with encoder indices {actual_indices}`."
                )
        return selected_indices

    # Otherwise return last `encoder_depth` indices
    return actual_indices[-encoder_depth:]


def get_encoder_depth_from_reductions(
    reductions: List[int], specified_depth: int
) -> int:
    max_reduction = max(reductions)
    if max_reduction not in [2, 4, 8, 16, 32]:
        raise ValueError(
            f"Not able to build encoder. Specified encoder has inappropriate reductions {reductions}."
        )

    depth_by_features = len(reductions)
    depth_by_reduction = int(max_reduction**0.5)
    encoder_max_depth = max(depth_by_reduction, depth_by_features)
    return min(encoder_max_depth, specified_depth)


class ConnectorLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: float):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        # 1x1 convolution to reduce number of channels (only if needed)
        if in_channels and out_channels and in_channels != out_channels:
            self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.reduce = nn.Identity()

        # rescale features to match spatial resolution
        if scale_factor and scale_factor != 1:
            self.rescale = nn.Upsample(
                scale_factor=scale_factor, mode="bilinear", align_corners=False
            )
        else:
            self.rescale = nn.Identity()

    def forward(self, x):
        x = self.reduce(x)
        x = self.rescale(x)
        return x


class Connector(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        scale_factors: List[float],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factors = scale_factors

        self.layers = nn.ModuleList(
            [
                ConnectorLayer(in_channels=in_ch, out_channels=out_ch, scale_factor=sf)
                for in_ch, out_ch, sf in zip(in_channels, out_channels, scale_factors)
            ]
        )

    def forward(self, features):
        return [layer(f) for layer, f in zip(self.layers, features)]


class PermuteIfNeeded(nn.Module):
    """Check if features have correct order of dimensions (N, C, H, W) and permute them if needed."""

    def __init__(self, channels: List[int]):
        super().__init__()
        self.channels = channels

    def forward(self, features: List[torch.Tensor]):
        last_dim = [f.shape[-1] for f in features]
        if last_dim == self.channels:
            features = [f.permute(0, 3, 1, 2) for f in features]
        return features


class Encoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        depth: int,
        out_indices: Optional[List[int]] = None,
    ):
        super().__init__()

        model_feature_info = model.feature_info
        model_reductions = model_feature_info.reduction()
        model_out_indices = model_feature_info.out_indices

        selected_out_indices = select_feature_indices(
            model_out_indices, depth, out_indices
        )
        corrected_depth = get_encoder_depth_from_reductions(model_reductions, depth)
        n_real_features = len(selected_out_indices)
        n_dummy_features = corrected_depth - n_real_features

        if corrected_depth != depth:
            logger.warning(
                f"Encoder depth is adjusted to `encoder_depth={corrected_depth}` "
                f"to match encoder features reductions {model_reductions}."
            )
        if n_dummy_features:
            logger.info(
                f"Encoder has `{n_dummy_features}` dummy feature(s), because the real number of "
                f"features `{n_real_features}` is less than `encoder_depth={corrected_depth}`."
            )

        if out_indices is None and len(selected_out_indices) < len(model_out_indices):
            logger.info(
                f"Selected encoder features indices {selected_out_indices} out of original {model_out_indices}. "
                f"Consider using `out_indices` argument to select specific features."
            )

        # set out indices for model with specified depth
        model.feature_info.out_indices = selected_out_indices

        # set attributes
        self.out_model_channels = model.feature_info.channels()
        self.out_model_reductions = model.feature_info.reduction()

        self.n_dummy_features = n_dummy_features
        self.dummy_features_reductions = [2**i for i in range(1, n_dummy_features + 1)]

        self.out_channels = [0] * n_dummy_features + self.out_model_channels
        self.out_reductions = self.dummy_features_reductions + self.out_model_reductions
        self.depth = corrected_depth

        # needed to select features from the list in forward
        self.out_indices = [model_out_indices.index(i) for i in selected_out_indices]

        # set modules
        self.model = model
        self.permute = PermuteIfNeeded(self.out_model_channels)

    def _get_dummy_feature(self, x, reduction):
        b, _, h, w = x.shape
        return torch.zeros(
            b, 0, h // reduction, w // reduction, device=x.device, dtype=x.dtype
        )

    def forward(self, x):
        features = self.model(x)
        features = [features[i] for i in self.out_indices]
        features = self.permute(features)
        dummy_features = [
            self._get_dummy_feature(x, r) for r in self.dummy_features_reductions
        ]
        return dummy_features + features


class TimmUniversalEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        out_indices: Optional[List[int]] = None,
        out_channels: Optional[List[int]] = None,
    ):
        super().__init__()

        # ----------------------------------------------------------
        # Initializing timm model with all out indices
        # ----------------------------------------------------------
        kwargs = dict(
            in_chans=in_channels,
            pretrained=pretrained,
            features_only=True,
            num_classes=0,
            global_pool="",
        )

        # not all timm models support output stride argument
        # since 32 is default value, we can drop it to make other models work
        if output_stride != 32:
            kwargs["output_stride"] = output_stride

        model = timm.create_model(name, **kwargs)
        model_base_name = name.split(".")[0]

        # patch for models with incorrect reductions
        if model_base_name in corrected_reductions:
            model_reductions = corrected_reductions[model_base_name]
            for model_feature_info, reduction in zip(
                model.feature_info.info, model_reductions
            ):
                model_feature_info["reduction"] = reduction

        # ----------------------------------------------------------
        # Create encoder to extract features
        # ----------------------------------------------------------

        try:
            encoder = Encoder(model, depth, out_indices)
        except Exception as e:
            raise ValueError(
                f"Can't create encoder with specified name `{name}`. "
                f"Please, select another `encoder_name`. "
                f"Encoder channels: {model.feature_info.channels()}. "
                f"Encoder reductions: {model_reductions}. "
                f"Error: {e}"
            )

        self.encoder = encoder
        depth = encoder.depth

        # ----------------------------------------------------------
        # Create connector to adjust features
        # ----------------------------------------------------------

        if out_channels is None:
            out_channels = encoder.out_channels

        if len(out_channels) < depth:
            raise ValueError(
                f"Invalid `encoder_out_channels` argument. Expected length is equal to `encoder_depth={depth}`. "
                f"Got {len(out_channels)} out channels {out_channels}."
            )
        elif len(out_channels) > depth:
            out_channels = out_channels[:depth]

        expected_reductions = [2**i for i in range(1, depth + 1)]
        expected_reductions = [
            min(reduction, output_stride) for reduction in expected_reductions
        ]
        actual_reductions = encoder.out_reductions

        scale_factors = [
            actual_reduction / expected_reduction
            for (expected_reduction, actual_reduction) in zip(
                expected_reductions, actual_reductions
            )
        ]

        self.connector = Connector(
            in_channels=encoder.out_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
        )

        self.depth = depth
        self.output_stride = min(output_stride, 2**depth)
        self.out_channels = [in_channels] + out_channels

    def forward(self, x):
        features = self.encoder(x)
        features = self.connector(features)
        return [x] + features


corrected_reductions = {
    "caformer_b36": [4, 8, 16, 32],
    "caformer_m36": [4, 8, 16, 32],
    "caformer_s18": [4, 8, 16, 32],
    "caformer_s36": [4, 8, 16, 32],
    "convformer_b36": [4, 8, 16, 32],
    "convformer_m36": [4, 8, 16, 32],
    "convformer_s18": [4, 8, 16, 32],
    "convformer_s36": [4, 8, 16, 32],
    "davit_base": [4, 8, 16, 32],
    "davit_giant": [4, 8, 16, 32],
    "davit_huge": [4, 8, 16, 32],
    "davit_large": [4, 8, 16, 32],
    "davit_small": [4, 8, 16, 32],
    "davit_tiny": [4, 8, 16, 32],
    "efficientformer_l1": [4, 8, 16, 32],
    "efficientformer_l3": [4, 8, 16, 32],
    "efficientformer_l7": [4, 8, 16, 32],
    "efficientvit_m0": [16, 32, 56],
    "efficientvit_m1": [16, 32, 56],
    "efficientvit_m2": [16, 32, 56],
    "efficientvit_m3": [16, 32, 56],
    "efficientvit_m4": [16, 32, 56],
    "efficientvit_m5": [16, 32, 56],
    "inception_resnet_v2": [2, 4, 8, 18, 42],
    "inception_v3": [2, 4, 8, 18, 42],
    "inception_v4": [2, 4, 8, 18, 42],
    "levit_128": [16, 32, 56],
    "levit_128s": [16, 32, 56],
    "levit_192": [16, 32, 56],
    "levit_256": [16, 32, 56],
    "levit_256d": [16, 32, 56],
    "levit_384": [16, 32, 56],
    "levit_512": [16, 32, 56],
    "levit_512d": [16, 32, 56],
    "levit_conv_128": [16, 32, 56],
    "levit_conv_128s": [16, 32, 56],
    "levit_conv_192": [16, 32, 56],
    "levit_conv_256": [16, 32, 56],
    "levit_conv_256d": [16, 32, 56],
    "levit_conv_384": [16, 32, 56],
    "levit_conv_512": [16, 32, 56],
    "levit_conv_512d": [16, 32, 56],
    "pit_b_224": [7, 14, 28],
    "pit_b_distilled_224": [7, 14, 28],
    "pit_s_224": [8, 16, 32],
    "pit_s_distilled_224": [8, 16, 32],
    "pit_ti_224": [8, 16, 32],
    "pit_ti_distilled_224": [8, 16, 32],
    "pit_xs_224": [8, 16, 32],
    "pit_xs_distilled_224": [8, 16, 32],
    "poolformer_m36": [4, 8, 16, 32],
    "poolformer_m48": [4, 8, 16, 32],
    "poolformer_s12": [4, 8, 16, 32],
    "poolformer_s24": [4, 8, 16, 32],
    "poolformer_s36": [4, 8, 16, 32],
    "poolformerv2_m36": [4, 8, 16, 32],
    "poolformerv2_m48": [4, 8, 16, 32],
    "poolformerv2_s12": [4, 8, 16, 32],
    "poolformerv2_s24": [4, 8, 16, 32],
    "poolformerv2_s36": [4, 8, 16, 32],
}


if __name__ == "__main__":
    # name = "swinv2_base_window8_256"
    # name = "resnet18"
    # name = "darknet53"
    name = "efficientformer_l1"
    # name = "xcit_tiny_24_p16_224"
    # name = "convformer_m36"
    # name = "efficientvit_m0" # check for error (has reduction 64)

    # model = timm.create_model(name, pretrained=False, features_only=True, num_classes=0, global_pool="")
    # encoder = Encoder(model, depth=5).eval()

    encoder = TimmUniversalEncoder(name, depth=5, output_stride=32)
    result = encoder(torch.randn(1, 3, 224, 224))
    print([x.shape for x in result])
