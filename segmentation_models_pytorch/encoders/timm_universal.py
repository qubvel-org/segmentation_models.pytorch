import timm
import torch
import torch.nn as nn
from typing import List, Optional


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


class TimmSimpleEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        depth: int,
    ):
        super().__init__()

        # set out indices for model with specified depth
        model_all_out_indices = model.feature_info.out_indices
        model_out_indices = model_all_out_indices[-depth:]
        model.feature_info.out_indices = model_out_indices

        # set attributes
        self.depth = depth
        self.out_channels = model.feature_info.channels()
        self.out_reductions = model.feature_info.reduction()

        # set modules
        self.model = model
        self.permute = PermuteIfNeeded(self.out_channels)
        
    def forward(self, x):
        features = self.model(x)
        features = self.permute(features)
        return features

    @staticmethod
    def is_suitable(model, required_depth):
        model_all_out_reductions = list(model.feature_info.reduction())
        supposed_all_reductions = [2, 4, 8, 16, 32]

        if required_depth > len(model_all_out_reductions) or required_depth > len(
            supposed_all_reductions
        ):
            return False
        elif (
            supposed_all_reductions[:required_depth]
            == model_all_out_reductions[:required_depth]
        ):
            return True
        return False


class TimmReducedFeaturesEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        depth: int,
    ):
        super().__init__()

        # set out indices for model with specified depth
        model_all_out_indices = model.feature_info.out_indices
        model_out_indices = model_all_out_indices[-(depth - 1) :]
        model.feature_info.out_indices = model_out_indices

        # set attributes
        self.depth = depth
        self.out_channels = [0] + model.feature_info.channels()
        self.out_reductions = [2] + model.feature_info.reduction()

        # set modules
        self.model = model
        self.permute = PermuteIfNeeded(model.feature_info.channels())

    def _get_dummy_feature(self, x):
        b, _, h, w = x.shape
        return torch.zeros(b, 0, h // 2, w // 2, device=x.device, dtype=x.dtype)

    def forward(self, x):
        features = self.model(x)
        features = self.permute(features)
        dummy_feature = self._get_dummy_feature(x)
        return [dummy_feature] + features

    @staticmethod
    def is_suitable(model, required_depth):
        model_all_out_reductions = list(model.feature_info.reduction())
        supposed_all_reductions = [4, 8, 16, 32]
        required_depth -= 1

        if required_depth > len(model_all_out_reductions) or required_depth > len(
            supposed_all_reductions
        ):
            return False
        elif (
            supposed_all_reductions[:required_depth]
            == model_all_out_reductions[:required_depth]
        ):
            return True
        return False


class TimmManyFeaturesEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        depth: int,
        out_indices: Optional[List[int]] = None,
    ):
        super().__init__()

        out_indices = out_indices or list(range(-depth, 0))  # last `depth` features
        if len(out_indices) != depth:
            raise ValueError(
                f"Invalid `encoder_out_indices={out_indices}` for encoder with `encoder_depth={depth}`. "
                f"Expected `encoder_out_indices` length is equal to `encoder_depth`."
            )

        # set out indices for model with specified depth
        model_all_out_indices = model.feature_info.out_indices
        try:
            model_out_indices = [model_all_out_indices[i] for i in out_indices]
        except IndexError:
            raise IndexError(
                f"Invalid selection on `out_indices={out_indices}` for encoder "
                f"with encoder indices {model_all_out_indices}`."
            )
        model.feature_info.out_indices = model_out_indices

        # set attributes
        self.depth = depth
        self.out_channels = model.feature_info.channels()
        self.out_reductions = model.feature_info.reduction()
        self.out_indices = out_indices

        # set modules
        self.model = model
        self.permute = PermuteIfNeeded(self.out_channels)

    def forward(self, x):
        features = self.model(x)
        features = [features[i] for i in self.out_indices]
        features = self.permute(features)
        return features

    @staticmethod
    def is_suitable(model, required_depth):
        model_all_out_reductions = model.feature_info.reduction()
        if len(model_all_out_reductions) >= required_depth:
            return True
        return False


class TimmUniversalEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        out_indices=None,
        out_channels=None,
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

        # ----------------------------------------------------------
        # Create encoder to extract features
        # ----------------------------------------------------------

        if TimmSimpleEncoder.is_suitable(model, depth):
            encoder = TimmSimpleEncoder(model, depth)
        elif TimmReducedFeaturesEncoder.is_suitable(model, depth):
            encoder = TimmReducedFeaturesEncoder(model, depth)
        elif TimmManyFeaturesEncoder.is_suitable(model, depth):
            encoder = TimmManyFeaturesEncoder(model, depth, out_indices)
        else:
            raise ValueError(
                f"Can't create encoder with specified name `{name}`. "
                f"Please, select another `encoder_name`. "
                f"Encoder channels: {model.feature_info.channels()}. "
                f"Encoder reductions: {model.feature_info.reduction()}. "
            )

        self.encoder = encoder

        # ----------------------------------------------------------
        # Create connector to adjust features
        # ----------------------------------------------------------

        if out_channels is None:
            out_channels = encoder.out_channels

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

        self.output_stride = min(output_stride, 2**depth)
        self.out_channels = [in_channels] + out_channels

    def forward(self, x):
        features = self.encoder(x)
        features = self.connector(features)
        return [x] + features
