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
    """Select encoder indices based on encoder depth and specified indices."""
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


def get_encoder_num_features_from_reductions(
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


class AdapterLayer(nn.Module):
    """Apply 1x1 convolution (if in_channels != out_channels) and rescale features
    if rescale factor is not equal to 1.
    """

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


class Adapter(nn.Module):
    """Adapter module to adjust features from encoder to match specified output channels and spatial resolutions.
    Foe each feature map in the list, applies 1x1 convolution (if in_channels != out_channels) and rescale features
    in spatial dim if rescale factor is not equal to 1.
    """

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
                AdapterLayer(in_channels=in_ch, out_channels=out_ch, scale_factor=sf)
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


class BaseFeatureExtractor(nn.Module):
    """Base feature extractor to extract features from timm model with specified depth and out indices."""

    def __init__(
        self,
        timm_model: nn.Module,
        num_features: int,
        out_indices: Optional[List[int]] = None,
    ):
        """Initialize feature extractor with specified depth and out indices.

        Args:
            model (nn.Module): timm model created with `features_only=True`
            num_features (int): number of features required to extract from timm model
                (if not enough, features list will be padded with dummy features)
            out_indices (Optional[List[int]], optional): Indices of features to extract from timm model.
                If None, last `num_features` features will be extracted. Defaults to None.
        """
        super().__init__()

        timm_model_all_out_reductions = timm_model.feature_info.reduction()
        timm_model_all_out_indices = timm_model.feature_info.out_indices

        timm_model_selected_out_indices = select_feature_indices(
            timm_model_all_out_indices, num_features, out_indices
        )
        corrected_num_features = get_encoder_num_features_from_reductions(
            timm_model_all_out_reductions, num_features
        )
        n_real_features = len(timm_model_selected_out_indices)
        n_dummy_features = corrected_num_features - n_real_features

        if corrected_num_features != num_features:
            logger.warning(
                f"Encoder depth is adjusted to `encoder_depth={corrected_num_features}` "
                f"to match `timm` model features reductions {timm_model_all_out_reductions}."
            )
        if n_dummy_features:
            logger.info(
                f"Encoder has {n_dummy_features} dummy feature(s), because the real number of "
                f"features ({n_real_features}) is less than specified encoder depth ({corrected_num_features})."
            )

        if out_indices is None and len(timm_model_selected_out_indices) < len(
            timm_model_all_out_indices
        ):
            logger.info(
                f"Selected encoder features indices {timm_model_selected_out_indices} out of timm model "
                f"feature indices {timm_model_all_out_indices}. Consider using `encoder_out_indices` argument to "
                f"select specific features. Use `model.encoder.visualize()` to see the full list of features."
            )

        # set out indices for model with specified depth
        timm_model.feature_info.out_indices = timm_model_selected_out_indices

        # set attributes

        # timm model
        self.timm_model_out_channels = timm_model.feature_info.channels()
        self.timm_model_out_reductions = timm_model.feature_info.reduction()

        # selected features with indexing based on timm feature info
        self.timm_model_selected_out_indices = timm_model_selected_out_indices

        # dummy features
        self.n_dummy_features = n_dummy_features
        self.dummy_features_reductions = [2**i for i in range(1, n_dummy_features + 1)]

        # feature extractor attributes
        self.out_channels = [0] * n_dummy_features + self.timm_model_out_channels
        self.out_reductions = (
            self.dummy_features_reductions + self.timm_model_out_reductions
        )
        self.out_num_features = corrected_num_features

        # selected features with indexing based on feature index in the list (for .forward() features collection)
        self._feature_out_indices = [
            timm_model_all_out_indices.index(i) for i in timm_model_selected_out_indices
        ]

        # PyTorch Modules
        self.model = timm_model
        self.permute = PermuteIfNeeded(self.timm_model_out_channels)

    def _get_dummy_feature(self, x, reduction):
        b, _, h, w = x.shape
        return torch.zeros(
            b, 0, h // reduction, w // reduction, device=x.device, dtype=x.dtype
        )

    def forward(self, x):
        features = self.model(x)
        features = [features[i] for i in self._feature_out_indices]
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

        # select info ONLY for features that are going out of the model
        # some features are not out for forward method
        timm_model_out_feature_info = [
            x
            for x in model.feature_info.info
            if x["index"] in model.feature_info.out_indices
        ]

        # ----------------------------------------------------------
        # Create feature_extractor from timm model
        # ----------------------------------------------------------

        try:
            feature_extractor = BaseFeatureExtractor(model, depth, out_indices)
        except Exception as e:
            raise ValueError(
                f"Can't create encoder with specified name `{name}`. "
                f"Please, select another `encoder_name`. "
                f"Encoder channels: {model.feature_info.channels()}. "
                f"Encoder reductions: {model.feature_info.reduction()}. "
                f"Error: {e}"
            )

        self.feature_extractor = feature_extractor
        depth = feature_extractor.out_num_features

        # ----------------------------------------------------------
        # Create adapter to adjust features
        # ----------------------------------------------------------

        if out_channels is None:
            out_channels = feature_extractor.out_channels

        if len(out_channels) < depth:
            raise ValueError(
                f"Invalid `encoder_out_channels` argument. Expected length is equal to `encoder_depth={depth}`. "
                f"Got {len(out_channels)} out channels {out_channels}."
            )
        elif len(out_channels) > depth:
            out_channels = out_channels[:depth]

        # calculate rescale factors to adjust spatial resolution of extracted features
        # for example features may have reductions [2, 16, 16, 16], but we need [2, 4, 8, 16]
        expected_reductions = [2**i for i in range(1, depth + 1)]
        expected_reductions = [
            min(reduction, output_stride) for reduction in expected_reductions
        ]
        actual_reductions = feature_extractor.out_reductions
        scale_factors = [
            actual_reduction / expected_reduction
            for (expected_reduction, actual_reduction) in zip(
                expected_reductions, actual_reductions
            )
        ]

        # create adapter
        self.adapter = Adapter(
            in_channels=feature_extractor.out_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
        )

        self.depth = depth
        self.output_stride = min(output_stride, 2**depth)
        self.out_channels = [in_channels] + out_channels

        # for visualization
        self._timm_model_out_feature_info = timm_model_out_feature_info
        self._expected_reductions = expected_reductions
        self._encoder_out_channels = out_channels

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.adapter(features)
        return [x] + features

    # This one is used for visualization only, the code is a bit messy
    def visualize(self) -> str:
        """Method to visualize features of extractor and adapter."""
        # for visualization
        model_selected_features_info = [
            x
            for x in self._timm_model_out_feature_info
            if x["index"] in self.feature_extractor.timm_model_selected_out_indices
        ]
        dummy_features_indices = list(
            range(-self.feature_extractor.n_dummy_features, 0)
        )
        dummy_features_info = [
            {"index": i, "reduction": r, "num_chs": 0, "module": "'dummy'"}
            for i, r in zip(
                dummy_features_indices, self.feature_extractor.dummy_features_reductions
            )
        ]
        encoder_features_info = dummy_features_info + model_selected_features_info

        connector_features_info = [
            {
                "index": encoder_feature["index"],
                "reduction": reduction,
                "num_chs": out_ch if encoder_feature["index"] >= 0 else 0,
                "module": "connector",
            }
            for encoder_feature, reduction, out_ch in zip(
                encoder_features_info,
                self._expected_reductions,
                self._encoder_out_channels,
            )
        ]

        model_features_dict = {x["index"]: x for x in self._timm_model_out_feature_info}
        encoder_features_dict = {x["index"]: x for x in encoder_features_info}
        connector_features_dict = {x["index"]: x for x in connector_features_info}
        all_indexes = (
            set(model_features_dict.keys())
            | set(encoder_features_dict.keys())
            | set(connector_features_dict.keys())
        )
        all_indexes = sorted(list(all_indexes))

        def feature_as_string(f):
            if f is None:
                return " " * 25
            reduction = f["reduction"]
            num_chs = f["num_chs"]
            ch_str = f"{num_chs:5d}"
            h_str = f"H // {reduction}".ljust(7)
            w_str = f"W // {reduction}".ljust(7)
            return f"({ch_str}, {h_str}, {w_str})"

        header = f"{'index / module ': <15}: {' Timm model features': <25} -> {' Selected features': <25} -> {' Adapted features': <25}\n"
        header += "-" * len(header) + "\n"
        rows = []
        for index in all_indexes:
            model_feature = model_features_dict.get(index)
            encoder_feature = encoder_features_dict.get(index)
            connector_feature = connector_features_dict.get(index)
            name = model_feature["module"] if model_feature is not None else "-none-"

            model_str = feature_as_string(model_feature)
            encoder_str = feature_as_string(encoder_feature)
            connector_str = feature_as_string(connector_feature)
            index_str = str(index) if index >= 0 else "x"
            name_str = f"({name})"
            rows.append(
                f"{index_str: >2} {name_str: >12}: {model_str} -> {encoder_str} -> {connector_str}"
            )

        return header + "\n".join(rows)


# Some models of timm (1.0.3) have incorrect reductions, this is a patch for them
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
    name = "swinv2_base_window8_256"
    # name = "resnet18"
    # name = "darknet53"
    # name = "efficientformer_l1"
    # name = "xcit_tiny_24_p16_224"
    # name = "convformer_m36"
    # name = "efficientvit_m0" # check for error (has reduction 64)
    # name = "eva02_large_patch14_clip_224"
    name = "vit_base_patch16_siglip_224"

    # model = timm.create_model(name, pretrained=False, features_only=True, num_classes=0, global_pool="")
    # encoder = Encoder(model, depth=5).eval()

    encoder = TimmUniversalEncoder(
        name,
        depth=5,
        output_stride=32,
        pretrained=False,
        out_channels=[32, 64, 128, 256, 512],
    )
    result = encoder(torch.randn(1, 3, 224, 224))
    print([x.shape for x in result])
    print(encoder.visualize())
