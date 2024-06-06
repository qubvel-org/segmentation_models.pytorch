import timm
import torch.nn as nn
from loguru import logger


class TimmUniversalEncoder(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
        out_indices=(-5, -4, -3, -2, -1),
    ):
        super().__init__()

        # for common case with depth=5 we have reductions [2, 4, 8, 16, 32]
        # if stride is not 32, we need to adjust the list of features reductions
        features_reductions = [2**i for i in range(1, depth + 1)]
        features_reductions = [
            min(reduction, output_stride) for reduction in features_reductions
        ]
        self._features_reductions = features_reductions
        logger.debug(
            f"`features_reductions`: `{features_reductions}` for `encoder_depth={depth}` and `output_stride={output_stride}`."
        )

        # Initializing timm model
        out_indices = list(out_indices)[:depth]
        logger.debug(
            f"Using `out_indices`: `{out_indices}` for `encoder_depth={depth}`."
        )

        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=out_indices,
        )

        # not all timm models support output stride argument
        # since 32 is default value, we can drop it to make other models work
        if output_stride == 32:
            kwargs.pop("output_stride")

        self.model = timm.create_model(name, **kwargs)

        # encoder reductions
        self._timm_model_features_reductions = self.model.feature_info.reduction()
        logger.debug(
            f"`timm_model_features_reductions`: `{self._timm_model_features_reductions}`"
        )

        # encoder out channels
        self._timm_model_out_channels = self.model.feature_info.channels()
        logger.debug(f"`timm_model_out_channels`: `{self._timm_model_out_channels}`")

        if len(self._timm_model_features_reductions) != len(self._features_reductions):
            raise RuntimeError("Misconfiguration of encoder features reductions.")

        self._scale_factors = [
            timm_reduction / required_reduction
            for (required_reduction, timm_reduction) in zip(
                self._features_reductions, self._timm_model_features_reductions
            )
        ]
        logger.debug(f"`rescale_factors`: `{self._scale_factors}`")

        # default attributes required for encoder
        self._in_channels = in_channels
        self._out_channels = [in_channels] + self._timm_model_out_channels
        self._depth = depth
        self._output_stride = output_stride
        logger.debug(f"`out_channels`: `{self._out_channels}`")

    def forward(self, x):
        encoder_features = self.model(x)
        output_features = [x]
        for scale_factor, stage_features in zip(self._scale_factors, encoder_features):
            if scale_factor != 1:
                logger.debug(
                    f"Rescaling features {stage_features.shape} with scale factor: {scale_factor}"
                )
                stage_features = nn.functional.interpolate(
                    stage_features,
                    scale_factor=scale_factor,
                    mode="bilinear",
                    align_corners=False,
                )
                logger.debug(f"Rescaled features shape: {stage_features.shape}")
            output_features.append(stage_features)
        return output_features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)
