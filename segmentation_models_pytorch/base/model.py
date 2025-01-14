import torch
from typing import TypeVar, Type

from . import initialization as init
from .hub_mixin import SMPHubMixin
from .utils import is_torch_compiling

T = TypeVar("T", bound="SegmentationModel")


class SegmentationModel(torch.nn.Module, SMPHubMixin):
    """Base class for all segmentation models."""

    _is_torch_scriptable = True
    _is_torch_exportable = True
    _is_torch_compilable = True

    # if model supports shape not divisible by 2 ^ n set to False
    requires_divisible_input_shape = True

    # Fix type-hint for models, to avoid HubMixin signature
    def __new__(cls: Type[T], *args, **kwargs) -> T:
        instance = super().__new__(cls, *args, **kwargs)
        return instance

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        """Check if the input shape is divisible by the output stride.
        If not, raise a RuntimeError.
        """
        if not self.requires_divisible_input_shape:
            return

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        if not (
            torch.jit.is_scripting() or torch.jit.is_tracing() or is_torch_compiling()
        ):
            self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x

    def load_state_dict(self, state_dict, **kwargs):
        # for compatibility of weights for
        # timm- ported encoders with TimmUniversalEncoder
        from segmentation_models_pytorch.encoders import TimmUniversalEncoder

        if not isinstance(self.encoder, TimmUniversalEncoder):
            return super().load_state_dict(state_dict, **kwargs)

        patterns = ["regnet", "res2", "resnest", "mobilenetv3", "gernet"]

        is_deprecated_encoder = any(
            self.encoder.name.startswith(pattern) for pattern in patterns
        )

        if is_deprecated_encoder:
            keys = list(state_dict.keys())
            for key in keys:
                new_key = key
                if key.startswith("encoder.") and not key.startswith("encoder.model."):
                    new_key = "encoder.model." + key.removeprefix("encoder.")
                if "gernet" in self.encoder.name:
                    new_key = new_key.replace(".stages.", ".stages_")
                state_dict[new_key] = state_dict.pop(key)

        return super().load_state_dict(state_dict, **kwargs)
