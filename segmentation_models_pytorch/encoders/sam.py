import math
import warnings
from typing import Mapping, Any

import torch
from segment_anything.modeling import ImageEncoderViT

from segmentation_models_pytorch.encoders._base import EncoderMixin


class SamVitEncoder(EncoderMixin, ImageEncoderViT):
    def __init__(self, name: str, **kwargs):
        patch_size = kwargs.get("patch_size", 16)
        n_skips = kwargs.pop("num_hidden_skips", int(self._get_scale_factor(patch_size)))
        super().__init__(**kwargs)
        self._name = name
        self._depth = kwargs["depth"]
        self._out_chans = kwargs.get("out_chans", 256)
        self._num_skips = n_skips
        self._validate_output(patch_size)

    @staticmethod
    def _get_scale_factor(patch_size: int) -> float:
        """Input image will be downscale by this factor"""
        return math.log(patch_size, 2)

    def _validate_output(self, patch_size: int):
        scale_factor = self._get_scale_factor(patch_size)
        if scale_factor != self._num_skips:
            raise ValueError(
                f"With {patch_size=} and {self._num_skips} skip connection layers, "
                "spatial dimensions of model output will not match input spatial dimensions"
            )

    @property
    def out_channels(self):
        # Fill up with leading zeros to be used in Unet
        return [0] * self._num_skips + [self._out_chans]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Return a list of tensors to match other encoders
        return [x, super().forward(x)]

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        # Exclude mask_decoder and prompt encoder weights
        # and remove 'image_encoder.' prefix
        state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if not k.startswith("mask_decoder") and not k.startswith("prompt_encoder")
        }
        missing, unused = super().load_state_dict(state_dict, strict=False)
        if len(missing) + len(unused) > 0:
            n_loaded = len(state_dict) - len(missing) - len(unused)
            warnings.warn(
                f"Only {n_loaded} out of pretrained {len(state_dict)} SAM image encoder modules are loaded. "
                f"Missing modules: {missing}. Unused modules: {unused}."
            )


sam_vit_encoders = {
    "sam-vit_h": {
        "encoder": SamVitEncoder,
        "pretrained_settings": {
            "sa-1b": {"url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"},
        },
        "params": dict(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        ),
    },
    "sam-vit_l": {
        "encoder": SamVitEncoder,
        "pretrained_settings": {
            "sa-1b": {"url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"},
        },
        "params": dict(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        ),
    },
    "sam-vit_b": {
        "encoder": SamVitEncoder,
        "pretrained_settings": {
            "sa-1b": {"url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"},
        },
        "params": dict(
            embed_dim=768,
            depth=12,
            num_heads=12,
            global_attn_indexes=[2, 5, 8, 11],
        ),
    },
}
