import math
import warnings
from typing import Mapping, Any

import torch
from segment_anything.modeling import ImageEncoderViT

from segmentation_models_pytorch.encoders._base import EncoderMixin


class SamVitEncoder(EncoderMixin, ImageEncoderViT):
    def __init__(self, **kwargs):
        self._vit_depth = kwargs.pop("vit_depth")
        self._encoder_depth = kwargs.get("depth", 5)
        kwargs.update({"depth": self._vit_depth})
        super().__init__(**kwargs)
        self._out_chans = kwargs.get("out_chans", 256)
        self._patch_size = kwargs.get("patch_size", 16)
        self._validate()

    @property
    def output_stride(self):
        return 32

    def _get_scale_factor(self) -> float:
        """Input image will be downscale by this factor"""
        return int(math.log(self._patch_size, 2))

    def _validate(self):
        # check vit depth
        if self._vit_depth not in [12, 24, 32]:
            raise ValueError(f"vit_depth must be one of [12, 24, 32], got {self._vit_depth}")
        # check output
        scale_factor = self._get_scale_factor()
        if scale_factor != self._encoder_depth:
            raise ValueError(
                f"With patch_size={self._patch_size} and depth={self._encoder_depth}, "
                "spatial dimensions of model output will not match input spatial dimensions. "
                "It is recommended to set encoder depth=4 with default vit patch_size=16."
            )

    @property
    def out_channels(self):
        # Fill up with leading zeros to be used in Unet
        scale_factor = self._get_scale_factor()
        return [0] * scale_factor + [self._out_chans]

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
            vit_depth=32,
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
            vit_depth=24,
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
            vit_depth=12,
            num_heads=12,
            global_attn_indexes=[2, 5, 8, 11],
        ),
    },
}
