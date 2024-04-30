import math
import warnings
from typing import Mapping, Any

import torch
from segment_anything.modeling import ImageEncoderViT
from torch import nn
from segment_anything.modeling.common import LayerNorm2d

from segmentation_models_pytorch.encoders._base import EncoderMixin


class SamVitEncoder(EncoderMixin, ImageEncoderViT):
    def __init__(self, **kwargs):
        self._vit_depth = kwargs.pop("vit_depth")
        self._encoder_depth = kwargs.get("depth", 5)
        kwargs.update({"depth": self._vit_depth})
        super().__init__(**kwargs)
        self._out_chans = kwargs.get("out_chans", 256)
        self._patch_size = kwargs.get("patch_size", 16)
        self._embed_dim = kwargs.get("embed_dim", 768)
        self._validate()
        self.intermediate_necks = nn.ModuleList(
            [self.init_neck(self._embed_dim, out_chan) for out_chan in self.out_channels[:-1]]
        )

    @staticmethod
    def init_neck(embed_dim: int, out_chans: int) -> nn.Module:
        # Use similar neck as in ImageEncoderViT
        return nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    @staticmethod
    def neck_forward(neck: nn.Module, x: torch.Tensor, scale_factor: float = 1) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        if scale_factor != 1.0:
            x = nn.functional.interpolate(x, scale_factor=scale_factor, mode="bilinear")
        return neck(x)

    def requires_grad_(self, requires_grad: bool = True):
        # Keep the intermediate necks trainable
        for param in self.parameters():
            param.requires_grad_(requires_grad)
        for param in self.intermediate_necks.parameters():
            param.requires_grad_(True)
        return self

    @property
    def output_stride(self):
        return 32

    @property
    def out_channels(self):
        return [self._out_chans // (2**i) for i in range(self._encoder_depth + 1)][::-1]

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

    def _get_scale_factor(self) -> float:
        """Input image will be downscale by this factor"""
        return int(math.log(self._patch_size, 2))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        features = []
        skip_steps = self._vit_depth // self._encoder_depth
        scale_factor = self._get_scale_factor()
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i % skip_steps == 0:
                # Double spatial dimension and halve number of channels
                neck = self.intermediate_necks[i // skip_steps]
                features.append(self.neck_forward(neck, x, scale_factor=2**scale_factor))
                scale_factor -= 1

        x = self.neck(x.permute(0, 3, 1, 2))
        features.append(x)

        return features

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        # Exclude mask_decoder and prompt encoder weights
        # and remove 'image_encoder.' prefix
        state_dict = {
            k.replace("image_encoder.", ""): v
            for k, v in state_dict.items()
            if not k.startswith("mask_decoder") and not k.startswith("prompt_encoder")
        }
        missing, unused = super().load_state_dict(state_dict, strict=False)
        missing = list(filter(lambda x: not x.startswith("intermediate_necks"), missing))
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
