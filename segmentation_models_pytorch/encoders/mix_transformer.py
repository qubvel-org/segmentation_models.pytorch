# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# Licensed under the NVIDIA Source Code License. For full license
# terms, please refer to the LICENSE file provided with this code
# or visit NVIDIA's official repository at
# https://github.com/NVlabs/SegFormer/tree/master.
#
# This code has been modified.
# ---------------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Sequence, List

from timm.layers import DropPath, to_2tuple, trunc_normal_


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, -1).transpose(1, 2)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.transpose(1, 2).view(batch_size, channels, height, width)
        else:
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} should be divided by num_heads {num_heads}."
        )

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = LayerNorm(dim)
        else:
            # for torchscript compatibility
            self.sr = nn.Identity()
            self.norm = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, N, C = x.shape
        q = (
            self.q(x)
            .reshape(batch_size, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(batch_size, C, height, width)
            x_ = self.sr(x_).reshape(batch_size, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(batch_size, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(batch_size, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
        x = x.transpose(1, 2).view(batch_size, -1, height, width)
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dims=[64, 128, 256, 512],
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=LayerNorm,
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0],
        )
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
        )
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
        )
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
        )

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[0],
                    num_heads=num_heads[0],
                    mlp_ratio=mlp_ratios[0],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0],
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[1],
                    num_heads=num_heads[1],
                    mlp_ratio=mlp_ratios[1],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1],
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[2],
                    num_heads=num_heads[2],
                    mlp_ratio=mlp_ratios[2],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2],
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.Sequential(
            *[
                Block(
                    dim=embed_dims[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratios[3],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3],
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        pass

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed1",
            "pos_embed2",
            "pos_embed3",
            "pos_embed4",
            "cls_token",
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []

        # stage 1
        x = self.patch_embed1(x)
        x = self.block1(x)
        x = self.norm1(x).contiguous()
        outs.append(x)

        # stage 2
        x = self.patch_embed2(x)
        x = self.block2(x)
        x = self.norm2(x).contiguous()
        outs.append(x)

        # stage 3
        x = self.patch_embed3(x)
        x = self.block3(x)
        x = self.norm3(x).contiguous()
        outs.append(x)

        # stage 4
        x = self.patch_embed4(x)
        x = self.block4(x)
        x = self.norm4(x).contiguous()
        outs.append(x)

        return outs

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.forward_features(x)
        # x = self.head(x)

        return features


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).view(batch_size, channels, height, width)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# ---------------------------------------------------------------
# End of NVIDIA code
# ---------------------------------------------------------------

from ._base import EncoderMixin  # noqa E402


class MixVisionTransformerEncoder(MixVisionTransformer, EncoderMixin):
    def __init__(
        self, out_channels: List[int], depth: int = 5, output_stride: int = 32, **kwargs
    ):
        if depth > 5 or depth < 1:
            raise ValueError(
                f"{self.__class__.__name__} depth should be in range [1, 5], got {depth}"
            )
        super().__init__(**kwargs)

        self._depth = depth
        self._in_channels = 3
        self._out_channels = out_channels
        self._output_stride = output_stride

    def get_stages(self) -> Dict[int, Sequence[torch.nn.Module]]:
        return {
            16: [self.patch_embed3, self.block3, self.norm3],
            32: [self.patch_embed4, self.block4, self.norm4],
        }

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # create dummy output for the first block
        batch_size, _, height, width = x.shape
        dummy = torch.empty(
            [batch_size, 0, height // 2, width // 2], dtype=x.dtype, device=x.device
        )

        features = [x, dummy]

        if self._depth >= 2:
            x = self.patch_embed1(x)
            x = self.block1(x)
            x = self.norm1(x)
            x = x.contiguous()
            features.append(x)

        if self._depth >= 3:
            x = self.patch_embed2(x)
            x = self.block2(x)
            x = self.norm2(x)
            x = x.contiguous()
            features.append(x)

        if self._depth >= 4:
            x = self.patch_embed3(x)
            x = self.block3(x)
            x = self.norm3(x)
            x = x.contiguous()
            features.append(x)

        if self._depth >= 5:
            x = self.patch_embed4(x)
            x = self.block4(x)
            x = self.norm4(x)
            x = x.contiguous()
            features.append(x)

        return features

    def load_state_dict(self, state_dict):
        state_dict.pop("head.weight", None)
        state_dict.pop("head.bias", None)
        return super().load_state_dict(state_dict)


mix_transformer_encoders = {
    "mit_b0": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/mit_b0.imagenet",
                "revision": "9ce53d104d92d75aabb00aae70677aaab67e7c84",
            }
        },
        "params": {
            "out_channels": [3, 0, 32, 64, 160, 256],
            "patch_size": 4,
            "embed_dims": [32, 64, 160, 256],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b1": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/mit_b1.imagenet",
                "revision": "a04bf4f13a549bce677cf79b04852e7510782817",
            }
        },
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [2, 2, 2, 2],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b2": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/mit_b2.imagenet",
                "revision": "868ab6f13871dcf8c3d9f90ee4519403475b65ef",
            }
        },
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [3, 4, 6, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b3": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/mit_b3.imagenet",
                "revision": "32558d12a65f1daa0ebcf4f4053c4285e2c1cbda",
            }
        },
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [3, 4, 18, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b4": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/mit_b4.imagenet",
                "revision": "3a3454e900a4b4f11dd60eeb59101a9a1a36b017",
            }
        },
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [3, 8, 27, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
    "mit_b5": {
        "encoder": MixVisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": {
                "repo_id": "smp-hub/mit_b5.imagenet",
                "revision": "ced04d96c586b6297fd59a7a1e244fc78fdb6531",
            }
        },
        "params": {
            "out_channels": [3, 0, 64, 128, 320, 512],
            "patch_size": 4,
            "embed_dims": [64, 128, 320, 512],
            "num_heads": [1, 2, 5, 8],
            "mlp_ratios": [4, 4, 4, 4],
            "qkv_bias": True,
            "norm_layer": partial(LayerNorm, eps=1e-6),
            "depths": [3, 6, 40, 3],
            "sr_ratios": [8, 4, 2, 1],
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
        },
    },
}
