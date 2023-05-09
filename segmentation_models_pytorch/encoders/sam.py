from segment_anything.modeling import ImageEncoderViT

from segmentation_models_pytorch.encoders._base import EncoderMixin


class SamVitEncoder(EncoderMixin, ImageEncoderViT):
    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self._name = name
        self._depth = kwargs["depth"]
        self._out_chans = kwargs.get("out_chans", 256)

    @property
    def out_channels(self):
        return [-1, self._out_chans]


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
