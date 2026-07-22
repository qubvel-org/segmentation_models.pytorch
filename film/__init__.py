"""FiLM (Feature-wise Linear Modulation) conditioning for `smp.Unet`.

Attaches a metadata-driven gamma/beta modulation to an existing
`segmentation_models_pytorch.Unet`, at the encoder (via forward hooks),
the decoder (via a forked `UnetDecoder`), or both — configurable per model,
never by editing SMP's own source.

Typical usage::

    from film import MetadataSchema, FiLMConfig, build_film_unet

    samples = [
        {"sensor_gain": 400.0, "capture_mode": "indoor"},
        {"sensor_gain": 900.0, "capture_mode": "outdoor"},
    ]
    schema = MetadataSchema.infer(samples)

    model = build_film_unet(
        schema,
        film_config=FiLMConfig(target="both"),
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=1,
        classes=2,
    )

    masks = model(images, metadata=[{"sensor_gain": 400.0, "capture_mode": "indoor"}])
"""

from .audit import AuditLog, collect_generators
from .generator import FiLMGenerator, apply_film
from .hooks import EncoderFiLM, MetaBox
from .decoder_blocks import FiLMConv2dReLU, FiLMUnetDecoder, FiLMUnetDecoderBlock
from .metadata_encoder import MetadataEncoder, UnknownCategoryError
from .metadata_schema import FieldSpec, MetadataSchema
from .wiring import FiLMConfig, FiLMUnet, build_film_unet

__all__ = [
    "FieldSpec",
    "MetadataSchema",
    "MetadataEncoder",
    "UnknownCategoryError",
    "FiLMGenerator",
    "apply_film",
    "EncoderFiLM",
    "MetaBox",
    "FiLMConv2dReLU",
    "FiLMUnetDecoderBlock",
    "FiLMUnetDecoder",
    "FiLMConfig",
    "FiLMUnet",
    "build_film_unet",
    "AuditLog",
    "collect_generators",
]
