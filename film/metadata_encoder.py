"""dict/JSON metadata -> fixed-order numeric vector `m`.

The only encoding logic in the whole `film/` package — both the encoder-hook
path and the decoder-fork path call through here, so a metadata field is
encoded exactly one way no matter where it's used.

Categorical fields are one-hot, not a learned embedding: the only learned
transform between raw metadata and (gamma, beta) is the FiLMGenerator's
single Linear layer (see generator.py). A learned embedding table ahead of
it would add a second, opaque learned mapping and break the "read W directly
off the weight matrix" auditability the design calls for.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from .metadata_schema import MetadataSchema


class UnknownCategoryError(ValueError):
    pass


class MetadataEncoder:
    def __init__(self, schema: MetadataSchema, unknown_category: str = "raise"):
        if unknown_category not in ("raise", "zero"):
            raise ValueError("unknown_category must be 'raise' or 'zero'")
        self.schema = schema
        self.unknown_category = unknown_category

    def encode(self, meta: Dict[str, Any]) -> torch.Tensor:
        chunks: List[float] = []
        for f in self.schema.fields:
            if f.name not in meta:
                raise KeyError(
                    f"metadata dict is missing required field {f.name!r} "
                    f"(schema fields: {self.schema.field_names})"
                )
            value = meta[f.name]
            if f.kind == "continuous":
                v = float(value)
                if f.mean is not None and f.std is not None:
                    v = (v - f.mean) / f.std
                chunks.append(v)
            else:
                one_hot = [0.0] * len(f.categories)
                try:
                    idx = f.categories.index(value)
                    one_hot[idx] = 1.0
                except ValueError:
                    if self.unknown_category == "raise":
                        raise UnknownCategoryError(
                            f"value {value!r} for categorical field {f.name!r} is not in "
                            f"the fitted vocabulary {f.categories} — refit the schema or "
                            f"pass unknown_category='zero' to fall back to an all-zero encoding"
                        )
                    # unknown_category == "zero": leave the one-hot vector all-zero
                chunks.extend(one_hot)
        return torch.tensor(chunks, dtype=torch.float32)

    def encode_batch(self, metas: Sequence[Dict[str, Any]]) -> torch.Tensor:
        if not metas:
            raise ValueError("encode_batch requires at least one metadata dict")
        return torch.stack([self.encode(m) for m in metas], dim=0)
