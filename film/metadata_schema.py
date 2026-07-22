"""Metadata data contract — single source of truth for what a metadata dict
looks like and how it maps onto a fixed-order numeric vector.

Generic by design: this is not tied to any particular domain's field names.
A field's encoding is picked from the Python type of the values observed for
it (``int``/``float`` -> continuous, everything else -> categorical), so the
same schema machinery works for sensor metadata, natural-image EXIF data,
or anything else a caller wants to condition on.

Imported by both the encoder-hook path (hooks.py) and the decoder-fork path
(decoder_blocks.py) so field order/encoding is defined exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class FieldSpec:
    """One metadata field's encoding rule.

    kind="continuous": encoded as a single scalar, optionally standardized
        with (mean, std) computed by ``MetadataSchema.fit``.
    kind="categorical": encoded as a one-hot vector over ``categories``
        (fixed at fit time — the vector length can't change after a
        ``FiLMGenerator`` has been sized against it).
    """

    name: str
    kind: str  # "continuous" | "categorical"
    mean: Optional[float] = None
    std: Optional[float] = None
    categories: Optional[List[Any]] = field(default=None)

    def __post_init__(self) -> None:
        if self.kind not in ("continuous", "categorical"):
            raise ValueError(
                f"FieldSpec.kind must be 'continuous' or 'categorical', got {self.kind!r}"
            )

    @property
    def dim(self) -> int:
        if self.kind == "continuous":
            return 1
        if self.categories is None:
            raise ValueError(
                f"categorical field {self.name!r} has no categories yet — call MetadataSchema.fit() first"
            )
        return len(self.categories)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "mean": self.mean,
            "std": self.std,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FieldSpec":
        return cls(**d)


def _infer_kind(value: Any) -> str:
    # bool is a subclass of int in Python — treat it as categorical, not continuous
    if isinstance(value, bool):
        return "categorical"
    if isinstance(value, (int, float)):
        return "continuous"
    return "categorical"


class MetadataSchema:
    """Fixed field order + encoding rules for dict-metadata -> vector `m`.

    Build via ``MetadataSchema.infer(samples)`` to auto-detect field kinds
    and fit normalization stats / category vocabularies from example data,
    or construct explicitly with a list of ``FieldSpec`` for full control.
    Field order is the order fields were first seen (or declared) — stable
    across calls, since vector positions must not shift once a
    ``FiLMGenerator`` has been sized against ``schema.dim``.
    """

    def __init__(self, fields: Optional[Sequence[FieldSpec]] = None):
        self.fields: List[FieldSpec] = list(fields) if fields is not None else []
        self._by_name: Dict[str, FieldSpec] = {f.name: f for f in self.fields}

    @property
    def field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    @property
    def dim(self) -> int:
        return sum(f.dim for f in self.fields)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __getitem__(self, name: str) -> FieldSpec:
        return self._by_name[name]

    @classmethod
    def infer(cls, samples: Sequence[Dict[str, Any]]) -> "MetadataSchema":
        """Auto-detect field kinds from a batch of example metadata dicts,
        then fit normalization stats / category vocabularies in one pass.
        """
        if not samples:
            raise ValueError("need at least one metadata sample to infer a schema from")

        names_in_order: List[str] = []
        seen = set()
        for sample in samples:
            for name in sample.keys():
                if name not in seen:
                    seen.add(name)
                    names_in_order.append(name)

        schema = cls()
        for name in names_in_order:
            values = [s[name] for s in samples if name in s]
            kind = _infer_kind(values[0])
            schema.fields.append(FieldSpec(name=name, kind=kind))
        schema._by_name = {f.name: f for f in schema.fields}
        schema.fit(samples)
        return schema

    def fit(self, samples: Sequence[Dict[str, Any]]) -> "MetadataSchema":
        """Compute (mean, std) for continuous fields and the category
        vocabulary for categorical fields, from example data. Mutates and
        returns self. Safe to call again to re-fit against new data.
        """
        for f in self.fields:
            values = [s[f.name] for s in samples if f.name in s]
            if not values:
                continue
            if f.kind == "continuous":
                values = [float(v) for v in values]
                n = len(values)
                mean = sum(values) / n
                variance = sum((v - mean) ** 2 for v in values) / n
                f.mean = mean
                f.std = variance**0.5 if variance > 0 else 1.0
            else:
                existing = list(f.categories) if f.categories else []
                for v in values:
                    if v not in existing:
                        existing.append(v)
                f.categories = existing
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {"fields": [f.to_dict() for f in self.fields]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetadataSchema":
        return cls([FieldSpec.from_dict(fd) for fd in d["fields"]])
