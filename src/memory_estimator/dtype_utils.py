"""Utilities for normalising dtype references and computing element sizes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ScalarType:
    """Simple representation of numeric storage characteristics."""

    name: str
    bits: float
    torch_dtype: torch.dtype | None = None

    @property
    def bytes(self) -> float:
        return self.bits / 8.0


# Known dtype aliases encountered in HF / vLLM configs.
_DTYPE_ALIASES: dict[str, str] = {
    "half": "float16",
    "bf16": "bfloat16",
    "float": "float32",
    "fp16": "float16",
    "fp32": "float32",
    "fp8": "float8_e4m3fn",
    "e4m3": "float8_e4m3fn",
    "e5m2": "float8_e5m2",
    "int4": "uint4",
    "nf4": "nf4",
}

# Canonical scalar descriptors.
_SCALAR_TYPES: dict[str, ScalarType] = {
    "float16": ScalarType("float16", 16, torch.float16),
    "bfloat16": ScalarType("bfloat16", 16, torch.bfloat16),
    "float32": ScalarType("float32", 32, torch.float32),
    "float64": ScalarType("float64", 64, torch.float64),
    "uint8": ScalarType("uint8", 8, torch.uint8),
    "int8": ScalarType("int8", 8, torch.int8),
    "float8_e4m3fn": ScalarType("float8_e4m3fn", 8),
    "float8_e5m2": ScalarType("float8_e5m2", 8),
    "uint4": ScalarType("uint4", 4),
    "nf4": ScalarType("nf4", 4),
    "int32": ScalarType("int32", 32, torch.int32),
    "int16": ScalarType("int16", 16, torch.int16),
    "int64": ScalarType("int64", 64, torch.int64),
}


def normalise_dtype(value: Any) -> ScalarType:
    """Normalise a user/config supplied dtype into a :class:`ScalarType`.

    The helper accepts ``torch.dtype`` instances, strings (case-insensitive),
    or objects exposing a ``dtype`` attribute. Quantised 4-bit dtypes are
    treated as occupying half a byte per element.
    """

    if isinstance(value, ScalarType):
        return value

    if isinstance(value, torch.dtype):
        for scalar in _SCALAR_TYPES.values():
            if scalar.torch_dtype is value:
                return scalar
        raise ValueError(f"Unsupported torch dtype: {value}")

    if hasattr(value, "dtype") and not isinstance(value, (str, bytes)):
        return normalise_dtype(value.dtype)

    if isinstance(value, bytes):
        value = value.decode()

    if isinstance(value, str):
        key = value.strip().lower()
        key = _DTYPE_ALIASES.get(key, key)
        scalar = _SCALAR_TYPES.get(key)
        if scalar:
            return scalar
        if key.startswith("float") and key[5:].isdigit():
            bits = float(key[5:])
            return ScalarType(key, bits)
        if key.startswith("int") and key[3:].isdigit():
            bits = float(key[3:])
            return ScalarType(key, bits)
        raise ValueError(f"Unknown dtype string: {value}")

    raise TypeError(f"Cannot interpret dtype from value of type {type(value)!r}")


def bytes_per_element(value: Any) -> float:
    """Return the number of bytes consumed by a single element."""

    return normalise_dtype(value).bytes
