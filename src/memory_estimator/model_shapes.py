"""Helpers to derive parameter shapes without materialising weights."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Iterable

# Safetensors dtype string → bytes per element
_SAFETENSORS_DTYPE_BYTES: dict[str, float] = {
    "F64": 8.0, "F32": 4.0, "F16": 2.0, "BF16": 2.0,
    "I64": 8.0, "I32": 4.0, "I16": 2.0, "I8": 1.0,
    "U8": 1.0, "BOOL": 1.0, "F8_E4M3": 1.0, "F8_E5M2": 1.0,
}


@dataclass
class ParameterShape:
    name: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return math.prod(int(d) for d in self.shape)


def _cache_dir(model_id: str) -> Path:
    """Return the HF cache directory for a model."""
    from huggingface_hub.constants import HF_HUB_CACHE

    # HF cache uses models--org--name directory structure
    folder = "models--" + model_id.replace("/", "--")
    return Path(HF_HUB_CACHE) / folder


def _cache_path(model_id: str, revision: str | None) -> Path:
    cache_dir = _cache_dir(model_id)
    rev = revision or "main"
    return cache_dir / f"memory_estimator_cache_{rev}.json"


def _load_cache(
    model_id: str, revision: str | None
) -> tuple[list[ParameterShape], int] | None:
    """Load cached parameter shapes if available."""
    path = _cache_path(model_id, revision)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        shapes = [
            ParameterShape(name=s["name"], shape=tuple(s["shape"]))
            for s in data["shapes"]
        ]
        return shapes, data["total_bytes"]
    except (KeyError, json.JSONDecodeError, TypeError):
        return None


def _save_cache(
    model_id: str, revision: str | None,
    shapes: list[ParameterShape], total_bytes: int,
) -> None:
    """Save parameter shapes to cache."""
    path = _cache_path(model_id, revision)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_id": model_id,
            "revision": revision or "main",
            "shapes": [{"name": s.name, "shape": list(s.shape)} for s in shapes],
            "total_bytes": total_bytes,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data))
    except OSError:
        pass  # Cache write failure is not fatal


def _fetch_from_hub(
    model_id: str, revision: str | None
) -> tuple[list[ParameterShape], int]:
    """Fetch parameter shapes from safetensors headers via HF Hub."""
    from huggingface_hub import get_safetensors_metadata

    meta = get_safetensors_metadata(model_id, revision=revision)
    shapes: list[ParameterShape] = []
    total_bytes = 0
    seen: set[str] = set()
    for file_meta in meta.files_metadata.values():
        for name, tensor_info in file_meta.tensors.items():
            if name not in seen:
                seen.add(name)
                shapes.append(ParameterShape(name=name, shape=tuple(tensor_info.shape)))
                numel = math.prod(tensor_info.shape)
                dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(tensor_info.dtype, 2.0)
                total_bytes += int(numel * dtype_bytes)
    return shapes, total_bytes


def collect_parameter_shapes(
    model_id: str, revision: str | None = None, use_cache: bool = True,
) -> tuple[list[ParameterShape], int]:
    """Collect parameter shapes and total bytes from safetensors file headers.

    Uses ``huggingface_hub.get_safetensors_metadata`` which fetches only the
    small header section of each safetensors shard, giving us tensor names,
    shapes, and dtypes without downloading the full weight files.

    Results are cached locally so subsequent runs skip the network requests.
    Pass ``use_cache=False`` to force re-fetching.

    Returns ``(shapes, total_bytes)`` where ``total_bytes`` is the exact
    on-disk byte count computed from tensor metadata.
    """
    if use_cache:
        cached = _load_cache(model_id, revision)
        if cached is not None:
            return cached

    shapes, total_bytes = _fetch_from_hub(model_id, revision)
    _save_cache(model_id, revision, shapes, total_bytes)
    return shapes, total_bytes


def count_total_parameters(shapes: Iterable[ParameterShape]) -> int:
    return sum(shape.numel for shape in shapes)
