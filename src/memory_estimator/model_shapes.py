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
) -> tuple[list[ParameterShape], int, int, int, int, float] | None:
    """Load cached parameter shapes if available."""
    path = _cache_path(model_id, revision)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if "replicated_bytes" not in data or "disk_bytes_per_element" not in data:
            return None
        shapes = [
            ParameterShape(name=s["name"], shape=tuple(s["shape"]))
            for s in data["shapes"]
        ]
        total_bytes = data["total_bytes"]
        expert_bytes = data.get("expert_bytes", 0)
        non_expert_bytes = data.get("non_expert_bytes", total_bytes)
        replicated_bytes = data.get("replicated_bytes", 0)
        disk_bpe = data.get("disk_bytes_per_element", 2.0)
        return shapes, total_bytes, expert_bytes, non_expert_bytes, replicated_bytes, disk_bpe
    except (KeyError, json.JSONDecodeError, TypeError):
        return None


def _save_cache(
    model_id: str, revision: str | None,
    shapes: list[ParameterShape], total_bytes: int,
    expert_bytes: int, non_expert_bytes: int,
    replicated_bytes: int, disk_bytes_per_element: float,
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
            "expert_bytes": expert_bytes,
            "non_expert_bytes": non_expert_bytes,
            "replicated_bytes": replicated_bytes,
            "disk_bytes_per_element": disk_bytes_per_element,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data))
    except OSError:
        pass  # Cache write failure is not fatal


def _is_expert_tensor(name: str) -> bool:
    """Return True if the tensor name belongs to a MoE expert layer."""
    return ".experts." in name.lower()


_REPLICATED_COMPONENTS = (
    "vision_tower.", "vision_model.", "image_encoder.",
    "visual.", "multi_modal_projector.", "mm_projector.",
    "vision_embed_tokens.", "img_projection.", "image_projection.",
    "vision_encoder.", "image_newline",
)


def _is_replicated_tensor(name: str) -> bool:
    """Return True if this tensor is replicated (not sharded) across TP ranks.

    Vision encoders and multimodal projectors are replicated on every GPU
    in vLLM rather than being split by tensor parallelism.
    """
    lower = name.lower()
    if lower.startswith(tuple(_REPLICATED_COMPONENTS)):
        return True
    return any(f".{comp}" in lower for comp in _REPLICATED_COMPONENTS)


def _fetch_from_hub(
    model_id: str, revision: str | None
) -> tuple[list[ParameterShape], int, int, int, int, float]:
    """Fetch parameter shapes from safetensors headers via HF Hub.

    Returns ``(shapes, total_bytes, expert_bytes, non_expert_bytes,
    replicated_bytes, disk_bytes_per_element)``.
    """
    from collections import Counter

    from huggingface_hub import get_safetensors_metadata

    meta = get_safetensors_metadata(model_id, revision=revision)
    shapes: list[ParameterShape] = []
    total_bytes = 0
    expert_bytes = 0
    non_expert_bytes = 0
    replicated_bytes = 0
    seen: set[str] = set()
    dtype_counts: Counter[str] = Counter()
    for file_meta in meta.files_metadata.values():
        for name, tensor_info in file_meta.tensors.items():
            if name not in seen:
                seen.add(name)
                shapes.append(ParameterShape(name=name, shape=tuple(tensor_info.shape)))
                numel = math.prod(tensor_info.shape)
                dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(tensor_info.dtype, 2.0)
                tensor_bytes = int(numel * dtype_bytes)
                total_bytes += tensor_bytes
                dtype_counts[tensor_info.dtype] += numel
                if _is_replicated_tensor(name):
                    replicated_bytes += tensor_bytes
                elif _is_expert_tensor(name):
                    expert_bytes += tensor_bytes
                else:
                    non_expert_bytes += tensor_bytes

    dominant_dtype = dtype_counts.most_common(1)[0][0] if dtype_counts else "BF16"
    disk_bpe = _SAFETENSORS_DTYPE_BYTES.get(dominant_dtype, 2.0)

    return shapes, total_bytes, expert_bytes, non_expert_bytes, replicated_bytes, disk_bpe


def collect_parameter_shapes(
    model_id: str, revision: str | None = None, use_cache: bool = True,
) -> tuple[list[ParameterShape], int, int, int, int, float]:
    """Collect parameter shapes and total bytes from safetensors file headers.

    Uses ``huggingface_hub.get_safetensors_metadata`` which fetches only the
    small header section of each safetensors shard, giving us tensor names,
    shapes, and dtypes without downloading the full weight files.

    Results are cached locally so subsequent runs skip the network requests.
    Pass ``use_cache=False`` to force re-fetching.

    Returns ``(shapes, total_bytes, expert_bytes, non_expert_bytes,
    replicated_bytes, disk_bytes_per_element)`` where ``total_bytes`` is the
    exact on-disk byte count, ``expert_bytes`` / ``non_expert_bytes`` split
    by MoE expert membership, ``replicated_bytes`` are vision/projector
    tensors that vLLM replicates across TP ranks, and
    ``disk_bytes_per_element`` is the predominant on-disk dtype size.
    """
    if use_cache:
        cached = _load_cache(model_id, revision)
        if cached is not None:
            return cached

    result = _fetch_from_hub(model_id, revision)
    shapes, total_bytes, expert_bytes, non_expert_bytes, replicated_bytes, disk_bpe = result
    _save_cache(model_id, revision, shapes, total_bytes, expert_bytes,
                non_expert_bytes, replicated_bytes, disk_bpe)
    return result


def count_total_parameters(shapes: Iterable[ParameterShape]) -> int:
    return sum(shape.numel for shape in shapes)
