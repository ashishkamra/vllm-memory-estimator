"""High-level orchestration for building memory estimates."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .buckets import build_memory_buckets
from .config_utils import max_model_len
from .dtype_utils import normalise_dtype
from .model_shapes import collect_parameter_shapes
from .model_summary import ModelSummary
from .quantization import parse_quantization
from .reports import MemoryEstimate
from .reports import build_estimate
from .validation import validate_positive_int


@dataclass
class EstimatorInputs:
    model_id: str
    max_seq_len: int | None = None
    max_active_seqs: int = 256
    revision: str | None = None
    enforce_eager: bool = False
    cudagraph_capture_sizes: list[int] | None = None
    max_num_batched_tokens: int | None = None
    kv_cache_dtype: str | None = None
    dtype: str | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False
    block_size: int | None = None
    use_cache: bool = True


_NESTED_CONFIG_KEYS = frozenset((
    "text_config", "vision_config", "language_model_config", "llm_config",
    "base_model_config", "model_config", "decoder", "encoder",
))


class _DictConfig:
    """Lightweight config wrapper when transformers is not installed.

    Only wraps known nested config keys as ``_DictConfig`` objects so that
    ``getattr()``-based attribute access works for architecture resolution.
    Everything else (like ``quantization_config``) stays as a plain dict.
    """

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict) and key in _NESTED_CONFIG_KEYS:
                setattr(self, key, _DictConfig(value))
            else:
                setattr(self, key, value)


def _load_config(inputs: EstimatorInputs):
    import json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        inputs.model_id, "config.json", revision=inputs.revision
    )
    with open(path) as f:
        data = json.load(f)
    return _DictConfig(data)


def _validate_inputs(inputs: EstimatorInputs) -> None:
    if inputs.max_seq_len is not None:
        validate_positive_int(inputs.max_seq_len, name="max_seq_len")
    validate_positive_int(inputs.max_active_seqs, name="max_active_seqs")
    validate_positive_int(inputs.tensor_parallel_size, name="tensor_parallel_size")
    validate_positive_int(inputs.pipeline_parallel_size, name="pipeline_parallel_size")
    validate_positive_int(inputs.data_parallel_size, name="data_parallel_size")
    if inputs.max_num_batched_tokens is not None:
        validate_positive_int(inputs.max_num_batched_tokens, name="max_num_batched_tokens")
    if inputs.cudagraph_capture_sizes is not None:
        if not inputs.cudagraph_capture_sizes:
            raise ValueError("cudagraph_capture_sizes requires at least one positive integer")
        for size in inputs.cudagraph_capture_sizes:
            validate_positive_int(size, name="cudagraph_capture_sizes")
    if inputs.block_size is not None:
        validate_positive_int(inputs.block_size, name="block_size")


def _apply_dtype_overrides(inputs: EstimatorInputs, quant_spec):
    """Apply CLI dtype overrides to the parsed quantization spec."""
    if inputs.kv_cache_dtype and inputs.kv_cache_dtype != "auto":
        quant_spec = dataclasses.replace(
            quant_spec, kv_cache_dtype=normalise_dtype(inputs.kv_cache_dtype)
        )

    if inputs.dtype and inputs.dtype != "auto":
        new_dtype = normalise_dtype(inputs.dtype)
        overrides = {"activation_dtype": new_dtype}
        if not quant_spec.is_quantized:
            overrides["weight_dtype"] = new_dtype
        quant_spec = dataclasses.replace(quant_spec, **overrides)

    return quant_spec


def prepare_summary(inputs: EstimatorInputs) -> ModelSummary:
    _validate_inputs(inputs)
    config = _load_config(inputs)

    seq_len = inputs.max_seq_len
    if seq_len is None:
        seq_len = max_model_len(config)

    quant_spec = parse_quantization(config)
    quant_spec = _apply_dtype_overrides(inputs, quant_spec)
    parameter_shapes, precomputed_bytes, expert_bytes, non_expert_bytes = (
        collect_parameter_shapes(
            inputs.model_id, revision=inputs.revision, use_cache=inputs.use_cache,
        )
    )
    return ModelSummary(
        model_id=inputs.model_id,
        config=config,
        quantization=quant_spec,
        parameter_shapes=parameter_shapes,
        max_active_seqs=inputs.max_active_seqs,
        max_seq_len=seq_len,
        enforce_eager=inputs.enforce_eager,
        cudagraph_capture_sizes=inputs.cudagraph_capture_sizes,
        max_num_batched_tokens=inputs.max_num_batched_tokens,
        precomputed_parameter_bytes=precomputed_bytes,
        tensor_parallel_size=inputs.tensor_parallel_size,
        pipeline_parallel_size=inputs.pipeline_parallel_size,
        data_parallel_size=inputs.data_parallel_size,
        enable_expert_parallel=inputs.enable_expert_parallel,
        expert_bytes=expert_bytes,
        non_expert_bytes=non_expert_bytes,
        block_size=inputs.block_size,
    )


def estimate_memory(summary: ModelSummary) -> MemoryEstimate:
    buckets = build_memory_buckets(
        summary.config,
        summary.precomputed_parameter_bytes,
        summary.max_active_seqs,
        summary.max_seq_len,
        summary.quantization,
        enforce_eager=summary.enforce_eager,
        cudagraph_capture_sizes=summary.cudagraph_capture_sizes,
        max_num_batched_tokens=summary.max_num_batched_tokens,
        tensor_parallel_size=summary.tensor_parallel_size,
        pipeline_parallel_size=summary.pipeline_parallel_size,
        data_parallel_size=summary.data_parallel_size,
        enable_expert_parallel=summary.enable_expert_parallel,
        expert_bytes=summary.expert_bytes,
        non_expert_bytes=summary.non_expert_bytes,
        block_size=summary.block_size,
    )
    return build_estimate(
        summary.model_id, buckets,
        tensor_parallel_size=summary.tensor_parallel_size,
        pipeline_parallel_size=summary.pipeline_parallel_size,
        data_parallel_size=summary.data_parallel_size,
        enable_expert_parallel=summary.enable_expert_parallel,
    )


def estimate_from_inputs(inputs: EstimatorInputs) -> tuple[ModelSummary, MemoryEstimate]:
    summary = prepare_summary(inputs)
    estimate = estimate_memory(summary)
    return summary, estimate
