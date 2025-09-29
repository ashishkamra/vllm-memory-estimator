"""High-level orchestration for building memory estimates."""
from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoConfig

from .buckets import build_memory_buckets
from .model_shapes import collect_parameter_shapes
from .model_summary import ModelSummary
from .quantization import QuantizationSpec
from .quantization import parse_quantization
from .reports import MemoryEstimate
from .reports import build_estimate


@dataclass
class EstimatorInputs:
    model_id: str
    max_seq_len: int
    max_active_seqs: int = 1
    revision: str | None = None
    trust_remote_code: bool = False


def _load_config(inputs: EstimatorInputs):
    return AutoConfig.from_pretrained(inputs.model_id,
                                      revision=inputs.revision,
                                      trust_remote_code=inputs.trust_remote_code)


def prepare_summary(inputs: EstimatorInputs) -> ModelSummary:
    config = _load_config(inputs)
    quant_spec = parse_quantization(config)
    parameter_shapes = collect_parameter_shapes(config,
                                                trust_remote_code=inputs.
                                                trust_remote_code)
    return ModelSummary(
        model_id=inputs.model_id,
        config=config,
        quantization=quant_spec,
        parameter_shapes=parameter_shapes,
        max_active_seqs=inputs.max_active_seqs,
        max_seq_len=inputs.max_seq_len,
    )


def estimate_memory(summary: ModelSummary) -> MemoryEstimate:
    buckets = build_memory_buckets(summary.config,
                                   summary.parameter_shapes,
                                   summary.max_active_seqs,
                                   summary.max_seq_len,
                                   summary.quantization)
    return build_estimate(summary.model_id, buckets)


def estimate_from_inputs(inputs: EstimatorInputs) -> tuple[ModelSummary, MemoryEstimate]:
    summary = prepare_summary(inputs)
    estimate = estimate_memory(summary)
    return summary, estimate
