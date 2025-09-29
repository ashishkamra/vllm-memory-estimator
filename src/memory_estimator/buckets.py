"""Memory accounting helpers grouped by logical category."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from typing import Sequence
from typing import Set

from transformers import PretrainedConfig

from .dtype_utils import bytes_per_element
from .model_shapes import ParameterShape
from .quantization import QuantizationSpec
from .quantization import per_group_overhead


_NESTED_CONFIG_KEYS: Sequence[str] = (
    "text_config",
    "vision_config",
    "language_model_config",
    "llm_config",
    "base_model_config",
    "model_config",
    "decoder",
    "encoder",
)


def _resolve_config_attr(config: PretrainedConfig,
                         names: Sequence[str],
                         visited: Set[int] | None = None) -> int | None:
    if visited is None:
        visited = set()
    ident = id(config)
    if ident in visited:
        return None
    visited.add(ident)

    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value

    for nested_key in _NESTED_CONFIG_KEYS:
        nested = getattr(config, nested_key, None)
        if nested is None:
            continue
        if isinstance(nested, (list, tuple, set)):
            for item in nested:
                if item is None:
                    continue
                result = _resolve_config_attr(item, names, visited)
                if result is not None:
                    return result
        else:
            result = _resolve_config_attr(nested, names, visited)
            if result is not None:
                return result
    return None


@dataclass
class MemoryBuckets:
    parameter_bytes: float
    activation_bytes: float
    kv_cache_bytes: float
    workspace_bytes: float

    @property
    def total_bytes(self) -> float:
        return self.parameter_bytes + self.activation_bytes + self.kv_cache_bytes + self.workspace_bytes


def _is_quantized_parameter(name: str) -> bool:
    lower = name.lower()
    if not lower.endswith("weight"):
        return False
    for keyword in ("norm", "rms", "layernorm", "embed", "embeddings",
                    "ln_f", "lora", "bias"):
        if keyword in lower:
            return False
    return True


def estimate_parameter_bytes(shapes: Iterable[ParameterShape],
                             quant_spec: QuantizationSpec) -> float:
    total = 0.0
    quantised = quant_spec.is_quantized
    dense_bytes = bytes_per_element(quant_spec.weight_dtype)
    quant_bytes = quant_spec.weight_bits / 8.0

    for shape in shapes:
        numel = shape.numel
        if quantised and _is_quantized_parameter(shape.name):
            total += numel * quant_bytes
            total += per_group_overhead(numel, quant_spec)
        else:
            total += numel * dense_bytes
    return total


def _hidden_size(config: PretrainedConfig) -> int:
    value = _resolve_config_attr(config, ("hidden_size", "n_embd",
                                          "model_dim", "d_model"))
    if value is not None:
        return int(value)
    raise AttributeError("Config does not expose a hidden size field")


def _intermediate_size(config: PretrainedConfig, hidden_size: int) -> int:
    value = _resolve_config_attr(config,
                                 ("intermediate_size", "ffn_dim", "n_inner",
                                  "d_ff"))
    if value is not None:
        return int(value)
    # fall back to GPT-like 4x width
    return int(hidden_size * 4)


def _num_layers(config: PretrainedConfig) -> int:
    value = _resolve_config_attr(config, ("num_hidden_layers", "n_layer",
                                          "num_layers", "decoder_layers"))
    if value is not None:
        return int(value)
    raise AttributeError("Config does not expose number of layers")


def _num_attention_heads(config: PretrainedConfig) -> tuple[int, int]:
    total = _resolve_config_attr(config,
                                 ("num_attention_heads", "n_head",
                                  "encoder_attention_heads"))
    if total is None:
        raise AttributeError("Config does not expose attention head count")
    kv = (_resolve_config_attr(config,
                               ("num_key_value_heads", "num_kv_heads"))
          or total)
    return int(total), int(kv)


def _head_dim(config: PretrainedConfig, hidden_size: int,
              num_heads: int) -> int:
    for attr in ("head_dim", "qk_head_dim", "attention_head_size"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    return hidden_size // num_heads


def estimate_activation_bytes(config: PretrainedConfig,
                              max_active_seqs: int,
                              max_seq_len: int,
                              quant_spec: QuantizationSpec) -> float:
    hidden = _hidden_size(config)
    intermediate = _intermediate_size(config, hidden)
    bytes_per_act = bytes_per_element(quant_spec.activation_dtype)

    tokens = max_active_seqs * max_seq_len

    # In inference the model processes layers sequentially, so peak activation
    # usage is dominated by the current hidden state buffer.
    peak_buffer = tokens * hidden * bytes_per_act

    moe_experts = getattr(config, "num_local_experts", None)
    if moe_experts is None:
        moe_experts = getattr(config, "num_experts", None)
    if moe_experts:
        topk = getattr(config, "num_experts_per_tok", 2)
        expert_hidden = getattr(config, "moe_intermediate_size", intermediate)
        peak_buffer += tokens * expert_hidden * topk * bytes_per_act * 0.5

    return peak_buffer * 1.10


def estimate_kv_cache_bytes(config: PretrainedConfig,
                             max_active_seqs: int,
                             max_seq_len: int,
                             quant_spec: QuantizationSpec) -> float:
    hidden = _hidden_size(config)
    layers = _num_layers(config)
    num_heads, kv_heads = _num_attention_heads(config)
    head_dim = _head_dim(config, hidden, num_heads)
    dtype_bytes = bytes_per_element(quant_spec.kv_cache_dtype)

    cache_elements = layers * max_active_seqs * max_seq_len * kv_heads * head_dim * 2
    total = cache_elements * dtype_bytes

    if quant_spec.kv_cache_dtype.bits <= 8 and quant_spec.kv_cache_scale_dtype:
        # Per-layer-per-head scaling factors.
        scales = layers * kv_heads
        total += scales * bytes_per_element(quant_spec.kv_cache_scale_dtype)
    return total



def build_memory_buckets(config: PretrainedConfig,
                         shapes: Iterable[ParameterShape],
                         max_active_seqs: int,
                         max_seq_len: int,
                         quant_spec: QuantizationSpec) -> MemoryBuckets:
    params = estimate_parameter_bytes(shapes, quant_spec)
    activations = estimate_activation_bytes(config, max_active_seqs,
                                            max_seq_len, quant_spec)
    kv_cache = estimate_kv_cache_bytes(config, max_active_seqs, max_seq_len,
                                       quant_spec)
    workspace = activations * 0.05
    return MemoryBuckets(params, activations, kv_cache, workspace)
