"""Helpers for extracting architecture attributes from HF configs."""
from __future__ import annotations

from typing import Any
from typing import Sequence

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


def resolve_config_attr(config: Any,
                        names: Sequence[str],
                        visited: set[int] | None = None) -> Any | None:
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
                result = resolve_config_attr(item, names, visited)
                if result is not None:
                    return result
        else:
            result = resolve_config_attr(nested, names, visited)
            if result is not None:
                return result
    return None


def hidden_size(config: Any) -> int:
    value = resolve_config_attr(config, ("hidden_size", "n_embd",
                                         "model_dim", "d_model"))
    if value is not None:
        return int(value)
    raise AttributeError("Config does not expose a hidden size field")


def intermediate_size(config: Any, hidden: int) -> int:
    value = resolve_config_attr(config,
                                ("intermediate_size", "ffn_dim", "n_inner",
                                 "d_ff"))
    if value is not None:
        return int(value)
    return int(hidden * 4)


def num_layers(config: Any) -> int:
    value = resolve_config_attr(config, ("num_hidden_layers", "n_layer",
                                         "num_layers", "decoder_layers"))
    if value is not None:
        return int(value)
    raise AttributeError("Config does not expose number of layers")


def num_attention_heads(config: Any) -> tuple[int, int]:
    total = resolve_config_attr(config,
                                ("num_attention_heads", "n_head",
                                 "encoder_attention_heads"))
    if total is None:
        raise AttributeError("Config does not expose attention head count")
    kv = (resolve_config_attr(config,
                              ("num_key_value_heads", "num_kv_heads"))
          or total)
    return int(total), int(kv)


def head_dim(config: Any, hidden: int,
             n_heads: int) -> int:
    value = resolve_config_attr(config,
                                ("head_dim", "qk_head_dim",
                                 "attention_head_size"))
    if value is not None:
        return int(value)
    return hidden // n_heads


def vocab_size(config: Any) -> int:
    value = resolve_config_attr(config, ("vocab_size",))
    if value is not None:
        return int(value)
    return 0


def sliding_window(config: Any) -> int | None:
    """Return the sliding window size, or None if not set."""
    value = resolve_config_attr(config, ("sliding_window",))
    if value is not None and int(value) > 0:
        return int(value)
    return None


def kv_lora_rank(config: Any) -> int:
    """Return the KV LoRA rank for MLA models, or 0 if not applicable."""
    value = resolve_config_attr(config, ("kv_lora_rank",))
    return int(value) if value is not None else 0


def qk_rope_head_dim(config: Any) -> int:
    """Return the RoPE head dimension for MLA models, or 0 if not applicable."""
    value = resolve_config_attr(config, ("qk_rope_head_dim",))
    return int(value) if value is not None else 0


def attention_chunk_size(config: Any) -> int | None:
    """Return the attention chunk size for chunked local attention, or None."""
    value = resolve_config_attr(config, ("attention_chunk_size",))
    if value is not None and int(value) > 0:
        return int(value)
    return None


def layers_block_type(config: Any) -> list[str] | None:
    """Return per-layer block types for hybrid models (e.g. Jamba).

    Returns a list like ``["attention", "mamba", "attention", ...]`` or
    ``None`` if the config does not declare layer types.
    """
    value = resolve_config_attr(config, ("layers_block_type",))
    if value is not None and isinstance(value, (list, tuple)):
        return list(value)
    return None


def no_rope_layers(config: Any) -> list[int] | None:
    """Return the no-RoPE layer mask for models like LLaMA-4.

    Returns a list of 0/1 values where 0 means the layer uses RoPE (full
    attention) and non-zero means NoPE (chunked local attention), or
    ``None`` if not set.
    """
    value = resolve_config_attr(config, ("no_rope_layers",))
    if value is not None and isinstance(value, (list, tuple)):
        return list(value)
    return None


def mamba_d_state(config: Any) -> int:
    """Return Mamba SSM state dimension, or 0 if not a Mamba model."""
    value = resolve_config_attr(config, ("mamba_d_state", "ssm_state_size",
                                         "state_size"))
    return int(value) if value is not None else 0


def mamba_d_conv(config: Any) -> int:
    """Return Mamba convolution kernel size, or 0 if not a Mamba model."""
    value = resolve_config_attr(config, ("mamba_d_conv", "conv_kernel"))
    return int(value) if value is not None else 0


def mamba_expand(config: Any) -> float:
    """Return Mamba expansion factor, or 0 if not a Mamba model."""
    value = resolve_config_attr(config, ("mamba_expand", "expand"))
    return float(value) if value is not None else 0.0


def mamba_n_groups(config: Any) -> int:
    """Return Mamba2 number of groups, or 1."""
    value = resolve_config_attr(config, ("mamba_n_groups", "n_groups"))
    return int(value) if value is not None else 1


def mamba_n_heads(config: Any) -> int:
    """Return Mamba2 number of heads, or 0."""
    value = resolve_config_attr(config, ("mamba_n_heads", "num_heads"))
    return int(value) if value is not None else 0


def mamba_head_dim(config: Any) -> int:
    """Return Mamba2 head dimension, or 0."""
    value = resolve_config_attr(config, ("mamba_d_head", "head_dim"))
    return int(value) if value is not None else 0


def model_type(config: Any) -> str:
    """Return the model_type string from config, or 'unknown'."""
    value = resolve_config_attr(config, ("model_type",))
    return str(value) if value is not None else "unknown"


def max_model_len(config: Any) -> int:
    """Return the model's maximum sequence length from config.

    Searches the same attributes vLLM uses to determine the default
    ``--max-model-len`` when it is not explicitly provided.
    """
    # Note: model_max_length is intentionally excluded — it is a tokenizer
    # attribute that can contain sentinel values (e.g. 1e30) and would produce
    # absurd memory estimates.
    value = resolve_config_attr(config, (
        "max_position_embeddings", "n_positions", "max_seq_len",
        "seq_length", "max_sequence_length",
    ))
    if value is not None:
        return int(value)
    raise AttributeError(
        "Config does not expose a maximum sequence length "
        "(e.g. max_position_embeddings). Use --max-model-len to specify it."
    )
