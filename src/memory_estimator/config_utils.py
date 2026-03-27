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
