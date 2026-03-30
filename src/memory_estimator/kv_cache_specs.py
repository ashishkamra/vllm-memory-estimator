"""KV cache estimation formulas matching upstream vLLM spec types.

Each function mirrors the memory formula of a corresponding KVCacheSpec
subclass in ``vllm/v1/kv_cache_interface.py``, without requiring PyTorch
or any vLLM runtime objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

from .config_utils import attention_chunk_size as _cfg_attention_chunk_size
from .config_utils import head_dim
from .config_utils import hidden_size
from .config_utils import intermediate_size
from .config_utils import kv_lora_rank as _cfg_kv_lora_rank
from .config_utils import layers_block_type as _cfg_layers_block_type
from .config_utils import mamba_d_conv as _cfg_mamba_d_conv
from .config_utils import mamba_d_state as _cfg_mamba_d_state
from .config_utils import mamba_expand as _cfg_mamba_expand
from .config_utils import mamba_head_dim as _cfg_mamba_head_dim
from .config_utils import mamba_n_groups as _cfg_mamba_n_groups
from .config_utils import mamba_n_heads as _cfg_mamba_n_heads
from .config_utils import model_type as _cfg_model_type
from .config_utils import no_rope_layers as _cfg_no_rope_layers
from .config_utils import num_attention_heads
from .config_utils import num_layers
from .config_utils import qk_rope_head_dim as _cfg_qk_rope_head_dim
from .config_utils import sliding_window as _cfg_sliding_window
from .dtype_utils import bytes_per_element
from .quantization import QuantizationSpec
from .vllm_defaults import DEFAULT_BLOCK_SIZE
from .vllm_defaults import DEFAULT_MAX_NUM_BATCHED_TOKENS

# Mamba2 models that use Mamba2-style state shapes.
_MAMBA2_MODEL_TYPES = frozenset({
    "mamba2", "bamba", "falcon_h1", "plamo2", "falcon_hybrid",
    "granitemoehybrid", "nemotron_h",
})

# Mamba1 models that use Mamba1-style state shapes.
_MAMBA1_MODEL_TYPES = frozenset({
    "mamba", "jamba", "falcon_mamba",
})

# FP8 MLA: fixed 656 bytes per token (512 fp8 NoPE + 16 scale + 128 RoPE).
_FP8_DS_MLA_BYTES_PER_TOKEN = 656


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class KVCacheResult:
    """Wraps a KV cache byte estimate with the spec type used."""

    total_bytes: float
    spec_type: str
    layer_groups: list[LayerGroupEstimate] = field(default_factory=list)


@dataclass
class LayerGroupEstimate:
    """Per-group breakdown for hybrid models."""

    spec_type: str
    num_layers: int
    bytes: float


# ---------------------------------------------------------------------------
# Per-spec formula functions
# ---------------------------------------------------------------------------

def _page_size_bytes(
    block_size: int,
    kv_heads: int,
    head_size: int,
    dtype_bytes: float,
    *,
    head_size_v: int | None = None,
) -> float:
    """Byte size of one KV page (K + V combined).

    Mirrors ``AttentionSpec.real_page_size_bytes`` / ``FullAttentionSpec``
    which uses ``head_size + head_size_v``.
    """
    hsv = head_size_v if head_size_v is not None else head_size
    return block_size * kv_heads * (head_size + hsv) * dtype_bytes


def _quant_scale_bytes(
    layers: int,
    kv_heads: int,
    quant_spec: QuantizationSpec,
) -> float:
    """Extra bytes for per-head quantization scales (K and V)."""
    if quant_spec.kv_cache_dtype.bits <= 8 and quant_spec.kv_cache_scale_dtype:
        scales = layers * kv_heads * 2
        return scales * bytes_per_element(quant_spec.kv_cache_scale_dtype)
    return 0.0


def full_attention_kv_bytes(
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    layers: int,
    kv_heads: int,
    head_size: int,
    block_size: int,
    *,
    head_size_v: int | None = None,
) -> float:
    """Mirrors ``FullAttentionSpec.max_memory_usage_bytes``.

    Per-layer: ``ceil(max_seq_len / block_size) * page_size_bytes``
    Total: per-layer * layers * max_active_seqs  + scale overhead
    """
    dtype_bytes = bytes_per_element(quant_spec.kv_cache_dtype)
    page = _page_size_bytes(block_size, kv_heads, head_size, dtype_bytes,
                            head_size_v=head_size_v)
    pages_per_seq = _cdiv(max_seq_len, block_size)
    total = layers * max_active_seqs * pages_per_seq * page
    total += _quant_scale_bytes(layers, kv_heads, quant_spec)
    return total


def sliding_window_kv_bytes(
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    layers: int,
    kv_heads: int,
    head_size: int,
    block_size: int,
    window: int,
    max_num_batched_tokens: int,
    *,
    head_size_v: int | None = None,
) -> float:
    """Mirrors ``SlidingWindowSpec.max_memory_usage_bytes``.

    Only the last ``window`` tokens need caching, plus newly scheduled
    tokens during chunked prefill.  An extra block accounts for
    misalignment at window boundaries.
    """
    dtype_bytes = bytes_per_element(quant_spec.kv_cache_dtype)
    page = _page_size_bytes(block_size, kv_heads, head_size, dtype_bytes,
                            head_size_v=head_size_v)
    num_tokens = min(window - 1 + max_num_batched_tokens, max_seq_len)
    pages_per_seq = _cdiv(num_tokens, block_size) + 1  # +1 for misalignment
    total = layers * max_active_seqs * pages_per_seq * page
    total += _quant_scale_bytes(layers, kv_heads, quant_spec)
    return total


def mla_kv_bytes(
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    layers: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    *,
    fp8_ds_mla: bool = False,
) -> float:
    """Mirrors ``MLAAttentionSpec.real_page_size_bytes``.

    MLA stores a single latent vector per token (no K+V split), with
    ``num_kv_heads=1`` and ``head_size = kv_lora_rank + qk_rope_head_dim``.

    When *fp8_ds_mla* is True, uses the fixed 656 bytes/token layout
    (512 fp8 NoPE + 16 scale bytes + 128 bf16 RoPE).
    """
    if fp8_ds_mla:
        page = block_size * _FP8_DS_MLA_BYTES_PER_TOKEN
    else:
        dtype_bytes = bytes_per_element(quant_spec.kv_cache_dtype)
        head_size = kv_lora_rank + qk_rope_head_dim
        page = block_size * 1 * head_size * dtype_bytes
    pages_per_seq = _cdiv(max_seq_len, block_size)
    total = layers * max_active_seqs * pages_per_seq * page
    if not fp8_ds_mla:
        total += _quant_scale_bytes(layers, 1, quant_spec)
    return total


def chunked_local_kv_bytes(
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    layers: int,
    kv_heads: int,
    head_size: int,
    block_size: int,
    chunk_size: int,
    max_num_batched_tokens: int,
    *,
    head_size_v: int | None = None,
) -> float:
    """Mirrors ``ChunkedLocalAttentionSpec.max_memory_usage_bytes``.

    Only ``chunk_size + max_num_batched_tokens`` tokens are cached,
    capped at ``max_seq_len``.
    """
    dtype_bytes = bytes_per_element(quant_spec.kv_cache_dtype)
    page = _page_size_bytes(block_size, kv_heads, head_size, dtype_bytes,
                            head_size_v=head_size_v)
    num_tokens = min(chunk_size + max_num_batched_tokens, max_seq_len)
    pages_per_seq = _cdiv(num_tokens, block_size)
    total = layers * max_active_seqs * pages_per_seq * page
    total += _quant_scale_bytes(layers, kv_heads, quant_spec)
    return total


def mamba_kv_bytes(
    max_active_seqs: int,
    max_seq_len: int,
    layers: int,
    state_bytes_per_layer: float,
    block_size: int,
) -> float:
    """Mirrors ``MambaSpec.max_memory_usage_bytes`` in "all" mode.

    Mamba uses state tensors rather than KV caches.  In the default "none"
    mode, only 1 state per sequence is kept (minimal).  In "all" mode
    (used with prefix caching), states are cached at every block boundary.

    We estimate "all" mode as the conservative upper bound.
    """
    pages_per_seq = _cdiv(max_seq_len, block_size)
    return layers * max_active_seqs * pages_per_seq * state_bytes_per_layer


def mamba1_state_bytes_per_layer(
    intermediate_size: int,
    d_state: int,
    d_conv: int,
    dtype_bytes: float,
    tp: int = 1,
) -> float:
    """Compute Mamba1 state size per layer (conv + temporal states).

    Mirrors ``mamba1_state_shape`` in ``mamba_utils.py``.
    """
    local_intermediate = intermediate_size // tp
    conv_elements = (d_conv - 1) * local_intermediate
    temporal_elements = local_intermediate * d_state
    return (conv_elements + temporal_elements) * dtype_bytes


def mamba2_state_bytes_per_layer(
    n_heads: int,
    mamba_head_dim: int,
    d_state: int,
    d_conv: int,
    intermediate_size: int,
    n_groups: int,
    dtype_bytes: float,
    tp: int = 1,
) -> float:
    """Compute Mamba2 state size per layer (conv + temporal states).

    Mirrors ``mamba2_state_shape`` in ``mamba_utils.py``.
    """
    conv_dim = intermediate_size + 2 * n_groups * d_state
    conv_elements = (d_conv - 1) * (conv_dim // tp)
    temporal_elements = (n_heads // tp) * mamba_head_dim * d_state
    return (conv_elements + temporal_elements) * dtype_bytes


# ---------------------------------------------------------------------------
# Detection and dispatch
# ---------------------------------------------------------------------------

def _is_mamba_model(config: Any) -> bool:
    """Check if the model is a pure Mamba (non-hybrid) model."""
    mt = _cfg_model_type(config)
    return mt in (_MAMBA1_MODEL_TYPES | _MAMBA2_MODEL_TYPES)


def _is_hybrid_model(config: Any) -> bool:
    """Check if the model has mixed layer types (attention + mamba, etc.)."""
    if _cfg_layers_block_type(config) is not None:
        return True
    if _cfg_no_rope_layers(config) is not None:
        return True
    return False


def _mamba_version(config: Any) -> int:
    """Return 1 for Mamba1 models, 2 for Mamba2 models, 0 if not Mamba."""
    mt = _cfg_model_type(config)
    if mt in _MAMBA1_MODEL_TYPES:
        return 1
    if mt in _MAMBA2_MODEL_TYPES:
        return 2
    return 0


def detect_kv_spec_type(config: Any) -> str:
    """Determine the KV cache spec type from model config attributes.

    Returns one of: ``"hybrid"``, ``"mla"``, ``"sliding_window"``,
    ``"chunked_local"``, ``"mamba"``, or ``"full"``.
    """
    # Hybrid check first — models with mixed layer types
    if _is_hybrid_model(config):
        return "hybrid"

    # Pure Mamba models
    if _is_mamba_model(config):
        return "mamba"

    # MLA check: DeepSeek-V2/V3 style latent attention
    lora_rank = _cfg_kv_lora_rank(config)
    rope_dim = _cfg_qk_rope_head_dim(config)
    if lora_rank > 0 and rope_dim > 0:
        return "mla"

    # Sliding window check
    window = _cfg_sliding_window(config)
    if window is not None:
        return "sliding_window"

    # Chunked local attention check
    chunk = _cfg_attention_chunk_size(config)
    if chunk is not None:
        return "chunked_local"

    return "full"


def _compute_hybrid_kv_bytes(
    config: Any,
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    block_size: int,
    batched: int,
    tp: int,
) -> KVCacheResult:
    """Compute KV cache for hybrid models with mixed layer types.

    Splits layers into groups by spec type, computes each group's memory
    using the appropriate formula, and sums the results.
    """
    hidden = hidden_size(config)
    n_heads, kv_heads = num_attention_heads(config)
    hdim = head_dim(config, hidden, n_heads)
    dtype_bytes = bytes_per_element(quant_spec.kv_cache_dtype)

    # Determine per-layer types
    block_types = _cfg_layers_block_type(config)
    nope_mask = _cfg_no_rope_layers(config)

    attn_layers = 0
    mamba_layers = 0
    chunked_layers = 0

    if block_types is not None:
        # Jamba-style: explicit per-layer type
        for bt in block_types:
            if bt.lower() in ("mamba", "mamba1", "mamba2"):
                mamba_layers += 1
            else:
                attn_layers += 1
    elif nope_mask is not None:
        # LLaMA-4-style: NoPE layers use chunked local, others use full
        chunk = _cfg_attention_chunk_size(config)
        for val in nope_mask:
            if val == 0:
                attn_layers += 1
            else:
                chunked_layers += 1
        # If no chunk_size declared despite NoPE mask, treat all as full
        if chunk is None:
            attn_layers += chunked_layers
            chunked_layers = 0

    groups: list[LayerGroupEstimate] = []
    total = 0.0

    # Full attention layers
    if attn_layers > 0:
        b = full_attention_kv_bytes(
            max_active_seqs, max_seq_len, quant_spec,
            attn_layers, kv_heads, hdim, block_size,
        )
        total += b
        groups.append(LayerGroupEstimate("full", attn_layers, b))

    # Chunked local attention layers (LLaMA-4 NoPE)
    if chunked_layers > 0:
        chunk = _cfg_attention_chunk_size(config)
        assert chunk is not None
        b = chunked_local_kv_bytes(
            max_active_seqs, max_seq_len, quant_spec,
            chunked_layers, kv_heads, hdim, block_size, chunk, batched,
        )
        total += b
        groups.append(LayerGroupEstimate("chunked_local", chunked_layers, b))

    # Mamba layers
    if mamba_layers > 0:
        mver = _mamba_version(config)
        inter = intermediate_size(config, hidden)
        d_state = _cfg_mamba_d_state(config)
        d_conv = _cfg_mamba_d_conv(config)
        expand = _cfg_mamba_expand(config)
        mamba_inter = int(hidden * expand) if expand > 0 else inter

        if mver == 2 or d_state >= 64:
            m_heads = _cfg_mamba_n_heads(config) or 1
            m_hdim = _cfg_mamba_head_dim(config) or d_state
            n_groups = _cfg_mamba_n_groups(config)
            state_per_layer = mamba2_state_bytes_per_layer(
                m_heads, m_hdim, d_state, d_conv, mamba_inter, n_groups,
                dtype_bytes, tp=tp,
            )
        else:
            state_per_layer = mamba1_state_bytes_per_layer(
                mamba_inter, d_state, d_conv, dtype_bytes, tp=tp,
            )
        b = mamba_kv_bytes(
            max_active_seqs, max_seq_len, mamba_layers,
            state_per_layer, block_size,
        )
        total += b
        groups.append(LayerGroupEstimate("mamba", mamba_layers, b))

    return KVCacheResult(total_bytes=total, spec_type="hybrid", layer_groups=groups)


def estimate_kv_cache_bytes_specaware(
    config: Any,
    max_active_seqs: int,
    max_seq_len: int,
    quant_spec: QuantizationSpec,
    block_size: int | None = None,
    max_num_batched_tokens: int | None = None,
    tensor_parallel_size: int = 1,
) -> KVCacheResult:
    """Dispatch to the right KV cache formula based on model config.

    This is the main entry point — it detects the spec type and calls the
    matching formula function.
    """
    bs = block_size if block_size is not None else DEFAULT_BLOCK_SIZE
    batched = (max_num_batched_tokens
               if max_num_batched_tokens is not None
               else max(DEFAULT_MAX_NUM_BATCHED_TOKENS, max_active_seqs))

    spec_type = detect_kv_spec_type(config)

    if spec_type == "hybrid":
        return _compute_hybrid_kv_bytes(
            config, max_active_seqs, max_seq_len, quant_spec, bs, batched,
            tp=tensor_parallel_size,
        )

    hidden = hidden_size(config)
    layers_ = num_layers(config)
    n_heads, kv_heads = num_attention_heads(config)
    hdim = head_dim(config, hidden, n_heads)
    dtype_bytes = bytes_per_element(quant_spec.kv_cache_dtype)

    if spec_type == "mamba":
        inter = intermediate_size(config, hidden)
        d_state = _cfg_mamba_d_state(config)
        d_conv = _cfg_mamba_d_conv(config)
        expand = _cfg_mamba_expand(config)
        mamba_inter = int(hidden * expand) if expand > 0 else inter
        mver = _mamba_version(config)

        if mver == 2 or d_state >= 64:
            m_heads = _cfg_mamba_n_heads(config) or 1
            m_hdim = _cfg_mamba_head_dim(config) or d_state
            n_groups = _cfg_mamba_n_groups(config)
            state_per_layer = mamba2_state_bytes_per_layer(
                m_heads, m_hdim, d_state, d_conv, mamba_inter, n_groups,
                dtype_bytes, tp=tensor_parallel_size,
            )
        else:
            state_per_layer = mamba1_state_bytes_per_layer(
                mamba_inter, d_state, d_conv, dtype_bytes,
                tp=tensor_parallel_size,
            )
        total = mamba_kv_bytes(
            max_active_seqs, max_seq_len, layers_, state_per_layer, bs,
        )
        return KVCacheResult(total_bytes=total, spec_type="mamba")

    if spec_type == "mla":
        lora_rank = _cfg_kv_lora_rank(config)
        rope_dim = _cfg_qk_rope_head_dim(config)
        total = mla_kv_bytes(
            max_active_seqs, max_seq_len, quant_spec,
            layers_, lora_rank, rope_dim, bs,
        )
        return KVCacheResult(total_bytes=total, spec_type="mla")

    if spec_type == "sliding_window":
        window = _cfg_sliding_window(config)
        assert window is not None
        total = sliding_window_kv_bytes(
            max_active_seqs, max_seq_len, quant_spec,
            layers_, kv_heads, hdim, bs, window, batched,
        )
        return KVCacheResult(total_bytes=total, spec_type="sliding_window")

    if spec_type == "chunked_local":
        chunk = _cfg_attention_chunk_size(config)
        assert chunk is not None
        total = chunked_local_kv_bytes(
            max_active_seqs, max_seq_len, quant_spec,
            layers_, kv_heads, hdim, bs, chunk, batched,
        )
        return KVCacheResult(total_bytes=total, spec_type="chunked_local")

    # Default: full attention
    total = full_attention_kv_bytes(
        max_active_seqs, max_seq_len, quant_spec,
        layers_, kv_heads, hdim, bs,
    )
    return KVCacheResult(total_bytes=total, spec_type="full")
