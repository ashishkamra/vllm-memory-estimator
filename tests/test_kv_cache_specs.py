"""Tests for spec-aware KV cache estimation."""
from memory_estimator.buckets import estimate_kv_cache_bytes
from memory_estimator.kv_cache_specs import chunked_local_kv_bytes
from memory_estimator.kv_cache_specs import detect_kv_spec_type
from memory_estimator.kv_cache_specs import estimate_kv_cache_bytes_specaware
from memory_estimator.kv_cache_specs import full_attention_kv_bytes
from memory_estimator.kv_cache_specs import mamba1_state_bytes_per_layer
from memory_estimator.kv_cache_specs import mamba2_state_bytes_per_layer
from memory_estimator.kv_cache_specs import mamba_kv_bytes
from memory_estimator.kv_cache_specs import mla_kv_bytes
from memory_estimator.kv_cache_specs import sliding_window_kv_bytes
from memory_estimator.quantization import parse_quantization

# ---------------------------------------------------------------------------
# Mock configs
# ---------------------------------------------------------------------------

class StandardConfig:
    """Llama-3-style standard dense model."""
    model_type = "llama"
    hidden_size = 4096
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    intermediate_size = 14336
    torch_dtype = "bfloat16"


class SlidingWindowConfig:
    """Mistral-style model with sliding window attention."""
    model_type = "mistral"
    hidden_size = 4096
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    intermediate_size = 14336
    sliding_window = 4096
    torch_dtype = "bfloat16"


class MLAConfig:
    """DeepSeek-V2/V3-style MLA model."""
    model_type = "deepseek_v2"
    hidden_size = 5120
    num_hidden_layers = 60
    num_attention_heads = 128
    num_key_value_heads = 128
    intermediate_size = 12288
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    torch_dtype = "bfloat16"


class ChunkedLocalConfig:
    """LLaMA-4-style model with chunked local attention."""
    model_type = "llama4"
    hidden_size = 4096
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8
    intermediate_size = 14336
    attention_chunk_size = 8192
    torch_dtype = "bfloat16"


class JambaConfig:
    """Jamba-style hybrid attention+mamba model."""
    model_type = "jamba"
    hidden_size = 4096
    num_hidden_layers = 8
    num_attention_heads = 32
    num_key_value_heads = 8
    intermediate_size = 14336
    layers_block_type = [
        "mamba", "attention", "mamba", "attention",
        "mamba", "attention", "mamba", "attention",
    ]
    mamba_d_state = 16
    mamba_d_conv = 4
    mamba_expand = 1.5
    torch_dtype = "bfloat16"


class Llama4HybridConfig:
    """LLaMA-4-style hybrid with NoPE layers."""
    model_type = "llama4"
    hidden_size = 4096
    num_hidden_layers = 8
    num_attention_heads = 32
    num_key_value_heads = 8
    intermediate_size = 14336
    attention_chunk_size = 8192
    no_rope_layers = [0, 0, 0, 1, 0, 0, 0, 1]  # layers 3,7 are NoPE/chunked
    torch_dtype = "bfloat16"


class PureMamba1Config:
    """Pure Mamba1 model (no attention)."""
    model_type = "mamba"
    hidden_size = 2560
    num_hidden_layers = 64
    num_attention_heads = 1
    intermediate_size = 5120
    mamba_d_state = 16
    mamba_d_conv = 4
    mamba_expand = 2.0
    torch_dtype = "float16"


class PureMamba2Config:
    """Pure Mamba2 model."""
    model_type = "mamba2"
    hidden_size = 2560
    num_hidden_layers = 64
    num_attention_heads = 64
    intermediate_size = 5120
    mamba_d_state = 128
    mamba_d_conv = 4
    mamba_n_heads = 64
    mamba_d_head = 64
    mamba_n_groups = 8
    mamba_expand = 2.0
    torch_dtype = "bfloat16"


class BambaConfig:
    """Bamba-style hybrid attention+mamba2 model."""
    model_type = "bamba"
    hidden_size = 4096
    num_hidden_layers = 8
    num_attention_heads = 32
    num_key_value_heads = 8
    intermediate_size = 14336
    layers_block_type = [
        "mamba", "mamba", "mamba", "attention",
        "mamba", "mamba", "mamba", "attention",
    ]
    mamba_d_state = 128
    mamba_d_conv = 4
    mamba_n_heads = 64
    mamba_d_head = 64
    mamba_n_groups = 8
    mamba_expand = 2.0
    torch_dtype = "bfloat16"


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

def test_detect_full_attention():
    assert detect_kv_spec_type(StandardConfig()) == "full"


def test_detect_sliding_window():
    assert detect_kv_spec_type(SlidingWindowConfig()) == "sliding_window"


def test_detect_mla():
    assert detect_kv_spec_type(MLAConfig()) == "mla"


def test_detect_chunked_local():
    assert detect_kv_spec_type(ChunkedLocalConfig()) == "chunked_local"


def test_detect_unknown_defaults_to_full():
    class UnknownConfig:
        model_type = "unknown_model"
        hidden_size = 1024
        num_hidden_layers = 12
        num_attention_heads = 16
        torch_dtype = "float16"

    assert detect_kv_spec_type(UnknownConfig()) == "full"


def test_detect_jamba_hybrid():
    assert detect_kv_spec_type(JambaConfig()) == "hybrid"


def test_detect_llama4_hybrid():
    assert detect_kv_spec_type(Llama4HybridConfig()) == "hybrid"


def test_detect_pure_mamba1():
    assert detect_kv_spec_type(PureMamba1Config()) == "mamba"


def test_detect_pure_mamba2():
    assert detect_kv_spec_type(PureMamba2Config()) == "mamba"


def test_detect_bamba_hybrid():
    assert detect_kv_spec_type(BambaConfig()) == "hybrid"


# ---------------------------------------------------------------------------
# Formula tests — full attention
# ---------------------------------------------------------------------------

def test_full_attention_formula():
    """Verify full attention matches hand-calculated value with block rounding."""
    quant_spec = parse_quantization(StandardConfig())
    # 32 layers, 8 kv_heads, hdim = 4096/32 = 128, bf16 = 2 bytes
    # block_size=16: pages_per_seq = ceil(4096 / 16) = 256
    # page_size = 16 * 8 * (128 + 128) * 2 = 16 * 8 * 256 * 2 = 65536
    # total = 32 * 4 * 256 * 65536 = 2,147,483,648
    result = full_attention_kv_bytes(
        max_active_seqs=4, max_seq_len=4096, quant_spec=quant_spec,
        layers=32, kv_heads=8, head_size=128, block_size=16,
    )
    expected = 32 * 4 * 256 * (16 * 8 * 256 * 2)
    assert result == expected


def test_full_attention_cross_check_with_old_formula():
    """New full attention formula should match old estimate_kv_cache_bytes
    when max_seq_len is a multiple of block_size (no rounding difference)."""
    cfg = StandardConfig()
    quant_spec = parse_quantization(cfg)
    seqs, seq_len = 4, 4096  # 4096 is a multiple of 16
    layers, kv_heads, hdim = 32, 8, 128

    old = estimate_kv_cache_bytes(seqs, seq_len, quant_spec, layers, kv_heads, hdim)
    new = full_attention_kv_bytes(seqs, seq_len, quant_spec, layers, kv_heads, hdim, block_size=16)
    assert new == old


def test_full_attention_block_rounding():
    """When seq_len is not a multiple of block_size, new formula rounds up."""
    cfg = StandardConfig()
    quant_spec = parse_quantization(cfg)
    seqs, seq_len = 4, 4097  # not a multiple of 16
    layers, kv_heads, hdim = 32, 8, 128

    old = estimate_kv_cache_bytes(seqs, seq_len, quant_spec, layers, kv_heads, hdim)
    new = full_attention_kv_bytes(seqs, seq_len, quant_spec, layers, kv_heads, hdim, block_size=16)
    # New should be >= old due to rounding up
    assert new >= old
    # But very close (within one block worth per layer per seq)
    page_size = 16 * kv_heads * (hdim + hdim) * 2  # one page
    max_overhead = layers * seqs * page_size
    assert new - old < max_overhead


# ---------------------------------------------------------------------------
# Formula tests — sliding window
# ---------------------------------------------------------------------------

def test_sliding_window_much_smaller_than_full():
    """Sliding window KV cache should be much smaller than full when
    max_seq_len >> sliding_window."""
    quant_spec = parse_quantization(SlidingWindowConfig())
    seqs, layers, kv_heads, hdim = 256, 32, 8, 128

    full = full_attention_kv_bytes(
        seqs, 32768, quant_spec, layers, kv_heads, hdim, block_size=16,
    )
    sw = sliding_window_kv_bytes(
        seqs, 32768, quant_spec, layers, kv_heads, hdim, block_size=16,
        window=4096, max_num_batched_tokens=2048,
    )
    # With window=4096 and max_seq_len=32768, should be ~6-8x smaller
    ratio = full / sw
    assert ratio > 5


def test_sliding_window_formula():
    """Verify sliding window matches hand-calculated value."""
    quant_spec = parse_quantization(SlidingWindowConfig())
    # window=4096, max_batched_tokens=2048
    # num_tokens = min(4096 - 1 + 2048, 32768) = min(6143, 32768) = 6143
    # pages_per_seq = ceil(6143 / 16) + 1 = 384 + 1 = 385
    # page_size = 16 * 8 * 256 * 2 = 65536
    # total = 32 * 1 * 385 * 65536 = 807,403,520
    result = sliding_window_kv_bytes(
        max_active_seqs=1, max_seq_len=32768, quant_spec=quant_spec,
        layers=32, kv_heads=8, head_size=128, block_size=16,
        window=4096, max_num_batched_tokens=2048,
    )
    expected = 32 * 1 * 385 * (16 * 8 * 256 * 2)
    assert result == expected


def test_sliding_window_capped_at_max_seq_len():
    """When window + batched_tokens > max_seq_len, cap at max_seq_len."""
    quant_spec = parse_quantization(SlidingWindowConfig())
    result_short = sliding_window_kv_bytes(
        max_active_seqs=1, max_seq_len=1000, quant_spec=quant_spec,
        layers=32, kv_heads=8, head_size=128, block_size=16,
        window=4096, max_num_batched_tokens=2048,
    )
    result_full = full_attention_kv_bytes(
        max_active_seqs=1, max_seq_len=1000, quant_spec=quant_spec,
        layers=32, kv_heads=8, head_size=128, block_size=16,
    )
    # Sliding window with short seq should be close to full (just +1 block overhead)
    page_size = 16 * 8 * 256 * 2
    assert result_short - result_full == 32 * page_size


# ---------------------------------------------------------------------------
# Formula tests — MLA
# ---------------------------------------------------------------------------

def test_mla_much_smaller_than_full():
    """MLA KV cache should be dramatically smaller than standard attention."""
    quant_spec = parse_quantization(MLAConfig())
    seqs, seq_len = 256, 4096
    layers = 60
    hdim = 5120 // 128  # = 40

    full = full_attention_kv_bytes(
        seqs, seq_len, quant_spec, layers, 128, hdim, block_size=16,
    )
    mla = mla_kv_bytes(
        seqs, seq_len, quant_spec, layers,
        kv_lora_rank=512, qk_rope_head_dim=64, block_size=16,
    )
    ratio = full / mla
    assert ratio > 10


def test_mla_formula():
    """Verify MLA matches hand-calculated value."""
    quant_spec = parse_quantization(MLAConfig())
    # head_size = 512 + 64 = 576, num_kv_heads = 1, bf16 = 2 bytes
    # page = 16 * 1 * 576 * 2 = 18432
    # pages_per_seq = ceil(4096 / 16) = 256
    # total = 60 * 4 * 256 * 18432 = 1,132,462,080
    result = mla_kv_bytes(
        max_active_seqs=4, max_seq_len=4096, quant_spec=quant_spec,
        layers=60, kv_lora_rank=512, qk_rope_head_dim=64, block_size=16,
    )
    expected = 60 * 4 * 256 * (16 * 1 * 576 * 2)
    assert result == expected


def test_mla_fp8_ds_mla():
    """Verify fp8_ds_mla uses fixed 656 bytes/token."""
    quant_spec = parse_quantization(MLAConfig())
    result = mla_kv_bytes(
        max_active_seqs=1, max_seq_len=4096, quant_spec=quant_spec,
        layers=60, kv_lora_rank=512, qk_rope_head_dim=64, block_size=16,
        fp8_ds_mla=True,
    )
    # page = 16 * 656 = 10496
    # pages_per_seq = ceil(4096 / 16) = 256
    # total = 60 * 1 * 256 * 10496 = 161,218,560
    expected = 60 * 1 * 256 * (16 * 656)
    assert result == expected


def test_mla_fp8_smaller_than_standard_mla():
    """fp8_ds_mla should be smaller than standard bf16 MLA."""
    quant_spec = parse_quantization(MLAConfig())
    standard = mla_kv_bytes(
        max_active_seqs=4, max_seq_len=4096, quant_spec=quant_spec,
        layers=60, kv_lora_rank=512, qk_rope_head_dim=64, block_size=16,
    )
    fp8 = mla_kv_bytes(
        max_active_seqs=4, max_seq_len=4096, quant_spec=quant_spec,
        layers=60, kv_lora_rank=512, qk_rope_head_dim=64, block_size=16,
        fp8_ds_mla=True,
    )
    # 656 < 576 * 2 = 1152, so fp8 should be ~43% smaller
    assert fp8 < standard


# ---------------------------------------------------------------------------
# Formula tests — chunked local
# ---------------------------------------------------------------------------

def test_chunked_local_smaller_than_full():
    """Chunked local should be smaller than full for long contexts."""
    quant_spec = parse_quantization(ChunkedLocalConfig())
    seqs, layers, kv_heads, hdim = 256, 32, 8, 128

    full = full_attention_kv_bytes(
        seqs, 131072, quant_spec, layers, kv_heads, hdim, block_size=16,
    )
    chunked = chunked_local_kv_bytes(
        seqs, 131072, quant_spec, layers, kv_heads, hdim, block_size=16,
        chunk_size=8192, max_num_batched_tokens=2048,
    )
    ratio = full / chunked
    assert ratio > 10


def test_chunked_local_formula():
    """Verify chunked local matches hand-calculated value."""
    quant_spec = parse_quantization(ChunkedLocalConfig())
    # chunk=8192, max_batched=2048, max_seq_len=131072
    # num_tokens = min(8192 + 2048, 131072) = 10240
    # pages_per_seq = ceil(10240 / 16) = 640
    # page_size = 16 * 8 * 256 * 2 = 65536
    # total = 32 * 2 * 640 * 65536 = 2,684,354,560
    result = chunked_local_kv_bytes(
        max_active_seqs=2, max_seq_len=131072, quant_spec=quant_spec,
        layers=32, kv_heads=8, head_size=128, block_size=16,
        chunk_size=8192, max_num_batched_tokens=2048,
    )
    expected = 32 * 2 * 640 * (16 * 8 * 256 * 2)
    assert result == expected


# ---------------------------------------------------------------------------
# Formula tests — Mamba
# ---------------------------------------------------------------------------

def test_mamba1_state_bytes():
    """Verify Mamba1 state size calculation."""
    # intermediate = 2560 * 2.0 = 5120, d_state=16, d_conv=4, bf16=2
    # conv: (4-1) * 5120 = 15360 elements
    # temporal: 5120 * 16 = 81920 elements
    # total: (15360 + 81920) * 2 = 194560 bytes
    result = mamba1_state_bytes_per_layer(
        intermediate_size=5120, d_state=16, d_conv=4, dtype_bytes=2.0,
    )
    expected = (3 * 5120 + 5120 * 16) * 2
    assert result == expected


def test_mamba1_state_bytes_unsharded():
    """State functions return unsharded totals (TP applied later by build_memory_buckets)."""
    result = mamba1_state_bytes_per_layer(
        intermediate_size=5120, d_state=16, d_conv=4, dtype_bytes=2.0,
    )
    # Should equal full unsharded size regardless of TP
    expected = (3 * 5120 + 5120 * 16) * 2
    assert result == expected


def test_mamba2_state_bytes():
    """Verify Mamba2 state size calculation."""
    # n_heads=64, head_dim=64, d_state=128, d_conv=4
    # intermediate=5120, n_groups=8, bf16=2
    # conv_dim = 5120 + 2 * 8 * 128 = 5120 + 2048 = 7168
    # conv: (4-1) * 7168 = 21504 elements
    # temporal: 64 * 64 * 128 = 524288 elements
    # total: (21504 + 524288) * 2 = 1,091,584 bytes
    result = mamba2_state_bytes_per_layer(
        n_heads=64, mamba_head_dim=64, d_state=128, d_conv=4,
        intermediate_size=5120, n_groups=8, dtype_bytes=2.0,
    )
    conv_elements = 3 * 7168
    temporal_elements = 64 * 64 * 128
    expected = (conv_elements + temporal_elements) * 2
    assert result == expected


def test_mamba_kv_bytes_formula():
    """Verify mamba_kv_bytes matches expected calculation."""
    state_per_layer = 1000.0
    # 64 layers, 4 seqs, seq_len=4096, block_size=16
    # pages_per_seq = ceil(4096/16) = 256
    # total = 64 * 4 * 256 * 1000 = 65,536,000
    result = mamba_kv_bytes(
        max_active_seqs=4, max_seq_len=4096, layers=64,
        state_bytes_per_layer=state_per_layer, block_size=16,
    )
    expected = 64 * 4 * 256 * 1000
    assert result == expected


def test_pure_mamba1_orchestrator():
    """Orchestrator detects and handles pure Mamba1 model."""
    cfg = PureMamba1Config()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 4096, quant_spec, block_size=16)
    assert result.spec_type == "mamba"
    assert result.total_bytes > 0


def test_pure_mamba2_orchestrator():
    """Orchestrator detects and handles pure Mamba2 model."""
    cfg = PureMamba2Config()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 4096, quant_spec, block_size=16)
    assert result.spec_type == "mamba"
    assert result.total_bytes > 0


# ---------------------------------------------------------------------------
# Formula tests — hybrid models
# ---------------------------------------------------------------------------

def test_jamba_hybrid_has_attn_and_mamba_groups():
    """Jamba hybrid should produce groups for both attention and mamba layers."""
    cfg = JambaConfig()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 4096, quant_spec, block_size=16)
    assert result.spec_type == "hybrid"
    assert len(result.layer_groups) == 2
    types = {g.spec_type for g in result.layer_groups}
    assert types == {"full", "mamba"}
    # 4 attention + 4 mamba
    by_type = {g.spec_type: g for g in result.layer_groups}
    assert by_type["full"].num_layers == 4
    assert by_type["mamba"].num_layers == 4


def test_jamba_hybrid_total_is_sum_of_groups():
    """Total should equal sum of group bytes."""
    cfg = JambaConfig()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 4096, quant_spec, block_size=16)
    group_sum = sum(g.bytes for g in result.layer_groups)
    assert abs(result.total_bytes - group_sum) < 1


def test_llama4_hybrid_has_full_and_chunked_groups():
    """LLaMA-4 hybrid should have full attention and chunked local groups."""
    cfg = Llama4HybridConfig()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 131072, quant_spec, block_size=16)
    assert result.spec_type == "hybrid"
    assert len(result.layer_groups) == 2
    types = {g.spec_type for g in result.layer_groups}
    assert types == {"full", "chunked_local"}
    by_type = {g.spec_type: g for g in result.layer_groups}
    assert by_type["full"].num_layers == 6
    assert by_type["chunked_local"].num_layers == 2


def test_llama4_hybrid_less_than_all_full():
    """Hybrid should use less memory than if all layers were full attention."""
    cfg_hybrid = Llama4HybridConfig()
    cfg_full = StandardConfig()
    cfg_full.num_hidden_layers = 8  # match layer count

    quant_spec = parse_quantization(cfg_hybrid)
    hybrid = estimate_kv_cache_bytes_specaware(
        cfg_hybrid, 4, 131072, quant_spec, block_size=16,
    )
    full = estimate_kv_cache_bytes_specaware(
        cfg_full, 4, 131072, quant_spec, block_size=16,
    )
    assert hybrid.total_bytes < full.total_bytes


def test_bamba_hybrid_has_attn_and_mamba2():
    """Bamba should produce groups with mamba2-style state."""
    cfg = BambaConfig()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 4096, quant_spec, block_size=16)
    assert result.spec_type == "hybrid"
    assert len(result.layer_groups) == 2
    by_type = {g.spec_type: g for g in result.layer_groups}
    assert by_type["full"].num_layers == 2
    assert by_type["mamba"].num_layers == 6


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

def test_orchestrator_full():
    cfg = StandardConfig()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 4096, quant_spec, block_size=16)
    assert result.spec_type == "full"
    assert result.total_bytes > 0


def test_orchestrator_sliding_window():
    cfg = SlidingWindowConfig()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 32768, quant_spec, block_size=16)
    assert result.spec_type == "sliding_window"
    assert result.total_bytes > 0


def test_orchestrator_mla():
    cfg = MLAConfig()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 4096, quant_spec, block_size=16)
    assert result.spec_type == "mla"
    assert result.total_bytes > 0


def test_orchestrator_chunked_local():
    cfg = ChunkedLocalConfig()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(cfg, 4, 131072, quant_spec, block_size=16)
    assert result.spec_type == "chunked_local"
    assert result.total_bytes > 0


def test_orchestrator_passes_max_num_batched_tokens():
    """Sliding window result should change with different max_num_batched_tokens."""
    cfg = SlidingWindowConfig()
    quant_spec = parse_quantization(cfg)
    r1 = estimate_kv_cache_bytes_specaware(cfg, 4, 32768, quant_spec, block_size=16,
                                           max_num_batched_tokens=2048)
    r2 = estimate_kv_cache_bytes_specaware(cfg, 4, 32768, quant_spec, block_size=16,
                                           max_num_batched_tokens=8192)
    assert r2.total_bytes > r1.total_bytes


# ---------------------------------------------------------------------------
# Integration: build_memory_buckets uses spec-aware estimation
# ---------------------------------------------------------------------------

def test_build_memory_buckets_reports_spec_type():
    from memory_estimator.buckets import build_memory_buckets

    # Standard model at 32K context
    cfg = StandardConfig()
    quant_spec = parse_quantization(cfg)
    buckets = build_memory_buckets(
        cfg, parameter_bytes=1_000_000, max_active_seqs=4,
        max_seq_len=32768, quant_spec=quant_spec,
    )
    assert buckets.kv_cache_spec_type == "full"

    # Sliding window model at same 32K context
    cfg_sw = SlidingWindowConfig()
    quant_spec_sw = parse_quantization(cfg_sw)
    buckets_sw = build_memory_buckets(
        cfg_sw, parameter_bytes=1_000_000, max_active_seqs=4,
        max_seq_len=32768, quant_spec=quant_spec_sw,
    )
    assert buckets_sw.kv_cache_spec_type == "sliding_window"
    # SW KV cache should be much smaller than full for 32K context
    assert buckets_sw.kv_cache_bytes < buckets.kv_cache_bytes


def test_build_memory_buckets_hybrid():
    from memory_estimator.buckets import build_memory_buckets

    cfg = JambaConfig()
    quant_spec = parse_quantization(cfg)
    buckets = build_memory_buckets(
        cfg, parameter_bytes=1_000_000, max_active_seqs=4,
        max_seq_len=4096, quant_spec=quant_spec,
    )
    assert buckets.kv_cache_spec_type == "hybrid"
    assert buckets.kv_cache_bytes > 0


def test_mamba_tp_no_double_division():
    """TP division should only happen once in build_memory_buckets, not in state functions.

    Regression test: previously mamba state functions divided by TP internally,
    then build_memory_buckets divided by tp*pp again, causing double-division.
    """
    from memory_estimator.buckets import build_memory_buckets

    cfg = PureMamba1Config()
    quant_spec = parse_quantization(cfg)

    tp1 = build_memory_buckets(
        cfg, parameter_bytes=1_000_000, max_active_seqs=4,
        max_seq_len=4096, quant_spec=quant_spec,
        tensor_parallel_size=1,
    )
    tp2 = build_memory_buckets(
        cfg, parameter_bytes=1_000_000, max_active_seqs=4,
        max_seq_len=4096, quant_spec=quant_spec,
        tensor_parallel_size=2,
    )
    # With TP=2, KV cache per GPU should be exactly half of TP=1
    assert abs(tp2.kv_cache_bytes - tp1.kv_cache_bytes / 2) < 1


def test_orchestrator_fp8_mla():
    """FP8 MLA should use the fixed 656 bytes/token layout via the orchestrator."""
    from memory_estimator.dtype_utils import normalise_dtype
    from memory_estimator.quantization import QuantizationSpec

    cfg = MLAConfig()
    fp8_quant = QuantizationSpec(
        method=None,
        weight_dtype=normalise_dtype("bfloat16"),
        activation_dtype=normalise_dtype("bfloat16"),
        kv_cache_dtype=normalise_dtype("fp8"),
        kv_cache_scale_dtype=None,
    )
    result = estimate_kv_cache_bytes_specaware(
        cfg, 4, 4096, fp8_quant, block_size=16,
    )
    assert result.spec_type == "mla"
    # Verify it used the fp8 formula: page = 16 * 656 = 10496
    # pages_per_seq = ceil(4096/16) = 256
    # total = 60 * 4 * 256 * 10496
    expected = 60 * 4 * 256 * (16 * 656)
    assert result.total_bytes == expected


def test_orchestrator_bf16_mla_not_fp8():
    """BF16 MLA should NOT use the FP8 fixed layout."""
    cfg = MLAConfig()
    quant_spec = parse_quantization(cfg)
    result = estimate_kv_cache_bytes_specaware(
        cfg, 4, 4096, quant_spec, block_size=16,
    )
    # BF16 MLA: page = 16 * 1 * 576 * 2 = 18432
    expected = 60 * 4 * 256 * (16 * 1 * 576 * 2)
    assert result.total_bytes == expected
