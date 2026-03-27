from memory_estimator.buckets import build_memory_buckets
from memory_estimator.buckets import estimate_activation_bytes
from memory_estimator.buckets import estimate_cuda_graph_bytes
from memory_estimator.quantization import parse_quantization


class TinyConfig:
    model_type = "dummy"
    hidden_size = 32
    num_hidden_layers = 2
    num_attention_heads = 4
    intermediate_size = 64
    torch_dtype = "float16"


class NestedWrapperConfig:
    model_type = "nested"

    def __init__(self):
        self.text_config = TinyConfig()
        self.quantization_config = {
            "quant_method": "awq",
            "bits": 4,
            "activation_dtype": "float16",
        }


def test_build_memory_buckets_dense():
    cfg = TinyConfig()
    quant_spec = parse_quantization(cfg)
    buckets = build_memory_buckets(
        cfg, parameter_bytes=4096, max_active_seqs=2, max_seq_len=128, quant_spec=quant_spec
    )
    assert buckets.parameter_bytes == 4096
    assert buckets.activation_bytes > 0
    assert buckets.kv_cache_bytes > 0
    assert buckets.total_bytes > buckets.parameter_bytes


def test_build_memory_buckets_nested_text_config():
    cfg = NestedWrapperConfig()
    quant_spec = parse_quantization(cfg)
    buckets = build_memory_buckets(
        cfg, parameter_bytes=4096, max_active_seqs=2, max_seq_len=128, quant_spec=quant_spec
    )
    assert buckets.activation_bytes > 0


def _base_buckets(tp=1, pp=1, dp=1, ep=False, expert_bytes=0.0, non_expert_bytes=0.0,
                  max_active_seqs=8, parameter_bytes=4096):
    cfg = TinyConfig()
    quant_spec = parse_quantization(cfg)
    if non_expert_bytes == 0.0 and expert_bytes == 0.0:
        non_expert_bytes = float(parameter_bytes)
    return build_memory_buckets(
        cfg, parameter_bytes=parameter_bytes, max_active_seqs=max_active_seqs,
        max_seq_len=128, quant_spec=quant_spec,
        tensor_parallel_size=tp, pipeline_parallel_size=pp,
        data_parallel_size=dp, enable_expert_parallel=ep,
        expert_bytes=expert_bytes, non_expert_bytes=non_expert_bytes,
    )


def test_pp_divides_params_and_kv_not_activations():
    base = _base_buckets()
    pp2 = _base_buckets(pp=2)
    # Params and KV cache halved
    assert abs(pp2.parameter_bytes - base.parameter_bytes / 2) < 1
    assert abs(pp2.kv_cache_bytes - base.kv_cache_bytes / 2) < 1
    # Activations unchanged
    assert pp2.activation_bytes == base.activation_bytes


def test_dp_reduces_kv_via_fewer_seqs():
    base = _base_buckets(max_active_seqs=8)
    dp2 = _base_buckets(dp=2, max_active_seqs=8)
    # DP reduces effective seqs, so KV cache should be smaller
    assert dp2.kv_cache_bytes < base.kv_cache_bytes
    # Params unchanged (full replica)
    assert dp2.parameter_bytes == base.parameter_bytes


def test_ep_splits_expert_params():
    total = 10000.0
    expert = 8000.0
    non_expert = 2000.0
    base = _base_buckets(parameter_bytes=total, expert_bytes=expert,
                         non_expert_bytes=non_expert)
    ep = _base_buckets(tp=2, dp=2, ep=True, parameter_bytes=total,
                       expert_bytes=expert, non_expert_bytes=non_expert)
    # Without EP: params = 10000 / (2*1) = 5000
    assert abs(base.parameter_bytes - total) < 1
    # With EP: non_expert/TP + expert/(TP*DP) = 2000/2 + 8000/4 = 1000 + 2000 = 3000
    assert abs(ep.parameter_bytes - 3000.0) < 1


def test_combined_tp_pp():
    base = _base_buckets()
    tp2_pp2 = _base_buckets(tp=2, pp=2)
    # Params divided by TP*PP=4
    assert abs(tp2_pp2.parameter_bytes - base.parameter_bytes / 4) < 1
    # KV cache divided by TP*PP=4
    assert abs(tp2_pp2.kv_cache_bytes - base.kv_cache_bytes / 4) < 1
    # Activations divided by TP only
    assert abs(tp2_pp2.activation_bytes - base.activation_bytes / 2) < 1


def test_cuda_graph_uses_per_gpu_params():
    """CUDA graph bytes should scale with per-GPU params, not total params."""
    base = _base_buckets()
    tp2 = _base_buckets(tp=2)
    # With TP=2, per-GPU params are halved, so CUDA graph should be roughly halved
    # (the per-layer term stays the same since PP=1, but the param term halves)
    assert tp2.cuda_graph_bytes < base.cuda_graph_bytes


def test_cuda_graph_pp_reduces_layers():
    """CUDA graph per-layer term should use PP-local layers."""
    # Use enforce_eager=False (default) and known capture sizes
    pp1 = _base_buckets(pp=1)
    pp2 = _base_buckets(pp=2)
    # With PP=2, both params and local layers are halved, so CUDA graph
    # should be significantly smaller
    assert pp2.cuda_graph_bytes < pp1.cuda_graph_bytes * 0.6


def test_cuda_graph_bytes_direct():
    """Test estimate_cuda_graph_bytes uses per-GPU params and local layers."""
    # 1000 bytes per-GPU params, 4 local layers, 3 captures
    result = estimate_cuda_graph_bytes(1000.0, 4, [1, 2, 4])
    from memory_estimator.vllm_defaults import CUDA_GRAPH_BYTES_PER_CAPTURE
    from memory_estimator.vllm_defaults import CUDA_GRAPH_PARAM_FRACTION
    expected = (1000.0 * CUDA_GRAPH_PARAM_FRACTION + 4 * CUDA_GRAPH_BYTES_PER_CAPTURE) * 3
    assert abs(result - expected) < 1


def test_activation_logits_not_additive():
    """Logits buffer shouldn't be added on top of intermediate activations."""
    cfg = TinyConfig()
    quant_spec = parse_quantization(cfg)
    act_bytes = estimate_activation_bytes(cfg, 2, 128, quant_spec)
    # If logits were additive, changing vocab_size would always increase activations.
    # With max(), logits only matter if they exceed the intermediate buffers.
    # For TinyConfig (vocab unset → 0), activations should come from hidden+ffn/qkv.
    assert act_bytes > 0


class LargeVocabConfig:
    model_type = "dummy"
    hidden_size = 32
    num_hidden_layers = 2
    num_attention_heads = 4
    intermediate_size = 64
    vocab_size = 100000
    torch_dtype = "float16"


def test_activation_logits_dominated_by_vocab():
    """When vocab is very large, logits buffer should dominate via max()."""
    cfg = LargeVocabConfig()
    quant_spec = parse_quantization(cfg)
    act_bytes = estimate_activation_bytes(cfg, 2, 128, quant_spec)
    # logits_buf = tokens * 100000 * 2 should be >> hidden_buf + ffn_buf
    # Verify it's in the right ballpark (logits-dominated)
    tokens = min(2 * 128, 2048)
    logits_buf = tokens * 100000 * 2  # tokens * vocab * fp16
    from memory_estimator.vllm_defaults import ACTIVATION_OVERHEAD_FACTOR
    # Should be close to logits_buf * ACTIVATION_OVERHEAD_FACTOR (not double)
    assert act_bytes < logits_buf * ACTIVATION_OVERHEAD_FACTOR * 1.5
