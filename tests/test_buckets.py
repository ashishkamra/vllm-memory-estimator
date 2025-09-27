from memory_estimator.buckets import build_memory_buckets
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
