from transformers import PretrainedConfig

from memory_estimator.buckets import build_memory_buckets
from memory_estimator.model_shapes import ParameterShape
from memory_estimator.quantization import parse_quantization


class TinyConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self):
        super().__init__()
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 64
        self.torch_dtype = "float16"


class NestedWrapperConfig(PretrainedConfig):
    model_type = "nested"

    def __init__(self):
        super().__init__()
        self.text_config = TinyConfig()
        # Provide minimal quantization metadata so the parser works.
        self.quantization_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "activation_dtype": "float16",
        }


SHAPES = [
    ParameterShape(name="layer.0.attn.weight", shape=(32, 32)),
    ParameterShape(name="layer.0.norm.weight", shape=(32, )),
    ParameterShape(name="layer.1.attn.weight", shape=(32, 32)),
]


def test_build_memory_buckets_dense():
    cfg = TinyConfig()
    quant_spec = parse_quantization(cfg)
    buckets = build_memory_buckets(cfg, SHAPES, max_active_seqs=2, max_seq_len=128,
                                   quant_spec=quant_spec)
    assert buckets.parameter_bytes > 0
    assert buckets.activation_bytes > buckets.parameter_bytes * 0.1
    assert buckets.kv_cache_bytes > 0
    assert buckets.total_bytes > buckets.parameter_bytes


def test_build_memory_buckets_nested_text_config():
    cfg = NestedWrapperConfig()
    quant_spec = parse_quantization(cfg)
    buckets = build_memory_buckets(cfg, SHAPES, max_active_seqs=2, max_seq_len=128,
                                   quant_spec=quant_spec)
    assert buckets.activation_bytes > 0
