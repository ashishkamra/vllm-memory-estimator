from transformers import PretrainedConfig

from memory_estimator.quantization import parse_quantization


class DummyConfig(PretrainedConfig):
    model_type = "dummy"

    def __init__(self):
        super().__init__()
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 64
        self.quantization_config = {
            "quant_method": "awq",
            "bits": 4,
            "group_size": 128,
            "activation_dtype": "float16",
            "scale_dtype": "float16",
            "zero_point_dtype": "int8",
            "kv_cache_dtype": "float8_e4m3fn",
            "kv_cache_scaling_dtype": "float16",
        }


def test_parse_quantization_awq():
    cfg = DummyConfig()
    spec = parse_quantization(cfg)
    assert spec.method == "awq"
    assert spec.weight_bits == 4
    assert spec.group_size == 128
    assert spec.scale_dtype and spec.scale_dtype.name == "float16"
    assert spec.zero_dtype and spec.zero_dtype.name == "int8"
    assert spec.kv_cache_dtype.name == "float8_e4m3fn"
    assert spec.kv_cache_scale_dtype and spec.kv_cache_scale_dtype.name == "float16"


class CompressedTensorConfig(PretrainedConfig):
    model_type = "phi3"

    def __init__(self):
        super().__init__()
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.intermediate_size = 64
        self.torch_dtype = "bfloat16"
        self.quantization_config = {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "input_activations": {
                        "type": "float",
                        "num_bits": 8,
                    },
                    "weights": {
                        "type": "float",
                        "num_bits": 8,
                        "group_size": None,
                    }
                }
            },
        }


def test_parse_quantization_compressed_tensors():
    cfg = CompressedTensorConfig()
    spec = parse_quantization(cfg)
    assert spec.method == "compressed-tensors"
    assert spec.weight_bits == 8
    assert spec.weight_dtype.bits == 8
    assert spec.activation_dtype.bits == 8
    assert spec.activation_bits == 8
