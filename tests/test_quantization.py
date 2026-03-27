from memory_estimator.quantization import parse_quantization


class DummyConfig:
    model_type = "dummy"
    hidden_size = 16
    num_hidden_layers = 2
    num_attention_heads = 4
    intermediate_size = 64
    quantization_config = {
        "quant_method": "awq",
        "bits": 4,
        "activation_dtype": "float16",
        "kv_cache_dtype": "float8_e4m3fn",
        "kv_cache_scaling_dtype": "float16",
    }


def test_parse_quantization_awq():
    cfg = DummyConfig()
    spec = parse_quantization(cfg)
    assert spec.method == "awq"
    assert spec.is_quantized
    assert spec.weight_dtype.bits == 4
    assert spec.activation_dtype.name == "float16"
    assert spec.kv_cache_dtype.name == "float8_e4m3fn"
    assert spec.kv_cache_scale_dtype and spec.kv_cache_scale_dtype.name == "float16"


class CompressedTensorConfig:
    model_type = "phi3"
    hidden_size = 16
    num_hidden_layers = 2
    num_attention_heads = 4
    intermediate_size = 64
    torch_dtype = "bfloat16"
    quantization_config = {
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
                },
            }
        },
    }


def test_parse_quantization_compressed_tensors():
    cfg = CompressedTensorConfig()
    spec = parse_quantization(cfg)
    assert spec.method == "compressed-tensors"
    assert spec.is_quantized
    assert spec.weight_dtype.bits == 8
    assert spec.activation_dtype.bits == 8


class DenseConfig:
    model_type = "llama"
    hidden_size = 16
    num_hidden_layers = 2
    num_attention_heads = 4
    intermediate_size = 64
    torch_dtype = "bfloat16"


def test_cli_quantization_override_fp8():
    """CLI -q fp8 should produce fp8 quantization when config has none."""
    cfg = DenseConfig()
    spec = parse_quantization(cfg, cli_quantization="fp8")
    assert spec.method == "fp8"
    assert spec.is_quantized
    assert spec.weight_dtype.bits == 8


def test_cli_quantization_override_awq():
    cfg = DenseConfig()
    spec = parse_quantization(cfg, cli_quantization="awq")
    assert spec.method == "awq"
    assert spec.weight_dtype.bits == 4


def test_cli_quantization_ignored_when_config_has_quant():
    """Config quantization takes precedence over CLI override."""
    cfg = DummyConfig()
    spec = parse_quantization(cfg, cli_quantization="fp8")
    # Config says awq/4-bit, not fp8/8-bit
    assert spec.method == "awq"
    assert spec.weight_dtype.bits == 4
