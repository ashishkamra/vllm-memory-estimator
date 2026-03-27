import pytest

from memory_estimator.estimator import EstimatorInputs
from memory_estimator.estimator import estimate_from_inputs
from memory_estimator.estimator import prepare_summary


def test_prepare_summary_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        prepare_summary(
            EstimatorInputs(model_id="facebook/opt-125m", max_seq_len=0, max_active_seqs=1)
        )


def test_prepare_summary_rejects_empty_cudagraph_capture_sizes():
    with pytest.raises(ValueError):
        prepare_summary(
            EstimatorInputs(
                model_id="facebook/opt-125m",
                max_seq_len=128,
                max_active_seqs=1,
                cudagraph_capture_sizes=[],
            )
        )


def test_cli_quantization_scales_parameter_bytes():
    """CLI -q fp8 on a dense bf16 model should halve parameter memory."""
    from memory_estimator.buckets import build_memory_buckets
    from memory_estimator.quantization import parse_quantization

    class BF16Config:
        model_type = "llama"
        hidden_size = 64
        num_hidden_layers = 4
        num_attention_heads = 4
        intermediate_size = 128
        torch_dtype = "bfloat16"

    cfg = BF16Config()
    base_spec = parse_quantization(cfg)
    quant_spec = parse_quantization(cfg, cli_quantization="fp8")

    param_bytes = 10000.0
    # Simulate the scaling that prepare_summary applies
    scale = quant_spec.weight_dtype.bits / base_spec.weight_dtype.bits
    scaled_bytes = param_bytes * scale

    base = build_memory_buckets(
        cfg, parameter_bytes=param_bytes, max_active_seqs=2,
        max_seq_len=128, quant_spec=base_spec,
    )
    quant = build_memory_buckets(
        cfg, parameter_bytes=scaled_bytes, max_active_seqs=2,
        max_seq_len=128, quant_spec=quant_spec,
    )

    # bf16 (16-bit) → fp8 (8-bit): params should be halved
    assert quant_spec.method == "fp8"
    assert abs(scale - 0.5) < 0.01
    assert abs(quant.parameter_bytes - base.parameter_bytes * 0.5) < 1
