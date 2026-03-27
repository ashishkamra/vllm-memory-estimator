import pytest

from memory_estimator.vllm_cmd_parser import parse_vllm_command


def test_parse_full_command():
    inputs = parse_vllm_command(
        "vllm serve facebook/opt-125m --max-model-len 2048 --max-num-seqs 4"
    )
    assert inputs.model_id == "facebook/opt-125m"
    assert inputs.max_seq_len == 2048
    assert inputs.max_active_seqs == 4


def test_parse_without_vllm_serve_prefix():
    inputs = parse_vllm_command("facebook/opt-125m --max-model-len 2048")
    assert inputs.model_id == "facebook/opt-125m"
    assert inputs.max_seq_len == 2048


def test_parse_model_via_flag():
    inputs = parse_vllm_command("--model facebook/opt-125m --max-model-len 2048")
    assert inputs.model_id == "facebook/opt-125m"


def test_parse_kv_cache_dtype():
    inputs = parse_vllm_command(
        "vllm serve m --max-model-len 2048 --kv-cache-dtype fp8"
    )
    assert inputs.kv_cache_dtype == "fp8"


def test_parse_tensor_parallel():
    inputs = parse_vllm_command(
        "vllm serve m --max-model-len 2048 -tp 4"
    )
    assert inputs.tensor_parallel_size == 4


def test_parse_enforce_eager():
    inputs = parse_vllm_command(
        "vllm serve m --max-model-len 2048 --enforce-eager"
    )
    assert inputs.enforce_eager is True


def test_parse_defaults():
    inputs = parse_vllm_command("vllm serve m --max-model-len 2048")
    assert inputs.max_active_seqs == 256
    assert inputs.tensor_parallel_size == 1
    assert inputs.pipeline_parallel_size == 1
    assert inputs.data_parallel_size == 1
    assert inputs.enable_expert_parallel is False
    assert inputs.enforce_eager is False
    assert inputs.kv_cache_dtype is None
    assert inputs.dtype is None
    assert inputs.block_size is None


def test_unknown_flags_ignored():
    inputs = parse_vllm_command(
        "vllm serve m --max-model-len 2048 --host 0.0.0.0 --port 8000 --api-key secret"
    )
    assert inputs.model_id == "m"
    assert inputs.max_seq_len == 2048


def test_missing_model_raises():
    with pytest.raises(ValueError, match="No model"):
        parse_vllm_command("--max-model-len 2048")


def test_missing_max_model_len_defaults_to_none():
    inputs = parse_vllm_command("vllm serve facebook/opt-125m")
    assert inputs.max_seq_len is None


def test_parse_block_size():
    inputs = parse_vllm_command(
        "vllm serve m --max-model-len 2048 --block-size 32"
    )
    assert inputs.block_size == 32


def test_parse_pipeline_parallel():
    inputs = parse_vllm_command(
        "vllm serve m --max-model-len 2048 -pp 4"
    )
    assert inputs.pipeline_parallel_size == 4


def test_parse_data_parallel():
    inputs = parse_vllm_command(
        "vllm serve m --max-model-len 2048 --data-parallel-size 2"
    )
    assert inputs.data_parallel_size == 2


def test_parse_expert_parallel():
    inputs = parse_vllm_command(
        "vllm serve m --max-model-len 2048 --enable-expert-parallel"
    )
    assert inputs.enable_expert_parallel is True


def test_parse_quantization():
    inputs = parse_vllm_command(
        "vllm serve m --max-model-len 2048 -q awq"
    )
    assert inputs.model_id == "m"
