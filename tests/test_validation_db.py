"""Tests for the validation database builder."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from memory_estimator.validation_db import CSVMetadata
from memory_estimator.validation_db import LogConfigData
from memory_estimator.validation_db import build_estimator_inputs_record
from memory_estimator.validation_db import build_validation_db
from memory_estimator.validation_db import cross_validate
from memory_estimator.validation_db import parse_csv_metadata
from memory_estimator.validation_db import parse_log_config
from memory_estimator.validation_db import parse_log_memory
from memory_estimator.validation_db import parse_runtime_args
from memory_estimator.validation_db import should_include
from memory_estimator.validation_db import strip_ansi

# ---------------------------------------------------------------------------
# Real log snippets (with ANSI escape codes preserved)
# ---------------------------------------------------------------------------

LLAMA_CONFIG_LINE = (
    "\x1b[1;36m(EngineCore_DP0 pid=279)\x1b[0;0m INFO 12-02 15:47:51"
    " [core.py:93] Initializing a V1 LLM engine (v0.11.2rc1+rhai0.cuda)"
    " with config: model='RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic',"
    " speculative_config=None,"
    " tokenizer='RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic',"
    " skip_tokenizer_init=False, tokenizer_mode=auto, revision=None,"
    " tokenizer_revision=None, trust_remote_code=True,"
    " dtype=torch.bfloat16, max_seq_len=8192, download_dir=None,"
    " load_format=auto, tensor_parallel_size=4,"
    " pipeline_parallel_size=1, data_parallel_size=1,"
    " disable_custom_all_reduce=False,"
    " quantization=compressed-tensors, enforce_eager=False,"
    " kv_cache_dtype=auto, device_config=cuda,"
    " enable_prefix_caching=False, enable_chunked_prefill=True,"
)

DEEPSEEK_CONFIG_LINE = (
    "(EngineCore_DP0 pid=265) INFO 12-03 17:30:13 [core.py:93]"
    " Initializing a V1 LLM engine (v0.11.2) with config:"
    " model='deepseek-ai/DeepSeek-R1-0528',"
    " speculative_config=None,"
    " tokenizer='deepseek-ai/DeepSeek-R1-0528',"
    " skip_tokenizer_init=False, tokenizer_mode=auto, revision=None,"
    " tokenizer_revision=None, trust_remote_code=True,"
    " dtype=torch.bfloat16, max_seq_len=8192, download_dir=None,"
    " load_format=auto, tensor_parallel_size=8,"
    " pipeline_parallel_size=1, data_parallel_size=1,"
    " disable_custom_all_reduce=False,"
    " quantization=fp8, enforce_eager=False,"
    " kv_cache_dtype=auto, device_config=cuda,"
    " enable_prefix_caching=False, enable_chunked_prefill=True,"
)

MEMORY_LINES = textwrap.dedent("""\
    \x1b[1;36m(Worker_TP0 pid=415)\x1b[0;0m INFO 12-02 15:48:37 [gpu_model_runner.py:3338] Model loading took 16.9619 GiB memory and 33.033432 seconds
    \x1b[1;36m(Worker_TP0 pid=415)\x1b[0;0m INFO 12-02 15:49:26 [gpu_worker.py:359] Available KV cache memory: 106.17 GiB
    \x1b[1;36m(EngineCore_DP0 pid=279)\x1b[0;0m INFO 12-02 15:49:27 [kv_cache_utils.py:1229] GPU KV cache size: 1,391,552 tokens
    \x1b[1;36m(EngineCore_DP0 pid=279)\x1b[0;0m INFO 12-02 15:49:27 [kv_cache_utils.py:1234] Maximum concurrency for 8,192 tokens per request: 169.87x
    \x1b[1;36m(Worker_TP0 pid=415)\x1b[0;0m INFO 12-02 15:49:35 [gpu_model_runner.py:4244] Graph capturing finished in 8 secs, took -0.17 GiB
""")

DEEPSEEK_MEMORY_LINES = textwrap.dedent("""\
    (Worker_TP0 pid=402) INFO 12-03 17:35:28 [gpu_model_runner.py:3338] Model loading took 79.7390 GiB memory and 263.495895 seconds
    (Worker_TP0 pid=402) INFO 12-03 17:37:45 [gpu_worker.py:359] Available KV cache memory: 41.60 GiB
    (EngineCore_DP0 pid=265) INFO 12-03 17:37:46 [kv_cache_utils.py:1229] GPU KV cache size: 635,648 tokens
    (EngineCore_DP0 pid=265) INFO 12-03 17:37:46 [kv_cache_utils.py:1234] Maximum concurrency for 8,192 tokens per request: 77.59x
    (Worker_TP0 pid=402) INFO 12-03 17:50:07 [gpu_model_runner.py:4244] Graph capturing finished in 35 secs, took 0.51 GiB
""")


# ---------------------------------------------------------------------------
# ANSI stripping
# ---------------------------------------------------------------------------

def test_strip_ansi_removes_codes():
    raw = "\x1b[1;36m(Worker_TP0 pid=415)\x1b[0;0m INFO loading"
    assert strip_ansi(raw) == "(Worker_TP0 pid=415) INFO loading"


def test_strip_ansi_noop_on_clean():
    text = "Model loading took 16.96 GiB memory"
    assert strip_ansi(text) == text


# ---------------------------------------------------------------------------
# Log config parser
# ---------------------------------------------------------------------------

def test_parse_log_config_llama():
    cfg = parse_log_config(LLAMA_CONFIG_LINE)
    assert cfg is not None
    assert cfg.model == "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
    assert cfg.dtype == "bfloat16"
    assert cfg.max_seq_len == 8192
    assert cfg.tensor_parallel_size == 4
    assert cfg.pipeline_parallel_size == 1
    assert cfg.data_parallel_size == 1
    assert cfg.quantization == "compressed-tensors"
    assert cfg.kv_cache_dtype == "auto"
    assert cfg.enforce_eager is False
    assert cfg.enable_prefix_caching is False
    assert cfg.enable_chunked_prefill is True
    assert cfg.vllm_version == "v0.11.2rc1+rhai0.cuda"


def test_parse_log_config_deepseek():
    cfg = parse_log_config(DEEPSEEK_CONFIG_LINE)
    assert cfg is not None
    assert cfg.model == "deepseek-ai/DeepSeek-R1-0528"
    assert cfg.quantization == "fp8"
    assert cfg.tensor_parallel_size == 8
    assert cfg.vllm_version == "v0.11.2"


def test_parse_log_config_returns_none_for_garbage():
    assert parse_log_config("some random log output\nno config here") is None


def test_parse_log_config_quantization_none():
    line = (
        "Initializing a V1 LLM engine (v0.16.0) with config:"
        " model='meta-llama/Llama-3.3-70B-Instruct',"
        " dtype=torch.bfloat16, max_seq_len=4096,"
        " tensor_parallel_size=4, pipeline_parallel_size=1,"
        " data_parallel_size=1, quantization=None,"
        " enforce_eager=True, kv_cache_dtype=auto,"
        " enable_prefix_caching=False, enable_chunked_prefill=True,"
    )
    cfg = parse_log_config(line)
    assert cfg is not None
    assert cfg.quantization is None
    assert cfg.enforce_eager is True


# ---------------------------------------------------------------------------
# Log memory parser
# ---------------------------------------------------------------------------

def test_parse_log_memory_all_fields():
    mem = parse_log_memory(MEMORY_LINES)
    assert mem is not None
    assert mem.model_load_gib == pytest.approx(16.9619)
    assert mem.available_kv_cache_gib == pytest.approx(106.17)
    assert mem.kv_cache_tokens == 1_391_552
    assert mem.max_concurrency_tokens == 8_192
    assert mem.max_concurrency_ratio == pytest.approx(169.87)
    assert mem.cudagraph_gib == pytest.approx(-0.17)


def test_parse_log_memory_deepseek():
    mem = parse_log_memory(DEEPSEEK_MEMORY_LINES)
    assert mem is not None
    assert mem.model_load_gib == pytest.approx(79.739)
    assert mem.kv_cache_tokens == 635_648
    assert mem.cudagraph_gib == pytest.approx(0.51)


def test_parse_log_memory_returns_none_without_model_load():
    assert parse_log_memory("Available KV cache memory: 100.0 GiB") is None


def test_parse_log_memory_minimal():
    text = "Model loading took 5.50 GiB memory and 10.0 seconds"
    mem = parse_log_memory(text)
    assert mem is not None
    assert mem.model_load_gib == pytest.approx(5.5)
    assert mem.available_kv_cache_gib is None
    assert mem.kv_cache_tokens is None
    assert mem.cudagraph_gib is None


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def test_parse_csv_metadata(tmp_path):
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "run,accelerator,model,TP,DP,uuid,runtime_args\n"
        "run1,H200,meta-llama/Llama-3.3-70B-Instruct,4.0,,uuid-1,max-model-len: 8192\n"
        "run2,H200,meta-llama/Llama-3.3-70B-Instruct,4.0,,uuid-1,max-model-len: 8192\n"
        "run3,MI300X,deepseek-ai/DeepSeek-R1,8.0,1.0,uuid-2,max-model-len: 4096\n"
    )
    result = parse_csv_metadata(csv_file)
    assert len(result) == 2
    assert result["uuid-1"].model == "meta-llama/Llama-3.3-70B-Instruct"
    assert result["uuid-1"].tp == 4
    assert result["uuid-1"].dp is None
    assert result["uuid-2"].accelerator == "MI300X"
    assert result["uuid-2"].dp == 1


# ---------------------------------------------------------------------------
# Runtime args parser
# ---------------------------------------------------------------------------

def test_parse_runtime_args_colon_format():
    args = "max-model-len: 8192; tensor-parallel-size: 8; trust-remote-code: True"
    result = parse_runtime_args(args)
    assert result["max-model-len"] == "8192"
    assert result["tensor-parallel-size"] == "8"
    assert result["trust-remote-code"] == "True"


def test_parse_runtime_args_equals_format():
    args = "trust-remote-code=True;max-model-len=10240;tensor-parallel-size=4"
    result = parse_runtime_args(args)
    assert result["max-model-len"] == "10240"
    assert result["tensor-parallel-size"] == "4"


def test_parse_runtime_args_empty():
    assert parse_runtime_args("") == {}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _csv_meta(model: str = "meta-llama/Llama-3.3-70B", accel: str = "H200") -> CSVMetadata:
    return CSVMetadata(model=model, accelerator=accel, tp=4, dp=None, runtime_args="")


def _log_config(model: str = "meta-llama/Llama-3.3-70B") -> LogConfigData:
    return LogConfigData(
        model=model, dtype="bfloat16", max_seq_len=4096,
        tensor_parallel_size=4, pipeline_parallel_size=1, data_parallel_size=1,
        quantization=None, kv_cache_dtype="auto", enforce_eager=False,
        enable_prefix_caching=False, enable_chunked_prefill=True,
    )


def _log_memory():
    from memory_estimator.validation_db import LogMemoryData
    return LogMemoryData(model_load_gib=10.0)


def test_should_include_valid():
    ok, reason = should_include(_csv_meta(), _log_config(), _log_memory())
    assert ok is True
    assert reason == ""


def test_should_exclude_tpu():
    ok, reason = should_include(_csv_meta(accel="TPU"), _log_config(), _log_memory())
    assert ok is False
    assert "TPU" in reason


def test_should_exclude_spyre():
    ok, reason = should_include(_csv_meta(accel="Spyre"), _log_config(), _log_memory())
    assert ok is False
    assert "Spyre" in reason


def test_should_exclude_openai():
    ok, reason = should_include(
        _csv_meta(model="openai/gpt-oss-120b"), _log_config(), _log_memory(),
    )
    assert ok is False
    assert "openai" in reason


def test_should_exclude_bart():
    ok, reason = should_include(
        _csv_meta(model="bart-large-cnn"), _log_config(), _log_memory(),
    )
    assert ok is False


def test_should_exclude_whisper():
    ok, reason = should_include(
        _csv_meta(model="openai/whisper-large-v3"), _log_config(), _log_memory(),
    )
    assert ok is False


def test_should_exclude_no_config():
    ok, reason = should_include(_csv_meta(), None, _log_memory())
    assert ok is False
    assert "no_config" in reason


def test_should_exclude_no_memory():
    ok, reason = should_include(_csv_meta(), _log_config(), None)
    assert ok is False
    assert "no_memory" in reason


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def test_cross_validate_matching():
    warnings = cross_validate(
        "test-uuid",
        _csv_meta(model="meta-llama/Llama-3.3-70B"),
        _log_config(model="meta-llama/Llama-3.3-70B"),
    )
    assert warnings == []


def test_cross_validate_subset_match():
    warnings = cross_validate(
        "test-uuid",
        _csv_meta(model="meta-llama/Llama-3.3-70B-Instruct"),
        _log_config(model="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"),
    )
    assert warnings == []


def test_cross_validate_mismatch():
    warnings = cross_validate(
        "test-uuid",
        _csv_meta(model="meta-llama/Llama-3.3-70B"),
        _log_config(model="deepseek-ai/DeepSeek-R1"),
    )
    assert len(warnings) == 1
    assert "CSV model" in warnings[0]


def test_cross_validate_tp_mismatch():
    meta = _csv_meta()
    meta.tp = 8
    cfg = _log_config()
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["tensor_parallel_size"] = 4
    warnings = cross_validate("test-uuid", meta, LogConfigData(**cfg_dict))
    assert any("TP" in w for w in warnings)


# ---------------------------------------------------------------------------
# EstimatorInputs reconstruction
# ---------------------------------------------------------------------------

def test_build_estimator_inputs_auto_kv():
    cfg = _log_config()
    record = build_estimator_inputs_record(cfg)
    assert record.kv_cache_dtype is None
    assert record.quantization is None
    assert record.dtype == "bfloat16"


def test_build_estimator_inputs_fp8_kv():
    cfg = LogConfigData(
        model="model/x", dtype="torch.bfloat16", max_seq_len=4096,
        tensor_parallel_size=4, pipeline_parallel_size=1, data_parallel_size=1,
        quantization="fp8", kv_cache_dtype="fp8", enforce_eager=False,
        enable_prefix_caching=False, enable_chunked_prefill=True,
    )
    record = build_estimator_inputs_record(cfg)
    assert record.kv_cache_dtype == "fp8"
    assert record.quantization == "fp8"
    assert record.dtype == "bfloat16"


# ---------------------------------------------------------------------------
# Integration: build from real log files
# ---------------------------------------------------------------------------

_LOCAL_LOG_DIR = Path("/home/akamra/workspace/vllm_memory_comparison/gpu-calc/.cli-logs")


@pytest.mark.skipif(
    not _LOCAL_LOG_DIR.exists(),
    reason="Local log directory not available",
)
def test_parse_real_log_file():
    """Parse a real log file and verify all fields are extracted."""
    from memory_estimator.validation_db import parse_log_file

    log_path = _LOCAL_LOG_DIR / "60aeae69-b737-44c2-bc2d-c7d1b09e8059.log"
    if not log_path.exists():
        pytest.skip("Expected log file not found")

    cfg, mem = parse_log_file(log_path)
    assert cfg is not None
    assert cfg.model == "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
    assert cfg.tensor_parallel_size == 4
    assert cfg.quantization == "compressed-tensors"
    assert mem is not None
    assert mem.model_load_gib == pytest.approx(16.9619)
    assert mem.available_kv_cache_gib == pytest.approx(106.17)
    assert mem.kv_cache_tokens == 1_391_552


@pytest.mark.skipif(
    not _LOCAL_LOG_DIR.exists(),
    reason="Local log directory not available",
)
def test_build_validation_db_with_real_data(tmp_path):
    """End-to-end build using real CSV and local log files."""
    csv_path = Path(__file__).resolve().parent.parent / "test_data" / "consolidated_dashboard.csv"
    if not csv_path.exists():
        pytest.skip("CSV file not found")

    output = tmp_path / "test_db.json"
    db = build_validation_db(
        csv_path=csv_path,
        log_dir=_LOCAL_LOG_DIR,
        output_path=output,
    )

    assert output.exists()
    assert db["metadata"]["records_included"] > 200
    assert db["metadata"]["records_included"] < 300

    data = json.loads(output.read_text())
    assert len(data["records"]) == db["metadata"]["records_included"]

    sample = next(iter(data["records"].values()))
    assert "csv_metadata" in sample
    assert "log_config" in sample
    assert "log_memory" in sample
    assert "estimator_inputs" in sample
    assert sample["log_memory"]["model_load_gib"] > 0
