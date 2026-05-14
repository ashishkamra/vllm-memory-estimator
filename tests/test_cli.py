from memory_estimator.cli import _build_parser


def test_cli_parses_command_string():
    parser = _build_parser()
    parsed = parser.parse_args([
        "estimate",
        "vllm serve facebook/opt-125m --max-model-len 2048 --max-num-seqs 4",
    ])
    assert parsed.command == "vllm serve facebook/opt-125m --max-model-len 2048 --max-num-seqs 4"
    assert parsed.json is False


def test_cli_parses_json_flag():
    parser = _build_parser()
    parsed = parser.parse_args([
        "estimate",
        "vllm serve facebook/opt-125m --max-model-len 2048",
        "--json",
    ])
    assert parsed.json is True


def test_cli_budget_parses_required_args():
    parser = _build_parser()
    parsed = parser.parse_args([
        "budget",
        "--model", "meta-llama/Llama-3.1-8B",
        "--gpu-memory-gib", "80",
    ])
    assert parsed.model == "meta-llama/Llama-3.1-8B"
    assert parsed.gpu_memory_gib == 80.0
    assert parsed.tensor_parallel_size == 1


def test_cli_budget_parses_tp_short():
    parser = _build_parser()
    parsed = parser.parse_args([
        "budget",
        "--model", "meta-llama/Llama-3.1-8B",
        "--gpu-memory-gib", "80",
        "--tp", "4",
    ])
    assert parsed.tensor_parallel_size == 4


def test_cli_budget_parses_seq_lengths():
    parser = _build_parser()
    parsed = parser.parse_args([
        "budget",
        "--model", "test/model",
        "--gpu-memory-gib", "80",
        "--seq-lengths", "1024,2048,4096",
    ])
    assert parsed.seq_lengths == "1024,2048,4096"
