from memory_estimator.cli import _build_arg_parser


def test_cli_parses_command_string():
    parser = _build_arg_parser()
    parsed = parser.parse_args([
        "vllm serve facebook/opt-125m --max-model-len 2048 --max-num-seqs 4"
    ])
    assert parsed.command == "vllm serve facebook/opt-125m --max-model-len 2048 --max-num-seqs 4"
    assert parsed.json is False


def test_cli_parses_json_flag():
    parser = _build_arg_parser()
    parsed = parser.parse_args([
        "vllm serve facebook/opt-125m --max-model-len 2048",
        "--json",
    ])
    assert parsed.json is True
