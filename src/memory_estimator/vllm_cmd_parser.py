"""Parse a ``vllm serve`` command string into EstimatorInputs."""

from __future__ import annotations

import argparse
import shlex

from .estimator import EstimatorInputs


def _build_vllm_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    parser.add_argument("model", nargs="?", default=None)
    parser.add_argument("--model", dest="model_flag", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--kv-cache-dtype", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--quantization", "-q", default=None)
    return parser


def parse_vllm_command(cmd: str) -> EstimatorInputs:
    """Parse a ``vllm serve`` command string into :class:`EstimatorInputs`.

    Strips leading ``vllm`` and ``serve`` tokens if present.  Unknown flags
    (e.g. ``--host``, ``--port``) are silently ignored.
    """
    tokens = shlex.split(cmd)

    # Strip leading "vllm" and "serve"
    while tokens and tokens[0] in ("vllm", "serve"):
        tokens.pop(0)

    parser = _build_vllm_parser()
    parsed, _ = parser.parse_known_args(tokens)

    model_id = parsed.model_flag or parsed.model
    if not model_id:
        raise ValueError("No model specified in vllm serve command")

    return EstimatorInputs(
        model_id=model_id,
        max_seq_len=parsed.max_model_len,
        max_active_seqs=parsed.max_num_seqs,
        revision=parsed.revision,
        enforce_eager=parsed.enforce_eager,
        cudagraph_capture_sizes=None,
        max_num_batched_tokens=parsed.max_num_batched_tokens,
        kv_cache_dtype=parsed.kv_cache_dtype,
        dtype=parsed.dtype,
        tensor_parallel_size=parsed.tensor_parallel_size,
        block_size=parsed.block_size,
    )
