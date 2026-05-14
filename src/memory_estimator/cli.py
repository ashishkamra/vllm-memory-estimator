"""Console entry point for the vLLM memory estimator."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from .budget import compute_budget
from .estimator import estimate_from_inputs
from .vllm_cmd_parser import parse_vllm_command

# ---------------------------------------------------------------------------
# estimate subcommand
# ---------------------------------------------------------------------------

def _run_estimate(parsed: argparse.Namespace) -> int:
    inputs = parse_vllm_command(parsed.command)
    if parsed.no_cache:
        inputs = dataclasses.replace(inputs, use_cache=False)
    summary, estimate = estimate_from_inputs(inputs)

    if parsed.json:
        payload = {
            "model": summary.model_id,
            "architecture": summary.architecture,
            "parameters": summary.parameter_count,
            "max_active_sequences": summary.max_active_seqs,
            "max_seq_len": summary.max_seq_len,
            "tensor_parallel_size": summary.tensor_parallel_size,
            "pipeline_parallel_size": summary.pipeline_parallel_size,
            "data_parallel_size": summary.data_parallel_size,
            "enable_expert_parallel": summary.enable_expert_parallel,
            "total_gpus": summary.total_gpus,
            "quantization": summary.quantization.as_payload(),
            "estimate_gib": estimate.as_dict(),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(estimate.render_table())
        print()
        print("Context:")
        print(f"  Model architecture : {summary.architecture}")
        print(f"  Parameter count    : {summary.parameter_count / 1e9:.3f} B")
        method = summary.quantization.method or "dense"
        print(f"  Quantization       : {method}")
        print(f"  Weight dtype       : {summary.quantization.weight_dtype.name}")
        print(f"  Activation dtype   : {summary.quantization.activation_dtype.name}")
        print(f"  KV cache dtype     : {summary.quantization.kv_cache_dtype.name}")
        print(f"  Max active sequences: {summary.max_active_seqs}")
        print(f"  Max sequence length: {summary.max_seq_len}")
        if summary.tensor_parallel_size > 1:
            print(f"  Tensor parallel    : {summary.tensor_parallel_size}")
        if summary.pipeline_parallel_size > 1:
            print(f"  Pipeline parallel  : {summary.pipeline_parallel_size}")
        if summary.data_parallel_size > 1:
            print(f"  Data parallel      : {summary.data_parallel_size}")
        if summary.enable_expert_parallel:
            ep_size = summary.tensor_parallel_size * summary.data_parallel_size
            print(f"  Expert parallel    : {ep_size}")
        if summary.total_gpus > 1:
            print(f"  Total GPUs         : {summary.total_gpus}")
        print(f"  Enforce eager      : {summary.enforce_eager}")

    return 0


# ---------------------------------------------------------------------------
# budget subcommand
# ---------------------------------------------------------------------------

def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _run_budget(parsed: argparse.Namespace) -> int:
    seq_lengths = _parse_int_list(parsed.seq_lengths) if parsed.seq_lengths else None
    seq_counts = _parse_int_list(parsed.seq_counts) if parsed.seq_counts else None

    result = compute_budget(
        model_id=parsed.model,
        gpu_memory_gib=parsed.gpu_memory_gib,
        tensor_parallel_size=parsed.tensor_parallel_size,
        pipeline_parallel_size=parsed.pipeline_parallel_size,
        data_parallel_size=parsed.data_parallel_size,
        enable_expert_parallel=parsed.enable_expert_parallel,
        quantization=parsed.quantization,
        dtype=parsed.dtype,
        kv_cache_dtype=parsed.kv_cache_dtype,
        enforce_eager=parsed.enforce_eager,
        block_size=parsed.block_size,
        revision=parsed.revision,
        use_cache=not parsed.no_cache,
        seq_lengths=seq_lengths,
        seq_counts=seq_counts,
        max_num_batched_tokens=parsed.max_num_batched_tokens,
    )

    if parsed.html:
        with open(parsed.html, "w") as f:
            f.write(result.render_html())
        print(f"HTML report written to {parsed.html}")

    if parsed.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(result.render_table())

    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memory-estimator",
        description=(
            "Estimate GPU memory requirements for serving Hugging Face models with vLLM.\n\n"
            "Subcommands:\n"
            "  estimate   Estimate memory for a vllm serve command\n"
            "  budget     Compute a token budget matrix for a model and GPU\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # --- estimate ---
    est = subparsers.add_parser(
        "estimate",
        help="Estimate memory for a vllm serve command",
        description=(
            "Estimate GPU memory for a vllm serve command string.\n\n"
            "Example:\n"
            '  memory-estimator estimate "vllm serve model --max-model-len 2000"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    est.add_argument(
        "command",
        help='vllm serve command string, e.g. "vllm serve model --max-model-len 2000"',
    )
    est.add_argument("--json", action="store_true", help="Emit results as JSON")
    est.add_argument(
        "--no-cache", action="store_true",
        help="Force re-fetching safetensors metadata",
    )
    est.set_defaults(func=_run_estimate)

    # --- budget ---
    bud = subparsers.add_parser(
        "budget",
        help="Compute a token budget matrix",
        description=(
            "Compute a token budget matrix showing estimated per-GPU memory\n"
            "for combinations of context length and concurrent sequences.\n\n"
            "Example:\n"
            "  memory-estimator budget --model meta-llama/Llama-3.1-8B --gpu-memory-gib 80"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bud.add_argument("--model", required=True, help="HuggingFace model ID")
    bud.add_argument(
        "--gpu-memory-gib", type=float, required=True,
        help="Available GPU memory in GiB",
    )
    bud.add_argument("--tp", "--tensor-parallel-size", type=int, default=1,
                     dest="tensor_parallel_size")
    bud.add_argument("--pp", "--pipeline-parallel-size", type=int, default=1,
                     dest="pipeline_parallel_size")
    bud.add_argument("--dp", "--data-parallel-size", type=int, default=1,
                     dest="data_parallel_size")
    bud.add_argument("--enable-expert-parallel", action="store_true", default=False)
    bud.add_argument("-q", "--quantization", default=None)
    bud.add_argument("--dtype", default=None)
    bud.add_argument("--kv-cache-dtype", default=None)
    bud.add_argument("--enforce-eager", action="store_true", default=False)
    bud.add_argument("--block-size", type=int, default=None)
    bud.add_argument("--revision", default=None)
    bud.add_argument("--max-num-batched-tokens", type=int, default=None)
    bud.add_argument(
        "--seq-lengths", default=None,
        help="Comma-separated context lengths to sweep (default: powers of 2 up to model max)",
    )
    bud.add_argument(
        "--seq-counts", default=None,
        help="Comma-separated concurrent sequence counts to sweep (default: 1,4,8,...,512)",
    )
    bud.add_argument("--json", action="store_true", help="Emit results as JSON")
    bud.add_argument("--html", default=None, metavar="FILE", help="Write HTML report to file")
    bud.add_argument("--no-cache", action="store_true",
                     help="Force re-fetching safetensors metadata")
    bud.set_defaults(func=_run_budget)

    return parser


def main(args: list[str] | None = None) -> int:
    raw_args = args if args is not None else sys.argv[1:]

    # Backwards compatibility: if the first argument is not a known subcommand,
    # treat the entire invocation as the legacy 'estimate' mode.
    subcommands = {"estimate", "budget"}
    if raw_args and raw_args[0] not in subcommands and raw_args[0] not in ("-h", "--help"):
        raw_args = ["estimate"] + list(raw_args)

    parser = _build_parser()
    parsed = parser.parse_args(raw_args)

    if not hasattr(parsed, "func"):
        parser.print_help()
        return 1

    return parsed.func(parsed)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
