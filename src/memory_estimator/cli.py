"""Console entry point for the vLLM memory estimator."""

from __future__ import annotations

import argparse
import dataclasses
import json

from .estimator import estimate_from_inputs
from .vllm_cmd_parser import parse_vllm_command


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate GPU memory requirements for serving a Hugging Face model with vLLM.\n\n"
            "Pass a vllm serve command string as the argument:\n"
            '  memory-estimator "vllm serve model --max-model-len 2000 --max-num-seqs 50"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        help='vllm serve command string, e.g. "vllm serve model --max-model-len 2000"',
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit results as JSON instead of a table"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force re-fetching safetensors metadata instead of using cached results",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    parsed = parser.parse_args(args)

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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
