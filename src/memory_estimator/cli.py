"""Console entry point for the vLLM memory estimator."""
from __future__ import annotations

import argparse
import json

from .estimator import EstimatorInputs
from .estimator import estimate_from_inputs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Estimate GPU memory requirements for serving a Hugging "
                     "Face model with vLLM."))
    parser.add_argument("--model",
                        required=True,
                        help="Hugging Face model repo or local path")
    parser.add_argument("--max-seq-len",
                        type=int,
                        required=True,
                        help="Maximum sequence length to support during serving"
                        )
    parser.add_argument("--max-active-seqs",
                        "--max-active-sequences",
                        "--batch-size",
                        dest="max_active_seqs",
                        type=int,
                        default=1,
                        help=(
                            "Peak number of in-flight sequences to size activations/KV cache"
                            " (alias: --batch-size)"
                        ))
    parser.add_argument("--revision",
                        default=None,
                        help="Specific HF revision (branch, tag, commit)")
    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        help="Allow models that ship custom code")
    parser.add_argument("--json",
                        action="store_true",
                        help="Emit results as JSON instead of a table")
    return parser


def main(args: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    parsed = parser.parse_args(args)

    inputs = EstimatorInputs(model_id=parsed.model,
                             max_seq_len=parsed.max_seq_len,
                             max_active_seqs=parsed.max_active_seqs,
                             revision=parsed.revision,
                             trust_remote_code=parsed.trust_remote_code)

    summary, estimate = estimate_from_inputs(inputs)

    if parsed.json:
        payload = {
            "model": summary.model_id,
            "architecture": summary.architecture,
            "parameters": summary.parameter_count,
            "max_active_sequences": summary.max_active_seqs,
            "max_seq_len": summary.max_seq_len,
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

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
