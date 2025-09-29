# vllm-memory-estimator

A lightweight utility for sizing GPU memory needs when serving Hugging Face
models with [vLLM](https://github.com/vllm-project/vllm). Given a model repo,
maximum sequence length, and optional max active sequence count, the tool infers
quantization/precision settings, reconstructs parameter shapes on the meta
device, and returns a range for parameters, activations, KV cache, and workspace
buffers.

## Quickstart

```bash
# Option 1: built-in venv
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# Option 2: uv (https://github.com/astral-sh/uv)
uv venv
source .venv/bin/activate
uv pip install -e .[dev]

python -m memory_estimator.cli --model meta-llama/Llama-3.1-8B --max-seq-len 4096 --max-active-seqs 4
```

Add `--json` for machine-readable output or `--trust-remote-code` when the
checkpoint depends on custom modules.

## How it Works

1. Loads the Hugging Face configuration (`AutoConfig`) and inspects
   `quantization_config`, handling mixed precision (INT4/INT8/FP8) and KV cache
   overrides.
2. Materialises the model graph on the meta device (via `accelerate`) to obtain
   parameter shapes without downloading full weights.
3. Applies quantization-aware accounting (grouped scales/zeros, RedHatAI INT4
   schemes, KV cache dtype) to compute parameter storage.
4. Estimates activation, workspace, and KV cache footprints using architecture
   metadata (hidden size, layer count, attention heads) plus the requested max
   active sequence count and context length, then inflates them into conservative
   lower/upper bounds.

## Notes & Limitations

- The estimator relies on `transformers`, `accelerate`, and `vllm` being
  importable. Install the matching vLLM version you plan to use for serving.
- Parameter accounting assumes standard quantization patterns (weights for GEMM
  layers, dense embeddings/norms). Extremely custom checkpoint structures may
  need manual review.
- Actual runtime consumption can exceed estimates due to CUDA allocator
  fragmentation or additional features (LoRA, speculative decoding). Leave
  buffer headroom in production deployments.

## Development

Run the unit tests with:

```bash
python -m pytest
```

Formatting and import sorting follow Ruff (`ruff check .`).
