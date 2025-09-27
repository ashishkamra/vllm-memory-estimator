# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU memory estimator for serving Hugging Face models with vLLM. Given a `vllm serve` command string, it parses memory-relevant flags, reads parameter metadata from safetensors file headers (no weight download), infers quantization/precision settings, and returns lower/upper bound estimates for parameters, activations, KV cache, workspace, and vLLM overhead.

## Commands

```bash
# Install
pip install -e .

# Install with dev deps
pip install -e ".[dev]"

# Run the estimator (pass a vllm serve command)
python -m memory_estimator.cli "vllm serve <model_id> --max-model-len <n> --max-num-seqs <n>" [--json]

# Tests (unit tests run without GPU; the CUDA profile test requires --profile-model)
python -m pytest
python -m pytest tests/test_dtype_utils.py           # single test file
python -m pytest -k test_normalise_dtype              # single test by name

# CUDA-dependent profiling test (requires GPU)
python -m pytest -m cuda --profile-model meta-llama/Llama-3.1-8B --profile-max-seq-len 4096 --profile-report

# Lint
ruff check .
```

## Architecture

The pipeline follows a linear flow: **CLI → vllm command parser → EstimatorInputs → ModelSummary → MemoryBuckets → MemoryEstimate**.

- **`cli.py`** — Argparse entry point (`memory-estimator` console script). Accepts a `vllm serve` command string + `--json` flag, calls `parse_vllm_command` → `estimate_from_inputs`, renders table or JSON.
- **`vllm_cmd_parser.py`** — Parses a `vllm serve` command string into `EstimatorInputs`. Uses `shlex.split` + argparse with `parse_known_args` to extract memory-relevant flags and ignore unknown ones.
- **`estimator.py`** — Orchestrator. Downloads `config.json` from HF Hub, parses quantization, collects parameter shapes from safetensors headers. `estimate_memory()` builds memory buckets and produces the final estimate.
- **`quantization.py`** — Parses `quantization_config` from HF configs into a `QuantizationSpec` dataclass with method, weight/activation/KV cache dtypes.
- **`dtype_utils.py`** — `ScalarType` dataclass and `normalise_dtype()` for converting string aliases and config values into a canonical bits/bytes representation.
- **`model_shapes.py`** — Reads safetensors file headers via `huggingface_hub.get_safetensors_metadata` for tensor shapes and exact byte counts. No weight download required.
- **`config_utils.py`** — Resolves architecture attributes (`hidden_size`, `num_layers`, etc.) by walking nested configs (text_config, vision_config, etc.).
- **`buckets.py`** — Core memory accounting. Takes precomputed parameter bytes from safetensors, estimates activations (including MoE), KV cache, workspace, and vLLM overhead. Applies tensor parallelism division.
- **`vllm_defaults.py`** — All vLLM-specific constants (workspace ratio, worker overhead, CUDA graph sizing, block size, confidence range factors).
- **`reports.py`** — Wraps raw byte counts into `MemoryEstimate` with nominal/lower/upper bounds per component.

## Key Design Decisions

- Parameter bytes come directly from safetensors file headers — exact on-disk size, no estimation needed.
- Config attribute resolution (`resolve_config_attr`) walks nested sub-configs for multimodal model support.
- `_DictConfig` wraps `config.json` as an attribute-accessible object; only known config keys (text_config, etc.) are recursively wrapped — data dicts like `quantization_config` stay as plain dicts.
- The `conftest.py` adds custom pytest options (`--profile-model`, `--profile-max-seq-len`, `--profile-max-active-seqs`, `--profile-report`) for GPU validation tests.

## Style

- Python ≥3.9, ruff with `line-length = 100`, isort with `force-single-line`.
- Uses `from __future__ import annotations` throughout.
- Dataclasses for all data containers (no Pydantic).
