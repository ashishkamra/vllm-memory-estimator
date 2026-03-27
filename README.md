# vllm-memory-estimator

A lightweight, offline utility that estimates GPU memory requirements for serving
Hugging Face models with [vLLM](https://github.com/vllm-project/vllm).

You pass your `vllm serve` command as a quoted string. The tool parses
memory-relevant flags (model, sequence length, batch size, dtype, parallelism,
etc.), reads model metadata from Hugging Face — no GPU or weight download
required — and produces lower/upper per-GPU ranges for every memory component:
parameters, activations, KV cache, workspace, and vLLM runtime overhead (CUDA
graphs, block tables, worker buffers).

Supports tensor parallelism (TP), pipeline parallelism (PP), data parallelism
(DP), and expert parallelism (EP) for MoE models — including DeepSeek-style
DP+EP (Wide EP) where attention is replicated and experts are distributed.

The only runtime dependency is `huggingface_hub`. No torch, transformers, or
vLLM installation needed.

## Installation

```bash
pip install -e .
```

For development (tests + linting):

```bash
pip install -e ".[dev]"
```

## Usage

### CLI

Pass your `vllm serve` command directly — unknown flags like `--host` or
`--port` are silently ignored:

```bash
memory-estimator "vllm serve openai/gpt-oss-120b --max-model-len 2000 --max-num-seqs 50"
```

Sample output:

```
Model: openai/gpt-oss-120b
----------------------------------
Parameters      :  60.77 GiB (57.73 – 63.81)
Activations     :   0.92 GiB ( 0.46 –  1.83)
KV Cache        :   6.87 GiB ( 6.73 –  7.00)
Workspace       :   0.05 GiB ( 0.02 –  0.15)
----------------------------------
Total (raw)     :  68.60 GiB (64.94 – 72.79)
vLLM overhead   :   8.01 GiB ( 4.00 – 16.01)
----------------------------------
Total (vLLM)    :  76.61 GiB (68.94 – 88.81)

Context:
  Model architecture : gpt_oss
  Parameter count    : 63.081 B
  Quantization       : mxfp4
  Weight dtype       : mxfp4
  Activation dtype   : float16
  KV cache dtype     : float16
  Max active sequences: 50
  Max sequence length: 2000
  Enforce eager      : False
```

Each row shows a nominal estimate plus a (lower – upper) confidence range.

Use `--kv-cache-dtype fp8` to halve KV cache memory, or parallelism flags to
see per-GPU estimates:

```bash
# FP8 KV cache — halves KV cache memory
memory-estimator "vllm serve openai/gpt-oss-120b --max-model-len 2000 --max-num-seqs 50 --kv-cache-dtype fp8"

# Tensor parallelism — shows per-GPU + total cluster memory
memory-estimator "vllm serve openai/gpt-oss-120b --max-model-len 2000 --max-num-seqs 50 -tp 2"

# Pipeline parallelism — splits layers across GPUs
memory-estimator "vllm serve openai/gpt-oss-120b --max-model-len 2000 --max-num-seqs 50 -tp 2 -pp 2"

# Data parallelism — full replica per GPU, fewer sequences per rank
memory-estimator "vllm serve openai/gpt-oss-120b --max-model-len 2000 --max-num-seqs 200 --data-parallel-size 4"

# Expert parallelism (MoE) — distributes experts, replicates attention
memory-estimator "vllm serve deepseek-ai/DeepSeek-V3 --max-model-len 4096 -tp 2 --data-parallel-size 4 --enable-expert-parallel"
```

Add `--json` for machine-readable output:

```bash
memory-estimator "vllm serve openai/gpt-oss-120b --max-model-len 2000 --max-num-seqs 50" --json
```

### Supported vLLM flags

The estimator parses these memory-relevant flags from the `vllm serve` command.
Unknown flags (e.g. `--host`, `--port`) are silently ignored.

| vLLM flag | What it affects |
|-----------|----------------|
| `--model` | Model to estimate (or positional first argument) |
| `--max-model-len` | Maximum sequence length (defaults to model's `max_position_embeddings`) |
| `--max-num-seqs` | Concurrent sequences for KV cache sizing (default: 256) |
| `--kv-cache-dtype` | KV cache precision — `fp8` halves cache memory |
| `--dtype` | Activation precision override |
| `--tensor-parallel-size` / `-tp` | Tensor parallelism — shards weights, KV cache, activations |
| `--pipeline-parallel-size` / `-pp` | Pipeline parallelism — splits layers across GPUs |
| `--data-parallel-size` | Data parallelism — full replica per GPU, fewer seqs per rank |
| `--enable-expert-parallel` | Expert parallelism (MoE) — distributes experts, replicates attention |
| `--enforce-eager` | Disable CUDA graphs (reduces overhead) |
| `--block-size` | Paged attention block size |
| `--max-num-batched-tokens` | Tokens per forward pass (controls activation sizing) |
| `--quantization` / `-q` | Quantization method override |
| `--revision` | Model revision / branch |

### Python API

```python
from memory_estimator.estimator import EstimatorInputs, estimate_from_inputs

summary, estimate = estimate_from_inputs(
    EstimatorInputs(
        model_id="openai/gpt-oss-120b",
        max_seq_len=2000,
        max_active_seqs=50,
        kv_cache_dtype="fp8",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
    )
)

print(estimate.render_table())
print(f"Per GPU (vLLM): {estimate.total_with_vllm.nominal_gib:.2f} GiB")
```

The `estimate` object exposes each component as a `MemoryComponentEstimate`
with `nominal_gib`, `lower_gib`, and `upper_gib` fields:

- `estimate.parameters` — model weights
- `estimate.activations` — forward pass activation buffers
- `estimate.kv_cache` — key/value cache
- `estimate.workspace` — temporary workspace / scratch memory
- `estimate.vllm_overhead` — CUDA graphs + block tables + worker overhead
- `estimate.total` — sum of first four (raw model memory)
- `estimate.total_with_vllm` — total including vLLM runtime overhead

Use `estimate.as_dict()` for a serialisable dictionary.

## How It Works

1. **Command parsing** — The `vllm serve` command string is tokenized with
   `shlex.split` and parsed with argparse using `parse_known_args` to extract
   memory-relevant flags. Unknown flags are ignored.

2. **Configuration loading** — Downloads `config.json` from the Hugging Face
   Hub and inspects `quantization_config` to determine quantization method,
   weight dtype, activation dtype, and KV cache dtype.

3. **Parameter byte counting** — Reads safetensors file headers via
   `huggingface_hub.get_safetensors_metadata` to obtain exact on-disk byte
   counts without downloading weights.

4. **Activation estimation** — Computes peak activation memory as:
   - Hidden state buffer (`tokens × hidden_size × bytes`)
   - Max of FFN intermediate and QKV projection buffers
   - Logits buffer (`tokens × vocab_size × bytes`)
   - MoE expert buffers (when applicable)

5. **KV cache estimation** — Straightforward formula:
   `layers × sequences × seq_len × kv_heads × head_dim × 2 × dtype_bytes`

6. **vLLM overhead estimation** — Models three runtime components:
   - **CUDA graphs**: proportional to parameter size and layer count
   - **Block tables**: per-sequence metadata for paged attention
   - **Worker overhead**: fixed ~300 MiB for vLLM worker process state

7. **Parallelism** — Applies per-GPU memory division based on the active
   parallelism strategy:

   | Strategy | Weights | KV Cache | Activations |
   |----------|---------|----------|-------------|
   | **TP** | /TP | /TP | /TP |
   | **PP** | /PP | /PP | unchanged |
   | **DP** | unchanged | seqs/DP | seqs/DP |
   | **EP** (MoE) | attention replicated, experts /(TP×DP) | /TP | /TP |

   For MoE models with EP enabled, expert vs non-expert parameters are
   identified by tensor name from the safetensors headers.

8. **Range construction** — Each component gets a confidence range to account
   for runtime variability (allocator fragmentation, framework overhead, etc.).

## Memory Components

| Component | What it covers |
|-----------|---------------|
| **Parameters** | Model weights as stored on disk (exact byte count from safetensors) |
| **Activations** | Hidden states, FFN intermediates, QKV projections, logits buffer |
| **KV Cache** | Per-layer key/value tensors for all active sequences × sequence length |
| **Workspace** | Temporary scratch buffers (~5% of activation memory) |
| **vLLM Overhead** | CUDA graph captures, block tables, worker process state |

## Project Structure

```
src/memory_estimator/
├── cli.py               # Command-line interface
├── vllm_cmd_parser.py   # vllm serve command string parser
├── estimator.py         # High-level API (EstimatorInputs → MemoryEstimate)
├── buckets.py           # Memory accounting by category
├── reports.py           # MemoryEstimate dataclass and table rendering
├── model_summary.py     # ModelSummary intermediate representation
├── model_shapes.py      # Safetensors-based parameter shape collection
├── quantization.py      # Quantization config parsing
├── config_utils.py      # Architecture attribute resolution
├── dtype_utils.py       # Dtype normalisation and byte-width helpers
└── vllm_defaults.py     # vLLM-specific constants

tests/
├── conftest.py               # Shared fixtures and CLI options
├── test_buckets.py           # Unit tests for memory bucket calculations
├── test_cli.py               # CLI argument parsing tests
├── test_dtype_utils.py       # Unit tests for dtype utilities
├── test_quantization.py      # Unit tests for quantization parsing
├── test_vllm_cmd_parser.py   # vllm serve command parser tests
├── test_memory_profile.py    # GPU integration test (PyTorch runtime comparison)
└── test_vllm_profile.py      # vLLM integration test (validates against vllm bench)
```

## Testing

### Unit tests (no GPU required)

```bash
pip install -e ".[dev]"
python -m pytest tests/ --ignore=tests/test_vllm_profile.py --ignore=tests/test_memory_profile.py -v
```

### GPU integration tests

Require a CUDA GPU. Compare estimated vs actual memory usage:

```bash
# PyTorch forward pass comparison
python -m pytest tests/test_memory_profile.py -v -m cuda --profile-report

# vLLM benchmark comparison (requires vLLM installed)
pip install -e ".[vllm,dev]"
python -m pytest tests/test_vllm_profile.py -v -s --profile-report

# Specify a different model
python -m pytest tests/test_vllm_profile.py -v -s \
    --profile-model openai-community/gpt2 \
    --profile-max-seq-len 256 \
    --profile-max-active-seqs 4 \
    --profile-report
```

### Linting

```bash
ruff check .
```

## Supported Models

The estimator works with any Hugging Face model whose `config.json` exposes
standard architecture attributes (`hidden_size`, `num_hidden_layers`,
`num_attention_heads`, etc.) and publishes weights in safetensors format. This
includes most decoder-only architectures:

- LLaMA / Llama 2 / Llama 3 family
- GPT-2 / GPT-NeoX / OPT
- Mistral / Mixtral (MoE)
- Qwen / Qwen2
- Phi / Phi-3
- Falcon
- Gemma

Quantized checkpoints (GPTQ, AWQ, compressed-tensors, FP8, MXFP4) are handled
automatically when `quantization_config` is present in the model config.

## Notes and Limitations

- The estimator runs entirely on CPU — no GPU is needed.
- The only dependency is `huggingface_hub`. No torch, transformers, or vLLM
  needed for estimation.
- Parameter memory is the exact on-disk size from safetensors file headers.
  Models without safetensors files are not supported.
- Actual runtime consumption can exceed estimates due to CUDA allocator
  fragmentation, LoRA adapters, speculative decoding, or tensor parallelism
  communication buffers. Leave headroom in production deployments.
- At large model scales (70B+), weights and KV cache dominate memory;
  activation estimates become proportionally less significant.

## License

See [LICENSE](LICENSE) for details.
