# Repository Guidelines
This guide explains how to contribute to vllm-memory-estimator with minimal friction. Follow these steps to keep the estimator accurate and maintainable.

## Project Structure & Module Organization
- `src/memory_estimator/` hosts runtime code; `estimator.py` combines bucket math with quantization helpers, `cli.py` wires flags to report generation, and `reports.py` renders human and JSON summaries.
- `tests/` mirrors the module layout with unit coverage for buckets, dtypes, quantization, and activation estimates.
- `pyproject.toml` centralizes dependencies, optional dev extras, Ruff, and pytest settings; update it when libraries or lint rules change.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate    # create local env
pip install -e .[dev]                                # install app + tooling
python -m memory_estimator.cli --model ... --max-seq-len ... --max-active-seqs ...  # run the estimator
python -m pytest                                     # execute unit tests
ruff check .                                         # lint and import sort
```
Run CLI examples against small Hugging Face models before scaling up to large checkpoints and note the CLI now reports lower–upper ranges for each memory bucket.

## Coding Style & Naming Conventions
- Target Python ≥3.9, 4-space indentation, and max line length 100 configured in Ruff.
- Keep imports sorted via Ruff’s single-line isort; prefer module-level functions over unused classes.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and descriptive filenames (`buckets.py`, `quantization.py`) when adding modules.

## Testing Guidelines
- Extend existing `tests/test_*.py` files or add new ones that mirror the module under test.
- Name tests `test_<behavior>()`; mark GPU-dependent cases with `@pytest.mark.cuda`.
- New features must include both happy-path and edge-case coverage; capture expected tensor shapes and dtype branches explicitly.
- Validate that `python -m pytest -m "not cuda"` remains green on CPU-only environments.

## Commit & Pull Request Guidelines
- Follow concise, imperative commits (e.g., `Add llama KV cache estimator`); group related changes and include relevant issue numbers in the body.
- Pull requests should summarize motivation, describe memory impact, note new flags, and paste example CLI output or JSON diff.
- Confirm lint/tests locally before opening a PR and mention any skipped checks with rationale.

## GPU & Remote Code Notes
- Prefer small public models for regression testing; document required weights if not publicly hosted.
- If a model demands custom modules, flag the `--trust-remote-code` usage in PRs and review third-party code paths for safety.
