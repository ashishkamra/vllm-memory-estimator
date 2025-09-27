import os
import re
import subprocess

import pytest

from memory_estimator.estimator import EstimatorInputs
from memory_estimator.estimator import estimate_from_inputs


def _has_vllm() -> bool:
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def _run_vllm_bench(
    model_id: str,
    max_model_len: int,
    batch_size: int,
    enforce_eager: bool = True,
    gpu_memory_utilization: float | None = None,
) -> tuple[int, str, str]:
    cmd = [
        "vllm", "bench", "latency",
        "--model", model_id,
        "--max-model-len", str(max_model_len),
        "--batch-size", str(batch_size),
        "--num-iters", "1",
        "--num-iters-warmup", "1",
        "--input-len", "16",
        "--output-len", "8",
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    if gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(gpu_memory_utilization)])

    env = os.environ.copy()
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, env=env,
    )
    return result.returncode, result.stdout, result.stderr


def _parse_model_load_gib(output: str) -> float | None:
    m = re.search(r"Model loading took ([\d.]+) GiB", output)
    if m:
        return float(m.group(1))
    return None


def _parse_memory_breakdown(output: str) -> dict[str, float]:
    breakdown = {}
    m = re.search(
        r"([\d.]+) GiB for weight.*?([\d.]+) GiB for peak activation.*?([\d.]+) GiB for non-torch",
        output,
    )
    if m:
        breakdown["weight_gib"] = float(m.group(1))
        breakdown["peak_activation_gib"] = float(m.group(2))
        breakdown["non_torch_gib"] = float(m.group(3))
    cg = re.search(r"([\d.]+) GiB for CUDAGraph", output)
    if cg:
        breakdown["cudagraph_gib"] = float(cg.group(1))
    return breakdown


def _is_oom(stderr: str) -> bool:
    return "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr


def _is_insufficient_memory(stderr: str) -> bool:
    return "less than desired GPU memory utilization" in stderr


def _compute_gpu_util() -> float:
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return round((free / total) - 0.05, 2)
    except Exception:
        return 0.85


@pytest.mark.cuda
@pytest.mark.skipif(not _has_vllm(), reason="vllm is not installed")
def test_vllm_estimates_align_with_runtime(request, profile_settings, profile_report_enabled):
    model_id = profile_settings.model_id
    max_seq_len = profile_settings.max_seq_len
    max_active_seqs = profile_settings.max_active_seqs

    _, estimate = estimate_from_inputs(
        EstimatorInputs(
            model_id=model_id,
            max_seq_len=max_seq_len,
            max_active_seqs=max_active_seqs,
            enforce_eager=True,
        ))

    seq_len = max_seq_len
    batch = max_active_seqs
    gpu_util = None
    returncode, stdout, stderr = _run_vllm_bench(model_id, seq_len, batch)

    if returncode != 0 and _is_insufficient_memory(stderr):
        gpu_util = _compute_gpu_util()
        returncode, stdout, stderr = _run_vllm_bench(
            model_id, seq_len, batch, gpu_memory_utilization=gpu_util)

    if returncode != 0 and (_is_oom(stderr) or _is_insufficient_memory(stderr)):
        seq_len = max(64, seq_len // 2)
        batch = 1
        gpu_util = gpu_util or _compute_gpu_util()
        returncode, stdout, stderr = _run_vllm_bench(
            model_id, seq_len, batch, gpu_memory_utilization=gpu_util)

    if returncode != 0:
        pytest.fail(
            f"vllm bench latency failed (exit {returncode}).\n"
            f"stderr (last 2000 chars):\n{stderr[-2000:]}"
        )

    combined = stdout + "\n" + stderr
    model_load_gib = _parse_model_load_gib(combined)
    if model_load_gib is None:
        pytest.skip("Could not parse 'Model loading took X GiB' from vLLM output")

    breakdown = _parse_memory_breakdown(combined)

    report_lines = [
        f"vLLM model loading: {model_load_gib:.3f} GiB",
    ]
    for k, v in breakdown.items():
        report_lines.append(f"vLLM {k}: {v:.3f} GiB")

    errors: list[str] = []

    report_lines.append(
        f"Estimated params: nominal {estimate.parameters.nominal_gib:.3f} GiB "
        f"[{estimate.parameters.lower_gib:.3f}, {estimate.parameters.upper_gib:.3f}]"
    )
    if not (estimate.parameters.lower_gib <= model_load_gib <= estimate.parameters.upper_gib):
        errors.append(
            f"Weight: vLLM {model_load_gib:.3f} GiB outside estimate "
            f"[{estimate.parameters.lower_gib:.3f}, {estimate.parameters.upper_gib:.3f}]"
        )

    if "peak_activation_gib" in breakdown:
        vllm_activation_gib = breakdown["peak_activation_gib"]
        est_act_lower = estimate.activations.lower_gib + estimate.workspace.lower_gib
        est_act_upper = estimate.activations.upper_gib + estimate.workspace.upper_gib
        est_act_nominal = estimate.activations.nominal_gib + estimate.workspace.nominal_gib
        report_lines.append(
            f"Estimated activations+workspace: nominal {est_act_nominal:.3f} GiB "
            f"[{est_act_lower:.3f}, {est_act_upper:.3f}]"
        )
        if not (est_act_lower <= vllm_activation_gib <= est_act_upper):
            errors.append(
                f"Activations: vLLM {vllm_activation_gib:.3f} GiB outside estimate "
                f"[{est_act_lower:.3f}, {est_act_upper:.3f}]"
            )

    if all(k in breakdown for k in ("weight_gib", "peak_activation_gib", "non_torch_gib")):
        vllm_non_kv_gib = (
            breakdown["weight_gib"]
            + breakdown["peak_activation_gib"]
            + breakdown["non_torch_gib"]
            + breakdown.get("cudagraph_gib", 0.0)
        )
        est_non_kv_lower = estimate.total_with_vllm.lower_gib - estimate.kv_cache.upper_gib
        est_non_kv_upper = estimate.total_with_vllm.upper_gib - estimate.kv_cache.lower_gib
        est_non_kv_nominal = estimate.total_with_vllm.nominal_gib - estimate.kv_cache.nominal_gib
        report_lines.append(
            f"Estimated non-KV total: nominal {est_non_kv_nominal:.3f} GiB "
            f"[{est_non_kv_lower:.3f}, {est_non_kv_upper:.3f}]"
        )
        report_lines.append(f"vLLM non-KV total: {vllm_non_kv_gib:.3f} GiB")
        if not (est_non_kv_lower <= vllm_non_kv_gib <= est_non_kv_upper):
            errors.append(
                f"Non-KV total: vLLM {vllm_non_kv_gib:.3f} GiB outside estimate "
                f"[{est_non_kv_lower:.3f}, {est_non_kv_upper:.3f}]"
            )

    if profile_report_enabled:
        header = (
            f"vLLM profile for {model_id} | max_seq_len={seq_len} | "
            f"max_active_seqs={batch}"
        )
        request.config.profile_reports.append((header, report_lines))

    if errors:
        pytest.fail("\n".join(errors))
