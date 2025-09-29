import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_estimator.estimator import EstimatorInputs, estimate_from_inputs

BYTES_PER_GIB = 1024 ** 3


def _bytes_to_gib(value: float) -> float:
    return value / BYTES_PER_GIB

def _iter_past_key_values(past_key_values):
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if isinstance(layer, (tuple, list)):
                yield layer
            elif hasattr(layer, "key") and hasattr(layer, "value"):
                yield layer.key, layer.value
            elif hasattr(layer, "keys") and hasattr(layer, "values"):
                yield layer.keys, layer.values
            else:
                raise TypeError("Unsupported KV cache layer structure")
    else:
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                yield layer
            elif hasattr(layer, "key") and hasattr(layer, "value"):
                yield layer.key, layer.value
            elif hasattr(layer, "keys") and hasattr(layer, "values"):
                yield layer.keys, layer.values
            else:
                raise TypeError("Unsupported KV cache layer structure")


def _sum_parameter_bytes(model: AutoModelForCausalLM) -> int:
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    return total


def _sum_kv_bytes(past_key_values) -> int:
    total = 0
    for key, value in _iter_past_key_values(past_key_values):
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


@pytest.mark.cuda
@pytest.mark.filterwarnings("ignore:torch_dtype is deprecated")
def test_estimates_align_with_runtime(request, profile_settings, profile_report_enabled):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model_id = profile_settings.model_id
    max_seq_len = profile_settings.max_seq_len
    max_active_seqs = profile_settings.max_active_seqs

    _, estimate = estimate_from_inputs(
        EstimatorInputs(
            model_id=model_id,
            max_seq_len=max_seq_len,
            max_active_seqs=max_active_seqs,
        ))

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Prepare random inputs matching the requested profile.
    input_ids = torch.randint(low=0,
                              high=tokenizer.vocab_size,
                              size=(max_active_seqs, max_seq_len),
                              device="cuda")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    after_forward_bytes = torch.cuda.memory_allocated()

    param_bytes = _sum_parameter_bytes(model)
    kv_cache_bytes = _sum_kv_bytes(outputs.past_key_values)
    activation_bytes = max(peak_bytes - after_forward_bytes, 0)
    workspace_bytes = max(peak_bytes - (param_bytes + kv_cache_bytes + activation_bytes), 0)
    total_bytes = peak_bytes

    estimates = {
        "parameters": estimate.parameters,
        "activations": estimate.activations,
        "kv_cache": estimate.kv_cache,
        "workspace": estimate.workspace,
        "total": estimate.total,
    }

    actuals = {
        "parameters": _bytes_to_gib(param_bytes),
        "activations": _bytes_to_gib(activation_bytes),
        "kv_cache": _bytes_to_gib(kv_cache_bytes),
        "workspace": _bytes_to_gib(workspace_bytes),
        "total": _bytes_to_gib(total_bytes),
    }

    report_lines = []
    for label, component in estimates.items():
        actual_gib = actuals[label]
        msg = (
            f"{label} actual {actual_gib:.3f} GiB outside range "
            f"({component.lower_gib:.3f}, {component.upper_gib:.3f})"
        )
        report_lines.append(
            f"{label:<11} actual {actual_gib:6.3f} GiB | "
            f"nominal {component.nominal_gib:6.3f} GiB "
            f"[{component.lower_gib:6.3f}, {component.upper_gib:6.3f}]"
        )
        assert component.lower_gib <= actual_gib <= component.upper_gib, msg

    if profile_report_enabled:
        header = (
            f"Profile report for {model_id} | max_seq_len={max_seq_len} | "
            f"max_active_seqs={max_active_seqs}"
        )
        request.config.profile_reports.append((header, report_lines))

    del outputs, input_ids, model, tokenizer
    torch.cuda.empty_cache()
