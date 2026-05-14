"""Tests for the validation runner."""
from __future__ import annotations

from pathlib import Path

import pytest

from memory_estimator.reports import MemoryComponentEstimate
from memory_estimator.reports import MemoryEstimate
from memory_estimator.validation_runner import ComparisonResult
from memory_estimator.validation_runner import _compare_record
from memory_estimator.validation_runner import compute_aggregate_stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_estimate(
    params_nominal: float = 10.0,
    params_lower: float = 9.5,
    params_upper: float = 10.5,
    kv_nominal: float = 5.0,
) -> MemoryEstimate:
    def _comp(n: float) -> MemoryComponentEstimate:
        return MemoryComponentEstimate(n, n * 0.9, n * 1.1)

    return MemoryEstimate(
        model_name="test-model",
        parameters=MemoryComponentEstimate(params_nominal, params_lower, params_upper),
        activations=_comp(1.0),
        kv_cache=MemoryComponentEstimate(kv_nominal, kv_nominal * 0.98, kv_nominal * 1.02),
        workspace=_comp(0.5),
        total=_comp(16.5),
        vllm_overhead=_comp(0.5),
        total_with_vllm=_comp(17.0),
        kv_cache_spec_type="full",
    )


def _make_record(
    model_load_gib: float = 10.0,
    available_kv_cache_gib: float | None = 50.0,
    kv_cache_tokens: int | None = 1_000_000,
    model_id: str = "test/model",
    max_seq_len: int = 4096,
    tp: int = 1,
    quantization: str | None = None,
    accelerator: str = "H200",
) -> dict:
    return {
        "uuid": "test-uuid-1234",
        "csv_metadata": {"model": model_id, "accelerator": accelerator, "tp": tp, "dp": None,
                         "runtime_args": ""},
        "log_config": {"model": model_id, "dtype": "bfloat16", "max_seq_len": max_seq_len,
                       "tensor_parallel_size": tp, "pipeline_parallel_size": 1,
                       "data_parallel_size": 1, "quantization": quantization,
                       "kv_cache_dtype": "auto", "enforce_eager": False,
                       "enable_prefix_caching": False, "enable_chunked_prefill": True,
                       "vllm_version": "v0.16.0"},
        "log_memory": {"model_load_gib": model_load_gib,
                       "available_kv_cache_gib": available_kv_cache_gib,
                       "kv_cache_tokens": kv_cache_tokens,
                       "max_concurrency_tokens": max_seq_len,
                       "max_concurrency_ratio": None, "cudagraph_gib": None},
        "estimator_inputs": {"model_id": model_id, "max_seq_len": max_seq_len,
                             "tensor_parallel_size": tp, "pipeline_parallel_size": 1,
                             "data_parallel_size": 1, "quantization": quantization,
                             "kv_cache_dtype": None, "enforce_eager": False, "dtype": "bfloat16"},
    }


# ---------------------------------------------------------------------------
# Comparison tests
# ---------------------------------------------------------------------------

def test_compare_record_within_bounds():
    estimate = _make_estimate(params_nominal=10.0, params_lower=9.5, params_upper=10.5)
    record = _make_record(model_load_gib=10.2)
    comp = _compare_record("uuid-1", record, estimate)

    assert comp.params_within_bounds is True
    assert comp.params_error_pct == pytest.approx(2.0, abs=0.01)
    assert comp.actual_model_load_gib == pytest.approx(10.2)
    assert comp.estimated_params_nominal_gib == pytest.approx(10.0)


def test_compare_record_outside_bounds_over():
    estimate = _make_estimate(params_nominal=10.0, params_lower=9.5, params_upper=10.5)
    record = _make_record(model_load_gib=11.0)
    comp = _compare_record("uuid-2", record, estimate)

    assert comp.params_within_bounds is False
    assert comp.params_error_pct == pytest.approx(10.0, abs=0.01)


def test_compare_record_outside_bounds_under():
    estimate = _make_estimate(params_nominal=10.0, params_lower=9.5, params_upper=10.5)
    record = _make_record(model_load_gib=9.0)
    comp = _compare_record("uuid-3", record, estimate)

    assert comp.params_within_bounds is False
    assert comp.params_error_pct == pytest.approx(-10.0, abs=0.01)


def test_compare_record_kv_sanity():
    estimate = _make_estimate(kv_nominal=5.0)
    record = _make_record(available_kv_cache_gib=50.0, kv_cache_tokens=1_000_000)
    comp = _compare_record("uuid-4", record, estimate)

    assert comp.actual_kv_per_token_bytes is not None
    assert comp.actual_kv_per_token_bytes == pytest.approx(50.0 * 1024**3 / 1_000_000)
    assert comp.estimated_kv_per_token_bytes is not None
    assert comp.kv_per_token_ratio is not None
    assert comp.kv_per_token_ratio > 0


def test_compare_record_kv_none_when_missing():
    estimate = _make_estimate()
    record = _make_record(available_kv_cache_gib=None, kv_cache_tokens=None)
    comp = _compare_record("uuid-5", record, estimate)

    assert comp.actual_kv_per_token_bytes is None
    assert comp.kv_per_token_ratio is None


def test_compare_record_metadata():
    estimate = _make_estimate()
    record = _make_record(model_id="meta-llama/Llama-70B", accelerator="MI300X",
                          tp=4, quantization="fp8", max_seq_len=8192)
    comp = _compare_record("uuid-6", record, estimate)

    assert comp.model_id == "meta-llama/Llama-70B"
    assert comp.accelerator == "MI300X"
    assert comp.tensor_parallel_size == 4
    assert comp.quantization == "fp8"
    assert comp.max_seq_len == 8192


# ---------------------------------------------------------------------------
# Aggregate stats tests
# ---------------------------------------------------------------------------

def _make_comparison(
    uuid: str,
    error_pct: float,
    within_bounds: bool,
    model_id: str = "model/a",
) -> ComparisonResult:
    return ComparisonResult(
        uuid=uuid, model_id=model_id, accelerator="H200",
        actual_model_load_gib=10.0, estimated_params_nominal_gib=10.0,
        estimated_params_lower_gib=9.5, estimated_params_upper_gib=10.5,
        params_within_bounds=within_bounds, params_error_pct=error_pct,
        actual_kv_per_token_bytes=None, estimated_kv_per_token_bytes=None,
        kv_per_token_ratio=None, kv_cache_spec_type="full",
        quantization=None, tensor_parallel_size=1, max_seq_len=4096,
    )


def test_aggregate_stats_basic():
    comps = [
        _make_comparison("u1", 2.0, True, "model/a"),
        _make_comparison("u2", -3.0, True, "model/a"),
        _make_comparison("u3", 8.0, False, "model/b"),
        _make_comparison("u4", 1.0, True, "model/b"),
    ]
    report = compute_aggregate_stats(comps, skipped=[], total_records=4)

    assert report.total_compared == 4
    assert report.total_skipped == 0
    assert report.params_in_bounds_count == 3
    assert report.params_in_bounds_pct == pytest.approx(75.0)
    assert report.params_mean_abs_error_pct == pytest.approx(3.5)
    assert report.params_median_abs_error_pct == pytest.approx(2.5)
    assert report.params_max_abs_error_pct == pytest.approx(8.0)
    assert len(report.worst_offenders) == 4
    assert report.worst_offenders[0].uuid == "u3"


def test_aggregate_per_model_stats():
    comps = [
        _make_comparison("u1", 2.0, True, "model/a"),
        _make_comparison("u2", -3.0, True, "model/a"),
        _make_comparison("u3", 8.0, False, "model/b"),
    ]
    report = compute_aggregate_stats(comps, skipped=[], total_records=3)

    assert "model/a" in report.per_model_stats
    assert report.per_model_stats["model/a"]["records"] == 2
    assert report.per_model_stats["model/a"]["in_bounds"] == 2
    assert report.per_model_stats["model/b"]["in_bounds"] == 0


def test_aggregate_with_skipped():
    comps = [_make_comparison("u1", 1.0, True)]
    skipped = [{"uuid": "u2", "model_id": "model/x", "reason": "gated"}]
    report = compute_aggregate_stats(comps, skipped=skipped, total_records=2)

    assert report.total_compared == 1
    assert report.total_skipped == 1
    assert report.total_records == 2


def test_report_render_summary():
    comps = [
        _make_comparison("u1", 2.0, True, "model/a"),
        _make_comparison("u2", -1.0, True, "model/a"),
    ]
    report = compute_aggregate_stats(comps, skipped=[], total_records=2)
    text = report.render_summary()

    assert "model/a" in text
    assert "TOTAL" in text
    assert "2/2" in text


def test_report_render_html():
    comps = [
        _make_comparison("u1", 2.0, True, "model/a"),
        _make_comparison("u2", 8.0, False, "model/b"),
    ]
    report = compute_aggregate_stats(comps, skipped=[], total_records=2)
    html_str = report.render_html()

    assert "<!DOCTYPE html>" in html_str
    assert "model/a" in html_str
    assert "model/b" in html_str
    assert "Validation Report" in html_str
    assert "PASS" in html_str
    assert "FAIL" in html_str


def test_report_to_dict():
    comps = [_make_comparison("u1", 2.0, True)]
    report = compute_aggregate_stats(comps, skipped=[], total_records=1)
    d = report.to_dict()

    assert "metadata" in d
    assert "aggregate" in d
    assert "comparisons" in d
    assert d["aggregate"]["params_in_bounds_count"] == 1
    assert len(d["comparisons"]) == 1


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).resolve().parent.parent / "test_data" / "validation_db.json"


@pytest.mark.skipif(not _DB_PATH.exists(), reason="Validation DB not available")
def test_run_validation_single_model():
    """Run validation on a single model and verify report structure."""
    from memory_estimator.validation_runner import run_validation

    report = run_validation(_DB_PATH, model_filter="Llama-3.3-70B-Instruct-FP8-dynamic")

    assert report.total_compared > 0
    assert report.total_records > 0
    assert len(report.comparisons) > 0
    assert all(c.model_id is not None for c in report.comparisons)
    assert all(c.actual_model_load_gib > 0 for c in report.comparisons)
    assert all(c.estimated_params_nominal_gib > 0 for c in report.comparisons)
