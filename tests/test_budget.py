"""Tests for the token budget matrix."""
from __future__ import annotations

import pytest

from memory_estimator.budget import BudgetCell
from memory_estimator.budget import BudgetResult
from memory_estimator.budget import _default_seq_lengths

# ---------------------------------------------------------------------------
# Default sweep ranges
# ---------------------------------------------------------------------------

def test_default_seq_lengths_power_of_two():
    result = _default_seq_lengths(4096)
    assert result == [256, 512, 1024, 2048, 4096]


def test_default_seq_lengths_non_power_of_two():
    result = _default_seq_lengths(5000)
    assert 256 in result
    assert 4096 in result
    assert result[-1] == 5000


def test_default_seq_lengths_small_model():
    result = _default_seq_lengths(512)
    assert result == [256, 512]


def test_default_seq_lengths_tiny_model():
    result = _default_seq_lengths(128)
    assert result == [128]


def test_default_seq_lengths_large_model():
    result = _default_seq_lengths(131072)
    assert result[0] == 256
    assert result[-1] == 131072
    assert len(result) >= 8


# ---------------------------------------------------------------------------
# BudgetCell
# ---------------------------------------------------------------------------

def test_budget_cell_fits():
    cell = BudgetCell(
        max_seq_len=4096, max_active_seqs=64,
        total_memory_gib=42.5, fits=True, remaining_gib=37.5,
        parameter_gib=15.0, activation_gib=2.0, kv_cache_gib=24.0, overhead_gib=1.5,
    )
    assert cell.fits is True
    assert cell.remaining_gib == pytest.approx(37.5)
    assert cell.total_memory_gib == pytest.approx(42.5)


def test_budget_cell_exceeds():
    cell = BudgetCell(
        max_seq_len=32768, max_active_seqs=256,
        total_memory_gib=95.0, fits=False, remaining_gib=-15.0,
        parameter_gib=15.0, activation_gib=5.0, kv_cache_gib=73.0, overhead_gib=2.0,
    )
    assert cell.fits is False
    assert cell.remaining_gib < 0


# ---------------------------------------------------------------------------
# BudgetResult helpers
# ---------------------------------------------------------------------------

def _make_result() -> BudgetResult:
    """Build a small synthetic BudgetResult for testing."""
    seq_lengths = [1024, 4096, 8192]
    seq_counts = [1, 32, 128]
    cells: list[list[BudgetCell]] = []
    max_seqs_list: list[int | None] = []
    gpu_mem = 80.0

    for sl in seq_lengths:
        row: list[BudgetCell] = []
        for sc in seq_counts:
            total = 15.0 + (sl / 1024) * (sc / 32) * 5.0
            fits = total <= gpu_mem
            row.append(BudgetCell(
                max_seq_len=sl, max_active_seqs=sc,
                total_memory_gib=total, fits=fits, remaining_gib=gpu_mem - total,
                parameter_gib=15.0, activation_gib=1.0,
                kv_cache_gib=total - 17.0, overhead_gib=1.0,
            ))
        cells.append(row)

        if sl == 1024:
            max_seqs_list.append(512)
        elif sl == 4096:
            max_seqs_list.append(128)
        else:
            max_seqs_list.append(32)

    return BudgetResult(
        model_id="test/model-8B",
        gpu_memory_gib=gpu_mem,
        architecture="llama",
        quantization_method="dense",
        kv_cache_spec_type="full",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        total_gpus=1,
        model_max_seq_len=131072,
        seq_lengths=seq_lengths,
        seq_counts=seq_counts,
        cells=cells,
        max_seqs=max_seqs_list,
    )


def test_budget_result_cell_lookup():
    result = _make_result()
    cell = result.cell(4096, 32)
    assert cell is not None
    assert cell.max_seq_len == 4096
    assert cell.max_active_seqs == 32


def test_budget_result_cell_lookup_missing():
    result = _make_result()
    assert result.cell(9999, 32) is None
    assert result.cell(4096, 9999) is None


def test_budget_result_max_seqs_at():
    result = _make_result()
    assert result.max_seqs_at(1024) == 512
    assert result.max_seqs_at(4096) == 128
    assert result.max_seqs_at(9999) is None


def test_budget_result_as_dict():
    result = _make_result()
    d = result.as_dict()

    assert d["model_id"] == "test/model-8B"
    assert d["gpu_memory_gib"] == 80.0
    assert d["architecture"] == "llama"
    assert d["tensor_parallel_size"] == 1
    assert d["model_max_seq_len"] == 131072
    assert len(d["seq_lengths"]) == 3
    assert len(d["seq_counts"]) == 3
    assert len(d["matrix"]) == 3
    assert len(d["matrix"][0]) == 3
    assert "total_gib" in d["matrix"][0][0]
    assert "fits" in d["matrix"][0][0]
    assert len(d["max_seqs_per_context"]) == 3


def test_budget_result_render_table():
    result = _make_result()
    table = result.render_table()

    assert "test/model-8B" in table
    assert "80.0 GiB" in table
    assert "1,024" in table
    assert "4,096" in table
    assert "8,192" in table
    assert "Max Seqs" in table
    assert "512" in table


def test_budget_result_render_table_exceeds():
    result = _make_result()
    table = result.render_table()
    has_exceeds = any(not c.fits for row in result.cells for c in row)
    if has_exceeds:
        assert "---" in table


def test_budget_result_render_table_multi_gpu():
    result = _make_result()
    result = BudgetResult(
        **{**result.__dict__,
           "tensor_parallel_size": 4,
           "total_gpus": 4}
    )
    table = result.render_table()
    assert "TP=4" in table


def test_budget_result_render_html():
    result = _make_result()
    html = result.render_html()

    assert "<!DOCTYPE html>" in html
    assert "test/model-8B" in html
    assert "80.0 GiB" in html
    assert "llama" in html
    assert "dense" in html
    assert "1,024" in html or "1024" in html
    assert "Max Seqs" in html


def test_budget_result_render_html_tooltip():
    result = _make_result()
    html = result.render_html()
    assert "Parameters:" in html
    assert "KV Cache:" in html
