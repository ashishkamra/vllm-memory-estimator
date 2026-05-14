"""Token budget matrix: sweep (seq_len, max_seqs) to find what fits in GPU memory."""
from __future__ import annotations

import dataclasses
import html as html_mod
from dataclasses import dataclass

from .config_utils import max_model_len
from .estimator import EstimatorInputs
from .estimator import estimate_memory
from .estimator import prepare_summary

_DEFAULT_SEQ_COUNTS = [1, 4, 8, 16, 32, 64, 128, 256, 512]

_MAX_SEQS_UPPER_BOUND = 4096


def _default_seq_lengths(model_max: int) -> list[int]:
    lengths: list[int] = []
    length = 256
    while length <= model_max:
        lengths.append(length)
        length *= 2
    if lengths and lengths[-1] != model_max:
        lengths.append(model_max)
    if not lengths:
        lengths.append(model_max)
    return lengths


@dataclass
class BudgetCell:
    max_seq_len: int
    max_active_seqs: int
    total_memory_gib: float
    fits: bool
    remaining_gib: float
    parameter_gib: float
    activation_gib: float
    kv_cache_gib: float
    overhead_gib: float


@dataclass
class BudgetResult:
    model_id: str
    gpu_memory_gib: float
    architecture: str
    quantization_method: str
    kv_cache_spec_type: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    enable_expert_parallel: bool
    total_gpus: int
    model_max_seq_len: int
    seq_lengths: list[int]
    seq_counts: list[int]
    cells: list[list[BudgetCell]]
    max_seqs: list[int | None]

    def cell(self, seq_len: int, seq_count: int) -> BudgetCell | None:
        try:
            row = self.seq_lengths.index(seq_len)
            col = self.seq_counts.index(seq_count)
        except ValueError:
            return None
        return self.cells[row][col]

    def max_seqs_at(self, seq_len: int) -> int | None:
        try:
            row = self.seq_lengths.index(seq_len)
        except ValueError:
            return None
        return self.max_seqs[row]

    def as_dict(self) -> dict:
        matrix = []
        for row in self.cells:
            matrix.append([
                {
                    "seq_len": c.max_seq_len,
                    "seqs": c.max_active_seqs,
                    "total_gib": round(c.total_memory_gib, 3),
                    "fits": c.fits,
                    "remaining_gib": round(c.remaining_gib, 3),
                    "parameter_gib": round(c.parameter_gib, 3),
                    "activation_gib": round(c.activation_gib, 3),
                    "kv_cache_gib": round(c.kv_cache_gib, 3),
                    "overhead_gib": round(c.overhead_gib, 3),
                }
                for c in row
            ])
        max_seqs_list = [
            {"seq_len": sl, "max_seqs": ms}
            for sl, ms in zip(self.seq_lengths, self.max_seqs)
        ]
        return {
            "model_id": self.model_id,
            "gpu_memory_gib": self.gpu_memory_gib,
            "architecture": self.architecture,
            "quantization": self.quantization_method,
            "kv_cache_spec_type": self.kv_cache_spec_type,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "enable_expert_parallel": self.enable_expert_parallel,
            "total_gpus": self.total_gpus,
            "model_max_seq_len": self.model_max_seq_len,
            "seq_lengths": self.seq_lengths,
            "seq_counts": self.seq_counts,
            "matrix": matrix,
            "max_seqs_per_context": max_seqs_list,
        }

    def render_table(self) -> str:
        gpu_mem = self.gpu_memory_gib
        header = f"Token Budget: {self.model_id}"
        dims: list[str] = []
        if self.tensor_parallel_size > 1:
            dims.append(f"TP={self.tensor_parallel_size}")
        if self.pipeline_parallel_size > 1:
            dims.append(f"PP={self.pipeline_parallel_size}")
        if self.data_parallel_size > 1:
            dims.append(f"DP={self.data_parallel_size}")
        if self.enable_expert_parallel:
            dims.append("EP")
        dim_str = f", {', '.join(dims)}" if dims else ""
        header += f" ({gpu_mem:.1f} GiB per GPU{dim_str})"

        col_headers = [f"{sc} seq" for sc in self.seq_counts] + ["Max Seqs"]
        col_w = max(max((len(h) for h in col_headers), default=7), 7)
        row_label_w = max(max((len(f"{sl:,}") for sl in self.seq_lengths), default=9), 9)

        sep_parts = ["─" * (row_label_w + 2)]
        for _ in col_headers:
            sep_parts.append("─" * (col_w + 2))
        sep_line = "┼".join(sep_parts)

        hdr_parts = [f"{'Context':>{row_label_w + 1}} "]
        for h in col_headers:
            hdr_parts.append(f" {h:>{col_w}} ")
        hdr_line = "│".join(hdr_parts)

        border = "═" * len(sep_line)
        lines = [header, border, hdr_line, sep_line]

        for row_idx, sl in enumerate(self.seq_lengths):
            parts = [f"{sl:>{row_label_w},} "]
            for col_idx, _sc in enumerate(self.seq_counts):
                c = self.cells[row_idx][col_idx]
                if c.fits:
                    val = f"{c.total_memory_gib:.1f}"
                else:
                    val = "---"
                parts.append(f" {val:>{col_w}} ")
            ms = self.max_seqs[row_idx]
            if ms is None or ms == 0:
                ms_str = "---"
            elif ms >= _MAX_SEQS_UPPER_BOUND:
                ms_str = f"{_MAX_SEQS_UPPER_BOUND}+"
            else:
                ms_str = str(ms)
            parts.append(f" {ms_str:>{col_w}} ")
            lines.append("│".join(parts))

        lines.append(border)
        lines.append(f"Values: estimated per-GPU memory (GiB). --- = exceeds {gpu_mem:.1f} GiB.")
        return "\n".join(lines)

    def render_html(self) -> str:
        return _render_html_report(self)


def _estimate_single(summary, seq_len: int, seqs: int, gpu_memory_gib: float) -> BudgetCell:
    modified = dataclasses.replace(summary, max_seq_len=seq_len, max_active_seqs=seqs)
    est = estimate_memory(modified)
    total = est.total_with_vllm.nominal_gib
    return BudgetCell(
        max_seq_len=seq_len,
        max_active_seqs=seqs,
        total_memory_gib=total,
        fits=total <= gpu_memory_gib,
        remaining_gib=gpu_memory_gib - total,
        parameter_gib=est.parameters.nominal_gib,
        activation_gib=est.activations.nominal_gib,
        kv_cache_gib=est.kv_cache.nominal_gib,
        overhead_gib=est.workspace.nominal_gib + est.vllm_overhead.nominal_gib,
    )


def _find_max_seqs(summary, seq_len: int, gpu_memory_gib: float) -> int | None:
    lo, hi = 1, _MAX_SEQS_UPPER_BOUND
    cell = _estimate_single(summary, seq_len, lo, gpu_memory_gib)
    if not cell.fits:
        return None
    cell = _estimate_single(summary, seq_len, hi, gpu_memory_gib)
    if cell.fits:
        return hi
    while lo < hi - 1:
        mid = (lo + hi) // 2
        cell = _estimate_single(summary, seq_len, mid, gpu_memory_gib)
        if cell.fits:
            lo = mid
        else:
            hi = mid
    return lo


def compute_budget(
    model_id: str,
    gpu_memory_gib: float,
    *,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    quantization: str | None = None,
    dtype: str | None = None,
    kv_cache_dtype: str | None = None,
    enforce_eager: bool = False,
    block_size: int | None = None,
    revision: str | None = None,
    use_cache: bool = True,
    seq_lengths: list[int] | None = None,
    seq_counts: list[int] | None = None,
    max_num_batched_tokens: int | None = None,
) -> BudgetResult:
    inputs = EstimatorInputs(
        model_id=model_id,
        max_seq_len=1,
        max_active_seqs=1,
        revision=revision,
        enforce_eager=enforce_eager,
        max_num_batched_tokens=max_num_batched_tokens,
        kv_cache_dtype=kv_cache_dtype,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        block_size=block_size,
        quantization=quantization,
        use_cache=use_cache,
    )
    summary = prepare_summary(inputs)
    model_max = max_model_len(summary.config)

    sweep_lengths = seq_lengths if seq_lengths is not None else _default_seq_lengths(model_max)
    sweep_counts = seq_counts if seq_counts is not None else list(_DEFAULT_SEQ_COUNTS)

    cells: list[list[BudgetCell]] = []
    for sl in sweep_lengths:
        row: list[BudgetCell] = []
        for sc in sweep_counts:
            row.append(_estimate_single(summary, sl, sc, gpu_memory_gib))
        cells.append(row)

    max_seqs_list: list[int | None] = []
    for sl in sweep_lengths:
        max_seqs_list.append(_find_max_seqs(summary, sl, gpu_memory_gib))

    sample_est = estimate_memory(
        dataclasses.replace(summary, max_seq_len=sweep_lengths[0], max_active_seqs=1)
    )

    return BudgetResult(
        model_id=model_id,
        gpu_memory_gib=gpu_memory_gib,
        architecture=summary.architecture,
        quantization_method=summary.quantization.method or "dense",
        kv_cache_spec_type=sample_est.kv_cache_spec_type,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        total_gpus=summary.total_gpus,
        model_max_seq_len=model_max,
        seq_lengths=sweep_lengths,
        seq_counts=sweep_counts,
        cells=cells,
        max_seqs=max_seqs_list,
    )


def _render_html_report(result: BudgetResult) -> str:
    esc = html_mod.escape
    gpu_mem = result.gpu_memory_gib

    dims: list[str] = []
    if result.tensor_parallel_size > 1:
        dims.append(f"TP={result.tensor_parallel_size}")
    if result.pipeline_parallel_size > 1:
        dims.append(f"PP={result.pipeline_parallel_size}")
    if result.data_parallel_size > 1:
        dims.append(f"DP={result.data_parallel_size}")
    if result.enable_expert_parallel:
        dims.append("EP")
    parallel_str = ", ".join(dims) if dims else "single GPU"

    rows_html: list[str] = []
    for row_idx, sl in enumerate(result.seq_lengths):
        tds: list[str] = [f'<td class="row-label">{sl:,}</td>']
        for col_idx in range(len(result.seq_counts)):
            c = result.cells[row_idx][col_idx]
            tooltip = (
                f"Parameters: {c.parameter_gib:.2f} GiB&#10;"
                f"KV Cache: {c.kv_cache_gib:.2f} GiB&#10;"
                f"Activations: {c.activation_gib:.2f} GiB&#10;"
                f"Overhead: {c.overhead_gib:.2f} GiB&#10;"
                f"Total: {c.total_memory_gib:.2f} GiB"
            )
            if c.fits:
                pct = c.total_memory_gib / gpu_mem
                green = int(200 * (1 - pct)) + 55
                green = max(55, min(255, green))
                style = f"background-color: rgba(0,{green},0,0.15);"
                val = f"{c.total_memory_gib:.1f}"
            else:
                style = "background-color: rgba(200,0,0,0.10); color: #999;"
                val = "---"
            tds.append(f'<td style="{style}" title="{tooltip}">{val}</td>')
        ms = result.max_seqs[row_idx]
        if ms is None or ms == 0:
            ms_str = "---"
        elif ms >= _MAX_SEQS_UPPER_BOUND:
            ms_str = f"{_MAX_SEQS_UPPER_BOUND}+"
        else:
            ms_str = f"{ms:,}"
        tds.append(f'<td class="max-seqs">{ms_str}</td>')
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    col_headers = "".join(
        f"<th>{sc} seq</th>" for sc in result.seq_counts
    ) + "<th>Max Seqs</th>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Token Budget: {esc(result.model_id)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2em auto; max-width: 1200px; color: #333; }}
  .summary {{ display: flex; flex-wrap: wrap; gap: 1em; margin-bottom: 2em; }}
  .card {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;
           padding: 1em 1.5em; min-width: 180px; }}
  .card .label {{ font-size: 0.85em; color: #666; margin-bottom: 0.2em; }}
  .card .value {{ font-size: 1.2em; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  th, td {{ border: 1px solid #dee2e6; padding: 0.5em 0.8em; text-align: right; }}
  th {{ background: #f1f3f5; position: sticky; top: 0; }}
  .row-label {{ font-weight: 600; text-align: right; background: #f8f9fa; }}
  .max-seqs {{ font-weight: 600; background: #e7f5ff; }}
  .footer {{ margin-top: 1em; font-size: 0.85em; color: #888; }}
</style>
</head>
<body>
<h1>Token Budget: {esc(result.model_id)}</h1>
<div class="summary">
  <div class="card">
    <div class="label">GPU Memory</div>
    <div class="value">{gpu_mem:.1f} GiB</div>
  </div>
  <div class="card">
    <div class="label">Parallelism</div>
    <div class="value">{esc(parallel_str)}</div>
  </div>
  <div class="card">
    <div class="label">Quantization</div>
    <div class="value">{esc(result.quantization_method)}</div>
  </div>
  <div class="card">
    <div class="label">Architecture</div>
    <div class="value">{esc(result.architecture)}</div>
  </div>
  <div class="card">
    <div class="label">KV Cache</div>
    <div class="value">{esc(result.kv_cache_spec_type)}</div>
  </div>
  <div class="card">
    <div class="label">Max Context</div>
    <div class="value">{result.model_max_seq_len:,}</div>
  </div>
</div>
<table>
<thead>
<tr><th>Context Length</th>{col_headers}</tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>
<p class="footer">
  Values: estimated per-GPU memory (GiB). <b>---</b> = exceeds {gpu_mem:.1f} GiB limit.
  Hover over cells for component breakdown.
</p>
</body>
</html>"""
