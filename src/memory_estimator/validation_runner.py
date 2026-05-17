"""Run the memory estimator against a validation database and compare results.

Loads records from a validation_db.json, calls estimate_from_inputs() for each
unique configuration, and produces a report comparing predicted vs actual
memory values from vLLM startup logs.
"""
from __future__ import annotations

import html
import json
import logging
import statistics
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

from .estimator import EstimatorInputs
from .estimator import estimate_from_inputs
from .reports import MemoryEstimate

logger = logging.getLogger(__name__)

_GIB = 1024**3


# ---------------------------------------------------------------------------
# GPU memory lookup (GiB per device)
# ---------------------------------------------------------------------------

_GPU_MEMORY_GIB: dict[str, float] = {
    "H200": 141.0,
    "H100": 80.0,
    "H100-80": 80.0,
    "B200": 192.0,
    "MI300X": 192.0,
    "A100-80": 80.0,
    "A100-40": 40.0,
    "A100": 80.0,
    "L40S": 48.0,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of comparing one validation record against estimator output."""
    uuid: str
    model_id: str
    accelerator: str
    actual_model_load_gib: float
    estimated_params_nominal_gib: float
    estimated_params_lower_gib: float
    estimated_params_upper_gib: float
    params_within_bounds: bool
    params_error_pct: float
    actual_kv_per_token_bytes: float | None
    estimated_kv_per_token_bytes: float | None
    kv_per_token_ratio: float | None
    kv_per_token_error_pct: float | None
    actual_overhead_gib: float | None
    estimated_overhead_gib: float | None
    overhead_error_pct: float | None
    kv_cache_spec_type: str
    quantization: str | None
    tensor_parallel_size: int
    max_seq_len: int
    error: str | None = None


@dataclass
class ValidationReport:
    """Aggregate report across all validation records."""
    comparisons: list[ComparisonResult]
    skipped: list[dict]
    total_records: int
    total_compared: int
    total_skipped: int
    params_in_bounds_count: int
    params_in_bounds_pct: float
    params_mean_abs_error_pct: float
    params_median_abs_error_pct: float
    params_max_abs_error_pct: float
    kv_compared_count: int
    kv_mean_abs_error_pct: float
    kv_median_abs_error_pct: float
    kv_max_abs_error_pct: float
    overhead_compared_count: int
    overhead_mean_abs_error_pct: float
    overhead_median_abs_error_pct: float
    overhead_max_abs_error_pct: float
    worst_offenders: list[ComparisonResult]
    per_model_stats: dict[str, dict]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_records": self.total_records,
                "total_compared": self.total_compared,
                "total_skipped": self.total_skipped,
            },
            "aggregate": {
                "params_in_bounds_count": self.params_in_bounds_count,
                "params_in_bounds_pct": round(self.params_in_bounds_pct, 2),
                "params_mean_abs_error_pct": round(self.params_mean_abs_error_pct, 2),
                "params_median_abs_error_pct": round(self.params_median_abs_error_pct, 2),
                "params_max_abs_error_pct": round(self.params_max_abs_error_pct, 2),
                "kv_compared_count": self.kv_compared_count,
                "kv_mean_abs_error_pct": round(self.kv_mean_abs_error_pct, 2),
                "kv_median_abs_error_pct": round(self.kv_median_abs_error_pct, 2),
                "kv_max_abs_error_pct": round(self.kv_max_abs_error_pct, 2),
                "overhead_compared_count": self.overhead_compared_count,
                "overhead_mean_abs_error_pct": round(self.overhead_mean_abs_error_pct, 2),
                "overhead_median_abs_error_pct": round(self.overhead_median_abs_error_pct, 2),
                "overhead_max_abs_error_pct": round(self.overhead_max_abs_error_pct, 2),
            },
            "per_model": self.per_model_stats,
            "comparisons": [asdict(c) for c in self.comparisons],
            "skipped": self.skipped,
            "worst_offenders": [asdict(c) for c in self.worst_offenders],
        }

    def render_summary(self) -> str:
        hdr = (
            f"{'Model':<50s} {'Records':>7s} {'In-bounds':>10s}"
            f" {'Wt err%':>8s} {'Wt max%':>8s}"
            f" {'KV err%':>8s} {'KV max%':>8s}"
            f" {'OH err%':>8s} {'OH max%':>8s}"
        )
        sep = "─" * len(hdr)
        lines = [sep, hdr, sep]

        for model in sorted(self.per_model_stats):
            s = self.per_model_stats[model]
            name = model if len(model) <= 48 else model[:46] + ".."
            in_b = f"{s['in_bounds']}/{s['records']}"
            kv_mean = f"{s['kv_mean_error_pct']:+7.1f}%" if s.get("kv_mean_error_pct") is not None else "     n/a"
            kv_max = f"{s['kv_max_abs_error_pct']:+7.1f}%" if s.get("kv_max_abs_error_pct") is not None else "     n/a"
            oh_mean = f"{s['oh_mean_error_pct']:+7.1f}%" if s.get("oh_mean_error_pct") is not None else "     n/a"
            oh_max = f"{s['oh_max_abs_error_pct']:+7.1f}%" if s.get("oh_max_abs_error_pct") is not None else "     n/a"
            lines.append(
                f"{name:<50s} {s['records']:>7d} {in_b:>10s}"
                f" {s['mean_error_pct']:>+7.1f}% {s['max_abs_error_pct']:>+7.1f}%"
                f" {kv_mean} {kv_max}"
                f" {oh_mean} {oh_max}"
            )

        lines.append(sep)
        in_b = f"{self.params_in_bounds_count}/{self.total_compared}"
        lines.append(
            f"{'TOTAL':<50s} {self.total_compared:>7d} {in_b:>10s}"
            f" {self.params_mean_abs_error_pct:>7.1f}% {self.params_max_abs_error_pct:>7.1f}%"
            f" {self.kv_mean_abs_error_pct:>7.1f}% {self.kv_max_abs_error_pct:>7.1f}%"
            f" {self.overhead_mean_abs_error_pct:>7.1f}%"
            f" {self.overhead_max_abs_error_pct:>7.1f}%"
        )
        lines.append(sep)

        if self.total_skipped > 0:
            lines.append(f"\nSkipped: {self.total_skipped} records")
            models = {s.get("model_id", "?") for s in self.skipped}
            for m in sorted(models):
                lines.append(f"  {m}")

        return "\n".join(lines)

    def render_html(self) -> str:
        return _render_html_report(self)


# ---------------------------------------------------------------------------
# Config key and input mapping
# ---------------------------------------------------------------------------

def _config_key(ei: dict) -> tuple:
    return (
        ei["model_id"], ei["max_seq_len"], ei["tensor_parallel_size"],
        ei["pipeline_parallel_size"], ei["data_parallel_size"],
        ei.get("quantization"), ei.get("kv_cache_dtype"),
        ei.get("enforce_eager", False), ei.get("dtype"),
    )


def _build_estimator_inputs(ei: dict) -> EstimatorInputs:
    return EstimatorInputs(
        model_id=ei["model_id"],
        max_seq_len=ei["max_seq_len"],
        tensor_parallel_size=ei.get("tensor_parallel_size", 1),
        pipeline_parallel_size=ei.get("pipeline_parallel_size", 1),
        data_parallel_size=ei.get("data_parallel_size", 1),
        quantization=ei.get("quantization"),
        kv_cache_dtype=ei.get("kv_cache_dtype"),
        enforce_eager=ei.get("enforce_eager", False),
        dtype=ei.get("dtype"),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _compare_record(
    uuid: str,
    record: dict,
    estimate: MemoryEstimate,
) -> ComparisonResult:
    log_mem = record["log_memory"]
    ei = record["estimator_inputs"]
    actual = log_mem["model_load_gib"]
    nominal = estimate.parameters.nominal_gib

    params_error_pct = ((actual - nominal) / nominal * 100) if nominal > 0 else 0.0
    within = estimate.parameters.lower_gib <= actual <= estimate.parameters.upper_gib

    actual_kv_pt: float | None = None
    est_kv_pt: float | None = None
    kv_ratio: float | None = None

    max_sl = ei["max_seq_len"]

    # Per-token KV comparison is only meaningful for specs where cache scales
    # linearly with sequence length (full, mla).  For sliding window, chunked
    # local, or mamba the cache size is decoupled from max_seq_len, so the
    # per-token metric would be misleading.
    kv_comparable = estimate.kv_cache_spec_type in ("full", "mla")

    if kv_comparable:
        avail_kv = log_mem.get("available_kv_cache_gib")
        kv_tokens = log_mem.get("kv_cache_tokens")
        if avail_kv and kv_tokens and kv_tokens > 0:
            actual_kv_pt = avail_kv * _GIB / kv_tokens

        max_active = ei.get("max_active_seqs") or 256
        total_tokens = max_active * max_sl
        if estimate.kv_cache.nominal_gib > 0 and total_tokens > 0:
            est_kv_pt = estimate.kv_cache.nominal_gib * _GIB / total_tokens

    kv_error_pct: float | None = None
    if actual_kv_pt and est_kv_pt and est_kv_pt > 0:
        kv_ratio = actual_kv_pt / est_kv_pt
        kv_error_pct = (actual_kv_pt - est_kv_pt) / est_kv_pt * 100

    # --- Non-weight overhead comparison ---
    # actual_overhead = usable_gpu_memory - weights - kv_cache_allocation
    # estimated_overhead = activations + workspace + vllm_overhead
    accelerator = record["csv_metadata"]["accelerator"]
    actual_oh: float | None = None
    est_oh: float | None = None
    oh_error: float | None = None

    gpu_mem = _GPU_MEMORY_GIB.get(accelerator)
    avail_kv_gib = log_mem.get("available_kv_cache_gib")
    gpu_util = log_mem.get("gpu_memory_utilization", 0.9)
    if gpu_mem and avail_kv_gib is not None:
        usable = gpu_mem * gpu_util
        actual_oh = usable - actual - avail_kv_gib
        est_oh = (estimate.activations.nominal_gib + estimate.workspace.nominal_gib
                  + estimate.vllm_overhead.nominal_gib)
        if est_oh > 0:
            oh_error = (actual_oh - est_oh) / est_oh * 100

    return ComparisonResult(
        uuid=uuid,
        model_id=ei["model_id"],
        accelerator=accelerator,
        actual_model_load_gib=actual,
        estimated_params_nominal_gib=nominal,
        estimated_params_lower_gib=estimate.parameters.lower_gib,
        estimated_params_upper_gib=estimate.parameters.upper_gib,
        params_within_bounds=within,
        params_error_pct=params_error_pct,
        actual_kv_per_token_bytes=actual_kv_pt,
        estimated_kv_per_token_bytes=est_kv_pt,
        kv_per_token_ratio=kv_ratio,
        kv_per_token_error_pct=kv_error_pct,
        actual_overhead_gib=actual_oh,
        estimated_overhead_gib=est_oh,
        overhead_error_pct=oh_error,
        kv_cache_spec_type=estimate.kv_cache_spec_type,
        quantization=ei.get("quantization"),
        tensor_parallel_size=ei.get("tensor_parallel_size", 1),
        max_seq_len=max_sl,
    )


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def _compute_per_model_stats(
    comparisons: list[ComparisonResult],
) -> dict[str, dict]:
    from collections import defaultdict
    by_model: dict[str, list[ComparisonResult]] = defaultdict(list)
    for c in comparisons:
        by_model[c.model_id].append(c)

    result: dict[str, dict] = {}
    for model, comps in sorted(by_model.items()):
        abs_errs = [abs(c.params_error_pct) for c in comps]
        signed_errs = [c.params_error_pct for c in comps]
        in_bounds = sum(1 for c in comps if c.params_within_bounds)
        kv_errs = [c.kv_per_token_error_pct for c in comps if c.kv_per_token_error_pct is not None]
        kv_abs_errs = [abs(e) for e in kv_errs]
        oh_errs = [c.overhead_error_pct for c in comps if c.overhead_error_pct is not None]
        oh_abs_errs = [abs(e) for e in oh_errs]
        result[model] = {
            "records": len(comps),
            "in_bounds": in_bounds,
            "in_bounds_pct": round(in_bounds / len(comps) * 100, 1) if comps else 0,
            "mean_error_pct": round(statistics.mean(signed_errs), 2),
            "mean_abs_error_pct": round(statistics.mean(abs_errs), 2),
            "median_abs_error_pct": round(statistics.median(abs_errs), 2),
            "max_abs_error_pct": round(max(abs_errs), 2),
            "kv_compared": len(kv_errs),
            "kv_mean_error_pct": round(statistics.mean(kv_errs), 2) if kv_errs else None,
            "kv_max_abs_error_pct": round(max(kv_abs_errs), 2) if kv_abs_errs else None,
            "oh_compared": len(oh_errs),
            "oh_mean_error_pct": round(statistics.mean(oh_errs), 2) if oh_errs else None,
            "oh_max_abs_error_pct": round(max(oh_abs_errs), 2) if oh_abs_errs else None,
        }
    return result


def compute_aggregate_stats(
    comparisons: list[ComparisonResult],
    skipped: list[dict],
    total_records: int,
) -> ValidationReport:
    abs_errs = [abs(c.params_error_pct) for c in comparisons]
    in_bounds = sum(1 for c in comparisons if c.params_within_bounds)
    n = len(comparisons)

    kv_errs = [c.kv_per_token_error_pct for c in comparisons
               if c.kv_per_token_error_pct is not None]
    kv_abs_errs = [abs(e) for e in kv_errs]

    oh_errs = [c.overhead_error_pct for c in comparisons
               if c.overhead_error_pct is not None]
    oh_abs_errs = [abs(e) for e in oh_errs]

    worst = sorted(comparisons, key=lambda c: abs(c.params_error_pct), reverse=True)[:10]
    per_model = _compute_per_model_stats(comparisons)

    return ValidationReport(
        comparisons=comparisons,
        skipped=skipped,
        total_records=total_records,
        total_compared=n,
        total_skipped=len(skipped),
        params_in_bounds_count=in_bounds,
        params_in_bounds_pct=round(in_bounds / n * 100, 2) if n else 0,
        params_mean_abs_error_pct=round(statistics.mean(abs_errs), 2) if abs_errs else 0,
        params_median_abs_error_pct=round(statistics.median(abs_errs), 2) if abs_errs else 0,
        params_max_abs_error_pct=round(max(abs_errs), 2) if abs_errs else 0,
        kv_compared_count=len(kv_errs),
        kv_mean_abs_error_pct=round(statistics.mean(kv_abs_errs), 2) if kv_abs_errs else 0,
        kv_median_abs_error_pct=round(statistics.median(kv_abs_errs), 2) if kv_abs_errs else 0,
        kv_max_abs_error_pct=round(max(kv_abs_errs), 2) if kv_abs_errs else 0,
        overhead_compared_count=len(oh_errs),
        overhead_mean_abs_error_pct=round(statistics.mean(oh_abs_errs), 2) if oh_abs_errs else 0,
        overhead_median_abs_error_pct=round(statistics.median(oh_abs_errs), 2) if oh_abs_errs else 0,
        overhead_max_abs_error_pct=round(max(oh_abs_errs), 2) if oh_abs_errs else 0,
        worst_offenders=worst,
        per_model_stats=per_model,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation(
    db_path: Path,
    *,
    model_filter: str | None = None,
) -> ValidationReport:
    """Run estimator against all configs in the validation DB and compare."""
    with open(db_path) as f:
        db = json.load(f)

    records = db["records"]
    if model_filter:
        records = {
            u: r for u, r in records.items()
            if model_filter.lower() in r["estimator_inputs"]["model_id"].lower()
        }

    config_to_uuids: dict[tuple, list[str]] = {}
    config_to_ei: dict[tuple, dict] = {}
    for uuid, rec in records.items():
        key = _config_key(rec["estimator_inputs"])
        config_to_uuids.setdefault(key, []).append(uuid)
        if key not in config_to_ei:
            config_to_ei[key] = rec["estimator_inputs"]

    sorted_keys = sorted(config_to_ei.keys(), key=lambda k: k[0])

    estimates: dict[tuple, MemoryEstimate | None] = {}
    skipped: list[dict] = []
    failed_models: set[str] = set()

    total_configs = len(sorted_keys)
    for i, key in enumerate(sorted_keys):
        ei = config_to_ei[key]
        model_id = ei["model_id"]

        if model_id in failed_models:
            estimates[key] = None
            for uid in config_to_uuids[key]:
                skipped.append({"uuid": uid, "model_id": model_id, "reason": "model_failed"})
            continue

        logger.info(
            "[%d/%d] Estimating %s (tp=%s seq=%s)",
            i + 1, total_configs, model_id,
            ei.get("tensor_parallel_size", 1), ei.get("max_seq_len"),
        )

        try:
            inputs = _build_estimator_inputs(ei)
            _, estimate = estimate_from_inputs(inputs)
            estimates[key] = estimate
        except Exception as exc:
            logger.warning("Failed to estimate %s: %s", model_id, exc)
            estimates[key] = None
            failed_models.add(model_id)
            for uid in config_to_uuids[key]:
                skipped.append({
                    "uuid": uid, "model_id": model_id,
                    "reason": str(exc)[:200],
                })

    comparisons: list[ComparisonResult] = []
    for key in sorted_keys:
        est = estimates[key]
        if est is None:
            continue
        for uuid in config_to_uuids[key]:
            comp = _compare_record(uuid, records[uuid], est)
            comparisons.append(comp)

    return compute_aggregate_stats(comparisons, skipped, len(records))


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _render_html_report(report: ValidationReport) -> str:
    h = html.escape

    rows_per_model = []
    for model in sorted(report.per_model_stats):
        s = report.per_model_stats[model]
        in_b = s["in_bounds"]
        total = s["records"]
        cls = "pass" if in_b == total else "fail"
        kv_mean = f"{s['kv_mean_error_pct']:+.2f}%" if s.get("kv_mean_error_pct") is not None else "-"
        kv_max = f"{s['kv_max_abs_error_pct']:+.2f}%" if s.get("kv_max_abs_error_pct") is not None else "-"
        rows_per_model.append(
            f'<tr class="{cls}">'
            f"<td>{h(model)}</td>"
            f"<td>{total}</td>"
            f"<td>{in_b}/{total}</td>"
            f"<td>{s['mean_error_pct']:+.2f}%</td>"
            f"<td>{s['max_abs_error_pct']:+.2f}%</td>"
            f"<td>{kv_mean}</td>"
            f"<td>{kv_max}</td>"
            f"</tr>"
        )

    rows_worst = []
    for c in report.worst_offenders:
        cls = "pass" if c.params_within_bounds else "fail"
        rows_worst.append(
            f'<tr class="{cls}">'
            f"<td>{h(c.uuid[:8])}</td>"
            f"<td>{h(c.model_id)}</td>"
            f"<td>{h(c.accelerator)}</td>"
            f"<td>{c.tensor_parallel_size}</td>"
            f"<td>{c.max_seq_len}</td>"
            f"<td>{h(str(c.quantization or '-'))}</td>"
            f"<td>{c.actual_model_load_gib:.2f}</td>"
            f"<td>{c.estimated_params_nominal_gib:.2f}</td>"
            f"<td>{c.params_error_pct:+.2f}%</td>"
            f"<td>{'PASS' if c.params_within_bounds else 'FAIL'}</td>"
            f"</tr>"
        )

    rows_all = []
    for c in sorted(report.comparisons, key=lambda x: (x.model_id, x.uuid)):
        cls = "pass" if c.params_within_bounds else "fail"
        act_kv = f"{c.actual_kv_per_token_bytes:.1f}" if c.actual_kv_per_token_bytes else "-"
        est_kv = f"{c.estimated_kv_per_token_bytes:.1f}" if c.estimated_kv_per_token_bytes else "-"
        kv_err = f"{c.kv_per_token_error_pct:+.2f}%" if c.kv_per_token_error_pct is not None else "-"
        rows_all.append(
            f'<tr class="{cls}">'
            f"<td>{h(c.uuid[:8])}</td>"
            f"<td>{h(c.model_id)}</td>"
            f"<td>{h(c.accelerator)}</td>"
            f"<td>{c.tensor_parallel_size}</td>"
            f"<td>{c.max_seq_len}</td>"
            f"<td>{h(str(c.quantization or '-'))}</td>"
            f"<td>{c.actual_model_load_gib:.2f}</td>"
            f"<td>{c.estimated_params_nominal_gib:.2f}</td>"
            f"<td>[{c.estimated_params_lower_gib:.2f}, {c.estimated_params_upper_gib:.2f}]</td>"
            f"<td>{c.params_error_pct:+.2f}%</td>"
            f"<td>{'PASS' if c.params_within_bounds else 'FAIL'}</td>"
            f"<td>{act_kv}</td>"
            f"<td>{est_kv}</td>"
            f"<td>{kv_err}</td>"
            f"</tr>"
        )

    skipped_rows = []
    for s in report.skipped:
        skipped_rows.append(
            f"<tr>"
            f"<td>{h(s.get('uuid', '?')[:8])}</td>"
            f"<td>{h(s.get('model_id', '?'))}</td>"
            f"<td>{h(s.get('reason', '?')[:80])}</td>"
            f"</tr>"
        )

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vLLM Memory Estimator Validation Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2rem; background: #fafafa; color: #333; }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ color: #16213e; margin-top: 2rem; }}
  .summary {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1rem 1.5rem;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 140px; }}
  .card .value {{ font-size: 1.8rem; font-weight: 700; }}
  .card .label {{ font-size: 0.85rem; color: #666; margin-top: 0.25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0 1.5rem; }}
  th, td {{ padding: 0.4rem 0.7rem; text-align: left; border-bottom: 1px solid #e0e0e0;
            font-size: 0.85rem; }}
  th {{ background: #16213e; color: #fff; position: sticky; top: 0; }}
  tr:hover {{ background: #f0f4ff; }}
  tr.pass td:last-child {{ color: #2e7d32; font-weight: 600; }}
  tr.fail td:last-child {{ color: #c62828; font-weight: 600; }}
  tr.fail {{ background: #fff3f3; }}
  .timestamp {{ color: #999; font-size: 0.8rem; }}
  th.sortable {{ cursor: pointer; user-select: none; }}
  th.sortable:hover {{ background: #1e2f50; }}
  th.sortable::after {{ content: "\\2195"; margin-left: 0.3em; opacity: 0.4; }}
  th.sort-asc::after {{ content: "\\2191"; opacity: 1; }}
  th.sort-desc::after {{ content: "\\2193"; opacity: 1; }}
</style>
<script>
document.addEventListener("DOMContentLoaded", function() {{
  document.querySelectorAll("table.sortable").forEach(function(table) {{
    var headers = table.querySelectorAll("th");
    headers.forEach(function(th, colIdx) {{
      th.classList.add("sortable");
      th.addEventListener("click", function() {{
        var tbody = table.querySelector("tbody");
        var rows = Array.from(tbody.querySelectorAll("tr"));
        var dir = th.classList.contains("sort-asc") ? "desc" : "asc";
        headers.forEach(function(h) {{ h.classList.remove("sort-asc", "sort-desc"); }});
        th.classList.add("sort-" + dir);
        rows.sort(function(a, b) {{
          var aText = a.children[colIdx].textContent.trim();
          var bText = b.children[colIdx].textContent.trim();
          var aNum = parseFloat(aText.replace(/[%+\\[\\],]/g, ""));
          var bNum = parseFloat(bText.replace(/[%+\\[\\],]/g, ""));
          var result;
          if (!isNaN(aNum) && !isNaN(bNum)) {{
            result = aNum - bNum;
          }} else {{
            result = aText.localeCompare(bText);
          }}
          return dir === "desc" ? -result : result;
        }});
        rows.forEach(function(row) {{ tbody.appendChild(row); }});
      }});
    }});
  }});
}});
</script>
</head>
<body>
<h1>vLLM Memory Estimator &mdash; Validation Report</h1>
<p class="timestamp">Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>

<div class="summary">
  <div class="card">
    <div class="value">{report.total_compared}</div>
    <div class="label">Records compared</div>
  </div>
  <div class="card">
    <div class="value">{report.total_skipped}</div>
    <div class="label">Skipped</div>
  </div>
</div>

<h2>Weight loading</h2>
<p style="margin:0.5rem 0 0.75rem; font-size:0.9rem; color:#555;">
  Compares <code>model_load_gib</code> from vLLM startup logs (actual weight load)
  against the estimator's predicted weight size.
</p>
<div class="summary">
  <div class="card">
    <div class="value">{report.params_in_bounds_pct:.1f}%</div>
    <div class="label">Weights in bounds</div>
  </div>
  <div class="card">
    <div class="value">{report.params_mean_abs_error_pct:.1f}%</div>
    <div class="label">Wt mean |error|</div>
  </div>
  <div class="card">
    <div class="value">{report.params_median_abs_error_pct:.1f}%</div>
    <div class="label">Wt median |error|</div>
  </div>
  <div class="card">
    <div class="value">{report.params_max_abs_error_pct:.1f}%</div>
    <div class="label">Wt max |error|</div>
  </div>
</div>

<h2>KV cache (per-token bytes)</h2>
<p style="margin:0.5rem 0 0.75rem; font-size:0.9rem; color:#555;">
  Compares per-token KV cache bytes: actual
  (<code>available_kv_cache_gib&nbsp;/&nbsp;kv_cache_tokens</code> from logs)
  vs estimator prediction.
</p>
<div class="summary">
  <div class="card">
    <div class="value">{report.kv_compared_count}</div>
    <div class="label">KV records compared</div>
  </div>
  <div class="card">
    <div class="value">{report.kv_mean_abs_error_pct:.1f}%</div>
    <div class="label">KV mean |error|</div>
  </div>
  <div class="card">
    <div class="value">{report.kv_median_abs_error_pct:.1f}%</div>
    <div class="label">KV median |error|</div>
  </div>
  <div class="card">
    <div class="value">{report.kv_max_abs_error_pct:.1f}%</div>
    <div class="label">KV max |error|</div>
  </div>
</div>

<h2>Per-model summary</h2>
<table class="sortable">
<thead>
<tr><th>Model</th><th>Records</th><th>Wt in bounds</th>
    <th>Wt mean err%</th><th>Wt max |err%|</th>
    <th>KV mean err%</th><th>KV max |err%|</th></tr>
</thead>
<tbody>
{"".join(rows_per_model)}
</tbody>
<tfoot>
<tr style="font-weight:700; border-top:2px solid #333">
  <td>TOTAL</td><td>{report.total_compared}</td>
  <td>{report.params_in_bounds_count}/{report.total_compared}</td>
  <td>{report.params_mean_abs_error_pct:.2f}%</td>
  <td>{report.params_max_abs_error_pct:.2f}%</td>
  <td>{report.kv_mean_abs_error_pct:.2f}%</td>
  <td>{report.kv_max_abs_error_pct:.2f}%</td>
</tr>
</tfoot>
</table>

<h2>Worst offenders (top 10 by |error|)</h2>
<table class="sortable">
<thead>
<tr><th>UUID</th><th>Model</th><th>Accel</th><th>TP</th><th>Seq len</th>
    <th>Quant</th><th>Actual Weights GiB</th><th>Est Weights GiB</th><th>Error</th><th>Bounds</th></tr>
</thead>
<tbody>
{"".join(rows_worst)}
</tbody>
</table>

{'<h2>Skipped records</h2><table class="sortable"><thead><tr><th>UUID</th><th>Model</th><th>Reason</th></tr></thead><tbody>'
 + "".join(skipped_rows) + "</tbody></table>" if skipped_rows else ""}

<h2>All comparisons</h2>
<table class="sortable">
<thead>
<tr><th>UUID</th><th>Model</th><th>Accel</th><th>TP</th><th>Seq len</th>
    <th>Quant</th><th>Actual Wt GiB</th><th>Est Wt GiB</th><th>Wt bounds</th>
    <th>Wt err%</th><th>Status</th>
    <th>Actual KV B/tok</th><th>Est KV B/tok</th><th>KV err%</th></tr>
</thead>
<tbody>
{"".join(rows_all)}
</tbody>
</table>

</body>
</html>"""
