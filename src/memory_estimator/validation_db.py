"""Build a validation database from vLLM startup logs and benchmark CSV metadata.

Parses memory-related data points from vLLM log files and combines them with
run metadata from consolidated_dashboard.csv to produce a JSON database for
validating the memory estimator against actual runtime values.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LogConfigData:
    """Configuration extracted from the vLLM engine initialisation line."""
    model: str
    dtype: str
    max_seq_len: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    quantization: str | None
    kv_cache_dtype: str
    enforce_eager: bool
    enable_prefix_caching: bool
    enable_chunked_prefill: bool
    vllm_version: str | None = None


@dataclass
class LogMemoryData:
    """Memory measurements extracted from a single vLLM startup log."""
    model_load_gib: float
    available_kv_cache_gib: float | None = None
    kv_cache_tokens: int | None = None
    max_concurrency_tokens: int | None = None
    max_concurrency_ratio: float | None = None
    cudagraph_gib: float | None = None
    gpu_memory_utilization: float = 0.9


@dataclass
class CSVMetadata:
    """Metadata from consolidated_dashboard.csv for a single UUID."""
    model: str
    accelerator: str
    tp: int | None
    dp: int | None
    runtime_args: str


@dataclass
class EstimatorInputsRecord:
    """Fields needed to construct an EstimatorInputs for this run."""
    model_id: str
    max_seq_len: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    quantization: str | None
    kv_cache_dtype: str | None
    enforce_eager: bool
    dtype: str | None


@dataclass
class ValidationRecord:
    """Complete per-UUID record combining CSV metadata and log data."""
    uuid: str
    csv_metadata: CSVMetadata
    log_config: LogConfigData
    log_memory: LogMemoryData
    estimator_inputs: EstimatorInputsRecord


# ---------------------------------------------------------------------------
# ANSI stripping
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Log parsers
# ---------------------------------------------------------------------------

_RE_INIT_LINE = re.compile(
    r"Initializing a(?:n)? V1 LLM engine \(([^)]+)\) with config:"
)
_RE_CFG_MODEL = re.compile(r"model='([^']+)'")
_RE_CFG_DTYPE = re.compile(r"dtype=(\S+?),")
_RE_CFG_MAX_SEQ_LEN = re.compile(r"max_seq_len=(\d+)")
_RE_CFG_TP = re.compile(r"tensor_parallel_size=(\d+)")
_RE_CFG_PP = re.compile(r"pipeline_parallel_size=(\d+)")
_RE_CFG_DP = re.compile(r"data_parallel_size=(\d+)")
_RE_CFG_QUANT = re.compile(r"quantization=(\S+?),")
_RE_CFG_KV_DTYPE = re.compile(r"kv_cache_dtype=(\S+?),")
_RE_CFG_EAGER = re.compile(r"enforce_eager=(\S+?),")
_RE_CFG_PREFIX = re.compile(r"enable_prefix_caching=(\S+?),")
_RE_CFG_CHUNKED = re.compile(r"enable_chunked_prefill=(\S+?),")

_RE_MODEL_LOAD = re.compile(r"Model loading took ([\d.]+) GiB")
_RE_KV_AVAILABLE = re.compile(r"Available KV cache memory: ([\d.]+) GiB")
_RE_KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
_RE_CONCURRENCY = re.compile(
    r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"
)
_RE_CUDAGRAPH = re.compile(
    r"Graph capturing finished in \d+ secs, took (-?[\d.]+) GiB"
)
_RE_GPU_MEM_UTIL = re.compile(r"gpu_memory_utilization['\"]?:\s*([\d.]+)")


def _parse_bool(val: str) -> bool:
    return val.strip().lower() == "true"


def _parse_optional_str(val: str) -> str | None:
    v = val.strip()
    if v.lower() == "none":
        return None
    return v


def parse_log_config(text: str) -> LogConfigData | None:
    """Parse the vLLM engine initialisation config line."""
    cleaned = strip_ansi(text)
    init_match = _RE_INIT_LINE.search(cleaned)
    if not init_match:
        return None

    version = init_match.group(1)
    line_start = init_match.start()
    config_text = cleaned[line_start:]

    model_m = _RE_CFG_MODEL.search(config_text)
    if not model_m:
        return None

    def _extract(pattern: re.Pattern[str], default: str = "") -> str:
        m = pattern.search(config_text)
        return m.group(1) if m else default

    dtype_raw = _extract(_RE_CFG_DTYPE, "torch.bfloat16")
    dtype = dtype_raw.replace("torch.", "")

    quant_raw = _extract(_RE_CFG_QUANT, "None")
    quantization = _parse_optional_str(quant_raw)

    return LogConfigData(
        model=model_m.group(1),
        dtype=dtype,
        max_seq_len=int(_extract(_RE_CFG_MAX_SEQ_LEN, "0")),
        tensor_parallel_size=int(_extract(_RE_CFG_TP, "1")),
        pipeline_parallel_size=int(_extract(_RE_CFG_PP, "1")),
        data_parallel_size=int(_extract(_RE_CFG_DP, "1")),
        quantization=quantization,
        kv_cache_dtype=_extract(_RE_CFG_KV_DTYPE, "auto"),
        enforce_eager=_parse_bool(_extract(_RE_CFG_EAGER, "False")),
        enable_prefix_caching=_parse_bool(_extract(_RE_CFG_PREFIX, "False")),
        enable_chunked_prefill=_parse_bool(_extract(_RE_CFG_CHUNKED, "True")),
        vllm_version=version,
    )


def parse_log_memory(text: str) -> LogMemoryData | None:
    """Parse memory reporting lines from vLLM log output."""
    cleaned = strip_ansi(text)

    ml = _RE_MODEL_LOAD.search(cleaned)
    if not ml:
        return None

    ka = _RE_KV_AVAILABLE.search(cleaned)
    kt = _RE_KV_TOKENS.search(cleaned)
    mc = _RE_CONCURRENCY.search(cleaned)
    cg = _RE_CUDAGRAPH.search(cleaned)
    gu = _RE_GPU_MEM_UTIL.search(cleaned)

    return LogMemoryData(
        model_load_gib=float(ml.group(1)),
        available_kv_cache_gib=float(ka.group(1)) if ka else None,
        kv_cache_tokens=int(kt.group(1).replace(",", "")) if kt else None,
        max_concurrency_tokens=int(mc.group(1).replace(",", "")) if mc else None,
        max_concurrency_ratio=float(mc.group(2)) if mc else None,
        cudagraph_gib=float(cg.group(1)) if cg else None,
        gpu_memory_utilization=float(gu.group(1)) if gu else 0.9,
    )


def parse_log_file(path: Path) -> tuple[LogConfigData | None, LogMemoryData | None]:
    """Read and parse a vLLM log file."""
    text = path.read_text(errors="replace")
    return parse_log_config(text), parse_log_memory(text)


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def _parse_float_to_int(val: str | None) -> int | None:
    if val is None:
        return None
    v = val.strip()
    if not v:
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def parse_csv_metadata(csv_path: Path) -> dict[str, CSVMetadata]:
    """Read the CSV and return one CSVMetadata per unique UUID."""
    result: dict[str, CSVMetadata] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uuid = row["uuid"].strip()
            if not uuid or uuid in result:
                continue
            result[uuid] = CSVMetadata(
                model=row["model"].strip(),
                accelerator=row["accelerator"].strip(),
                tp=_parse_float_to_int(row.get("TP", "")),
                dp=_parse_float_to_int(row.get("DP", "")),
                runtime_args=row.get("runtime_args", "").strip(),
            )
    return result


def parse_runtime_args(runtime_args: str) -> dict[str, str]:
    """Parse the runtime_args string into a key-value dict."""
    result: dict[str, str] = {}
    if not runtime_args:
        return result

    if ": " in runtime_args and "=" not in runtime_args.split(";")[0]:
        for part in runtime_args.split(";"):
            part = part.strip()
            if ": " in part:
                k, v = part.split(": ", 1)
                result[k.strip()] = v.strip()
    else:
        for part in runtime_args.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                result[k.strip()] = v.strip()
    return result


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

_EXCLUDED_ACCELERATORS = frozenset({"Spyre", "TPU"})
_EXCLUDED_MODEL_PREFIXES = ("openai/", "bart-")
_EXCLUDED_MODEL_SUBSTRINGS = ("whisper",)


def should_include(
    csv_meta: CSVMetadata,
    log_config: LogConfigData | None,
    log_memory: LogMemoryData | None,
) -> tuple[bool, str]:
    """Return (include, reason) for a UUID."""
    if csv_meta.accelerator in _EXCLUDED_ACCELERATORS:
        return False, f"excluded_accelerator:{csv_meta.accelerator}"

    model = csv_meta.model
    for prefix in _EXCLUDED_MODEL_PREFIXES:
        if model.startswith(prefix):
            return False, f"excluded_model_prefix:{prefix}"
    for sub in _EXCLUDED_MODEL_SUBSTRINGS:
        if sub in model.lower():
            return False, f"excluded_model_substring:{sub}"

    if log_config is None:
        return False, "log_parse_failure:no_config"
    if log_memory is None:
        return False, "log_parse_failure:no_memory"

    return True, ""


def cross_validate(
    uuid: str,
    csv_meta: CSVMetadata,
    log_config: LogConfigData,
) -> list[str]:
    """Return warning messages for mismatches between CSV and log."""
    warnings: list[str] = []

    csv_model_base = csv_meta.model.split("/")[-1].lower()
    log_model_base = log_config.model.split("/")[-1].lower()
    if csv_model_base != log_model_base:
        if not (csv_model_base in log_model_base or log_model_base in csv_model_base):
            warnings.append(
                f"[{uuid}] CSV model '{csv_meta.model}' vs log model '{log_config.model}'"
            )

    if csv_meta.tp is not None and csv_meta.tp != log_config.tensor_parallel_size:
        warnings.append(
            f"[{uuid}] CSV TP={csv_meta.tp} vs log TP={log_config.tensor_parallel_size}"
        )

    return warnings


# ---------------------------------------------------------------------------
# EstimatorInputs reconstruction
# ---------------------------------------------------------------------------

def build_estimator_inputs_record(log_config: LogConfigData) -> EstimatorInputsRecord:
    """Map log config data to EstimatorInputs fields."""
    kv_dtype = log_config.kv_cache_dtype
    if kv_dtype == "auto":
        kv_dtype = None

    dtype = log_config.dtype
    if dtype:
        dtype = dtype.replace("torch.", "")

    return EstimatorInputsRecord(
        model_id=log_config.model,
        max_seq_len=log_config.max_seq_len,
        tensor_parallel_size=log_config.tensor_parallel_size,
        pipeline_parallel_size=log_config.pipeline_parallel_size,
        data_parallel_size=log_config.data_parallel_size,
        quantization=log_config.quantization,
        kv_cache_dtype=kv_dtype,
        enforce_eager=log_config.enforce_eager,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# S3 download
# ---------------------------------------------------------------------------

def download_logs_from_s3(
    bucket: str,
    prefix: str,
    local_dir: Path,
    needed_uuids: set[str],
) -> int:
    """Download log files from S3 for UUIDs not already cached locally.

    Returns the count of newly downloaded files.
    """
    try:
        import boto3
        from botocore.exceptions import BotoCoreError
        from botocore.exceptions import ClientError
    except ImportError:
        logger.warning("boto3 not installed — skipping S3 download")
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    try:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.rsplit("/", 1)[-1]
                uuid = filename.replace(".log", "")
                if uuid not in needed_uuids:
                    continue

                local_path = local_dir / filename
                if local_path.exists():
                    continue

                logger.info("Downloading %s", key)
                s3.download_file(bucket, key, str(local_path))
                downloaded += 1

    except (BotoCoreError, ClientError) as e:
        logger.warning("S3 download failed: %s", e)

    return downloaded


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_validation_db(
    csv_path: Path,
    log_dir: Path,
    output_path: Path,
    *,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    download_s3: bool = False,
) -> dict[str, Any]:
    """Build the validation database and save as JSON.

    Returns the full database dict.
    """
    csv_metadata = parse_csv_metadata(csv_path)
    logger.info("Parsed %d unique UUIDs from CSV", len(csv_metadata))

    if download_s3 and s3_bucket and s3_prefix:
        existing_logs = {p.stem for p in log_dir.glob("*.log")}
        needed = set(csv_metadata.keys()) - existing_logs
        if needed:
            n = download_logs_from_s3(s3_bucket, s3_prefix, log_dir, needed)
            logger.info("Downloaded %d new log files from S3", n)

    log_files = {p.stem: p for p in log_dir.glob("*.log")}
    logger.info("Found %d log files in %s", len(log_files), log_dir)

    records: dict[str, dict[str, Any]] = {}
    all_warnings: list[dict[str, str]] = []
    exclusion_counts: dict[str, int] = {}
    no_log_count = 0

    for uuid, csv_meta in csv_metadata.items():
        if uuid not in log_files:
            no_log_count += 1
            continue

        log_config, log_memory = parse_log_file(log_files[uuid])

        include, reason = should_include(csv_meta, log_config, log_memory)
        if not include:
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1
            continue

        assert log_config is not None
        assert log_memory is not None

        warns = cross_validate(uuid, csv_meta, log_config)
        for w in warns:
            all_warnings.append({"uuid": uuid, "message": w})
            logger.warning(w)

        estimator_inputs = build_estimator_inputs_record(log_config)

        record = ValidationRecord(
            uuid=uuid,
            csv_metadata=csv_meta,
            log_config=log_config,
            log_memory=log_memory,
            estimator_inputs=estimator_inputs,
        )
        records[uuid] = asdict(record)

    db: dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "csv_source": str(csv_path),
            "log_source": str(log_dir),
            "total_csv_uuids": len(csv_metadata),
            "total_logs_available": len(log_files),
            "logs_not_found": no_log_count,
            "records_included": len(records),
            "exclusion_reasons": exclusion_counts,
        },
        "records": records,
        "warnings": all_warnings,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(db, f, indent=2)

    logger.info(
        "Built validation DB: %d records included, %d excluded, %d without logs",
        len(records),
        sum(exclusion_counts.values()),
        no_log_count,
    )
    return db
