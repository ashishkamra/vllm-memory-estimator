#!/usr/bin/env python3
"""Build the validation test database from CSV + vLLM log files.

Usage:
    python scripts/build_validation_db.py
    python scripts/build_validation_db.py --download-s3
    python scripts/build_validation_db.py --csv path/to/csv --log-dir path/to/logs
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_estimator.validation_db import build_validation_db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validation test database")
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "consolidated_dashboard.csv",
        help="Path to consolidated_dashboard.csv",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "logs",
        help="Directory containing cached vLLM log files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "validation_db.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--download-s3",
        action="store_true",
        help="Download missing logs from S3 before building",
    )
    parser.add_argument(
        "--s3-bucket",
        default="psap-model-furnace",
        help="S3 bucket name (default: psap-model-furnace)",
    )
    parser.add_argument(
        "--s3-prefix",
        default="logs/",
        help="S3 key prefix (default: logs/)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    db = build_validation_db(
        csv_path=args.csv,
        log_dir=args.log_dir,
        output_path=args.output,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        download_s3=args.download_s3,
    )

    meta = db["metadata"]
    print(f"\nValidation DB built: {args.output}")
    print(f"  CSV UUIDs:        {meta['total_csv_uuids']}")
    print(f"  Logs available:   {meta['total_logs_available']}")
    print(f"  Logs not found:   {meta['logs_not_found']}")
    print(f"  Records included: {meta['records_included']}")
    if meta["exclusion_reasons"]:
        print("  Exclusions:")
        for reason, count in sorted(meta["exclusion_reasons"].items()):
            print(f"    {reason}: {count}")
    if db["warnings"]:
        print(f"  Warnings: {len(db['warnings'])}")


if __name__ == "__main__":
    main()
