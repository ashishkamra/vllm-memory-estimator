#!/usr/bin/env python3
"""Run the memory estimator against the validation database and report accuracy.

Usage:
    python scripts/run_validation.py
    python scripts/run_validation.py --model "Llama-3.3-70B"
    python scripts/run_validation.py --output test_data/validation_report.json
    python scripts/run_validation.py --html test_data/validation_report.html
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from memory_estimator.validation_runner import run_validation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run memory estimator against validation database",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=PROJECT_ROOT / "test_data" / "validation_db.json",
        help="Path to validation_db.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write JSON report to this file",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Write HTML report to this file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_stdout",
        help="Print full JSON report to stdout",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Filter to models containing this substring",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-record details",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    if not args.db.exists():
        print(f"Error: validation DB not found at {args.db}", file=sys.stderr)
        print("Run: python scripts/build_validation_db.py --download-s3", file=sys.stderr)
        sys.exit(1)

    report = run_validation(args.db, model_filter=args.model)

    if args.json_stdout:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render_summary())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nJSON report written to {args.output}")

    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        with open(args.html, "w") as f:
            f.write(report.render_html())
        print(f"HTML report written to {args.html}")


if __name__ == "__main__":
    main()
