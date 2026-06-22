#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nydsge.benchmark_compare import (
    BenchmarkCrossMachineSummary,
    build_summary,
    load_benchmark_report,
    speedup_reference_from_reference,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cross-machine benchmark report JSON files.",
    )
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        required=True,
        help="Benchmark report JSON path. Pass one or more report files.",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Optional kernel filter for rows in summary and speedup tables.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce strict command signatures across all report files.",
    )
    parser.add_argument(
        "--baseline-machine",
        default=None,
        help="Optional baseline machine label when computing speedups.",
    )
    parser.add_argument(
        "--baseline-backend",
        default=None,
        help="Optional baseline backend when computing speedups.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the JSON payload.",
    )
    return parser.parse_args(argv)


def build_compare_payload(
    report_paths: list[Path],
    *,
    kernel: str | None,
    strict: bool,
    baseline_machine: str | None,
    baseline_backend: str | None,
) -> dict[str, Any]:
    reports = [load_benchmark_report(path) for path in report_paths]
    summary: BenchmarkCrossMachineSummary = build_summary(
        reports,
        kernel=kernel,
        enforce_signature=strict,
    )

    baseline = (
        None
        if baseline_machine is None or baseline_backend is None
        else (baseline_machine, baseline_backend)
    )

    speedups = speedup_reference_from_reference(
        summary.rows,
        baseline=baseline,
    )

    speedup_payload = [
        {
            "kernel": key[0],
            "backend": key[1],
            "device": key[2],
            "horizon": key[3],
            "repeats": key[4],
            "dtype": key[5],
            "entries": [
                {
                    "machine": machine,
                    "elapsed_seconds": elapsed_seconds,
                    "speedup_vs_baseline": speedup,
                }
                for machine, elapsed_seconds, speedup in values
            ],
        }
        for key, values in sorted(speedups.items())
    ]

    payload = {
        "summary": summary.to_dict(),
        "speedups": speedup_payload,
    }
    return payload


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = build_compare_payload(
        args.report,
        kernel=args.kernel,
        strict=args.strict,
        baseline_machine=args.baseline_machine,
        baseline_backend=args.baseline_backend,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
