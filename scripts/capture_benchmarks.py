#!/usr/bin/env python

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from nydsge.benchmark_capture import (
    BenchmarkCaptureResult,
    capture_benchmarks,
)


def _run_command(command: Sequence[str]) -> None:
    process = subprocess.run(
        list(command),
        check=False,
        text=True,
        capture_output=True,
    )
    if process.returncode != 0:
        details = "\n".join(
            part for part in (process.stdout, process.stderr) if part and part.strip()
        )
        raise RuntimeError(
            "Benchmark capture command failed."
            f" command={list(command)!r} code={process.returncode}"
            + (f" details={details}" if details else "")
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture native benchmark reports for cross-machine comparison.",
    )
    parser.add_argument("--kernel", default="all", help="Benchmark kernel: forecast, all, etc.")
    parser.add_argument("--horizon", type=int, default=40, help="Forecast horizon.")
    parser.add_argument("--periods", type=int, default=40, help="Kalman periods.")
    parser.add_argument("--batches", type=int, default=8, help="Kalman batch count.")
    parser.add_argument("--draws", type=int, default=2, help="Hard-target draws.")
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats.")
    parser.add_argument(
        "--label",
        default=None,
        help="Machine label override (default: auto-detected).",
    )
    parser.add_argument(
        "--include-pseudo",
        action="store_true",
        help="Include pseudo-observable forecast outputs in captures.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional Julia benchmark baseline JSON file.",
    )
    parser.add_argument(
        "--capture-julia-baseline",
        action="store_true",
        help="Run Julia benchmark script and attach baseline in one pass.",
    )
    parser.add_argument(
        "--julia-executable",
        default="julia",
        help="Julia executable (default: julia).",
    )
    parser.add_argument(
        "--julia-version",
        default=None,
        help="Optional Julia version selector (e.g., 1.8).",
    )
    parser.add_argument(
        "--julia-project",
        default="tools/oracle_julia",
        help="Julia oracle project root.",
    )
    parser.add_argument(
        "--julia-script",
        default="tools/oracle_julia/benchmark_model1002.jl",
        help="Julia benchmark script path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/benchmarks"),
        help="Output directory for local and baseline-aware report files.",
    )
    parser.add_argument("--timestamp", default=None, help="Custom capture timestamp.")
    return parser.parse_args(args=argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result: BenchmarkCaptureResult = capture_benchmarks(
        output_dir=args.output_dir,
        kernel=args.kernel,
        horizon=args.horizon,
        periods=args.periods,
        batches=args.batches,
        draws=args.draws,
        repeats=args.repeats,
        include_pseudo=args.include_pseudo,
        label=args.label,
        baseline_path=args.baseline,
        capture_julia_baseline=args.capture_julia_baseline,
        julia_executable=args.julia_executable,
        julia_project=args.julia_project,
        julia_script=Path(args.julia_script),
        julia_version=args.julia_version,
        timestamp=args.timestamp,
        command_runner=_run_command,
        python_executable=sys.executable,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
