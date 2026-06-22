from __future__ import annotations

import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nydsge.runtime import RuntimeStatus, runtime_report


@dataclass(frozen=True)
class BenchmarkCaptureResult:
    label: str
    kernel: str
    timestamp: str
    raw_output: Path
    julia_baseline_output: Path | None
    speedup_output: Path | None
    raw_command: list[str]
    speedup_command: list[str] | None
    julia_command: list[str] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kernel": self.kernel,
            "timestamp": self.timestamp,
            "raw_output": str(self.raw_output),
            "julia_baseline_output": (
                None if self.julia_baseline_output is None else str(self.julia_baseline_output)
            ),
            "speedup_output": None if self.speedup_output is None else str(self.speedup_output),
            "raw_command": self.raw_command,
            "speedup_command": self.speedup_command,
            "julia_command": self.julia_command,
        }


def infer_machine_label(
    *,
    runtime_statuses: list[RuntimeStatus] | None = None,
    platform_name: str | None = None,
) -> str:
    """Infer the canonical machine label used for benchmark filenames."""

    statuses = runtime_statuses or runtime_report()
    host = (platform_name or platform.system()).lower()
    if host == "windows":
        if any(
            status.backend == "torch" and status.device == "cuda" and status.available
            for status in statuses
        ):
            return "windows-cuda"
        return "windows-cpu"
    if host == "darwin":
        if any(
            status.backend == "torch" and status.device == "mps" and status.available
            for status in statuses
        ):
            return "macos-mps"
        return "macos-cpu"
    if host == "linux":
        has_cuda = any(
            status.backend == "torch" and status.device == "cuda" and status.available
            for status in statuses
        ) or any(
            status.backend == "jax" and status.device == "cuda" and status.available
            for status in statuses
        )
        return "linux-cuda" if has_cuda else "linux-cpu"
    if any(status.backend == "torch" and status.device == "cuda" for status in statuses):
        return "other-cuda"
    return "other-cpu"


def default_capture_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def build_python_benchmark_command(
    *,
    python_executable: str,
    kernel: str,
    horizon: int,
    periods: int,
    batches: int,
    draws: int,
    repeats: int,
    include_pseudo: bool,
    output_path: Path,
    baseline_path: Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "nydsge.cli",
        "bench",
        "--kernel",
        kernel,
        "--horizon",
        str(horizon),
        "--periods",
        str(periods),
        "--batches",
        str(batches),
        "--draws",
        str(draws),
        "--repeats",
        str(repeats),
        "--output",
        str(output_path),
    ]
    if include_pseudo:
        command.append("--include-pseudo")
    if baseline_path is not None:
        command.extend(["--baseline", str(baseline_path)])
    return command


def build_julia_baseline_command(
    *,
    julia_executable: str,
    julia_project: str,
    julia_script: Path,
    horizon: int,
    repeats: int,
    output_path: Path,
    julia_version: str | None = None,
) -> list[str]:
    command: list[str] = [julia_executable]
    if julia_version is not None:
        command.append(f"+{julia_version}")
    command.extend(
        [
            f"--project={julia_project}",
            str(julia_script),
            "--kernel",
            "forecast",
            "--horizon",
            str(horizon),
            "--repeats",
            str(repeats),
            "--out",
            str(output_path),
        ]
    )
    return command


def capture_benchmarks(
    *,
    output_dir: Path,
    kernel: str,
    horizon: int = 40,
    periods: int = 40,
    batches: int = 8,
    draws: int = 2,
    repeats: int = 3,
    include_pseudo: bool = False,
    label: str | None = None,
    baseline_path: Path | None = None,
    capture_julia_baseline: bool = False,
    julia_executable: str = "julia",
    julia_project: str = "tools/oracle_julia",
    julia_script: Path = Path("tools/oracle_julia/benchmark_model1002.jl"),
    julia_version: str | None = None,
    python_executable: str,
    timestamp: str | None = None,
    runtime_statuses: list[RuntimeStatus] | None = None,
    command_runner: Callable[[Sequence[str]], None],
) -> BenchmarkCaptureResult:
    """Run a local benchmark capture and optional Julia baseline attachment."""

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_label = label or infer_machine_label(
        runtime_statuses=runtime_statuses,
    )
    selected_timestamp = timestamp or default_capture_timestamp()

    raw_output = output_dir / f"{selected_label}_{kernel}_{selected_timestamp}_local.json"
    baseline = baseline_path
    julia_output: Path | None = None
    if capture_julia_baseline and baseline is None:
        julia_output = output_dir / f"{selected_label}_julia_forecast_{selected_timestamp}.json"
        command_runner(
            build_julia_baseline_command(
                julia_executable=julia_executable,
                julia_project=julia_project,
                julia_script=julia_script,
                horizon=horizon,
                repeats=repeats,
                output_path=julia_output,
                julia_version=julia_version,
            )
        )
        baseline = julia_output

    raw_command = build_python_benchmark_command(
        python_executable=python_executable,
        kernel=kernel,
        horizon=horizon,
        periods=periods,
        batches=batches,
        draws=draws,
        repeats=repeats,
        include_pseudo=include_pseudo,
        output_path=raw_output,
        baseline_path=None,
    )
    command_runner(raw_command)

    speedup_command: list[str] | None = None
    speedup_output: Path | None = None
    if baseline is not None:
        speedup_output = (
            output_dir / f"{selected_label}_{kernel}_vs_julia_{selected_timestamp}.json"
        )
        speedup_command = build_python_benchmark_command(
            python_executable=python_executable,
            kernel=kernel,
            horizon=horizon,
            periods=periods,
            batches=batches,
            draws=draws,
            repeats=repeats,
            include_pseudo=include_pseudo,
            output_path=speedup_output,
            baseline_path=baseline,
        )
        command_runner(speedup_command)

    return BenchmarkCaptureResult(
        label=selected_label,
        kernel=kernel,
        timestamp=selected_timestamp,
        raw_output=raw_output,
        julia_baseline_output=julia_output,
        speedup_output=speedup_output,
        raw_command=raw_command,
        speedup_command=speedup_command,
        julia_command=None
        if julia_output is None
        else build_julia_baseline_command(
            julia_executable=julia_executable,
            julia_project=julia_project,
            julia_script=julia_script,
            horizon=horizon,
            repeats=repeats,
            output_path=julia_output,
            julia_version=julia_version,
        ),
    )
