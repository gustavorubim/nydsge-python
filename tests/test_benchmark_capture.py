from __future__ import annotations

import subprocess
from pathlib import Path

from nydsge.benchmark_capture import (
    build_julia_baseline_command,
    build_python_benchmark_command,
    capture_benchmarks,
    infer_machine_label,
)
from nydsge.runtime import RuntimeStatus


def _capture_commands(
    calls: list[list[str]],
):
    def _run(command: list[str] | tuple[str, ...], **_: object) -> subprocess.CompletedProcess[str]:
        commands = list(command)
        calls.append(commands)
        return subprocess.CompletedProcess(
            args=commands,
            returncode=0,
            stdout="ok",
            stderr="",
        )

    return _run


def test_infer_machine_label_prefers_available_accelerator_targets() -> None:
    statuses = [
        RuntimeStatus("platform", "native", True, "native"),
        RuntimeStatus("torch", "cuda", True, "cuda"),
        RuntimeStatus("torch", "mps", True, "mps"),
        RuntimeStatus("jax", "cuda", True, "jax cuda"),
    ]

    assert infer_machine_label(runtime_statuses=statuses, platform_name="Windows") == "windows-cuda"
    assert infer_machine_label(runtime_statuses=statuses, platform_name="darwin") == "macos-mps"
    assert infer_machine_label(runtime_statuses=statuses, platform_name="Linux") == "linux-cuda"
    assert infer_machine_label(runtime_statuses=statuses, platform_name="plan9") == "other-cuda"


def test_build_python_benchmark_command_includes_inputs_and_baseline() -> None:
    command = build_python_benchmark_command(
        python_executable="py",
        kernel="forecast",
        horizon=10,
        periods=5,
        batches=2,
        draws=3,
        repeats=4,
        include_pseudo=True,
        output_path=Path("benchmark.json"),
        baseline_path=Path("baseline.json"),
    )

    assert command[:4] == ["py", "-m", "nydsge.cli", "bench"]
    assert "--kernel" in command and "forecast" in command
    assert "--horizon" in command and "10" in command
    assert "--periods" in command and "5" in command
    assert "--batches" in command and "2" in command
    assert "--draws" in command and "3" in command
    assert "--repeats" in command and "4" in command
    assert "--include-pseudo" in command
    assert command.count("--output") == 1
    assert command.count("--baseline") == 1


def test_build_julia_baseline_command_respects_version_and_paths() -> None:
    command = build_julia_baseline_command(
        julia_executable="julia",
        julia_project="tools/oracle_julia",
        julia_script=Path("tools/oracle_julia/benchmark_model1002.jl"),
        horizon=20,
        repeats=2,
        output_path=Path("out.json"),
        julia_version="1.10",
    )

    assert command[0] == "julia"
    assert command[1] == "+1.10"
    assert "--project=tools/oracle_julia" in command
    assert str(Path("tools/oracle_julia/benchmark_model1002.jl")) in command
    assert command[-4:] == ["--repeats", "2", "--out", "out.json"]


def test_capture_benchmarks_runs_local_and_baseline_passes(tmp_path: Path) -> None:
    calls: list[list[str]] = []
    statuses = [
        RuntimeStatus("platform", "native", True, "native"),
        RuntimeStatus("torch", "cuda", True, "cuda"),
    ]
    result = capture_benchmarks(
        output_dir=tmp_path / "reports",
        kernel="all",
        horizon=7,
        periods=7,
        batches=3,
        draws=2,
        repeats=2,
        include_pseudo=False,
        label="linux-cuda",
        capture_julia_baseline=True,
        baseline_path=None,
        timestamp="2026-06-22",
        runtime_statuses=statuses,
        command_runner=_capture_commands(calls),
        julia_script=Path("tools/oracle_julia/benchmark_model1002.jl"),
        julia_version="1.8",
        python_executable="py",
    )

    assert result.label == "linux-cuda"
    assert result.timestamp == "2026-06-22"
    assert result.raw_output.name == "linux-cuda_all_2026-06-22_local.json"
    assert result.julia_baseline_output is not None
    assert result.julia_baseline_output.name == "linux-cuda_julia_forecast_2026-06-22.json"
    assert result.speedup_output is not None
    assert result.speedup_output.name == "linux-cuda_all_vs_julia_2026-06-22.json"
    assert len(calls) == 3
    assert "--out" in calls[0]
    assert "--output" in calls[1]
    assert "--output" in calls[2]
    assert "--baseline" in calls[2]
    assert "--baseline" in calls[2]
    assert any(item.endswith(".json") for item in calls[0][-2:])
