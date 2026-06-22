from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from nydsge.benchmark_compare import (
    BenchmarkCrossMachineReport,
    BenchmarkCrossMachineRow,
    BenchmarkResultRecord,
    build_summary,
    infer_machine_from_report_path,
    load_benchmark_report,
    speedup_reference_from_reference,
)


def _write_report(path: Path, *, machine_label: str) -> Path:
    payload = {
        "schema_version": 1,
        "created_utc": "2026-06-22T00:00:00Z",
        "command": {
            "kernel": "forecast",
            "horizon": 1,
            "periods": 1,
            "batches": 1,
            "draws": 1,
            "repeats": 3,
            "dtype": "float64",
            "include_pseudo": False,
        },
        "platform": {
            "system": "Linux",
            "release": "99",
            "machine": "x86_64",
            "processor": "",
            "python_version": "3.12.0",
        },
        "runtime_statuses": [],
        "results": [
            {
                "backend": "numpy",
                "device": "cpu",
                "kernel": "forecast",
                "horizon": 1,
                "repeats": 3,
                "dtype": "float64",
                "available": True,
                "skipped": False,
                "reason": "ok",
                "elapsed_seconds": 1.0,
            }
        ],
    }

    report_path = path / f"{machine_label}_forecast_2026-06-22_local.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def test_infer_machine_from_report_path_prefers_filename_prefix() -> None:
    assert (
        infer_machine_from_report_path(Path("linux-cpu_forecast_2026-06-22_local.json"))
        == "linux-cpu"
    )
    assert infer_machine_from_report_path(Path("justreport.json")) == "justreport"


def test_load_benchmark_report_parses_payload_and_infers_machine(tmp_path: Path) -> None:
    report_path = _write_report(tmp_path, machine_label="linux-cpu")
    report = load_benchmark_report(report_path)

    assert report.machine == "linux-cpu"
    assert report.path == str(report_path)
    assert report.schema_version == 1
    assert report.results[0].machine == "linux-cpu"
    assert report.results[0].backend == "numpy"


def test_build_summary_enforces_signature_across_reports(tmp_path: Path) -> None:
    first = _write_report(tmp_path, machine_label="linux-cpu")
    second = _write_report(tmp_path, machine_label="windows-cpu")
    second_payload = json.loads(
        (tmp_path / "windows-cpu_forecast_2026-06-22_local.json").read_text(
            encoding="utf-8",
        )
    )
    second_payload["command"]["repeats"] = 4
    (tmp_path / "windows-cpu_forecast_2026-06-22_local.json").write_text(
        json.dumps(second_payload),
        encoding="utf-8",
    )

    reports = [
        load_benchmark_report(first),
        load_benchmark_report(second),
    ]

    with pytest.raises(ValueError, match="command signature differs"):
        build_summary(reports, enforce_signature=True)

    summary = build_summary(reports, enforce_signature=False)
    assert {row.machine for row in summary.rows} == {"linux-cpu", "windows-cpu"}


def test_build_summary_can_filter_kernel() -> None:
    report = BenchmarkCrossMachineReport(
        machine="linux-cpu",
        path="tmp",
        schema_version=1,
        command={
            "kernel": "all",
            "horizon": 2,
            "periods": 1,
            "batches": 1,
            "draws": 1,
            "repeats": 3,
            "dtype": "float64",
            "include_pseudo": False,
        },
        platform={},
        runtime_statuses=[],
        results=[
            BenchmarkResultRecord(
                kernel="forecast",
                backend="numpy",
                device="cpu",
                horizon=2,
                repeats=3,
                dtype="float64",
                available=True,
                skipped=False,
                reason="ok",
                elapsed_seconds=1.2,
                machine="linux-cpu",
            ),
            BenchmarkResultRecord(
                kernel="kalman",
                backend="numpy",
                device="cpu",
                horizon=2,
                repeats=3,
                dtype="float64",
                available=True,
                skipped=False,
                reason="ok",
                elapsed_seconds=2.4,
                machine="linux-cpu",
            ),
        ],
    )

    summary = build_summary([report], kernel="forecast")
    assert len(summary.rows) == 1
    assert summary.rows[0].kernel == "forecast"


def test_speedup_reference_from_reference_defaults_to_fastest_baseline() -> None:
    rows = [
        BenchmarkCrossMachineRow(
            kernel="forecast",
            backend="numpy",
            device="cpu",
            horizon=1,
            repeats=3,
            dtype="float64",
            machine="linux-cpu",
            elapsed_seconds=2.0,
            skipped=False,
            available=True,
            reason="",
        ),
        BenchmarkCrossMachineRow(
            kernel="forecast",
            backend="numpy",
            device="cpu",
            horizon=1,
            repeats=3,
            dtype="float64",
            machine="windows-cpu",
            elapsed_seconds=1.0,
            skipped=False,
            available=True,
            reason="",
        ),
        BenchmarkCrossMachineRow(
            kernel="forecast",
            backend="numpy",
            device="cpu",
            horizon=1,
            repeats=3,
            dtype="float64",
            machine="linux-cpu-cuda",
            elapsed_seconds=None,
            skipped=True,
            available=False,
            reason="cuda unavailable",
        ),
    ]

    speedups = speedup_reference_from_reference(rows)
    speedup_key = next(
        key for key in speedups if key == ("forecast", "numpy", "cpu", 1, 3, "float64")
    )
    entries = speedups[speedup_key]
    assert entries[0] == ("windows-cpu", 1.0, 1.0)
    assert entries[1] == ("linux-cpu", 2.0, 0.5)


def test_compare_script_writes_payload(tmp_path: Path) -> None:
    first = _write_report(tmp_path, machine_label="linux-cpu")
    second = _write_report(tmp_path, machine_label="windows-cpu")
    output = tmp_path / "compare.json"

    command = [
        sys.executable,
        str(Path("scripts/compare_benchmark_reports.py")),
        "--report",
        str(first),
        "--report",
        str(second),
        "--output",
        str(output),
        "--baseline-machine",
        "windows-cpu",
        "--baseline-backend",
        "numpy",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    assert result.returncode == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["signature"]["kernel"] == "forecast"
    assert payload["summary"]["rows"][0]["machine"] == "linux-cpu"
    assert {entry["machine"] for entry in payload["speedups"][0]["entries"]} == {
        "linux-cpu",
        "windows-cpu",
    }
