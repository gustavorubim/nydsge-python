from __future__ import annotations

from pathlib import Path

import pytest

import nydsge.runtime as runtime
from nydsge.purity import audit_runtime_purity
from nydsge.runtime import RuntimeConfig, RuntimeStatus, UnsupportedRuntimeError, runtime_report


def test_numpy_cpu_resolves() -> None:
    resolved = RuntimeConfig(backend="numpy", device="cpu").resolve()
    assert resolved.backend == "numpy"
    assert resolved.device == "cpu"


def test_numpy_cuda_is_rejected() -> None:
    with pytest.raises(UnsupportedRuntimeError):
        RuntimeConfig(backend="numpy", device="cuda").resolve()


def test_runtime_report_contains_native_targets() -> None:
    pairs = {(row.backend, row.device) for row in runtime_report()}
    assert ("platform", "native") in pairs
    assert ("torch", "cuda") in pairs
    assert ("torch", "mps") in pairs
    assert ("jax", "cuda") in pairs


def test_runtime_rejects_wsl_environment(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "is_wsl", lambda: True)

    status = runtime.native_environment_status()
    assert not status.available
    assert "WSL" in status.reason

    with pytest.raises(UnsupportedRuntimeError, match="WSL"):
        RuntimeConfig(backend="numpy", device="cpu").resolve()


def test_runtime_purity_audit_passes_package_sources() -> None:
    report = audit_runtime_purity(Path("src/nydsge"))

    assert report.passed
    assert report.checked_files > 0
    assert report.findings == ()


def test_runtime_purity_audit_reports_prohibited_runtime_references(tmp_path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "bad.py").write_text(
        "import subprocess\nsubprocess.run(['julia', 'script.jl'])\n",
        encoding="utf-8",
    )

    report = audit_runtime_purity(package)

    assert not report.passed
    assert report.checked_files == 1
    assert report.findings[0].pattern == "subprocess.run"
    assert report.findings[0].line == 2


def test_runtime_purity_audit_reports_wsl_executable_hooks(tmp_path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "bad.py").write_text('runner = "wsl.exe"\n', encoding="utf-8")

    report = audit_runtime_purity(package)

    assert not report.passed
    assert report.findings[0].pattern == "wsl.exe"


def test_torch_mps_float64_is_rejected_even_when_mps_available(monkeypatch) -> None:
    original_status_for = runtime._status_for

    def fake_status_for(backend: str, device: str) -> RuntimeStatus:
        if backend == "torch" and device == "mps":
            return RuntimeStatus("torch", "mps", True, "test MPS")
        return original_status_for(backend, device)

    monkeypatch.setattr(runtime, "_status_for", fake_status_for)

    with pytest.raises(UnsupportedRuntimeError, match="float64"):
        RuntimeConfig(backend="torch", device="mps", dtype="float64").resolve()

    resolved = RuntimeConfig(backend="torch", device="mps", dtype="float32").resolve()
    assert resolved.backend == "torch"
    assert resolved.device == "mps"
