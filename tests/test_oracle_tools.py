from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_julia_exporter_is_include_safe() -> None:
    source = (ROOT / "tools" / "oracle_julia" / "export_model1002.jl").read_text(encoding="utf-8")

    assert "function export_model1002_main(args = ARGS)" in source
    assert "if is_main_script()" in source
    assert "export_model1002_main()" in source


def test_julia_benchmark_model1002_emits_python_baseline_contract() -> None:
    source = (ROOT / "tools" / "oracle_julia" / "benchmark_model1002.jl").read_text(
        encoding="utf-8"
    )

    assert "benchmark_forecast_entry" in source
    assert "forecast(" in source
    assert '"results"' in source
    assert '"elapsed_seconds"' in source
    assert '"elapsed_samples_seconds"' in source
    assert 'write_json_report(options["out"], report)' in source
