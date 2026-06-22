from __future__ import annotations

import json
from typing import Any

import pytest

from nydsge.bench import (
    BenchmarkResult,
    apply_benchmark_baselines,
    benchmark_batched_kalman_targets,
    benchmark_forecast_targets,
    benchmark_hard_target_replay_targets,
    benchmark_kalman_targets,
    benchmark_reference_report,
    compare_backend_parity_targets,
    load_benchmark_baselines,
)
from nydsge.runtime import RuntimeStatus


def test_benchmark_forecast_targets_runs_available_numpy_and_skips_unavailable() -> None:
    results = benchmark_forecast_targets(
        horizon=1,
        repeats=1,
        include_pseudo=True,
        statuses=[
            RuntimeStatus("numpy", "cpu", True, "test numpy"),
            RuntimeStatus("torch", "cuda", False, "no CUDA"),
        ],
    )

    assert len(results) == 2
    assert results[0].backend == "numpy"
    assert results[0].kernel == "forecast"
    assert results[0].available
    assert not results[0].skipped
    assert results[0].elapsed_seconds is not None
    assert results[0].states_shape == (1, 84)
    assert results[0].observables_shape == (1, 19)
    assert results[0].pseudo_observables_shape == (1, 21)
    assert results[1].backend == "torch"
    assert results[1].kernel == "forecast"
    assert results[1].device == "cuda"
    assert results[1].skipped
    assert results[1].elapsed_seconds is None


def test_benchmark_forecast_targets_validates_inputs() -> None:
    with pytest.raises(ValueError, match="horizon"):
        benchmark_forecast_targets(horizon=-1)

    with pytest.raises(ValueError, match="repeats"):
        benchmark_forecast_targets(repeats=0)


def test_benchmark_kalman_targets_runs_available_numpy_and_skips_unavailable() -> None:
    results = benchmark_kalman_targets(
        periods=1,
        repeats=1,
        statuses=[
            RuntimeStatus("numpy", "cpu", True, "test numpy"),
            RuntimeStatus("torch", "cuda", False, "no CUDA"),
        ],
    )

    assert len(results) == 2
    assert results[0].backend == "numpy"
    assert results[0].kernel == "kalman"
    assert results[0].available
    assert not results[0].skipped
    assert results[0].elapsed_seconds is not None
    assert results[0].states_shape == (1, 84)
    assert results[0].observables_shape == (1, 19)
    assert results[1].backend == "torch"
    assert results[1].kernel == "kalman"
    assert results[1].device == "cuda"
    assert results[1].skipped
    assert results[1].elapsed_seconds is None


def test_benchmark_kalman_targets_validates_inputs() -> None:
    with pytest.raises(ValueError, match="periods"):
        benchmark_kalman_targets(periods=0)

    with pytest.raises(ValueError, match="repeats"):
        benchmark_kalman_targets(repeats=0)


def test_benchmark_batched_kalman_targets_runs_available_numpy_and_skips_unavailable() -> None:
    results = benchmark_batched_kalman_targets(
        periods=1,
        batches=2,
        repeats=1,
        statuses=[
            RuntimeStatus("numpy", "cpu", True, "test numpy"),
            RuntimeStatus("torch", "cuda", False, "no CUDA"),
        ],
    )

    assert len(results) == 2
    assert results[0].backend == "numpy"
    assert results[0].kernel == "kalman-batch"
    assert results[0].available
    assert not results[0].skipped
    assert results[0].elapsed_seconds is not None
    assert results[0].states_shape == (2, 1, 84)
    assert results[0].observables_shape == (2, 1, 19)
    assert results[0].batches == 2
    assert results[1].backend == "torch"
    assert results[1].kernel == "kalman-batch"
    assert results[1].device == "cuda"
    assert results[1].skipped
    assert results[1].elapsed_seconds is None
    assert results[1].batches == 2


def test_benchmark_batched_kalman_targets_validates_inputs() -> None:
    with pytest.raises(ValueError, match="periods"):
        benchmark_batched_kalman_targets(periods=0)

    with pytest.raises(ValueError, match="batches"):
        benchmark_batched_kalman_targets(batches=0)

    with pytest.raises(ValueError, match="repeats"):
        benchmark_batched_kalman_targets(repeats=0)


def test_benchmark_hard_target_replay_targets_runs_available_numpy_and_skips_unavailable() -> None:
    results = benchmark_hard_target_replay_targets(
        horizon=1,
        periods=1,
        draws=1,
        repeats=1,
        statuses=[
            RuntimeStatus("numpy", "cpu", True, "test numpy"),
            RuntimeStatus("torch", "cuda", False, "no CUDA"),
        ],
    )

    assert len(results) == 2
    assert results[0].backend == "numpy"
    assert results[0].kernel == "hard-target"
    assert results[0].available
    assert not results[0].skipped
    assert results[0].elapsed_seconds is not None
    assert results[0].states_shape == (1, 84)
    assert results[0].observables_shape == (1, 19)
    assert results[0].draws == 1
    assert results[0].artifacts == 8
    assert results[1].backend == "torch"
    assert results[1].kernel == "hard-target"
    assert results[1].device == "cuda"
    assert results[1].skipped
    assert results[1].elapsed_seconds is None
    assert results[1].draws == 1


def test_benchmark_hard_target_replay_targets_validates_inputs() -> None:
    with pytest.raises(ValueError, match="horizon"):
        benchmark_hard_target_replay_targets(horizon=-1)

    with pytest.raises(ValueError, match="periods"):
        benchmark_hard_target_replay_targets(periods=0)

    with pytest.raises(ValueError, match="draws"):
        benchmark_hard_target_replay_targets(draws=0)

    with pytest.raises(ValueError, match="repeats"):
        benchmark_hard_target_replay_targets(repeats=0)


def test_load_and_apply_benchmark_baselines(tmp_path) -> None:
    baseline_path = tmp_path / "julia_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "name": "julia-1.8",
                "results": [
                    {
                        "kernel": "forecast",
                        "horizon": 2,
                        "dtype": "float64",
                        "elapsed_seconds": 4.0,
                    },
                    {
                        "kernel": "kalman-batch",
                        "horizon": 2,
                        "batches": 3,
                        "elapsed_seconds": 9.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    baselines = load_benchmark_baselines(baseline_path)
    results = apply_benchmark_baselines(
        [
            BenchmarkResult(
                backend="numpy",
                device="cpu",
                available=True,
                skipped=False,
                reason="ok",
                horizon=2,
                repeats=1,
                dtype="float64",
                kernel="forecast",
                elapsed_seconds=2.0,
            ),
            BenchmarkResult(
                backend="numpy",
                device="cpu",
                available=True,
                skipped=False,
                reason="ok",
                horizon=2,
                repeats=1,
                dtype="float64",
                kernel="kalman-batch",
                elapsed_seconds=3.0,
                batches=3,
            ),
        ],
        baselines,
    )

    assert results[0].baseline_name == "julia-1.8"
    assert results[0].baseline_elapsed_seconds == 4.0
    assert results[0].speedup_vs_baseline == 2.0
    assert results[1].baseline_elapsed_seconds == 9.0
    assert results[1].speedup_vs_baseline == 3.0


def test_load_benchmark_baselines_accepts_julia_oracle_report(tmp_path) -> None:
    baseline_path = tmp_path / "julia_benchmark_model1002.json"
    baseline_path.write_text(
        json.dumps(
            {
                "name": "julia-model1002-ss10",
                "source": "tools/oracle_julia/benchmark_model1002.jl",
                "results": [
                    {
                        "name": "julia-model1002-ss10-forecast",
                        "backend": "julia",
                        "device": "cpu",
                        "kernel": "forecast",
                        "horizon": 2,
                        "repeats": 3,
                        "dtype": "float64",
                        "elapsed_seconds": 1.5,
                        "mean_elapsed_seconds": 1.7,
                        "elapsed_samples_seconds": [1.5, 1.7, 1.9],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    baselines = load_benchmark_baselines(baseline_path)

    assert len(baselines) == 1
    assert baselines[0].name == "julia-model1002-ss10-forecast"
    assert baselines[0].kernel == "forecast"
    assert baselines[0].horizon == 2
    assert baselines[0].dtype == "float64"
    assert baselines[0].elapsed_seconds == 1.5


def test_load_benchmark_baselines_validates_schema(tmp_path) -> None:
    baseline_path = tmp_path / "bad.json"
    baseline_path.write_text(
        json.dumps({"results": [{"kernel": "forecast", "elapsed_seconds": 0.0}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="elapsed_seconds"):
        load_benchmark_baselines(baseline_path)


def test_benchmark_reference_report_records_command_platform_and_results() -> None:
    results = [
        BenchmarkResult(
            backend="numpy",
            device="cpu",
            available=True,
            skipped=False,
            reason="ok",
            horizon=1,
            repeats=1,
            dtype="float64",
            kernel="forecast",
            elapsed_seconds=0.01,
        )
    ]

    report = benchmark_reference_report(
        results,
        kernel="forecast",
        horizon=1,
        periods=2,
        batches=3,
        draws=4,
        repeats=1,
        dtype="float64",
        include_pseudo=True,
        baseline_path="oracle.json",
    )

    assert report["schema_version"] == 1
    assert report["created_utc"].endswith("Z")
    assert report["command"]["kernel"] == "forecast"
    assert report["command"]["baseline_path"] == "oracle.json"
    assert report["command"]["include_pseudo"] is True
    assert report["platform"]["system"]
    assert report["runtime_statuses"]
    assert report["runtime_statuses"][0]["backend"] == "platform"
    assert report["results"][0]["backend"] == "numpy"
    assert report["results"][0]["elapsed_seconds"] == 0.01


def test_compare_backend_parity_targets_passes_numpy_and_skips_unavailable() -> None:
    results = compare_backend_parity_targets(
        kernel="all",
        horizon=1,
        periods=1,
        include_pseudo=True,
        statuses=[
            RuntimeStatus("numpy", "cpu", True, "test numpy"),
            RuntimeStatus("torch", "cuda", False, "no CUDA"),
        ],
    )

    assert len(results) == 4
    passed = [result for result in results if result.backend == "numpy"]
    skipped = [result for result in results if result.backend == "torch"]
    assert {result.kernel for result in passed} == {"forecast", "kalman"}
    assert all(result.passed for result in passed)
    assert all(not result.skipped for result in passed)
    assert all(result.max_abs_diff == 0.0 for result in passed)
    assert all(result.skipped for result in skipped)
    assert all(result.passed for result in skipped)


def test_backend_target_runners_exclude_platform_status_rows() -> None:
    statuses = [
        RuntimeStatus("platform", "native", True, "native platform"),
        RuntimeStatus("numpy", "cpu", True, "test numpy"),
    ]

    forecast_results = benchmark_forecast_targets(horizon=1, repeats=1, statuses=statuses)
    kalman_results = benchmark_kalman_targets(periods=1, repeats=1, statuses=statuses)
    batched_results = benchmark_batched_kalman_targets(
        periods=1,
        batches=1,
        repeats=1,
        statuses=statuses,
    )
    hard_target_results = benchmark_hard_target_replay_targets(
        horizon=1,
        periods=1,
        draws=1,
        repeats=1,
        statuses=statuses,
    )
    parity_results = compare_backend_parity_targets(
        kernel="forecast",
        horizon=1,
        periods=1,
        statuses=statuses,
    )

    assert {result.backend for result in forecast_results} == {"numpy"}
    assert {result.backend for result in kalman_results} == {"numpy"}
    assert {result.backend for result in batched_results} == {"numpy"}
    assert {result.backend for result in hard_target_results} == {"numpy"}
    assert {result.backend for result in parity_results} == {"numpy"}


def test_compare_backend_parity_targets_validates_inputs() -> None:
    bad_kernel: Any = "bad"
    with pytest.raises(ValueError, match="kernel"):
        compare_backend_parity_targets(kernel=bad_kernel)

    with pytest.raises(ValueError, match="horizon"):
        compare_backend_parity_targets(horizon=-1)

    with pytest.raises(ValueError, match="periods"):
        compare_backend_parity_targets(periods=0)

    with pytest.raises(ValueError, match="tolerances"):
        compare_backend_parity_targets(atol=-1.0)
