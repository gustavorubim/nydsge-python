from __future__ import annotations

from typing import Any

import pytest

from nydsge.bench import (
    benchmark_forecast_targets,
    benchmark_kalman_targets,
    compare_backend_parity_targets,
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
    parity_results = compare_backend_parity_targets(
        kernel="forecast",
        horizon=1,
        periods=1,
        statuses=statuses,
    )

    assert {result.backend for result in forecast_results} == {"numpy"}
    assert {result.backend for result in kalman_results} == {"numpy"}
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
