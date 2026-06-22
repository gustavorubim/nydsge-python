from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np

from nydsge.backends import get_backend
from nydsge.forecast import ForecastOutput, forecast_linear_system
from nydsge.kalman import KalmanResult, kalman_log_likelihood
from nydsge.models import Model1002
from nydsge.runtime import (
    BackendName,
    DeviceName,
    DTypeName,
    RuntimeConfig,
    RuntimeStatus,
    UnsupportedRuntimeError,
    runtime_report,
)
from nydsge.solve import compute_system

ParityKernel = Literal["forecast", "kalman", "all"]


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    device: str
    available: bool
    skipped: bool
    reason: str
    horizon: int
    repeats: int
    dtype: str
    kernel: str = "forecast"
    elapsed_seconds: float | None = None
    states_shape: tuple[int, int] | None = None
    observables_shape: tuple[int, int] | None = None
    pseudo_observables_shape: tuple[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["states_shape"] = None if self.states_shape is None else list(self.states_shape)
        payload["observables_shape"] = (
            None if self.observables_shape is None else list(self.observables_shape)
        )
        payload["pseudo_observables_shape"] = (
            None if self.pseudo_observables_shape is None else list(self.pseudo_observables_shape)
        )
        return payload


@dataclass(frozen=True)
class BackendParityResult:
    backend: str
    device: str
    kernel: str
    dtype: str
    available: bool
    skipped: bool
    passed: bool
    reason: str
    atol: float
    rtol: float
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def benchmark_forecast_targets(
    *,
    horizon: int = 40,
    repeats: int = 3,
    dtype: DTypeName = "float64",
    include_pseudo: bool = False,
    statuses: list[RuntimeStatus] | None = None,
) -> list[BenchmarkResult]:
    if horizon < 0:
        msg = "Benchmark horizon must be nonnegative."
        raise ValueError(msg)
    if repeats <= 0:
        msg = "Benchmark repeats must be positive."
        raise ValueError(msg)

    target_statuses = _backend_statuses(statuses)
    system = compute_system(Model1002())
    initial_state = np.zeros(system.transition.TTT.shape[0], dtype=np.float64)
    results: list[BenchmarkResult] = []

    for status in target_statuses:
        if not status.available:
            results.append(
                _skipped_result(
                    status,
                    horizon=horizon,
                    repeats=repeats,
                    dtype=dtype,
                    kernel="forecast",
                )
            )
            continue
        try:
            runtime = RuntimeConfig(
                backend=cast(BackendName, status.backend),
                device=cast(DeviceName, status.device),
                dtype=dtype,
            )
            try:
                resolved = runtime.resolve()
            except UnsupportedRuntimeError as err:
                results.append(
                    _skipped_result(
                        status,
                        horizon=horizon,
                        repeats=repeats,
                        dtype=dtype,
                        kernel="forecast",
                        reason=str(err),
                    )
                )
                continue
            backend = get_backend(resolved)
            forecast_linear_system(
                system,
                initial_state,
                horizon=min(1, horizon),
                include_pseudo=include_pseudo,
                backend=backend,
            )
            started = perf_counter()
            output = None
            for _ in range(repeats):
                output = forecast_linear_system(
                    system,
                    initial_state,
                    horizon=horizon,
                    include_pseudo=include_pseudo,
                    backend=backend,
                )
            elapsed = perf_counter() - started
            if output is None:
                msg = "Benchmark did not produce output."
                raise RuntimeError(msg)
            results.append(
                BenchmarkResult(
                    backend=status.backend,
                    device=status.device,
                    available=True,
                    skipped=False,
                    reason=status.reason,
                    horizon=horizon,
                    repeats=repeats,
                    dtype=dtype,
                    kernel="forecast",
                    elapsed_seconds=elapsed,
                    states_shape=output.states.shape,
                    observables_shape=output.observables.shape,
                    pseudo_observables_shape=(
                        None
                        if output.pseudo_observables is None
                        else output.pseudo_observables.shape
                    ),
                )
            )
        except Exception as err:  # pragma: no cover - depends on optional native runtimes.
            results.append(
                BenchmarkResult(
                    backend=status.backend,
                    device=status.device,
                    available=False,
                    skipped=False,
                    reason=f"benchmark failed: {err}",
                    horizon=horizon,
                    repeats=repeats,
                    dtype=dtype,
                    kernel="forecast",
                )
            )

    return results


def compare_backend_parity_targets(
    *,
    kernel: ParityKernel = "all",
    horizon: int = 40,
    periods: int = 40,
    dtype: DTypeName = "float64",
    include_pseudo: bool = False,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
    statuses: list[RuntimeStatus] | None = None,
) -> list[BackendParityResult]:
    if kernel not in {"forecast", "kalman", "all"}:
        msg = "Parity kernel must be forecast, kalman, or all."
        raise ValueError(msg)
    if horizon < 0:
        msg = "Parity forecast horizon must be nonnegative."
        raise ValueError(msg)
    if periods <= 0:
        msg = "Parity Kalman periods must be positive."
        raise ValueError(msg)
    if atol < 0.0 or rtol < 0.0:
        msg = "Parity tolerances must be nonnegative."
        raise ValueError(msg)

    target_statuses = _backend_statuses(statuses)
    results: list[BackendParityResult] = []
    if kernel in {"forecast", "all"}:
        results.extend(
            _compare_forecast_parity(
                target_statuses,
                horizon=horizon,
                dtype=dtype,
                include_pseudo=include_pseudo,
                atol=atol,
                rtol=rtol,
            )
        )
    if kernel in {"kalman", "all"}:
        results.extend(
            _compare_kalman_parity(
                target_statuses,
                periods=periods,
                dtype=dtype,
                atol=atol,
                rtol=rtol,
            )
        )
    return results


def benchmark_kalman_targets(
    *,
    periods: int = 40,
    repeats: int = 3,
    dtype: DTypeName = "float64",
    statuses: list[RuntimeStatus] | None = None,
) -> list[BenchmarkResult]:
    if periods <= 0:
        msg = "Benchmark periods must be positive."
        raise ValueError(msg)
    if repeats <= 0:
        msg = "Benchmark repeats must be positive."
        raise ValueError(msg)

    target_statuses = _backend_statuses(statuses)
    system = compute_system(Model1002())
    data = np.zeros((periods, system.measurement.ZZ.shape[0]), dtype=np.float64)
    results: list[BenchmarkResult] = []

    for status in target_statuses:
        if not status.available:
            results.append(
                _skipped_result(
                    status,
                    horizon=periods,
                    repeats=repeats,
                    dtype=dtype,
                    kernel="kalman",
                )
            )
            continue
        try:
            runtime = RuntimeConfig(
                backend=cast(BackendName, status.backend),
                device=cast(DeviceName, status.device),
                dtype=dtype,
            )
            try:
                resolved = runtime.resolve()
            except UnsupportedRuntimeError as err:
                results.append(
                    _skipped_result(
                        status,
                        horizon=periods,
                        repeats=repeats,
                        dtype=dtype,
                        kernel="kalman",
                        reason=str(err),
                    )
                )
                continue
            backend = get_backend(resolved)
            kalman_log_likelihood(system, data[:1], backend=backend)
            started = perf_counter()
            output = None
            for _ in range(repeats):
                output = kalman_log_likelihood(system, data, backend=backend)
            elapsed = perf_counter() - started
            if output is None:
                msg = "Benchmark did not produce output."
                raise RuntimeError(msg)
            results.append(
                BenchmarkResult(
                    backend=status.backend,
                    device=status.device,
                    available=True,
                    skipped=False,
                    reason=status.reason,
                    horizon=periods,
                    repeats=repeats,
                    dtype=dtype,
                    kernel="kalman",
                    elapsed_seconds=elapsed,
                    states_shape=output.filtered_states.shape,
                    observables_shape=data.shape,
                )
            )
        except Exception as err:  # pragma: no cover - depends on optional native runtimes.
            results.append(
                BenchmarkResult(
                    backend=status.backend,
                    device=status.device,
                    available=False,
                    skipped=False,
                    reason=f"benchmark failed: {err}",
                    horizon=periods,
                    repeats=repeats,
                    dtype=dtype,
                    kernel="kalman",
                )
            )

    return results


def _compare_forecast_parity(
    statuses: list[RuntimeStatus],
    *,
    horizon: int,
    dtype: DTypeName,
    include_pseudo: bool,
    atol: float,
    rtol: float,
) -> list[BackendParityResult]:
    system = compute_system(Model1002())
    initial_state = np.zeros(system.transition.TTT.shape[0], dtype=np.float64)
    reference = forecast_linear_system(
        system,
        initial_state,
        horizon=horizon,
        include_pseudo=include_pseudo,
    )
    return [
        _compare_status_to_reference(
            status,
            kernel="forecast",
            dtype=dtype,
            atol=atol,
            rtol=rtol,
            run_target=lambda backend: forecast_linear_system(
                system,
                initial_state,
                horizon=horizon,
                include_pseudo=include_pseudo,
                backend=backend,
            ),
            arrays=lambda output: _forecast_arrays(cast(ForecastOutput, output), include_pseudo),
            reference_arrays=_forecast_arrays(reference, include_pseudo),
        )
        for status in statuses
    ]


def _backend_statuses(statuses: list[RuntimeStatus] | None) -> list[RuntimeStatus]:
    return [
        status
        for status in (runtime_report() if statuses is None else statuses)
        if status.backend in {"numpy", "torch", "jax"}
    ]


def _compare_kalman_parity(
    statuses: list[RuntimeStatus],
    *,
    periods: int,
    dtype: DTypeName,
    atol: float,
    rtol: float,
) -> list[BackendParityResult]:
    system = compute_system(Model1002())
    data = np.zeros((periods, system.measurement.ZZ.shape[0]), dtype=np.float64)
    reference = kalman_log_likelihood(system, data)
    return [
        _compare_status_to_reference(
            status,
            kernel="kalman",
            dtype=dtype,
            atol=atol,
            rtol=rtol,
            run_target=lambda backend: kalman_log_likelihood(system, data, backend=backend),
            arrays=lambda output: _kalman_arrays(cast(KalmanResult, output)),
            reference_arrays=_kalman_arrays(reference),
        )
        for status in statuses
    ]


def _compare_status_to_reference(
    status: RuntimeStatus,
    *,
    kernel: str,
    dtype: DTypeName,
    atol: float,
    rtol: float,
    run_target: Any,
    arrays: Any,
    reference_arrays: tuple[np.ndarray, ...],
) -> BackendParityResult:
    if not status.available:
        return _skipped_parity_result(status, kernel=kernel, dtype=dtype, atol=atol, rtol=rtol)
    try:
        runtime = RuntimeConfig(
            backend=cast(BackendName, status.backend),
            device=cast(DeviceName, status.device),
            dtype=dtype,
        )
        try:
            resolved = runtime.resolve()
        except UnsupportedRuntimeError as err:
            return _skipped_parity_result(
                status,
                kernel=kernel,
                dtype=dtype,
                atol=atol,
                rtol=rtol,
                reason=str(err),
            )
        target_arrays = arrays(run_target(get_backend(resolved)))
        max_abs, max_rel, passed = _compare_array_groups(
            reference_arrays,
            target_arrays,
            atol=atol,
            rtol=rtol,
        )
        return BackendParityResult(
            backend=status.backend,
            device=status.device,
            kernel=kernel,
            dtype=dtype,
            available=True,
            skipped=False,
            passed=passed,
            reason=status.reason,
            atol=atol,
            rtol=rtol,
            max_abs_diff=max_abs,
            max_rel_diff=max_rel,
        )
    except Exception as err:  # pragma: no cover - depends on optional native runtimes.
        return BackendParityResult(
            backend=status.backend,
            device=status.device,
            kernel=kernel,
            dtype=dtype,
            available=False,
            skipped=False,
            passed=False,
            reason=f"parity check failed: {err}",
            atol=atol,
            rtol=rtol,
        )


def _forecast_arrays(forecast: ForecastOutput, include_pseudo: bool) -> tuple[np.ndarray, ...]:
    arrays = [forecast.states, forecast.observables]
    if include_pseudo:
        if forecast.pseudo_observables is None:
            msg = "Forecast parity expected pseudo-observables."
            raise RuntimeError(msg)
        arrays.append(forecast.pseudo_observables)
    return tuple(np.asarray(array, dtype=np.float64) for array in arrays)


def _kalman_arrays(kalman: KalmanResult) -> tuple[np.ndarray, ...]:
    return (
        np.asarray([kalman.log_likelihood], dtype=np.float64),
        np.asarray(kalman.filtered_states, dtype=np.float64),
        np.asarray(kalman.filtered_covariances, dtype=np.float64),
    )


def _compare_array_groups(
    reference_arrays: tuple[np.ndarray, ...],
    target_arrays: tuple[np.ndarray, ...],
    *,
    atol: float,
    rtol: float,
) -> tuple[float, float, bool]:
    if len(reference_arrays) != len(target_arrays):
        msg = "Parity array groups must have the same length."
        raise ValueError(msg)
    max_abs = 0.0
    max_rel = 0.0
    passed = True
    for reference, target in zip(reference_arrays, target_arrays, strict=True):
        if reference.shape != target.shape:
            return float("inf"), float("inf"), False
        diff = np.abs(reference - target)
        if diff.size:
            max_abs = max(max_abs, float(np.nanmax(diff)))
            denominator = np.maximum(np.abs(reference), np.finfo(np.float64).eps)
            max_rel = max(max_rel, float(np.nanmax(diff / denominator)))
        if not np.allclose(reference, target, atol=atol, rtol=rtol, equal_nan=True):
            passed = False
    return max_abs, max_rel, passed


def _skipped_parity_result(
    status: RuntimeStatus,
    *,
    kernel: str,
    dtype: str,
    atol: float,
    rtol: float,
    reason: str | None = None,
) -> BackendParityResult:
    return BackendParityResult(
        backend=status.backend,
        device=status.device,
        kernel=kernel,
        dtype=dtype,
        available=False,
        skipped=True,
        passed=True,
        reason=status.reason if reason is None else reason,
        atol=atol,
        rtol=rtol,
    )


def _skipped_result(
    status: RuntimeStatus,
    *,
    horizon: int,
    repeats: int,
    dtype: str,
    kernel: str,
    reason: str | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        backend=status.backend,
        device=status.device,
        available=False,
        skipped=True,
        reason=status.reason if reason is None else reason,
        horizon=horizon,
        repeats=repeats,
        dtype=dtype,
        kernel=kernel,
    )
