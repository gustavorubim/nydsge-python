from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np

from nydsge.backends import get_backend
from nydsge.estimate import estimate as estimate_model
from nydsge.forecast import (
    ForecastOutput,
    compute_meansbands,
    forecast_linear_system,
    forecast_one,
)
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
    states_shape: tuple[int, ...] | None = None
    observables_shape: tuple[int, ...] | None = None
    pseudo_observables_shape: tuple[int, ...] | None = None
    batches: int = 1
    draws: int = 0
    artifacts: int | None = None
    baseline_name: str | None = None
    baseline_elapsed_seconds: float | None = None
    speedup_vs_baseline: float | None = None

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
class BenchmarkBaseline:
    name: str
    kernel: str
    elapsed_seconds: float
    horizon: int | None = None
    dtype: str | None = None
    batches: int | None = None
    draws: int | None = None


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


def benchmark_reference_report(
    results: list[BenchmarkResult],
    *,
    kernel: str,
    horizon: int,
    periods: int,
    batches: int,
    draws: int,
    repeats: int,
    dtype: str,
    include_pseudo: bool,
    baseline_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a durable benchmark report for local and cross-machine references."""

    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "command": {
            "kernel": kernel,
            "horizon": horizon,
            "periods": periods,
            "batches": batches,
            "draws": draws,
            "repeats": repeats,
            "dtype": dtype,
            "include_pseudo": include_pseudo,
            "baseline_path": None if baseline_path is None else str(baseline_path),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
        },
        "results": [result.to_dict() for result in results],
    }


def write_benchmark_reference_report(
    path: Path | str,
    results: list[BenchmarkResult],
    *,
    kernel: str,
    horizon: int,
    periods: int,
    batches: int,
    draws: int,
    repeats: int,
    dtype: str,
    include_pseudo: bool,
    baseline_path: Path | str | None = None,
) -> dict[str, Any]:
    report = benchmark_reference_report(
        results,
        kernel=kernel,
        horizon=horizon,
        periods=periods,
        batches=batches,
        draws=draws,
        repeats=repeats,
        dtype=dtype,
        include_pseudo=include_pseudo,
        baseline_path=baseline_path,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


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


def load_benchmark_baselines(path: Path | str) -> tuple[BenchmarkBaseline, ...]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    default_name = source.stem
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        default_name = str(payload.get("name", payload.get("source", default_name)))
        if isinstance(payload.get("results"), list):
            entries = payload["results"]
        elif isinstance(payload.get("baselines"), list):
            entries = payload["baselines"]
        else:
            entries = [payload]
    else:
        msg = "Benchmark baseline must be a JSON object or list."
        raise ValueError(msg)

    baselines: list[BenchmarkBaseline] = []
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            msg = f"Benchmark baseline entry {index} must be an object."
            raise ValueError(msg)
        kernel = str(item.get("kernel", "")).strip()
        if not kernel:
            msg = f"Benchmark baseline entry {index} is missing kernel."
            raise ValueError(msg)
        if "elapsed_seconds" not in item:
            msg = f"Benchmark baseline entry {index} is missing elapsed_seconds."
            raise ValueError(msg)
        elapsed_seconds = float(item["elapsed_seconds"])
        if not np.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
            msg = f"Benchmark baseline entry {index} elapsed_seconds must be positive."
            raise ValueError(msg)
        baselines.append(
            BenchmarkBaseline(
                name=str(
                    item.get("baseline_name", item.get("name", item.get("backend", default_name)))
                ),
                kernel=kernel,
                elapsed_seconds=elapsed_seconds,
                horizon=_optional_int(item.get("horizon")),
                dtype=None if item.get("dtype") is None else str(item["dtype"]),
                batches=_optional_int(item.get("batches")),
                draws=_optional_int(item.get("draws")),
            )
        )
    return tuple(baselines)


def apply_benchmark_baselines(
    results: list[BenchmarkResult],
    baselines: tuple[BenchmarkBaseline, ...],
) -> list[BenchmarkResult]:
    if not baselines:
        return results
    matched: list[BenchmarkResult] = []
    for result in results:
        baseline = _matching_baseline(result, baselines)
        if baseline is None:
            matched.append(result)
            continue
        speedup = None
        if result.elapsed_seconds is not None and result.elapsed_seconds > 0.0:
            speedup = float(baseline.elapsed_seconds / result.elapsed_seconds)
        matched.append(
            replace(
                result,
                baseline_name=baseline.name,
                baseline_elapsed_seconds=baseline.elapsed_seconds,
                speedup_vs_baseline=speedup,
            )
        )
    return matched


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


def benchmark_batched_kalman_targets(
    *,
    periods: int = 40,
    batches: int = 8,
    repeats: int = 3,
    dtype: DTypeName = "float64",
    statuses: list[RuntimeStatus] | None = None,
) -> list[BenchmarkResult]:
    if periods <= 0:
        msg = "Benchmark periods must be positive."
        raise ValueError(msg)
    if batches <= 0:
        msg = "Benchmark batches must be positive."
        raise ValueError(msg)
    if repeats <= 0:
        msg = "Benchmark repeats must be positive."
        raise ValueError(msg)

    target_statuses = _backend_statuses(statuses)
    system = compute_system(Model1002())
    data = np.zeros((batches, periods, system.measurement.ZZ.shape[0]), dtype=np.float64)
    results: list[BenchmarkResult] = []

    for status in target_statuses:
        if not status.available:
            results.append(
                _skipped_result(
                    status,
                    horizon=periods,
                    repeats=repeats,
                    dtype=dtype,
                    kernel="kalman-batch",
                    batches=batches,
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
                        kernel="kalman-batch",
                        reason=str(err),
                        batches=batches,
                    )
                )
                continue
            backend = get_backend(resolved)
            _run_batched_kalman(system, data[:1, :1], backend)
            started = perf_counter()
            output: tuple[KalmanResult, ...] | None = None
            for _ in range(repeats):
                output = _run_batched_kalman(system, data, backend)
            elapsed = perf_counter() - started
            if output is None or not output:
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
                    kernel="kalman-batch",
                    elapsed_seconds=elapsed,
                    states_shape=(batches, *output[0].filtered_states.shape),
                    observables_shape=data.shape,
                    batches=batches,
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
                    kernel="kalman-batch",
                    batches=batches,
                )
            )

    return results


def benchmark_hard_target_replay_targets(
    *,
    horizon: int = 2,
    periods: int = 2,
    draws: int = 2,
    repeats: int = 1,
    dtype: DTypeName = "float64",
    statuses: list[RuntimeStatus] | None = None,
) -> list[BenchmarkResult]:
    if horizon < 0:
        msg = "Benchmark horizon must be nonnegative."
        raise ValueError(msg)
    if periods <= 0:
        msg = "Benchmark periods must be positive."
        raise ValueError(msg)
    if draws <= 0:
        msg = "Benchmark draws must be positive."
        raise ValueError(msg)
    if repeats <= 0:
        msg = "Benchmark repeats must be positive."
        raise ValueError(msg)

    target_statuses = _backend_statuses(statuses)
    results: list[BenchmarkResult] = []

    for status in target_statuses:
        if not status.available:
            results.append(
                _skipped_result(
                    status,
                    horizon=horizon,
                    repeats=repeats,
                    dtype=dtype,
                    kernel="hard-target",
                    draws=draws,
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
                        kernel="hard-target",
                        reason=str(err),
                        draws=draws,
                    )
                )
                continue
            resolved_runtime = RuntimeConfig(
                backend=cast(BackendName, resolved.backend),
                device=cast(DeviceName, resolved.device),
                dtype=resolved.dtype,
            )
            _run_hard_target_replay(
                resolved_runtime,
                periods=periods,
                horizon=min(1, horizon),
                draws=1,
            )
            started = perf_counter()
            output = None
            for _ in range(repeats):
                output = _run_hard_target_replay(
                    resolved_runtime,
                    periods=periods,
                    horizon=horizon,
                    draws=draws,
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
                    kernel="hard-target",
                    elapsed_seconds=elapsed,
                    states_shape=output["states_shape"],
                    observables_shape=output["observables_shape"],
                    batches=1,
                    draws=draws,
                    artifacts=output["artifacts"],
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
                    kernel="hard-target",
                    draws=draws,
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


def _run_batched_kalman(
    system: Any,
    data: np.ndarray,
    backend: Any,
) -> tuple[KalmanResult, ...]:
    return tuple(
        kalman_log_likelihood(system, data[batch_index], backend=backend)
        for batch_index in range(data.shape[0])
    )


def _run_hard_target_replay(
    runtime: RuntimeConfig,
    *,
    periods: int,
    horizon: int,
    draws: int,
) -> dict[str, Any]:
    model = Model1002(runtime=runtime)
    compute_system(model)
    data = np.zeros((periods, len(model.observables)), dtype=np.float64)
    shock_samples = np.zeros(
        (draws, horizon, len(model.indexes.exogenous_shocks)),
        dtype=np.float64,
    )
    mode_output_vars = ["forecastobs", "forecaststates", "histobs", "histstates"]
    mode_forecast = forecast_one(
        model,
        input_type="mode",
        cond_type="none",
        output_vars=mode_output_vars,
        horizon=horizon,
        data=data,
    )
    compute_meansbands(
        model,
        "mode",
        "none",
        ["forecastobs"],
        horizon=horizon,
        source="forecastobs",
        data=data,
    )
    compute_meansbands(
        model,
        "mode",
        "none",
        ["histobs"],
        horizon=horizon,
        source="histobs",
        data=data,
    )
    estimate_model(model, data)
    full_forecast = forecast_one(
        model,
        input_type="full",
        cond_type="none",
        output_vars=mode_output_vars,
        horizon=horizon,
        data=data,
        shock_samples=shock_samples,
    )
    compute_meansbands(
        model,
        "full",
        "none",
        ["forecastobs"],
        horizon=horizon,
        source="forecastobs",
        data=data,
        shock_samples=shock_samples,
    )
    compute_meansbands(
        model,
        "full",
        "none",
        ["histobs"],
        horizon=horizon,
        source="histobs",
        data=data,
        shock_samples=shock_samples,
    )
    return {
        "states_shape": full_forecast.states.shape,
        "observables_shape": full_forecast.observables.shape,
        "mode_states_shape": mode_forecast.states.shape,
        "artifacts": 8,
    }


def _matching_baseline(
    result: BenchmarkResult,
    baselines: tuple[BenchmarkBaseline, ...],
) -> BenchmarkBaseline | None:
    for baseline in baselines:
        if baseline.kernel != result.kernel:
            continue
        if baseline.horizon is not None and baseline.horizon != result.horizon:
            continue
        if baseline.dtype is not None and baseline.dtype != result.dtype:
            continue
        if baseline.batches is not None and baseline.batches != result.batches:
            continue
        if baseline.draws is not None and baseline.draws != result.draws:
            continue
        return baseline
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


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
    batches: int = 1,
    draws: int = 0,
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
        batches=batches,
        draws=draws,
    )
