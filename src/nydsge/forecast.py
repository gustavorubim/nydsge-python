from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from nydsge.backends import ArrayBackend, NumpyBackend, get_backend
from nydsge.core import DSGEModel, NotPortedError, Parameter
from nydsge.data import (
    date_labels_for_sample,
    df_to_matrix,
    load_data,
    reverse_transform_observables,
    reverse_transform_pseudo_observables,
)
from nydsge.estimate import MetropolisHastingsResult
from nydsge.kalman import (
    KalmanResult,
    kalman_log_likelihood,
    model_process_covariances,
    pseudo_observables_from_states,
    smooth_kalman_result,
)
from nydsge.solve import System, compute_system


@dataclass(frozen=True)
class ForecastOutput:
    states: np.ndarray
    observables: np.ndarray
    pseudo_observables: np.ndarray | None = None
    conditional_shocks: np.ndarray | None = None
    conditional_states: np.ndarray | None = None
    conditional_observables: np.ndarray | None = None
    state_samples: np.ndarray | None = None
    observable_samples: np.ndarray | None = None
    pseudo_observable_samples: np.ndarray | None = None
    history_states: np.ndarray | None = None
    history_observables: np.ndarray | None = None
    history_pseudo_observables: np.ndarray | None = None
    history_state_samples: np.ndarray | None = None
    history_observable_samples: np.ndarray | None = None
    history_pseudo_observable_samples: np.ndarray | None = None
    log_likelihood: float | None = None


@dataclass(frozen=True)
class MeansBands:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class ShockConditioningResult:
    shocks: np.ndarray
    states: np.ndarray
    observables: np.ndarray
    residuals: np.ndarray
    max_abs_error: float
    rank: int


@dataclass(frozen=True)
class _ForecastDataSplit:
    observations: np.ndarray
    history_periods: int
    conditional_periods: int
    start_date: Any | None = None


def forecast_linear_system(
    system: System,
    initial_state: np.ndarray,
    *,
    horizon: int,
    shocks: np.ndarray | None = None,
    include_pseudo: bool = False,
    backend: ArrayBackend | None = None,
) -> ForecastOutput:
    if horizon < 0:
        msg = "Forecast horizon must be nonnegative."
        raise ValueError(msg)
    array_backend = backend or NumpyBackend()
    state = np.asarray(initial_state, dtype=np.float64)
    transition = system.transition
    measurement = system.measurement
    if state.shape != (transition.TTT.shape[0],):
        msg = f"Initial state must have shape {(transition.TTT.shape[0],)}."
        raise ValueError(msg)
    shock_matrix = _normalize_shock_matrix(
        shocks,
        horizon=horizon,
        n_shocks=transition.RRR.shape[1],
    )
    backend_state = array_backend.array(state)
    backend_ttt = array_backend.array(transition.TTT)
    backend_rrr = array_backend.array(transition.RRR)
    backend_ccc = array_backend.array(transition.CCC)
    backend_zz = array_backend.array(measurement.ZZ)
    backend_dd = array_backend.array(measurement.DD)
    backend_shocks = array_backend.array(shock_matrix)

    states = np.zeros((horizon, state.shape[0]), dtype=np.float64)
    observables = np.zeros((horizon, measurement.ZZ.shape[0]), dtype=np.float64)
    pseudo_observables = None
    backend_zz_pseudo = None
    backend_dd_pseudo = None
    if include_pseudo:
        if system.pseudo_measurement is None:
            msg = "System does not include pseudo-measurement matrices."
            raise ValueError(msg)
        backend_zz_pseudo = array_backend.array(system.pseudo_measurement.ZZ_pseudo)
        backend_dd_pseudo = array_backend.array(system.pseudo_measurement.DD_pseudo)
        pseudo_observables = np.zeros(
            (horizon, system.pseudo_measurement.ZZ_pseudo.shape[0]),
            dtype=np.float64,
        )
    for period in range(horizon):
        backend_state = (
            array_backend.matmul(backend_ttt, backend_state)
            + array_backend.matmul(backend_rrr, backend_shocks[period])
            + backend_ccc
        )
        states[period] = array_backend.as_numpy(backend_state)
        observables[period] = array_backend.as_numpy(
            array_backend.matmul(backend_zz, backend_state) + backend_dd
        )
        if pseudo_observables is not None and backend_zz_pseudo is not None:
            pseudo_observables[period] = array_backend.as_numpy(
                array_backend.matmul(backend_zz_pseudo, backend_state) + backend_dd_pseudo
            )
    return ForecastOutput(
        states=states,
        observables=observables,
        pseudo_observables=pseudo_observables,
    )


def forecast_linear_system_samples(
    system: System,
    initial_state: np.ndarray,
    *,
    horizon: int,
    draws: int,
    seed: int | None = None,
    shock_samples: np.ndarray | None = None,
    include_pseudo: bool = False,
    backend: ArrayBackend | None = None,
) -> ForecastOutput:
    if horizon < 0:
        msg = "Forecast horizon must be nonnegative."
        raise ValueError(msg)
    n_shocks = system.transition.RRR.shape[1]
    if shock_samples is None:
        if draws <= 0:
            msg = "Forecast draws must be positive."
            raise ValueError(msg)
        rng = np.random.default_rng(seed)
        shock_covariance = np.asarray(system.measurement.QQ, dtype=np.float64)
        shocks = rng.multivariate_normal(
            mean=np.zeros(shock_covariance.shape[0], dtype=np.float64),
            cov=shock_covariance,
            size=(draws, horizon),
            check_valid="raise",
        )
    else:
        shocks = _normalize_shock_samples(
            shock_samples,
            horizon=horizon,
            n_shocks=n_shocks,
        )
        if shocks.shape[0] == 0:
            msg = "Shock samples must include at least one draw."
            raise ValueError(msg)
        if draws > 0 and draws != shocks.shape[0]:
            msg = (
                "Forecast draws must match explicit shock sample draws: "
                f"{draws} != {shocks.shape[0]}."
            )
            raise ValueError(msg)
        draws = shocks.shape[0]
    state_samples = np.zeros((draws, horizon, system.transition.TTT.shape[0]), dtype=np.float64)
    observable_samples = np.zeros(
        (draws, horizon, system.measurement.ZZ.shape[0]),
        dtype=np.float64,
    )
    pseudo_observable_samples = None
    if include_pseudo:
        if system.pseudo_measurement is None:
            msg = "System does not include pseudo-measurement matrices."
            raise ValueError(msg)
        pseudo_observable_samples = np.zeros(
            (draws, horizon, system.pseudo_measurement.ZZ_pseudo.shape[0]),
            dtype=np.float64,
        )

    for draw in range(draws):
        output = forecast_linear_system(
            system,
            initial_state,
            horizon=horizon,
            shocks=shocks[draw],
            include_pseudo=include_pseudo,
            backend=backend,
        )
        state_samples[draw] = output.states
        observable_samples[draw] = output.observables
        if pseudo_observable_samples is not None:
            if output.pseudo_observables is None:
                msg = "Pseudo-observable sample forecast did not produce pseudo outputs."
                raise RuntimeError(msg)
            pseudo_observable_samples[draw] = output.pseudo_observables

    return ForecastOutput(
        states=np.mean(state_samples, axis=0),
        observables=np.mean(observable_samples, axis=0),
        pseudo_observables=(
            None
            if pseudo_observable_samples is None
            else np.mean(pseudo_observable_samples, axis=0)
        ),
        state_samples=state_samples,
        observable_samples=observable_samples,
        pseudo_observable_samples=pseudo_observable_samples,
    )


def solve_shocks_for_observable_targets(
    system: System,
    initial_state: np.ndarray,
    targets: np.ndarray,
) -> ShockConditioningResult:
    target_array = np.asarray(targets, dtype=np.float64)
    if target_array.ndim != 2:
        msg = "Conditional targets must have shape (periods, observables)."
        raise ValueError(msg)
    if target_array.shape[1] != system.measurement.ZZ.shape[0]:
        msg = (
            "Conditional target observable count does not match measurement matrix rows: "
            f"{target_array.shape[1]} != {system.measurement.ZZ.shape[0]}"
        )
        raise ValueError(msg)

    horizon = target_array.shape[0]
    n_shocks = system.transition.RRR.shape[1]
    baseline = forecast_linear_system(system, initial_state, horizon=horizon)
    observed_mask = np.isfinite(target_array)
    if not np.any(observed_mask):
        residuals = np.full_like(target_array, np.nan, dtype=np.float64)
        return ShockConditioningResult(
            shocks=np.zeros((horizon, n_shocks), dtype=np.float64),
            states=baseline.states,
            observables=baseline.observables,
            residuals=residuals,
            max_abs_error=0.0,
            rank=0,
        )

    design = _observable_shock_design(system, horizon=horizon)
    rows = design[observed_mask].reshape(int(np.count_nonzero(observed_mask)), -1)
    target_residual = (target_array - baseline.observables)[observed_mask]
    solution, _, rank, _ = np.linalg.lstsq(rows, target_residual, rcond=None)
    shocks = solution.reshape(horizon, n_shocks)
    conditioned = forecast_linear_system(system, initial_state, horizon=horizon, shocks=shocks)
    residuals = np.full_like(target_array, np.nan, dtype=np.float64)
    residuals[observed_mask] = (conditioned.observables - target_array)[observed_mask]
    return ShockConditioningResult(
        shocks=shocks,
        states=conditioned.states,
        observables=conditioned.observables,
        residuals=residuals,
        max_abs_error=float(np.nanmax(np.abs(residuals[observed_mask]))),
        rank=int(rank),
    )


def build_zlb_conditional_observations(
    model: DSGEModel,
    policy_rate_path: Any,
    *,
    floor: float = 0.0,
    rate_units: str = "annualized",
    include_anticipated: bool = True,
) -> np.ndarray:
    """Build full-conditioning observables for a ZLB/market-implied FFR path.

    The returned matrix is in model units and uses NaN for unconstrained
    observables. `policy_rate_path` and `floor` are annualized percentage rates
    by default, matching raw FFR reporting units.
    """
    rates = np.asarray(policy_rate_path, dtype=np.float64)
    if rates.ndim != 1:
        msg = "Policy rate path must be a 1D array."
        raise ValueError(msg)
    if rates.size == 0:
        msg = "Policy rate path must include at least one period."
        raise ValueError(msg)
    units = rate_units.casefold()
    if units == "annualized":
        internal_rates = np.maximum(rates, float(floor)) / 4.0
    elif units in {"quarterly", "model"}:
        internal_rates = np.maximum(rates, float(floor))
    else:
        msg = "ZLB rate units must be 'annualized', 'quarterly', or 'model'."
        raise ValueError(msg)

    observable_names = list(model.observables)
    try:
        nominal_rate_column = observable_names.index("obs_nominalrate")
    except ValueError as err:
        msg = "Model does not include obs_nominalrate for ZLB conditioning."
        raise KeyError(msg) from err
    observations = np.full((internal_rates.size, len(observable_names)), np.nan)
    observations[:, nominal_rate_column] = internal_rates

    if include_anticipated:
        for name in observable_names:
            if not name.startswith("obs_nominalrate") or name == "obs_nominalrate":
                continue
            try:
                anticipated_horizon = int(name.removeprefix("obs_nominalrate"))
            except ValueError:
                continue
            column = observable_names.index(name)
            for period in range(internal_rates.size):
                target_period = period + anticipated_horizon
                if target_period < internal_rates.size:
                    observations[period, column] = internal_rates[target_period]
    return observations


def forecast_parameter_draws(
    model: DSGEModel,
    sampler: MetropolisHastingsResult,
    *,
    horizon: int,
    initial_state: np.ndarray | None = None,
    include_pseudo: bool = False,
    data: Any | None = None,
    check_empty_columns: bool = True,
    cond_type: str = "none",
    conditional_periods: int | None = None,
    history_method: str = "filtered",
) -> ForecastOutput:
    if horizon < 0:
        msg = "Forecast horizon must be nonnegative."
        raise ValueError(msg)
    parameter_draws = np.asarray(sampler.parameter_draws, dtype=np.float64)
    if parameter_draws.ndim != 2:
        msg = "Sampler parameter_draws must have shape (draws, parameters)."
        raise ValueError(msg)
    if parameter_draws.shape[0] == 0:
        msg = "Sampler must include at least one retained draw."
        raise ValueError(msg)
    if parameter_draws.shape[1] != len(sampler.parameter_names):
        msg = "Sampler parameter_draws column count must match parameter_names."
        raise ValueError(msg)

    original_parameters = dict(model.parameters)
    original_steady_state = dict(model.steady_state)
    state_samples = None
    observable_samples = None
    pseudo_observable_samples = None
    history_state_samples = None
    history_observable_samples = None
    history_pseudo_observable_samples = None
    log_likelihood_samples = None
    selected_history_method = _normalize_history_method(history_method)
    normalized_cond_type = _normalize_cond_type(cond_type)
    if data is None and normalized_cond_type != "none":
        data = load_data(model, check_empty_columns=check_empty_columns)
    data_split = None
    forecast_horizon = horizon
    if data is not None:
        data_split = _forecast_data_split(
            model,
            data,
            check_empty_columns=check_empty_columns,
            cond_type=normalized_cond_type,
            conditional_periods=conditional_periods,
        )
        if data_split.conditional_periods > forecast_horizon:
            msg = (
                "Conditional data periods cannot exceed the requested forecast "
                f"horizon: {data_split.conditional_periods} > {forecast_horizon}."
            )
            raise ValueError(msg)
        forecast_horizon -= data_split.conditional_periods
    try:
        for draw_index, parameter_values in enumerate(parameter_draws):
            _set_model_parameter_values(
                model,
                original_parameters,
                sampler.parameter_names,
                parameter_values,
            )
            system = compute_system(model)
            start = _initial_forecast_state(system, initial_state)
            history_states = None
            history_observables = None
            history_pseudo_observables = None
            log_likelihood = None
            if data_split is not None:
                filtered = kalman_log_likelihood(
                    system,
                    data_split.observations,
                    process_covariances=model_process_covariances(
                        model,
                        system,
                        data_split.observations.shape[0],
                        start_date=data_split.start_date,
                    ),
                    backend=get_backend(model.runtime),
                )
                log_likelihood = filtered.log_likelihood
                filtered_history_states = _states_for_history_method(
                    system,
                    filtered,
                    selected_history_method,
                )
                historical_states = filtered_history_states[: data_split.history_periods]
                if initial_state is None and filtered_history_states.shape[0] > 0:
                    start = filtered_history_states[-1].copy()
                history_states = historical_states
                history_observables = data_split.observations[: data_split.history_periods]
                if include_pseudo:
                    history_pseudo_observables = pseudo_observables_from_states(
                        system,
                        historical_states,
                    )
            output = forecast_linear_system(
                system,
                start,
                horizon=forecast_horizon,
                include_pseudo=include_pseudo,
                backend=get_backend(model.runtime),
            )
            if state_samples is None:
                state_samples = np.zeros(
                    (parameter_draws.shape[0], *output.states.shape),
                    dtype=np.float64,
                )
                observable_samples = np.zeros(
                    (parameter_draws.shape[0], *output.observables.shape),
                    dtype=np.float64,
                )
                if include_pseudo:
                    if output.pseudo_observables is None:
                        msg = "Posterior forecast did not produce pseudo-observables."
                        raise RuntimeError(msg)
                    pseudo_observable_samples = np.zeros(
                        (parameter_draws.shape[0], *output.pseudo_observables.shape),
                        dtype=np.float64,
                    )
                if data_split is not None and history_states is not None:
                    history_state_samples = np.zeros(
                        (parameter_draws.shape[0], *history_states.shape),
                        dtype=np.float64,
                    )
                    if history_observables is None:
                        msg = "History observable samples were not initialized."
                        raise RuntimeError(msg)
                    history_observable_samples = np.zeros(
                        (parameter_draws.shape[0], *history_observables.shape),
                        dtype=np.float64,
                    )
                    log_likelihood_samples = np.zeros(parameter_draws.shape[0], dtype=np.float64)
                    if include_pseudo:
                        if history_pseudo_observables is None:
                            msg = "History pseudo-observable samples were not initialized."
                            raise RuntimeError(msg)
                        history_pseudo_observable_samples = np.zeros(
                            (parameter_draws.shape[0], *history_pseudo_observables.shape),
                            dtype=np.float64,
                        )
            state_samples[draw_index] = output.states
            if observable_samples is None:
                msg = "Observable samples were not initialized."
                raise RuntimeError(msg)
            observable_samples[draw_index] = output.observables
            if pseudo_observable_samples is not None:
                if output.pseudo_observables is None:
                    msg = "Posterior forecast did not produce pseudo-observables."
                    raise RuntimeError(msg)
                pseudo_observable_samples[draw_index] = output.pseudo_observables
            if history_state_samples is not None:
                if (
                    history_observable_samples is None
                    or log_likelihood_samples is None
                    or history_states is None
                    or history_observables is None
                    or log_likelihood is None
                ):
                    msg = "History samples were not initialized."
                    raise RuntimeError(msg)
                history_state_samples[draw_index] = history_states
                history_observable_samples[draw_index] = history_observables
                log_likelihood_samples[draw_index] = log_likelihood
                if history_pseudo_observable_samples is not None:
                    if history_pseudo_observables is None:
                        msg = "History pseudo-observable samples were not produced."
                        raise RuntimeError(msg)
                    history_pseudo_observable_samples[draw_index] = history_pseudo_observables
    finally:
        _restore_model_parameters(model, original_parameters, original_steady_state)

    if state_samples is None or observable_samples is None:
        msg = "Posterior forecast did not produce samples."
        raise RuntimeError(msg)
    return ForecastOutput(
        states=np.mean(state_samples, axis=0),
        observables=np.mean(observable_samples, axis=0),
        pseudo_observables=(
            None
            if pseudo_observable_samples is None
            else np.mean(pseudo_observable_samples, axis=0)
        ),
        state_samples=state_samples,
        observable_samples=observable_samples,
        pseudo_observable_samples=pseudo_observable_samples,
        history_states=(
            None if history_state_samples is None else np.mean(history_state_samples, axis=0)
        ),
        history_observables=(
            None
            if history_observable_samples is None
            else np.mean(history_observable_samples, axis=0)
        ),
        history_pseudo_observables=(
            None
            if history_pseudo_observable_samples is None
            else np.mean(history_pseudo_observable_samples, axis=0)
        ),
        history_state_samples=history_state_samples,
        history_observable_samples=history_observable_samples,
        history_pseudo_observable_samples=history_pseudo_observable_samples,
        log_likelihood=(
            None if log_likelihood_samples is None else float(np.mean(log_likelihood_samples))
        ),
    )


def meansbands_from_forecast(
    forecast: ForecastOutput,
    *,
    source: str = "observables",
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> MeansBands:
    samples = _forecast_sample_array(forecast, source)
    if samples is not None:
        return meansbands_from_samples(
            samples,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
    values = _forecast_array(forecast, source)
    return MeansBands(mean=values.copy(), lower=values.copy(), upper=values.copy())


def meansbands_from_samples(
    samples: np.ndarray,
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> MeansBands:
    if not 0.0 <= lower_quantile <= upper_quantile <= 1.0:
        msg = "Quantiles must satisfy 0 <= lower_quantile <= upper_quantile <= 1."
        raise ValueError(msg)
    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.ndim != 3:
        msg = "Samples must have shape (draws, horizon, variables)."
        raise ValueError(msg)
    if sample_array.shape[0] == 0:
        msg = "Samples must include at least one draw."
        raise ValueError(msg)
    return MeansBands(
        mean=np.nanmean(sample_array, axis=0),
        lower=np.nanquantile(sample_array, lower_quantile, axis=0),
        upper=np.nanquantile(sample_array, upper_quantile, axis=0),
    )


def reverse_transform_forecast(model: DSGEModel, forecast: ForecastOutput) -> ForecastOutput:
    return ForecastOutput(
        states=np.asarray(forecast.states, dtype=np.float64).copy(),
        observables=reverse_transform_observables(model, forecast.observables),
        pseudo_observables=(
            None
            if forecast.pseudo_observables is None
            else reverse_transform_pseudo_observables(model, forecast.pseudo_observables)
        ),
        conditional_shocks=(
            None
            if forecast.conditional_shocks is None
            else np.asarray(forecast.conditional_shocks, dtype=np.float64).copy()
        ),
        conditional_states=(
            None
            if forecast.conditional_states is None
            else np.asarray(forecast.conditional_states, dtype=np.float64).copy()
        ),
        conditional_observables=(
            None
            if forecast.conditional_observables is None
            else reverse_transform_observables(model, forecast.conditional_observables)
        ),
        state_samples=(
            None
            if forecast.state_samples is None
            else np.asarray(forecast.state_samples, dtype=np.float64).copy()
        ),
        observable_samples=(
            None
            if forecast.observable_samples is None
            else reverse_transform_observables(
                model,
                _flatten_forecast_samples(forecast.observable_samples),
            ).reshape(forecast.observable_samples.shape)
        ),
        pseudo_observable_samples=(
            None
            if forecast.pseudo_observable_samples is None
            else reverse_transform_pseudo_observables(
                model,
                _flatten_forecast_samples(forecast.pseudo_observable_samples),
            ).reshape(forecast.pseudo_observable_samples.shape)
        ),
        history_states=(
            None
            if forecast.history_states is None
            else np.asarray(forecast.history_states, dtype=np.float64).copy()
        ),
        history_observables=(
            None
            if forecast.history_observables is None
            else reverse_transform_observables(model, forecast.history_observables)
        ),
        history_pseudo_observables=(
            None
            if forecast.history_pseudo_observables is None
            else reverse_transform_pseudo_observables(model, forecast.history_pseudo_observables)
        ),
        history_state_samples=(
            None
            if forecast.history_state_samples is None
            else np.asarray(forecast.history_state_samples, dtype=np.float64).copy()
        ),
        history_observable_samples=(
            None
            if forecast.history_observable_samples is None
            else reverse_transform_observables(
                model,
                _flatten_forecast_samples(forecast.history_observable_samples),
            ).reshape(forecast.history_observable_samples.shape)
        ),
        history_pseudo_observable_samples=(
            None
            if forecast.history_pseudo_observable_samples is None
            else reverse_transform_pseudo_observables(
                model,
                _flatten_forecast_samples(forecast.history_pseudo_observable_samples),
            ).reshape(forecast.history_pseudo_observable_samples.shape)
        ),
        log_likelihood=forecast.log_likelihood,
    )


def reverse_transform_meansbands(
    model: DSGEModel,
    meansbands: MeansBands,
    *,
    source: str = "observables",
) -> MeansBands:
    if source in _pseudo_observable_sources():
        transform = reverse_transform_pseudo_observables
    elif source in _observable_sources():
        transform = reverse_transform_observables
    else:
        msg = f"Cannot reverse-transform means/bands source: {source}"
        raise ValueError(msg)
    return MeansBands(
        mean=transform(model, meansbands.mean),
        lower=transform(model, meansbands.lower),
        upper=transform(model, meansbands.upper),
    )


def forecast_one(
    model: DSGEModel,
    input_type: str,
    cond_type: str,
    output_vars: list[str],
    *,
    check_empty_columns: bool = True,
    horizon: int | None = None,
    initial_state: np.ndarray | None = None,
    shocks: np.ndarray | None = None,
    shock_samples: np.ndarray | None = None,
    data: Any | None = None,
    history_method: str = "filtered",
    conditional_periods: int | None = None,
    draws: int = 0,
    seed: int | None = None,
    sampler: MetropolisHastingsResult | None = None,
) -> ForecastOutput:
    if model.spec == "m1002" and model.subspec == "ss10":
        if input_type not in {"mode", "full"}:
            msg = (
                "Only input_type='mode' or input_type='full' are ported for "
                "Model1002 ss10 forecasts."
            )
            raise NotPortedError(msg)
        normalized_cond_type = _normalize_cond_type(cond_type)
        if input_type == "full" and draws <= 0 and sampler is None and shock_samples is None:
            msg = "input_type='full' requires draws > 0, shock_samples, or a sampler result."
            raise ValueError(msg)
        if sampler is not None and input_type != "full":
            msg = "Sampler-draw forecasts require input_type='full'."
            raise ValueError(msg)
        if sampler is not None and draws > 0:
            msg = "Use either structural shock draws or sampler draws, not both."
            raise ValueError(msg)
        if sampler is not None and shock_samples is not None:
            msg = "Use either structural shock samples or sampler draws, not both."
            raise ValueError(msg)
        selected_history_method = _normalize_history_method(history_method)
        forecast_horizon = int(
            horizon if horizon is not None else model.get_setting("forecast_horizon", 40)
        )
        include_pseudo = _any_output_requested(
            output_vars,
            _forecast_pseudo_output_names() | _history_pseudo_output_names(),
        )
        if sampler is not None:
            if shocks is not None or shock_samples is not None:
                msg = "Explicit shocks are not supported with sampler-draw forecasts."
                raise ValueError(msg)
            forecast = forecast_parameter_draws(
                model,
                sampler,
                horizon=forecast_horizon,
                initial_state=initial_state,
                include_pseudo=include_pseudo,
                data=(
                    data
                    if data is not None
                    else (
                        load_data(model, check_empty_columns=check_empty_columns)
                        if normalized_cond_type != "none" or _requires_history(output_vars)
                        else None
                    )
                ),
                check_empty_columns=check_empty_columns,
                cond_type=normalized_cond_type,
                conditional_periods=conditional_periods,
                history_method=selected_history_method,
            )
            return ForecastOutput(
                states=forecast.states,
                observables=forecast.observables,
                pseudo_observables=forecast.pseudo_observables,
                state_samples=forecast.state_samples,
                observable_samples=forecast.observable_samples,
                pseudo_observable_samples=forecast.pseudo_observable_samples,
                history_states=(
                    forecast.history_states
                    if _output_requested(output_vars, "histstates")
                    else None
                ),
                history_observables=(
                    forecast.history_observables
                    if _output_requested(output_vars, "histobs")
                    else None
                ),
                history_pseudo_observables=(
                    forecast.history_pseudo_observables
                    if _any_output_requested(output_vars, _history_pseudo_output_names())
                    else None
                ),
                history_state_samples=(
                    forecast.history_state_samples
                    if _output_requested(output_vars, "histstates")
                    else None
                ),
                history_observable_samples=(
                    forecast.history_observable_samples
                    if _output_requested(output_vars, "histobs")
                    else None
                ),
                history_pseudo_observable_samples=(
                    forecast.history_pseudo_observable_samples
                    if _any_output_requested(output_vars, _history_pseudo_output_names())
                    else None
                ),
                log_likelihood=forecast.log_likelihood,
            )

        system = compute_system(model)
        start = _initial_forecast_state(system, initial_state)
        history_states = None
        history_observables = None
        history_pseudo_observables = None
        conditional_shocks = None
        conditional_states = None
        conditional_observables = None
        log_likelihood = None
        if data is not None or _requires_history(output_vars):
            data_split = _forecast_data_split(
                model,
                data,
                check_empty_columns=check_empty_columns,
                cond_type=normalized_cond_type,
                conditional_periods=conditional_periods,
            )
            if data_split.conditional_periods > forecast_horizon:
                msg = (
                    "Conditional data periods cannot exceed the requested forecast "
                    f"horizon: {data_split.conditional_periods} > {forecast_horizon}."
                )
                raise ValueError(msg)
            forecast_horizon -= data_split.conditional_periods
            filter_observations = (
                data_split.observations[: data_split.history_periods]
                if normalized_cond_type == "full" and data_split.conditional_periods > 0
                else data_split.observations
            )
            if filter_observations.shape[0] > 0:
                filtered = kalman_log_likelihood(
                    system,
                    filter_observations,
                    process_covariances=model_process_covariances(
                        model,
                        system,
                        filter_observations.shape[0],
                        start_date=data_split.start_date,
                    ),
                    backend=get_backend(model.runtime),
                )
                log_likelihood = filtered.log_likelihood
                filtered_history_states = _states_for_history_method(
                    system,
                    filtered,
                    selected_history_method,
                )
                historical_states = filtered_history_states[: data_split.history_periods]
                if initial_state is None and filtered_history_states.shape[0] > 0:
                    start = filtered_history_states[-1].copy()
            else:
                historical_states = np.zeros((0, system.transition.TTT.shape[0]))
            if _output_requested(output_vars, "histstates"):
                history_states = historical_states
            if _output_requested(output_vars, "histobs"):
                history_observables = data_split.observations[: data_split.history_periods]
            if _any_output_requested(output_vars, _history_pseudo_output_names()):
                history_pseudo_observables = pseudo_observables_from_states(
                    system,
                    historical_states,
                )
            if normalized_cond_type == "full" and data_split.conditional_periods > 0:
                conditional_targets = data_split.observations[data_split.history_periods :]
                conditioning = solve_shocks_for_observable_targets(
                    system,
                    start,
                    conditional_targets,
                )
                conditional_shocks = conditioning.shocks
                conditional_states = conditioning.states
                conditional_observables = conditioning.observables
                if conditional_states.shape[0] > 0:
                    start = conditional_states[-1].copy()

        if input_type == "full":
            if shocks is not None:
                msg = "Explicit shocks are only supported for input_type='mode'."
                raise ValueError(msg)
            forecast = forecast_linear_system_samples(
                system,
                start,
                horizon=forecast_horizon,
                draws=draws,
                seed=seed,
                shock_samples=shock_samples,
                include_pseudo=include_pseudo,
                backend=get_backend(model.runtime),
            )
        else:
            forecast = forecast_linear_system(
                system,
                start,
                horizon=forecast_horizon,
                shocks=shocks,
                include_pseudo=include_pseudo,
                backend=get_backend(model.runtime),
            )
        return ForecastOutput(
            states=forecast.states,
            observables=forecast.observables,
            pseudo_observables=forecast.pseudo_observables,
            conditional_shocks=conditional_shocks,
            conditional_states=conditional_states,
            conditional_observables=conditional_observables,
            state_samples=forecast.state_samples,
            observable_samples=forecast.observable_samples,
            pseudo_observable_samples=forecast.pseudo_observable_samples,
            history_states=history_states,
            history_observables=history_observables,
            history_pseudo_observables=history_pseudo_observables,
            history_state_samples=_repeat_history_samples(
                history_states,
                forecast.state_samples,
            ),
            history_observable_samples=_repeat_history_samples(
                history_observables,
                forecast.observable_samples,
            ),
            history_pseudo_observable_samples=_repeat_history_samples(
                history_pseudo_observables,
                forecast.pseudo_observable_samples,
            ),
            log_likelihood=log_likelihood,
        )

    msg = (
        f"Forecast driver for {model.spec} {model.subspec} is not ported yet. "
        "The linear forecast kernel exists, but model-specific filtering, forecast inputs, "
        "and output serialization still need translation."
    )
    raise NotPortedError(msg)


def compute_meansbands(
    model: DSGEModel,
    input_type: str,
    cond_type: str,
    output_vars: list[str],
    *,
    check_empty_columns: bool = True,
    horizon: int | None = None,
    initial_state: np.ndarray | None = None,
    shocks: np.ndarray | None = None,
    shock_samples: np.ndarray | None = None,
    source: str = "observables",
    data: Any | None = None,
    history_method: str = "filtered",
    conditional_periods: int | None = None,
    draws: int = 0,
    seed: int | None = None,
    sampler: MetropolisHastingsResult | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> MeansBands:
    if model.spec == "m1002" and model.subspec == "ss10":
        if input_type not in {"mode", "full"}:
            msg = (
                "Only input_type='mode' or input_type='full' are ported for "
                "Model1002 ss10 means/bands."
            )
            raise NotPortedError(msg)
        forecast = forecast_one(
            model,
            input_type=input_type,
            cond_type=cond_type,
            output_vars=output_vars,
            horizon=horizon,
            initial_state=initial_state,
            shocks=shocks,
            shock_samples=shock_samples,
            data=data,
            check_empty_columns=check_empty_columns,
            history_method=history_method,
            conditional_periods=conditional_periods,
            draws=draws,
            seed=seed,
            sampler=sampler,
        )
        return meansbands_from_forecast(
            forecast,
            source=source,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    msg = (
        f"Means/bands for {model.spec} {model.subspec} are not ported yet. "
        "Required translations: full-distribution forecast loading, reverse transforms, "
        "and DSGE.jl MeansBands parity fixtures."
    )
    raise NotPortedError(msg)


def _repeat_history_samples(
    history: np.ndarray | None,
    forecast_samples: np.ndarray | None,
) -> np.ndarray | None:
    if history is None or forecast_samples is None:
        return None
    return np.repeat(
        np.asarray(history, dtype=np.float64)[np.newaxis, :, :],
        forecast_samples.shape[0],
        axis=0,
    )


def _forecast_array(forecast: ForecastOutput, source: str) -> np.ndarray:
    if source in _observable_sources():
        return np.asarray(forecast.observables, dtype=np.float64)
    if source in {"states", "forecaststates"}:
        return np.asarray(forecast.states, dtype=np.float64)
    if source in _pseudo_observable_sources():
        if forecast.pseudo_observables is None:
            msg = "Forecast output does not include pseudo-observables."
            raise ValueError(msg)
        return np.asarray(forecast.pseudo_observables, dtype=np.float64)
    if source in {"history_observables", "histobs"}:
        if forecast.history_observables is None:
            msg = "Forecast output does not include historical observables."
            raise ValueError(msg)
        return np.asarray(forecast.history_observables, dtype=np.float64)
    if source in _history_pseudo_sources():
        if forecast.history_pseudo_observables is None:
            msg = "Forecast output does not include historical pseudo-observables."
            raise ValueError(msg)
        return np.asarray(forecast.history_pseudo_observables, dtype=np.float64)
    if source in {"history_states", "histstates"}:
        if forecast.history_states is None:
            msg = "Forecast output does not include historical states."
            raise ValueError(msg)
        return np.asarray(forecast.history_states, dtype=np.float64)
    msg = f"Unsupported forecast source: {source}"
    raise ValueError(msg)


def _forecast_sample_array(forecast: ForecastOutput, source: str) -> np.ndarray | None:
    if source in _observable_sources():
        return (
            None
            if forecast.observable_samples is None
            else np.asarray(forecast.observable_samples, dtype=np.float64)
        )
    if source in {"states", "forecaststates"}:
        return (
            None
            if forecast.state_samples is None
            else np.asarray(forecast.state_samples, dtype=np.float64)
        )
    if source in _pseudo_observable_sources():
        return (
            None
            if forecast.pseudo_observable_samples is None
            else np.asarray(forecast.pseudo_observable_samples, dtype=np.float64)
        )
    if source in {"history_observables", "histobs"}:
        return (
            None
            if forecast.history_observable_samples is None
            else np.asarray(forecast.history_observable_samples, dtype=np.float64)
        )
    if source in {"history_states", "histstates"}:
        return (
            None
            if forecast.history_state_samples is None
            else np.asarray(forecast.history_state_samples, dtype=np.float64)
        )
    if source in _history_pseudo_sources():
        return (
            None
            if forecast.history_pseudo_observable_samples is None
            else np.asarray(forecast.history_pseudo_observable_samples, dtype=np.float64)
        )
    return None


def _set_model_parameter_values(
    model: DSGEModel,
    original_parameters: dict[str, Parameter],
    parameter_names: tuple[str, ...],
    parameter_values: np.ndarray,
) -> None:
    if parameter_values.shape != (len(parameter_names),):
        msg = f"Parameter values must have shape {(len(parameter_names),)}."
        raise ValueError(msg)
    missing = [name for name in parameter_names if name not in original_parameters]
    if missing:
        msg = "Sampler references unknown parameter(s): " + ", ".join(missing)
        raise KeyError(msg)
    for name, value in zip(parameter_names, parameter_values, strict=True):
        model.parameters[name] = replace(original_parameters[name], value=float(value))


def _restore_model_parameters(
    model: DSGEModel,
    original_parameters: dict[str, Parameter],
    original_steady_state: dict[str, float],
) -> None:
    for name, parameter in original_parameters.items():
        model.parameters[name] = parameter
    model.steady_state.clear()
    model.steady_state.update(original_steady_state)


def _flatten_forecast_samples(samples: np.ndarray) -> np.ndarray:
    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.ndim != 3:
        msg = "Forecast samples must have shape (draws, horizon, variables)."
        raise ValueError(msg)
    return sample_array.reshape(
        sample_array.shape[0] * sample_array.shape[1], sample_array.shape[2]
    )


def _initial_forecast_state(system: System, initial_state: np.ndarray | None) -> np.ndarray:
    if initial_state is None:
        return np.zeros(system.transition.TTT.shape[0], dtype=np.float64)
    return np.asarray(initial_state, dtype=np.float64)


def _normalize_shock_matrix(
    shocks: np.ndarray | None,
    *,
    horizon: int,
    n_shocks: int,
) -> np.ndarray:
    if shocks is None:
        return np.zeros((horizon, n_shocks), dtype=np.float64)
    shock_array = np.asarray(shocks, dtype=np.float64)
    if shock_array.ndim != 2:
        msg = "Shocks must be a 2D matrix."
        raise ValueError(msg)
    if shock_array.shape[1] == n_shocks:
        periods_by_shocks = shock_array
    elif shock_array.shape[0] == n_shocks:
        periods_by_shocks = shock_array.T
    else:
        msg = (
            f"Shocks must have one dimension equal to the number of structural shocks ({n_shocks})."
        )
        raise ValueError(msg)
    normalized = np.zeros((horizon, n_shocks), dtype=np.float64)
    copied_periods = min(horizon, periods_by_shocks.shape[0])
    normalized[:copied_periods, :] = periods_by_shocks[:copied_periods, :]
    return normalized


def _normalize_shock_samples(
    shock_samples: np.ndarray,
    *,
    horizon: int,
    n_shocks: int,
) -> np.ndarray:
    sample_array = np.asarray(shock_samples, dtype=np.float64)
    if sample_array.ndim != 3:
        msg = "Shock samples must have shape (draws, periods, shocks)."
        raise ValueError(msg)
    if sample_array.shape[2] == n_shocks:
        draws_by_periods_by_shocks = sample_array
    elif sample_array.shape[1] == n_shocks:
        draws_by_periods_by_shocks = np.transpose(sample_array, (0, 2, 1))
    else:
        msg = (
            "Shock samples must have one non-draw dimension equal to the number "
            f"of structural shocks ({n_shocks})."
        )
        raise ValueError(msg)
    normalized = np.zeros(
        (draws_by_periods_by_shocks.shape[0], horizon, n_shocks),
        dtype=np.float64,
    )
    copied_periods = min(horizon, draws_by_periods_by_shocks.shape[1])
    normalized[:, :copied_periods, :] = draws_by_periods_by_shocks[:, :copied_periods, :]
    return normalized


def _observable_shock_design(system: System, *, horizon: int) -> np.ndarray:
    if horizon < 0:
        msg = "Forecast horizon must be nonnegative."
        raise ValueError(msg)
    transition = system.transition
    measurement = system.measurement
    n_observables = measurement.ZZ.shape[0]
    n_shocks = transition.RRR.shape[1]
    design = np.zeros((horizon, n_observables, horizon, n_shocks), dtype=np.float64)
    for shock_period in range(horizon):
        state_effects = transition.RRR.copy()
        for target_period in range(shock_period, horizon):
            design[target_period, :, shock_period, :] = measurement.ZZ @ state_effects
            state_effects = transition.TTT @ state_effects
    return design


def _states_for_history_method(
    system: System,
    filtered: KalmanResult,
    history_method: str,
) -> np.ndarray:
    if history_method == "filtered":
        return filtered.filtered_states
    smoothed = smooth_kalman_result(system, filtered)
    if smoothed.smoothed_states is None:
        msg = "Kalman smoother did not return smoothed states."
        raise RuntimeError(msg)
    return smoothed.smoothed_states


def _normalize_history_method(value: str) -> str:
    history_method = str(value).lower()
    if history_method not in {"filtered", "smoothed"}:
        msg = "History method must be 'filtered' or 'smoothed'."
        raise ValueError(msg)
    return history_method


def _normalize_cond_type(value: str) -> str:
    cond_type = str(value).lower()
    if cond_type not in {"none", "semi", "full"}:
        msg = "Conditioning type must be 'none', 'semi', or 'full'."
        raise NotPortedError(msg)
    return cond_type


def _forecast_data_split(
    model: DSGEModel,
    data: Any | None,
    *,
    check_empty_columns: bool,
    cond_type: str,
    conditional_periods: int | None,
) -> _ForecastDataSplit:
    if data is None:
        data = load_data(model, check_empty_columns=check_empty_columns)
    if hasattr(data, "columns"):
        if cond_type == "none":
            observations = df_to_matrix(model, data, in_sample=True)
            return _ForecastDataSplit(
                observations=observations,
                history_periods=observations.shape[0],
                conditional_periods=0,
                start_date=_first_sample_date(model, data, in_sample=True),
            )
        history = df_to_matrix(model, data, in_sample=True)
        conditional = df_to_matrix(model, data, in_sample=False)
        if conditional.shape[0] == 0:
            msg = "Conditional forecasts require dated rows at or after date_forecast_start."
            raise ValueError(msg)
        observations = np.vstack([history, conditional])
        start_date = _first_sample_date(model, data, in_sample=True)
        if start_date is None:
            start_date = _first_sample_date(model, data, in_sample=False)
        return _ForecastDataSplit(
            observations=observations,
            history_periods=history.shape[0],
            conditional_periods=conditional.shape[0],
            start_date=start_date,
        )
    observations = np.asarray(data, dtype=np.float64)
    if observations.ndim != 2:
        msg = "Forecast data must be a DataFrame or a 2D observable matrix."
        raise ValueError(msg)
    if cond_type == "none":
        return _ForecastDataSplit(
            observations=observations,
            history_periods=observations.shape[0],
            conditional_periods=0,
        )
    if conditional_periods is None:
        msg = "Undated matrix data requires conditional_periods for conditional forecasts."
        raise ValueError(msg)
    if conditional_periods < 0 or conditional_periods > observations.shape[0]:
        msg = "conditional_periods must be between zero and the number of data rows."
        raise ValueError(msg)
    return _ForecastDataSplit(
        observations=observations,
        history_periods=observations.shape[0] - conditional_periods,
        conditional_periods=conditional_periods,
    )


def _first_sample_date(model: DSGEModel, data: Any, *, in_sample: bool) -> str | None:
    labels = date_labels_for_sample(model, data, in_sample=in_sample)
    if not labels:
        return None
    return labels[0]


def _requires_history(output_vars: list[str]) -> bool:
    return any(
        _output_requested(output_vars, name)
        for name in ("histobs", "histstates", *_history_pseudo_output_names())
    )


def _output_requested(output_vars: list[str], name: str) -> bool:
    requested = {str(item) for item in output_vars}
    return name in requested


def _any_output_requested(output_vars: list[str], names: set[str]) -> bool:
    requested = {str(item) for item in output_vars}
    return bool(requested.intersection(names))


def _observable_sources() -> set[str]:
    return {"observables", "forecastobs"}


def _pseudo_observable_sources() -> set[str]:
    return {"pseudo_observables", "forecastpseudo", "forecastpseudoobs", "pseudoobs"}


def _history_pseudo_sources() -> set[str]:
    return {"history_pseudo_observables", "histpseudo", "histpseudoobs"}


def _forecast_pseudo_output_names() -> set[str]:
    return _pseudo_observable_sources()


def _history_pseudo_output_names() -> set[str]:
    return _history_pseudo_sources()
