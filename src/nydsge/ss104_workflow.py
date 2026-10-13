"""Fixed-parameter ss104 history and conditional mean forecast orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nydsge.forecast import ForecastOutput, forecast_linear_system
from nydsge.kalman import (
    kalman_log_likelihood,
    model_process_covariances,
    observables_from_states,
    pseudo_observables_from_states,
    smooth_kalman_result,
)
from nydsge.models.ss104 import initialize_gap_observations
from nydsge.regimes import RegimeSwitchingSystem, regime_indices_from_dates
from nydsge.solve import compute_system


def prepare_ss104_observations(model, data):
    """Apply the same dated gap initialization and historical ZLB masks for fitting/forecasting."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("ss104 history requires a dated pandas DataFrame.")
    frame = data.copy()
    frame.index = pd.PeriodIndex(frame.pop("date") if "date" in frame else frame.index, freq="Q")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("History quarters must be unique and ordered.")
    if len(frame) and not frame.index.equals(
        pd.period_range(frame.index[0], frame.index[-1], freq="Q")
    ):
        raise ValueError("History must have a contiguous quarterly grid; use NaN for missing data.")
    frame = initialize_gap_observations(model, frame)
    zlb = (frame.index >= pd.Period("2020Q4", freq="Q")) & (
        frame.index <= pd.Period("2021Q4", freq="Q")
    )
    rates = ["obs_nominalrate", *(f"obs_nominalrate{i}" for i in range(1, 7))]
    frame.loc[zlb, [name for name in rates if name in frame]] = np.nan
    missing = set(model.observables) - set(frame)
    if missing:
        raise ValueError(f"Missing transformed observable columns: {sorted(missing)}")
    return frame.loc[:, list(model.observables)]


def forecast_ss104(
    model,
    *,
    data=None,
    horizon=None,
    cond_type="none",
    initial_state=None,
    shocks=None,
    history_method="smoothed",
    system=None,
    conditional_periods=None,
):
    """Filter/smooth supplied transformed observations and forecast along the dated regimes.

    Missing observations are retained. Full/semi conditioning uses only the
    configured conditioning channels and expected FFR at/after forecast start.
    Conditional quarters update the Gaussian state distribution; their smoothed
    state estimates are returned separately, as in forecast_one's ss10 API.
    No parameters are fitted and no random draws are made.
    """
    if cond_type not in ("none", "full", "semi"):
        raise ValueError("ss104 cond_type must be none, full, or semi.")
    if history_method not in ("filtered", "smoothed"):
        raise ValueError("history_method must be filtered or smoothed.")
    horizon = int(model.get_setting("forecast_horizons", 40) if horizon is None else horizon)
    if horizon < 0:
        raise ValueError("Forecast horizon must be nonnegative.")
    system = compute_system(model) if system is None else system
    if not isinstance(system, RegimeSwitchingSystem):
        raise TypeError("ss104 forecasting requires a RegimeSwitchingSystem.")
    forecast_start = pd.Period(model.get_setting("date_forecast_start"), freq="Q")
    if not isinstance(forecast_start, pd.Period):
        raise ValueError("date_forecast_start must be a valid quarter.")
    n = system[1, "TTT"].shape[0]
    start = np.zeros(n) if initial_state is None else np.asarray(initial_state, dtype=float)
    history_states = history_obs = history_pseudo = None
    cond_states = cond_obs = None
    likelihood = None
    count_cond = 0
    if data is not None:
        frame = prepare_ss104_observations(model, data)
        if frame.empty or frame.index[-1] < forecast_start - 1:
            raise ValueError("History must reach the quarter preceding forecast start.")
        count_cond = int((frame.index >= forecast_start).sum())
        if conditional_periods is not None and count_cond != conditional_periods:
            raise ValueError("conditional_periods disagrees with dated data.")
        if count_cond > horizon:
            raise ValueError("Conditional data exceed the requested forecast horizon.")
        if count_cond and cond_type == "none":
            raise ValueError("Data at/after forecast start require full/semi conditioning.")
        if count_cond:
            names = model.get_setting(
                "cond_full_names" if cond_type == "full" else "cond_semi_names"
            )
            names = set(names) | {f"obs_exp_nominalrate{i}" for i in range(1, 7)}
            frame.loc[
                frame.index >= forecast_start, [k for k in model.observables if k not in names]
            ] = np.nan
        observations = frame.loc[:, list(model.observables)].to_numpy(dtype=float)
        indices = regime_indices_from_dates(model.get_setting("regime_dates"), frame.index)
        presample = int(
            (frame.index < pd.Period(model.get_setting("date_mainsample_start"), freq="Q")).sum()
        )
        filtered = kalman_log_likelihood(
            system,
            observations,
            regime_indices=indices,
            initial_state=initial_state,
            log_likelihood_start=presample,
            process_covariances=model_process_covariances(
                model, system, len(frame), start_date=frame.index[0]
            ),
        )
        smooth = smooth_kalman_result(system, filtered, data=observations)
        if smooth.smoothed_states is None:
            raise ValueError("The ss104 smoother did not return smoothed states.")
        states = (
            smooth.smoothed_states if history_method == "smoothed" else filtered.filtered_states
        )
        nhist = len(frame) - count_cond
        history_states, history_obs = states[:nhist], observations[:nhist]
        history_pseudo = pseudo_observables_from_states(
            system, history_states, regime_indices=indices[:nhist]
        )
        if count_cond:
            cond_states = smooth.smoothed_states[nhist:]
            cond_obs = observables_from_states(system, cond_states, regime_indices=indices[nhist:])
        start = smooth.final_smoothed_state if count_cond else states[-1].copy()
        likelihood = filtered.log_likelihood
    future_dates = pd.period_range(
        forecast_start + count_cond, periods=horizon - count_cond, freq="Q"
    )
    path = regime_indices_from_dates(model.get_setting("regime_dates"), future_dates)
    prediction = forecast_linear_system(
        system,
        start,
        horizon=horizon - count_cond,
        shocks=shocks,
        include_pseudo=True,
        regime_indices=path,
    )
    return ForecastOutput(
        prediction.states,
        prediction.observables,
        prediction.pseudo_observables,
        conditional_states=cond_states,
        conditional_observables=cond_obs,
        history_states=history_states,
        history_observables=history_obs,
        history_pseudo_observables=history_pseudo,
        log_likelihood=likelihood,
    )
