"""Conditional ensemble and Gaussian shock-allocation regression contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nydsge.estimate import MetropolisHastingsResult
from nydsge.forecast import forecast_one, historical_decomposition
from nydsge.models import Model1002
from nydsge.runtime import RuntimeConfig
from nydsge.solve import Measurement, System, Transition, compute_system


def _repeated_sampler(model):
    return MetropolisHastingsResult(
        parameter_names=("alpha",),
        estimation_draws=np.zeros((2, 1)),
        parameter_draws=np.full((2, 1), model.parameters["alpha"].value),
        log_posterior=np.zeros(2),
        accepted=np.ones(2, dtype=bool),
        acceptance_rate=1.0,
        proposal_covariance=np.eye(1),
        seed=3,
        burnin=0,
    )


@pytest.mark.parametrize("conditioning", ["none", "semi", "full"])
def test_identical_parameter_ensemble_matches_point_conditioning(conditioning):
    model = Model1002("ss10")
    system = compute_system(model)
    assert isinstance(system, System)
    data = np.tile(system["DD"], (2, 1))
    if conditioning != "none":
        data[1] = np.nan
        data[1, model.observables["obs_corepce"] - 1] = 0.9
    kwargs: dict[str, Any] = dict(
        data=data, horizon=3, conditional_periods=0 if conditioning == "none" else 1
    )
    point = forecast_one(model, "mode", conditioning, ["forecastobs", "histstates"], **kwargs)
    ensemble = forecast_one(
        model,
        "full",
        conditioning,
        ["forecastobs", "histstates"],
        sampler=_repeated_sampler(model),
        **kwargs,
    )
    assert ensemble.uncertainty_sources == ("parameter_draws",)
    assert "convergence is not certified" in (ensemble.parameter_draw_provenance or "")
    np.testing.assert_array_equal(ensemble.observables, point.observables)
    np.testing.assert_array_equal(ensemble.states, point.states)
    np.testing.assert_array_equal(ensemble.history_states, point.history_states)
    if conditioning != "none":
        assert ensemble.conditional_states is not None
        np.testing.assert_array_equal(ensemble.conditional_states, point.conditional_states)
        np.testing.assert_array_equal(
            ensemble.conditional_observables, point.conditional_observables
        )
    if conditioning == "full":
        np.testing.assert_array_equal(ensemble.conditional_shocks, point.conditional_shocks)


def test_dated_ss10_forecasts_reject_stale_or_missing_origin_rows():
    model = Model1002(
        "ss10",
        settings={
            "date_presample_start": "2018Q1",
            "date_mainsample_start": "2018Q1",
            "date_forecast_start": "2018Q4",
        },
    )
    columns = list(model.observables)
    stale = pd.DataFrame(0.0, index=pd.period_range("2018Q1", "2018Q2", freq="Q"), columns=columns)
    with pytest.raises(ValueError, match="quarter before forecast"):
        forecast_one(model, "mode", "none", ["forecastobs"], data=stale, horizon=2)
    missing_origin = pd.DataFrame(
        0.0, index=pd.PeriodIndex(["2018Q3", "2019Q1"], freq="Q"), columns=columns
    )
    with pytest.raises(
        ValueError, match="begin at forecast start|contiguous|quarter before forecast"
    ):
        forecast_one(model, "mode", "full", ["forecastobs"], data=missing_origin, horizon=2)


@pytest.mark.parametrize("variances", [[1.0, 100.0], [0.0, 4.0]])
def test_rank_deficient_hd_uses_covariance_weights_and_zero_variance(variances, monkeypatch):
    import nydsge.forecast as module

    model: Any = SimpleNamespace(
        spec="test",
        runtime=RuntimeConfig(),
        get_setting=lambda key, default=None: {"date_mainsample_start": "2000Q1"}.get(key, default),
    )
    system = System(
        Transition(np.array([[0.5]]), np.ones((1, 2)), np.zeros(1)),
        Measurement(np.ones((1, 1)), np.zeros(1), np.diag(variances), np.zeros((1, 1))),
    )
    monkeypatch.setattr(module, "compute_system", lambda _: system)
    output = historical_decomposition(model, data=np.array([[1.0], [2.0], [1.0]]))
    expected = np.asarray(variances) / np.sum(variances)
    np.testing.assert_allclose(output.smoothed_shocks[1], 1.5 * expected, atol=1e-14)
    assert output.reconciliation_max_abs_error < 1e-12
    if variances[0] == 0:
        np.testing.assert_array_equal(output.smoothed_shocks[:, 0], 0)


def test_full_column_rank_hd_preserves_unique_recovery(monkeypatch):
    import nydsge.forecast as module

    model: Any = SimpleNamespace(
        spec="test", runtime=RuntimeConfig(), get_setting=lambda key, default=None: default
    )
    system = System(
        Transition(np.array([[0.5]]), np.array([[2.0]]), np.zeros(1)),
        Measurement(np.ones((1, 1)), np.zeros(1), np.array([[100.0]]), np.zeros((1, 1))),
    )
    monkeypatch.setattr(module, "compute_system", lambda _: system)
    output = historical_decomposition(model, data=np.array([[1.0], [2.0], [1.0]]))
    np.testing.assert_allclose(output.smoothed_shocks[:, 0], [0.0, 0.75, 0.0], atol=1e-14)
    assert output.reconciliation_max_abs_error < 1e-12


def test_historical_decomposition_accepts_partial_dated_window_before_origin():
    model = Model1002(
        "ss10",
        settings={
            "date_presample_start": "2018Q1",
            "date_mainsample_start": "2018Q1",
            "date_forecast_start": "2018Q4",
        },
    )
    system = compute_system(model)
    assert isinstance(system, System)
    frame = pd.DataFrame(np.tile(system["DD"], (2, 1)), columns=list(model.observables))
    frame.insert(0, "date", ["2018Q1", "2018Q2"])
    decomposition = historical_decomposition(model, data=frame)
    assert decomposition.smoothed_states.shape[0] == 2
    assert np.isfinite(decomposition.smoothed_states).all()
    with pytest.raises(ValueError, match="quarter before forecast"):
        forecast_one(model, "mode", "none", ["forecastobs"], data=frame, horizon=1)


def test_ss10_future_innovation_uncertainty_survives_kernel_wrapper_and_bands():
    from nydsge.forecast import compute_meansbands, meansbands_from_forecast

    model = Model1002("ss10")
    output = forecast_one(model, "full", "none", ["forecastobs"], horizon=3, draws=4, seed=1)
    assert output.observable_samples is not None
    assert np.max(np.ptp(output.observable_samples, axis=0)) > 0
    assert output.uncertainty_sources == ("future_model_innovations",)
    assert output.parameter_draw_provenance is None
    bands = compute_meansbands(model, "full", "none", ["forecastobs"], horizon=3, draws=4, seed=1)
    assert bands.uncertainty_sources == output.uncertainty_sources
    assert bands.parameter_draw_provenance is None
    np.testing.assert_array_equal(bands.mean, output.observables)
    np.testing.assert_allclose(bands.lower, np.quantile(output.observable_samples, 0.05, axis=0))
    zero = forecast_one(
        model,
        "full",
        "none",
        ["forecastobs"],
        horizon=3,
        shock_samples=np.zeros((2, 3, len(model.indexes.exogenous_shocks))),
    )
    assert zero.uncertainty_sources == ("future_model_innovations",)
    assert zero.parameter_draw_provenance is None
    assert meansbands_from_forecast(zero).uncertainty_sources == zero.uncertainty_sources


@pytest.mark.parametrize("conditioning", ["full", "semi"])
@pytest.mark.parametrize("input_type", ["mode", "full"])
def test_ss10_conditional_forecast_requires_explicit_inputs_before_solving(
    conditioning, input_type, monkeypatch
):
    import nydsge.forecast as module

    model = Model1002("ss10")
    sampler = _repeated_sampler(model) if input_type == "full" else None

    def unexpected_solve(_):
        raise AssertionError("Missing conditioning inputs must fail before a solve.")

    monkeypatch.setattr(module, "compute_system", unexpected_solve)
    with pytest.raises(ValueError, match="require explicit"):
        forecast_one(model, input_type, conditioning, ["forecastobs"], horizon=2, sampler=sampler)
    if sampler is not None:
        with pytest.raises(ValueError, match="require explicit"):
            module.forecast_parameter_draws(model, sampler, horizon=2, cond_type=conditioning)
