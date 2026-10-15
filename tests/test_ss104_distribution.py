from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from nydsge.forecast import compute_meansbands, forecast_one, meansbands_from_forecast
from nydsge.models import Model1002
from nydsge.solve import compute_system
from nydsge.ss104_distribution import _parameter_model
from nydsge.ss104_estimation import ss104_parameter_catalog
from nydsge.ss104_workflow import forecast_ss104


@pytest.fixture(scope="module")
def solved():
    model = Model1002("ss104")
    return model, compute_system(model)


def full(model, **kwargs):
    return forecast_one(model, "full", "none", ["forecastobs"], horizon=3, **kwargs)


def test_zero_innovations_reproduce_mode_and_fixed_bands(solved, monkeypatch):
    model, system = solved
    monkeypatch.setattr("nydsge.ss104_distribution.compute_system", lambda _: system)
    mode = forecast_ss104(model, system=system, horizon=3)
    output = full(model, shock_samples=np.zeros((2, 3, 48)))
    np.testing.assert_array_equal(output.observables, mode.observables)
    bands = meansbands_from_forecast(output)
    np.testing.assert_array_equal(bands.lower, bands.upper)
    assert bands.uncertainty_sources == ("future_model_innovations",)


def test_seeded_innovations_and_meansbands_dispatch(solved, monkeypatch):
    model, system = solved
    monkeypatch.setattr("nydsge.ss104_distribution.compute_system", lambda _: system)
    a, b = full(model, draws=5, seed=12), full(model, draws=5, seed=12)
    np.testing.assert_array_equal(a.observable_samples, b.observable_samples)
    assert np.max(np.ptp(a.observable_samples, axis=0)) > 0
    bands = compute_meansbands(model, "full", "none", ["forecastobs"], horizon=3, draws=5, seed=12)
    np.testing.assert_array_equal(bands.mean, a.observables)
    np.testing.assert_allclose(bands.lower, np.quantile(a.observable_samples, 0.05, axis=0))


def test_parameter_model_space_regime_isolation(solved):
    model, _ = solved
    catalog = ss104_parameter_catalog(model)
    name = next(k for k in catalog if "@" in k)
    value = catalog[name].value * 1.001
    changed = _parameter_model(model, [value], [name])
    assert ss104_parameter_catalog(changed)[name].value == pytest.approx(value)
    assert ss104_parameter_catalog(model)[name].value == catalog[name].value
    assert changed.get_setting("parameter_regimes") is not model.get_setting("parameter_regimes")


def test_supplied_parameters_produce_actual_native_forecasts(solved):
    model, _ = solved
    catalog = ss104_parameter_catalog(model)
    name = "gamma"
    original = catalog[name].value
    values = np.array([[original], [original * 1.001]])
    output = full(
        model,
        parameter_draws=values,
        parameter_names=[name],
        parameter_draw_provenance="synthetic boundary test, not posterior",
    )
    assert output.observable_samples.shape == (2, 3, 28)
    assert np.max(np.abs(np.diff(output.observable_samples, axis=0))) > 0
    assert model.parameters[name].value == original
    assert output.uncertainty_sources == ("supplied_parameters",)
    assert output.log_likelihood is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"parameter_draws": np.zeros((0, 1)), "parameter_names": ["gamma"]},
        {"draws": -1},
        {"shock_samples": np.zeros((1, 2, 48))},
        {"parameter_draws": [[0.3]], "parameter_names": ["gamma"]},
    ],
)
def test_invalid_ensembles_fail_closed(solved, monkeypatch, kwargs):
    model, system = solved
    monkeypatch.setattr("nydsge.ss104_distribution.compute_system", lambda _: system)
    with pytest.raises(ValueError):
        full(model, **kwargs)


def test_failed_draw_does_not_mutate_caller(solved, monkeypatch):
    model, _ = solved
    before = deepcopy(model.parameters)
    regimes = deepcopy(model.get_setting("parameter_regimes"))

    def fail(candidate):
        assert candidate is not model
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr("nydsge.ss104_distribution.compute_system", fail)
    with pytest.raises(RuntimeError, match="synthetic solver failure"):
        full(
            model,
            parameter_draws=[[model.parameters["gamma"].value]],
            parameter_names=["gamma"],
            parameter_draw_provenance="test",
        )
    assert model.parameters == before
    assert model.get_setting("parameter_regimes") == regimes


def test_conditional_window_excluded_from_future_shock_samples(solved, monkeypatch):
    model, system = solved
    monkeypatch.setattr("nydsge.ss104_distribution.compute_system", lambda _: system)
    start = pd.Period(model.get_setting("date_forecast_start"), freq="Q")
    assert isinstance(start, pd.Period)
    frame = pd.DataFrame(
        np.nan, index=pd.period_range(start - 2, start, freq="Q"), columns=list(model.observables)
    )
    shocks = np.zeros((2, 2, 48))
    shocks[0, :, model.indexes.exogenous_shocks["rm_ait_sh"] - 1] = 0.01
    output = forecast_one(
        model,
        "full",
        "semi",
        ["forecastobs"],
        horizon=3,
        data=frame,
        shock_samples=shocks,
    )
    reference = forecast_ss104(
        model, system=system, data=frame, horizon=3, cond_type="semi", shocks=shocks[0]
    )
    assert output.observable_samples is not None
    np.testing.assert_array_equal(output.observable_samples[0], reference.observables)
    assert output.observable_samples is not None
    assert output.conditional_states is not None
    assert output.history_state_samples is not None
    assert output.observable_samples.shape == (2, 2, 28)
    assert output.conditional_states.shape == (1, 110)
    assert output.history_state_samples.shape == (2, 2, 110)


def test_zero_horizon_and_zero_variance_shocks(solved, monkeypatch):
    from nydsge.ss104_distribution import _future_shocks

    model, system = solved
    monkeypatch.setattr("nydsge.ss104_distribution.compute_system", lambda _: system)
    output = forecast_one(model, "full", "none", ["forecastobs"], horizon=0, draws=2, seed=2)
    assert output.observable_samples is not None
    assert output.observable_samples.shape == (2, 0, 28)
    baseline = forecast_ss104(model, system=system, horizon=3)
    shocks = _future_shocks(model, system, baseline, np.random.default_rng(2))
    from nydsge.regimes import regime_indices_from_dates

    start = pd.Period(model.get_setting("date_forecast_start"), freq="Q")
    path = regime_indices_from_dates(
        model.get_setting("regime_dates"), pd.period_range(start, periods=3, freq="Q")
    )
    for t, regime in enumerate(path):
        zero = np.diag(system[regime, "QQ"]) == 0
        np.testing.assert_array_equal(shocks[t, zero], 0)


def test_unknown_and_out_of_bounds_draws_fail_without_mutation(solved):
    model, _ = solved
    original = model.parameters["gamma"].value
    with pytest.raises(ValueError, match="Unknown/free"):
        _parameter_model(model, [1.0], ["not_a_parameter"])
    with pytest.raises(ValueError, match="Out-of-bounds"):
        _parameter_model(model, [-100.0], ["gamma"])
    assert model.parameters["gamma"].value == original


def test_future_anticipated_policy_shocks_share_filter_dated_availability(solved):
    from nydsge.forecast import ForecastOutput
    from nydsge.kalman import model_process_covariances, model_shock_covariance
    from nydsge.ss104_distribution import _future_shocks

    model, system = solved
    historical = deepcopy(model)
    historical.set_setting("date_forecast_start", "2008Q3")
    prediction = ForecastOutput(np.zeros((2, 110)), np.zeros((2, 28)))
    anticipated = [historical.indexes.exogenous_shocks[f"rm_shl{i}"] - 1 for i in range(1, 7)]
    pre = model_shock_covariance(historical, system[1], date="2008Q3")
    post = model_shock_covariance(historical, system[1], date="2008Q4")
    np.testing.assert_array_equal(pre[anticipated], 0)
    assert np.all(np.diag(post)[anticipated] > 0)
    shocks = _future_shocks(historical, system, prediction, np.random.default_rng(17))
    np.testing.assert_array_equal(shocks[0, anticipated], 0)
    assert np.all(shocks[1, anticipated] != 0)
    process = model_process_covariances(historical, system, 2, start_date="2008Q3")
    assert process is not None
    rrr = system[1, "RRR"]
    np.testing.assert_array_equal(process[0], rrr @ pre @ rrr.T)
    np.testing.assert_array_equal(process[1], rrr @ post @ rrr.T)
    # No dated mask is applied to contemporary runs; the archived QQ stays intact.
    np.testing.assert_array_equal(
        model_shock_covariance(model, system[29], date="2026Q4"), system[29, "QQ"]
    )
    np.testing.assert_array_equal(system[1, "QQ"], post)
