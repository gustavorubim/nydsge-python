from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nydsge.models import Model1002
from nydsge.parameters import parameter_log_prior
from nydsge.solve import compute_system
from nydsge.ss104_estimation import (
    evaluate_ss104_posterior,
    ss104_estimation_vector,
    ss104_parameter_catalog,
    update_ss104_parameters,
)
from nydsge.ss104_workflow import forecast_ss104, prepare_ss104_observations


def test_regime_coordinates_update_independently_and_count_prior_once():
    model = Model1002("ss104")
    catalog = ss104_parameter_catalog(model)
    assert len(catalog) == 96
    assert "sigma_phi@2" in catalog and "sigma_phi@3" in catalog
    assert "sigma_phi" not in catalog and "sigma_phi@1" not in catalog
    assert "sigma_corepce@2" not in catalog
    original = dict(model.get_setting("parameter_regimes")["sigma_phi"])
    x = ss104_estimation_vector(catalog)
    x[list(catalog).index("sigma_phi@2")] += np.log(1.1)
    x[list(catalog).index("sigma_g@1")] += np.log(1.2)
    update_ss104_parameters(model, catalog, x)
    assert model.regime_snapshot(2).numeric_value("sigma_phi") == pytest.approx(440)
    assert model.regime_snapshot(5).numeric_value("sigma_phi") == original[3].value
    assert model.get_setting("parameter_regimes")["sigma_phi"][1] == original[1]
    assert model.parameters["sigma_g"].value == pytest.approx(1.2 * catalog["sigma_g@1"].value)
    assert model.regime_snapshot(2).parameters["sigma_g"].value == catalog["sigma_g@2"].value
    current = ss104_parameter_catalog(model)
    expected = sum(
        parameter_log_prior(p)
        for name, p in model.parameters.items()
        if name not in model.get_setting("parameter_regimes")
    ) + sum(
        parameter_log_prior(p)
        for entries in model.get_setting("parameter_regimes").values()
        for p in entries.values()
    )
    assert sum(parameter_log_prior(p) for p in current.values()) == pytest.approx(expected)
    update_ss104_parameters(model, catalog, ss104_estimation_vector(catalog))
    assert {
        r: p.value for r, p in model.get_setting("parameter_regimes")["sigma_phi"].items()
    } == pytest.approx({r: p.value for r, p in original.items()})


def test_estimation_likelihood_matches_m6_forecast_and_presample():
    model = Model1002(
        "ss104",
        settings={
            "date_presample_start": "2019-Q4",
            "date_mainsample_start": "2020-Q1",
        },
    )
    system = compute_system(model)
    frame = pd.DataFrame(
        np.nan, index=pd.period_range("2019Q4", "2026Q2", freq="Q"), columns=list(model.observables)
    )
    frame["obs_corepce"] = 0.6
    frame["obs_nominalrate"] = 0.9
    frame.loc["2019Q4", "obs_corepce"] = 1.8
    posterior, likelihood, prior, filtered = evaluate_ss104_posterior(model, frame, system=system)
    forecast = forecast_ss104(model, data=frame, system=system, horizon=2)
    assert likelihood == forecast.log_likelihood
    assert posterior == likelihood + prior
    assert likelihood == pytest.approx(filtered.log_likelihood_by_period[1:].sum())
    prepared = prepare_ss104_observations(model, frame)
    assert prepared.loc["2020Q2", ["obs_pgap", "obs_ygap"]].tolist() == [-0.125, -12]
    assert prepared.loc["2020Q4":"2021Q4", "obs_nominalrate"].isna().all()
    with pytest.raises(ValueError, match="before forecast start"):
        evaluate_ss104_posterior(
            model, frame.reindex(pd.period_range("2019Q4", "2026Q3", freq="Q")), system=system
        )


def test_steady_state_cache_invalidates_structural_changes(monkeypatch):
    model = Model1002("ss104")
    calls = []
    compute = model.compute_steady_state

    def counted():
        calls.append(1)
        return compute()

    monkeypatch.setattr(model, "compute_steady_state", counted)
    first = model.steadystate()["z_star"]
    model.steadystate()
    assert len(calls) == 1
    model.parameters["gamma"] = replace(model.parameters["gamma"], value=0.4)
    assert model.steadystate()["z_star"] != first
    assert len(calls) == 2
    model.parameters["sigma_g"] = replace(model.parameters["sigma_g"], value=3)
    model.steadystate()
    assert len(calls) == 2


def test_local_survey_data_is_sparse_and_not_backfilled():
    from nydsge.survey_ffr import expected_ffr_observations

    path = Path(__file__).resolve().parents[1] / "data/public/ss104_survey_ffr_local.csv"
    records = pd.read_csv(path)
    dates = pd.period_range("2021Q4", "2026Q3", freq="Q")
    observed = expected_ffr_observations(records, dates, as_of="2026-10-12")
    assert observed.notna().sum().tolist() == [2] * 6
    assert observed.loc["2024Q3"].tolist() == pytest.approx(
        [1.1575, 1.0325, 0.97, 0.9075, 0.845, 0.845]
    )
    assert observed.loc["2026Q3"].tolist() == pytest.approx([0.9075] * 5 + [0.845])
    assert observed.loc["2026Q2"].isna().all()
