"""Reporting and conditioning contracts for the fixed-default local forecast."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from forecast_ss104_nyfed_params import (  # noqa: E402
    forecast_tables,
    semi_conditioned_history,
)
from ss104_local import build_local_model  # noqa: E402


def reporting_inputs():
    dates = pd.period_range("2026Q3", periods=40, freq="Q")
    pseudo = pd.DataFrame(
        {"PseudoGDP": 0.4, "PseudoCorePCE": 0.5, "NominalFFR": 0.9, "OutputGap": -2.0},
        index=dates,
    )
    history = pd.DataFrame(
        {"obs_gdp": [0.2, 0.3], "obs_corepce": [0.6, 0.7]},
        index=pd.period_range("2026Q1", "2026Q2", freq="Q"),
    )
    return pseudo, history


def test_annual_growth_uses_actual_first_half_and_exact_compounding():
    pseudo, history = reporting_inputs()
    quarterly, annual = forecast_tables(pseudo, history, 0.1)
    # Independent level-ratio arithmetic: Q1/Q2 actuals, Q3/Q4 model growth.
    gdp_ratio = np.prod(np.exp(np.array([0.3, 0.4, 0.5, 0.5]) / 100))
    pce_ratio = np.prod(np.exp(np.array([0.6, 0.7, 0.5, 0.5]) / 100))
    assert annual.iloc[0].gdp_ss104 == pytest.approx(100 * (gdp_ratio - 1))
    assert annual.iloc[0].corepce_ss104 == pytest.approx(100 * (pce_ratio - 1))
    assert quarterly.iloc[0].gdp_ss104 == pytest.approx(100 * (np.exp(0.005) ** 4 - 1))
    assert quarterly.iloc[0].ffr_ss104 == pytest.approx(3.6)
    assert list(annual.year) == list(range(2026, 2036))
    assert str(quarterly.index[-1]) == "2036Q2"


def test_missing_actual_quarter_cannot_silently_become_zero_growth():
    pseudo, history = reporting_inputs()
    history.loc["2026Q2", "obs_gdp"] = np.nan
    with pytest.raises(ValueError, match="Incomplete gdp growth in 2026"):
        forecast_tables(pseudo, history, 0.1)


def test_semi_conditioning_retains_rates_and_survey_but_excludes_nowcasts():
    model = build_local_model()
    dates = pd.period_range("2026Q2", "2026Q3", freq="Q")
    history = pd.DataFrame(0.7, index=dates, columns=list(model.observables))
    history.loc["2026Q3", "obs_exp_nominalrate2"] = np.nan
    effective = semi_conditioned_history(model, history)
    pd.testing.assert_series_equal(effective.iloc[0], history.iloc[0])
    assert np.isnan(effective.loc["2026Q3", "obs_gdp"])
    assert np.isnan(effective.loc["2026Q3", "obs_corepce"])
    assert np.isnan(effective.loc["2026Q3", "obs_exp_nominalrate2"])
    for name in ("obs_spread", "obs_nominalrate", "obs_longrate", "obs_exp_nominalrate1"):
        assert effective.loc["2026Q3", name] == 0.7
    assert history.loc["2026Q3", "obs_gdp"] == 0.7


def test_committed_python_julia_paths_match_at_explicit_precision():
    tables = Path(__file__).resolve().parents[1] / "data/estimates"
    for kind, keys in (
        ("quarterly", ("gdp", "corepce", "ffr")),
        ("annual", ("gdp", "corepce")),
    ):
        ours = pd.read_csv(tables / f"ss104_nyfed_params_forecast_{kind}.csv")
        julia = pd.read_csv(tables / f"ss104_nyfed_params_julia_forecast_{kind}.csv")
        date = "quarter" if kind == "quarterly" else "year"
        pd.testing.assert_series_equal(ours[date], julia[date])
        for key in keys:
            np.testing.assert_allclose(
                ours[f"{key}_ss104"], julia[f"{key}_julia"], atol=1e-5, rtol=0
            )
