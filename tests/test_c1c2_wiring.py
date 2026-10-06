from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nydsge.data import df_to_matrix
from nydsge.economy import (
    SHOCK_GROUPS,
    _model_shock_groups,
    _restrict_conditioning_quarters,
    load_quarterly_economy_config,
)
from nydsge.estimate import estimate, estimation_parameter_names
from nydsge.forecast import solve_shocks_for_observable_targets
from nydsge.models import Model1002
from nydsge.public_sources import load_public_sources_csv, merge_public_sources
from nydsge.scenarios import current_public_observables
from nydsge.solve import compute_system

ROOT = Path(__file__).resolve().parents[1]


def _model(n_ant: int = 0) -> Model1002:
    return Model1002(
        "ss10",
        settings={
            "date_presample_start": "1959-Q3",
            "date_mainsample_start": "1960-Q1",
            "date_forecast_start": "2026-Q3",
            "n_mon_anticipated_shocks": n_ant,
        },
    )


def test_no_anticipated_shocks_drops_ois_rows_and_shocks() -> None:
    model = _model(0)
    assert "obs_nominalrate1" not in model.observables
    assert not any(name.startswith("rm_shl") for name in model.indexes.exogenous_shocks)
    names = estimation_parameter_names(model)
    assert "sigma_r_m" in names
    assert not any(name.startswith("sigma_r_m") and name != "sigma_r_m" for name in names)
    system = compute_system(model)
    zz = system.measurement.ZZ
    obs = list(model.observables)
    for row in ("obs_longinflation", "obs_longrate", "obs_tfp"):
        assert np.abs(zz[obs.index(row)]).sum() > 0


def test_public_data_likelihood_is_finite_on_extended_sample() -> None:
    model = _model(0)
    levels = pd.read_csv(ROOT / "data/public/fred_levels_20261006.csv")
    levels = levels.loc[(levels["date"] >= "1959-Q2") & (levels["date"] <= "2026-Q2")]
    levels = merge_public_sources(
        levels.reset_index(drop=True),
        load_public_sources_csv(ROOT / "data/public/nonfred_sources_20261006.csv"),
    )
    obs = current_public_observables(model, levels)
    obs = obs.loc[obs["date"] >= "1959-Q3"]
    data = df_to_matrix(model, obs, include_presample=True)
    assert data.shape == (268, 13)  # 1959Q3 (presample) .. 2026Q2
    for name in ("obs_longinflation", "obs_longrate", "obs_tfp"):
        assert np.isfinite(data[:, list(model.observables).index(name)]).sum() > 150
    result = estimate(model, data, start_date="1959-Q3")
    assert np.isfinite(result.log_posterior)


def test_saved_c1c2_mode_is_complete_and_consistent() -> None:
    path = ROOT / "data/estimates/m1002_ss10_c1c2_mode.json"
    payload = json.loads(path.read_text())
    model = _model(0)
    names = estimation_parameter_names(model)
    assert tuple(payload["estimated_parameters"]) == names
    assert payload["n_mon_anticipated_shocks"] == 0
    assert payload["log_posterior"] > payload["log_posterior_start"]
    for name in names:
        bounds = model.parameters[name].value_bounds
        assert bounds is not None
        lo, hi = bounds
        assert lo <= payload["parameter_values"][name] <= hi


def test_model_shock_groups_follow_model_shocks() -> None:
    shocks = list(_model(0).indexes.exogenous_shocks)
    groups = _model_shock_groups(shocks)
    assert groups["Monetary policy"] == ("rm_sh",)
    assert {s for v in groups.values() for s in v} == set(shocks)
    full = _model_shock_groups(list(_model(6).indexes.exogenous_shocks))
    assert full == SHOCK_GROUPS


def test_restrict_conditioning_quarters_keeps_only_named_observables() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-Q2", "2026-Q3"],
            "obs_gdp": [1.0, 2.0],
            "obs_nominalrate": [0.9, 0.92],
            "obs_spread": [0.4, 0.41],
        }
    )
    out = _restrict_conditioning_quarters(
        frame, model_end_date="2026-Q2", keep=("obs_nominalrate", "obs_spread")
    )
    assert out.loc[0, "obs_gdp"] == 1.0
    assert np.isnan(out.loc[1, "obs_gdp"])
    assert out.loc[1, "obs_nominalrate"] == 0.92
    with pytest.raises(ValueError, match="Unknown conditioning"):
        _restrict_conditioning_quarters(frame, model_end_date="2026-Q2", keep=("obs_x",))


def test_qq_weighted_conditioning_is_weighted_minimum_norm() -> None:
    model = _model(0)
    system = compute_system(model)
    n_obs = system.measurement.ZZ.shape[0]
    start = np.zeros(system.transition.TTT.shape[0])
    targets = np.full((2, n_obs), np.nan)
    hours = list(model.observables).index("obs_hours")
    targets[:, hours] = [-0.5, -1.0]
    scale = np.sqrt(np.clip(np.diag(system.measurement.QQ), 0.0, None))
    raw = solve_shocks_for_observable_targets(system, start, targets)
    weighted = solve_shocks_for_observable_targets(system, start, targets, shock_scale=scale)
    assert raw.max_abs_error < 1e-8
    assert weighted.max_abs_error < 1e-8
    active = scale > 0
    z_raw = raw.shocks[:, active] / scale[active]
    z_w = weighted.shocks[:, active] / scale[active]
    # weighted solution minimises the standardized norm among exact solutions
    assert np.sum(z_w**2) <= np.sum(z_raw**2) + 1e-12
    assert np.all(weighted.shocks[:, ~active] == 0.0)
    with pytest.raises(ValueError, match="shock_scale"):
        solve_shocks_for_observable_targets(system, start, targets, shock_scale=scale[:-1])


def test_c1c2_config_parses_new_options() -> None:
    config = load_quarterly_economy_config(ROOT / "configs/quarterly_economy_c1c2.json")
    assert config.n_mon_anticipated_shocks == 0
    assert config.conditioning_end_date == "2026-Q3"
    assert "obs_nominalrate" in config.conditioning_observables
    assert config.conditional_shock_weighting == "qq"
    assert config.public_sources_path is not None and config.public_sources_path.exists()
    legacy = load_quarterly_economy_config(ROOT / "configs/quarterly_economy.json")
    assert legacy.n_mon_anticipated_shocks is None
    assert legacy.conditional_shock_weighting == "raw"
    assert legacy.public_sources_path is None
