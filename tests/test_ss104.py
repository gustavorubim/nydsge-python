from __future__ import annotations

import numpy as np
import pytest

from nydsge.kalman import kalman_log_likelihood, smooth_kalman_result
from nydsge.models import Model1002
from nydsge.regimes import regime_indices_from_dates
from nydsge.solve import compute_system


@pytest.fixture(scope="module")
def ss104_system():
    model = Model1002("ss104")
    return model, compute_system(model)


def test_covid_calibration_and_proportional_anticipation():
    model = Model1002("ss104")
    for reg, expected in (
        (1, (0, 0, 0)),
        (2, (400, 5, 4)),
        (4, (400, 5, 4)),
        (5, (4, 0.05, 0.04)),
        (6, (0, 0, 0)),
    ):
        m = model.regime_snapshot(reg)
        assert tuple(m.numeric_value(f"sigma_{k}") for k in ("phi", "ziid", "biidc")) == expected
        c = m.equilibrium_matrices()
        for key in ("phi", "ziid", "biidc"):
            row = m.indexes.equilibrium_conditions[f"eq_{key}l1"] - 1
            col = m.indexes.exogenous_shocks[f"{key}_sh"] - 1
            assert c.Psi[row, col] == (2 if reg == 2 else 0)
        assert m.numeric_value("sigma_biidc1") == (4 if reg in (4, 5) else 0)
    with pytest.raises(ValueError, match="Unknown ss104"):
        model.regime_snapshot(30)


def test_covid_equations_and_regime_filter(ss104_system):
    model, system = ss104_system
    assert len(system.systems) == 29
    max_residual = 0.0
    for reg in range(1, 30):
        m = model.regime_snapshot(reg)
        c = m.equilibrium_matrices()
        t = system[reg].transition
        n = len(m.indexes.endogenous_states)
        rows = [
            m.indexes.equilibrium_conditions[f"eq_{k}"] - 1
            for k in ("ziid", "biidc", "phi", "Ephi")
        ]
        max_residual = max(
            max_residual, float(np.max(np.abs((c.Gamma0 @ t.TTT[:n, :n] - c.Gamma1)[rows])))
        )
        for key in ("phi", "ziid", "biidc"):
            col = m.indexes.exogenous_shocks[f"{key}_sh"] - 1
            assert system[reg, "QQ"][col, col] == m.numeric_value(f"sigma_{key}") ** 2
    assert max_residual < 1e-10
    path = regime_indices_from_dates(
        model.get_setting("regime_dates"),
        ["2019Q4", "2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1"],
    )
    data = np.tile(system[1, "DD"], (len(path), 1))
    data[:, 13:] = np.nan
    data[np.asarray(path) >= 5, model.observables["obs_nominalrate"] - 1] = np.nan
    filtered = kalman_log_likelihood(system, data, regime_indices=path)
    smooth = smooth_kalman_result(system, filtered)
    assert np.isfinite(filtered.log_likelihood)
    assert np.isfinite(smooth.smoothed_states).all()
    print(f"COVID canonical residual={max_residual:.3e}; QQ max error=0")


def test_inflation_error_ar1_differenced_measurement_and_regime_boundaries(ss104_system):
    model, system = ss104_system
    state = model.indexes.endogenous_states_augmented["e_meas_pi_t"] - 1
    lag = model.indexes.endogenous_states_augmented["e_meas_pi_t1"] - 1
    shock = model.indexes.exogenous_shocks["meas_pi_sh"] - 1
    iid = model.indexes.exogenous_shocks["corepce_sh"] - 1
    for reg in range(1, 30):
        assert system[reg, "TTT"][state, state] == (0 if reg == 1 else 0.232)
        assert system[reg, "TTT"][lag, state] == 1
        assert system[reg, "RRR"][state, shock] == 1
        assert system[reg, "QQ"][shock, shock] == (0.0999**2 if 2 <= reg <= 10 else 0)
        assert system[reg, "QQ"][iid, iid] == (0 if 2 <= reg <= 10 else 0.0999**2)
        for obs in ("obs_gdpdeflator", "obs_corepce"):
            row = model.observables[obs] - 1
            np.testing.assert_array_equal(system[reg, "ZZ"][row, [state, lag]], [1, -1])
    # An isolated inflation-error innovation decays and is differenced by both inflation rows.
    x = system[2, "RRR"][:, shock]
    row = model.observables["obs_corepce"] - 1
    actual = []
    for _ in range(4):
        actual.append(system[2, "ZZ"][row] @ x)
        x = system[2, "TTT"] @ x
    np.testing.assert_allclose(
        actual, [1, 0.232 - 1, 0.232**2 - 0.232, 0.232**3 - 0.232**2], atol=1e-15, rtol=0
    )
    print("Inflation AR/ZZ/QQ calibration max error=0; impulse max error<1e-15")


def test_flexible_ait_policy_history_and_gap_initialization(ss104_system):
    import pandas as pd

    from nydsge.models.ss104 import initialize_gap_observations

    model, system = ss104_system
    rho = np.exp(np.log(0.5) / 10)
    assert model.get_setting("credibility")[5] == 0
    assert model.get_setting("credibility")[17] == 0.5
    assert model.get_setting("credibility")[29] == 1
    m = model.regime_snapshot(29)
    c = m.equilibrium_matrices()
    eq, state = m.indexes.equilibrium_conditions, m.indexes.endogenous_states
    assert c.Gamma0[eq["eq_mp"] - 1, state["pgap_t"] - 1] == pytest.approx(-4 * (1 - rho) * 0.1)
    assert c.Gamma0[eq["eq_mp"] - 1, state["ygap_t"] - 1] == pytest.approx(-3 * (1 - rho) * 0.1)
    assert c.Gamma1[eq["eq_mp"] - 1, state["R_t"] - 1] == 0.9
    maximum = 0.0
    for r in range(5, 30):
        # Upstream temporary solve freezes predictable equations at regime 5.
        c = model.regime_snapshot(5 if r < 10 else r).equilibrium_matrices()
        rows = np.all(c.Pi == 0, axis=1)
        n = len(state)
        t = system[r].transition
        maximum = max(
            maximum,
            np.max(np.abs((c.Gamma0 @ t.TTT[:n, :n] - c.Gamma1)[rows])),
            np.max(np.abs((c.Gamma0 @ t.RRR[:n] - c.Psi)[rows])),
            np.max(np.abs((c.Gamma0 @ t.CCC[:n] - c.C)[rows])),
        )
    assert maximum < 1e-9
    for r in range(5, 10):
        i = state["R_t"] - 1
        np.testing.assert_allclose(system[r, "TTT"][i], 0, atol=1e-10)
        assert system[r, "CCC"][i] + model.numeric_value("Rstarn") == pytest.approx(0.025)
    frame = pd.DataFrame(index=pd.period_range("2020Q1", "2020Q4", freq="Q"))
    gaps = initialize_gap_observations(model, frame)
    np.testing.assert_array_equal(gaps.notna().sum().values, [1, 1])
    np.testing.assert_array_equal(gaps.loc["2020Q2"].values, [-0.125, -12.0])
    print(f"AIT equation residual={maximum:.3e}; credibility/gap calibration error=0")


def test_expected_ffr_horizons_and_anticipated_shock_handoff(ss104_system):
    from nydsge.regimes import k_periods_ahead_expectations

    model, system = ss104_system
    assert len(model.observables) == 28
    rate = model.indexes.endogenous_states["R_t"] - 1
    for r in (1, 9, 10, 27, 28, 29):
        for h in range(1, 7):
            power, _ = k_periods_ahead_expectations(
                tuple(s.transition for s in system.systems),
                h,
                reg=r,
                information_set=system.information_sets[r - 1],
            )
            row = model.observables[f"obs_exp_nominalrate{h}"] - 1
            expected = power[rate].copy()
            expected[model.indexes.endogenous_states_augmented[f"e_exp_rm{h}"] - 1] += 1
            np.testing.assert_allclose(system[r, "ZZ"][row], expected, atol=1e-12, rtol=0)
            sigma = model.regime_snapshot(r).numeric_value(f"sigma_exp_rm{h}")
            assert sigma == (0 if r < 10 else 0.0375 + 0.00625 * h)
            assert model.regime_snapshot(r).numeric_value(f"sigma_ait_r_m{h}") == (
                0.2 if 9 <= r <= model.get_setting("cur_model_regime") else 0
            )
    print("Expected FFR ZZ/QQ calibration max error=0 at 1-6Q; n_ant=6")


def test_full_ss104_conditional_forecast_workflow(ss104_system):
    import pandas as pd

    from nydsge.ss104_workflow import forecast_ss104

    model, system = ss104_system
    dates = pd.period_range("2019Q4", "2026Q3", freq="Q")
    frame = pd.DataFrame(np.nan, index=dates, columns=list(model.observables))
    for i, name in enumerate(list(model.observables)[:13]):
        frame[name] = system[1, "DD"][i] + 0.01 * np.sin(np.arange(len(dates)) + i)
    frame.loc["2026Q3", "obs_gdp"] = 0.65
    frame.loc["2026Q3", "obs_corepce"] = 0.70
    output = forecast_ss104(model, system=system, data=frame, horizon=8, cond_type="full")
    assert output.states.shape == (7, 110)
    assert output.observables.shape == (7, 28)
    assert output.history_states.shape == (27, 110)
    assert output.conditional_states.shape == (1, 110)
    assert np.isfinite(output.states).all()
    assert np.isfinite(output.log_likelihood)
    gap = model.indexes.endogenous_states["pgap_t"] - 1
    ygap = model.indexes.endogenous_states["ygap_t"] - 1
    np.testing.assert_allclose(output.history_states[2, [gap, ygap]], [-0.125, -12], atol=1e-9)
    # Conditioning with finite conditional errors should pull both channels
    # toward the supplied nowcast, while allowing nonzero residuals.
    gdp = model.observables["obs_gdp"] - 1
    pce = model.observables["obs_corepce"] - 1
    assert abs(output.conditional_observables[0, gdp] - 0.65) < 0.1
    assert abs(output.conditional_observables[0, pce] - 0.70) < 0.1
    with pytest.raises(ValueError, match="require full/semi"):
        forecast_ss104(model, system=system, data=frame)


def test_ss104_presample_anticipated_policy_variances(ss104_system):
    from nydsge.kalman import model_process_covariances

    model, system = ss104_system
    covariances = model_process_covariances(model, system, 2, start_date="2008Q3")
    qq = system[1, "QQ"].copy()
    ant = [model.indexes.exogenous_shocks[f"rm_shl{h}"] - 1 for h in range(1, 7)]
    qq[ant, :] = qq[:, ant] = 0
    r = system[1, "RRR"]
    np.testing.assert_array_equal(covariances[0], r @ qq @ r.T)
    np.testing.assert_array_equal(covariances[1], r @ system[1, "QQ"] @ r.T)


@pytest.mark.parametrize(
    "vector_id", [None, 1, 2, 3], ids=["default", "vector_1", "vector_2", "vector_3"]
)
def test_ss104_source_julia_oracle_parity(ss104_system, vector_id):
    import json
    from pathlib import Path

    import h5py

    from nydsge.ss104_estimation import ss104_parameter_catalog, update_ss104_parameters
    from tools.oracle_julia.compare_ss104 import compare

    filename = "ss104_default_oracle.h5" if vector_id is None else f"ss104_vector_{vector_id}.h5"
    path = Path(__file__).parent / "fixtures" / filename
    if vector_id is None:
        model, system = ss104_system
    else:
        with h5py.File(path) as fixture:
            vector = json.loads(fixture["metadata/json"].asstr()[()])["parameter_vector"]
        model = Model1002("ss104")
        catalog = ss104_parameter_catalog(model)
        assert vector["coordinates"] == list(catalog)
        assert vector["id"] == f"vector_{vector_id}"
        update_ss104_parameters(model, catalog, vector["estimation_values"])
        updated = ss104_parameter_catalog(model)
        assert vector["parameters"] == {k: p.value for k, p in updated.items()}
        assert all(
            not p.fixed and p.value_bounds[0] <= p.value <= p.value_bounds[1]
            for p in updated.values()
        )
        assert sum(p.value != catalog[k].value for k, p in updated.items()) >= 90
        system = compute_system(model)
    report = compare(path, model=model, system=system)
    assert report["metadata"]["dsge_commit"] == "c09dfe881f8809f65b8c0f08a761ec6cc456a6fb"
    assert report["metadata"]["statespace_commit"] == "794fb02ff45a62537d8d5d037c1d12b3aa7e4574"
    assert len(report["regimes"]) == 29
    assert len(model.parameters) == 130
    assert len(model.pseudo_observables) == 26
    assert report["max_abs"]["parameters"] == 0
    for errors in report["regimes"].values():
        for name, error in errors.items():
            assert error <= (
                5e-10
                if name in ("TTT", "RRR", "CCC", "ZZ", "DD", "ZZ_pseudo", "DD_pseudo")
                else 1e-12
            ), (
                name,
                error,
            )
    for name in ("history_loglik", "conditional_loglik"):
        assert report["workflow"][name]["error"] < 1e-6
    for name in ("forecast_states_error", "conditional_forecast_states_error"):
        assert report["workflow"][name] < (1e-6 if vector_id is None else 2e-6)
    for name in (
        "forecast_pseudo_observables_error",
        "conditional_forecast_pseudo_observables_error",
    ):
        assert report["workflow"][name] < 1e-6
    for name in ("forecast_observables_error", "conditional_forecast_observables_error"):
        assert report["workflow"][name] < (1e-7 if vector_id is None else 5e-7)
    for name in ("history_smoothed_states_error", "conditional_smoothed_states_error"):
        assert report["workflow"][name] < (5e-6 if vector_id is None else 1e-5)
    for label in ("history", "conditional"):
        for state in ("predicted_states", "filtered_states"):
            assert report["workflow"][f"{label}_{state}_error"] < 2e-6
        for covariance in ("filtered_covariances", "predicted_covariances"):
            key = f"{label}_{covariance}"
            # Absolute error protects small entries; max-entry normalization
            # captures scale amplification from the COVID variances (sigma=400).
            assert report["workflow"][f"{key}_error"] < 2e-4
            tolerance = 1e-6 if covariance == "filtered_covariances" else 5e-10
            assert report["workflow"][f"{key}_relative_error"] < tolerance
