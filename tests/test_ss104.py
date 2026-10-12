from __future__ import annotations

import numpy as np
import pytest

from nydsge.kalman import kalman_log_likelihood, smooth_kalman_result
from nydsge.models import Model1002
from nydsge.regimes import regime_indices_from_dates
from nydsge.solve import compute_system


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


def test_covid_equations_and_regime_filter():
    model = Model1002("ss104")
    system = compute_system(model)
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
    filtered = kalman_log_likelihood(system, data, regime_indices=path)
    smooth = smooth_kalman_result(system, filtered)
    assert np.isfinite(filtered.log_likelihood)
    assert np.isfinite(smooth.smoothed_states).all()
    print(f"COVID canonical residual={max_residual:.3e}; QQ max error=0")


def test_inflation_error_ar1_differenced_measurement_and_regime_boundaries():
    model = Model1002("ss104")
    system = compute_system(model)
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
