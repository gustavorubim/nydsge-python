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
