"""Independent joint-Gaussian checks of native dated-regime attribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag

from general_analytics_tools.decomposition import export_decomposition
from nydsge.decomposition import regime_disturbance_attribution
from nydsge.kalman import kalman_log_likelihood, smooth_kalman_result
from nydsge.regimes import RegimeSwitchingSystem
from nydsge.solve import Measurement, System, Transition


def fixture():
    systems = []
    for a, q in [(0.7, 0.2), (0.4, 1.2)]:
        systems.append(
            System(
                Transition(
                    np.array([[a, 0.1], [0.0, 0.5]]),
                    np.array([[1.0, 0.2, 0.0], [0.3, 1.0, 0.0]]),
                    np.array([0.1, -0.05]),
                ),
                Measurement(
                    np.array([[1.0, 0.4], [-0.2, 1.0]]),
                    np.array([0.7, -0.3]),
                    np.diag([q, 0.15, 0.0]),
                    np.diag([0.1, 0.3]),
                ),
            )
        )
    return RegimeSwitchingSystem(tuple(systems))


@pytest.mark.parametrize("path", [[1], [1, 2, 2, 1]])
def test_regime_attribution_matches_independent_joint_gaussian(path):
    system = fixture()
    nperiod = len(path)
    active = system.period_systems(path, nperiod)
    m0 = np.array([0.2, -0.1])
    p0 = np.array([[0.5, 0.1], [0.1, 0.3]])
    q = np.array([s.measurement.QQ for s in active])
    data = np.tile([0.5, -0.4], (nperiod, 1))
    if nperiod > 1:
        data[1, 0] = np.nan
        data[2, :] = np.nan  # Entirely missing quarter across a regime change.
    filtered = kalman_log_likelihood(
        system, data, regime_indices=path, initial_state=m0, initial_covariance=p0
    )
    states = smooth_kalman_result(system, filtered, data=data).smoothed_states
    assert states is not None
    result = regime_disturbance_attribution(
        system,
        filtered,
        states,
        data,
        np.array(path),
        shock_covariances=q,
        initial_state=m0,
        initial_covariance=p0,
    )
    # Construct full latent prior [x_-1, eps_0, ..., eps_T] and its observation map.
    prior = block_diag(p0, *q)
    latent_mean = np.r_[m0, np.zeros(nperiod * 3)]
    mapping = np.zeros((2, len(latent_mean)))
    mapping[:, :2] = np.eye(2)
    deterministic = np.zeros(2)
    state_maps, means, measurement_maps, offsets = [], [], [], []
    for t, s in enumerate(active):
        mapping = s.transition.TTT @ mapping
        mapping[:, 2 + 3 * t : 2 + 3 * (t + 1)] += s.transition.RRR
        deterministic = s.transition.TTT @ deterministic + s.transition.CCC
        state_maps.append(mapping.copy())
        means.append(deterministic.copy())
        measurement_maps.append(s.measurement.ZZ @ mapping)
        offsets.append(s.measurement.ZZ @ deterministic + s.measurement.DD)
    h = np.vstack(measurement_maps)
    d = np.concatenate(offsets)
    e = block_diag(*(s.measurement.EE for s in active))
    mask = np.isfinite(data.ravel())
    hobs = h[mask]
    innovation = data.ravel()[mask] - hobs @ latent_mean - d[mask]
    posterior = latent_mean + prior @ hobs.T @ np.linalg.solve(
        hobs @ prior @ hobs.T + e[np.ix_(mask, mask)], innovation
    )
    expected_states = np.array([a @ posterior + b for a, b in zip(state_maps, means, strict=True)])
    np.testing.assert_allclose(states, expected_states, atol=2e-14, rtol=0)
    np.testing.assert_allclose(result["shocks"], posterior[2:].reshape(-1, 3), atol=2e-14, rtol=0)
    assert np.max(abs(result["remainder"])) < 2e-14
    assert np.max(abs(result["state_remainder"])) < 2e-14
    assert np.all(result["shocks"][:, 2] == 0)  # Zero prior support stays zero.
    assert np.isnan(result["measurement_gap"]).sum() == np.isnan(data).sum()
    # The annual-rate/log-growth factor belongs solely at the export boundary.
    table, gates = export_decomposition(
        result,
        pd.period_range("2000Q1", periods=nperiod, freq="Q"),
        ["obs_gdp", "obs_nominalrate"],
        ["g_sh", "b_sh", "rm_sh"],
    )
    assert all(v["status"] == "reconstruction_pass" for v in gates.values())
    total = table.groupby(["quarter", "variable"]).contribution.sum()
    assert len(total) == 2 * nperiod
    for t, quarter in enumerate(pd.period_range("2000Q1", periods=nperiod, freq="Q")):
        for i, name in enumerate(["obs_gdp", "obs_nominalrate"]):
            assert total.loc[(str(quarter), name)] == pytest.approx(
                4 * result["fitted"][t, i], abs=2e-14
            )


def test_attribution_rejects_wrong_initialization_covariance_and_regimes():
    system = fixture()
    data = np.array([[0.2, -0.4]])
    f = kalman_log_likelihood(system, data, regime_indices=[1])
    states = smooth_kalman_result(system, f, data=data).smoothed_states
    assert states is not None
    qq = np.array([system.systems[0].measurement.QQ])
    args = (system, f, states, data, np.array([1]))
    with pytest.raises(ValueError, match="Initial state"):
        regime_disturbance_attribution(*args, shock_covariances=qq, initial_state=np.ones(2))
    with pytest.raises(ValueError, match="inconsistent"):
        regime_disturbance_attribution(
            *args, shock_covariances=qq, process_covariances=np.zeros((1, 2, 2))
        )
    with pytest.raises(ValueError, match="regimes"):
        regime_disturbance_attribution(system, f, states, data, np.array([2]), shock_covariances=qq)
    with pytest.raises(ValueError, match="semidefinite"):
        regime_disturbance_attribution(*args, shock_covariances=-qq)
    with pytest.raises(ValueError, match="never infinite"):
        regime_disturbance_attribution(
            system, f, states, np.array([[np.inf, 0.0]]), np.array([1]), shock_covariances=qq
        )
