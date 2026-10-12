from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nydsge.core import NotPortedError
from nydsge.kalman import (
    kalman_log_likelihood,
    observables_from_states,
    smooth_kalman_result,
)
from nydsge.models import Model1002
from nydsge.regimes import (
    RegimeSwitchingSystem,
    compute_regime_system,
    compute_tvis_system,
    gensys2,
    gensys_to_predictable_form,
    k_periods_ahead_expectations,
    k_periods_ahead_expected_sums,
    regime_indices_from_dates,
    solve_regime_switching,
)
from nydsge.solve import CanonicalSystem, Measurement, System, Transition, compute_system, gensys


def policy_canonical(phi: float = 1.5, rate: float = 0.0) -> CanonicalSystem:
    """Small NK system: states x, pi, i, Epi; i = phi*pi + rate."""
    g0 = np.array(
        [
            [1.0, 0.0, 1.0, -1.0],
            [-0.1, 1.0, 0.0, -0.99],
            [0.0, -phi, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    g1 = np.zeros((4, 4))
    g1[0, 0] = 0.5
    g1[3, 3] = 1.0
    return CanonicalSystem(
        g0,
        g1,
        np.array([0.0, 0.0, rate, 0.0]),
        np.array([[1.0], [0.0], [0.0], [0.0]]),
        np.array([[0.0], [0.0], [0.0], [1.0]]),
    )


def scalar_canonical(value: float) -> CanonicalSystem:
    return CanonicalSystem(np.eye(1), np.array([[value]]), np.zeros(1), np.eye(1), np.zeros((1, 0)))


def state_measurement(reg, transition, transitions, information_set):
    n = transition.TTT.shape[0]
    return Measurement(np.eye(n), np.zeros(n), np.eye(transition.RRR.shape[1]), np.eye(n) * 0.1)


def varied_systems() -> RegimeSwitchingSystem:
    return RegimeSwitchingSystem(
        (
            System(
                Transition(np.array([[0.6, 0.1], [0.0, 0.3]]), np.eye(2), np.array([0.2, -0.1])),
                Measurement(
                    np.eye(2), np.array([0.3, 0.1]), np.diag([0.4, 0.2]), np.diag([0.1, 0.2])
                ),
            ),
            System(
                Transition(
                    np.array([[0.2, -0.2], [0.1, 0.7]]),
                    np.array([[0.5, 0.1], [0.0, 1.2]]),
                    np.array([-0.1, 0.4]),
                ),
                Measurement(
                    np.array([[1.0, 0.5], [-0.2, 1.0]]),
                    np.array([-0.4, 0.2]),
                    np.diag([0.8, 0.1]),
                    np.diag([0.3, 0.05]),
                ),
            ),
        )
    )


@pytest.mark.parametrize("subspec", ["ss10", "ss104"])
def test_one_regime_is_bit_identical_to_existing_model_solver_filter_and_smoother(subspec):
    model = Model1002(subspec)
    if subspec == "ss104":
        model = model.regime_snapshot(1)
    existing = compute_system(model)
    regime = compute_system(model, regime_models=(model,), tvis=True, information_sets=((1,),))
    assert isinstance(existing, System)
    assert isinstance(regime, RegimeSwitchingSystem)
    for name in ("TTT", "RRR", "CCC", "ZZ", "DD", "QQ", "EE", "ZZ_pseudo", "DD_pseudo"):
        np.testing.assert_array_equal(regime[1, name], existing[name])
    t = solve_regime_switching((model.equilibrium_matrices(),))[0]
    old_t = gensys(model.equilibrium_matrices()).transition
    for name in ("TTT", "RRR", "CCC"):
        np.testing.assert_array_equal(getattr(t, name), getattr(old_t, name))
    data = np.random.default_rng(1).normal(size=(8, existing["ZZ"].shape[0]))
    if subspec == "ss104":
        data[:, 13:] = np.nan
    data[1, 2] = np.nan
    data[3] = np.nan
    for process in (
        None,
        np.repeat((existing["RRR"] @ existing["QQ"] @ existing["RRR"].T)[None], 8, axis=0),
    ):
        old = smooth_kalman_result(
            existing,
            kalman_log_likelihood(
                existing, data, process_covariances=process, log_likelihood_start=2
            ),
        )
        new = smooth_kalman_result(
            regime,
            kalman_log_likelihood(
                regime, data, process_covariances=process, log_likelihood_start=2
            ),
        )
        assert new.log_likelihood == old.log_likelihood
        for name in (
            "log_likelihood_by_period",
            "predicted_states",
            "predicted_covariances",
            "filtered_states",
            "filtered_covariances",
            "smoothed_states",
            "smoothed_covariances",
        ):
            np.testing.assert_array_equal(getattr(new, name), getattr(old, name))


def test_two_regime_temporary_zlb_solves_despite_indeterminate_permanent_zlb():
    zlb, permanent = policy_canonical(0.0, -0.25), policy_canonical()
    assert gensys(zlb).eu == (1, 0)
    with pytest.raises(NotPortedError, match="Regime 1"):
        solve_regime_switching((zlb, permanent))
    system = compute_regime_system(
        (zlb, permanent),
        state_measurement,
        temporary_regimes=((1,),),
        information_sets=((1, 2), (2,)),
    )
    zlb_t, permanent_t = [s.transition for s in system.systems]
    np.testing.assert_allclose(zlb_t.TTT[2], 0.0, atol=1e-14)
    np.testing.assert_allclose(zlb_t.RRR[2], 0.0, atol=1e-14)
    assert zlb_t.CCC[2] == -0.25
    # Independently impose Epi_t = permanent pi transition applied to s_t.
    for state, shock in (
        (np.array([0.8, 0.0, 0.0, 0.0]), np.array([0.3])),
        (np.zeros(4), np.zeros(1)),
    ):
        next_state = zlb_t.TTT @ state + zlb_t.RRR @ shock + zlb_t.CCC
        expected_pi = (permanent_t.TTT @ next_state + permanent_t.CCC)[1]
        assert next_state[3] == pytest.approx(expected_pi)
        assert next_state[0] + next_state[2] - next_state[3] == pytest.approx(
            0.5 * state[0] + shock[0]
        )
        assert next_state[1] == pytest.approx(0.1 * next_state[0] + 0.99 * expected_pi)
    data = np.zeros((5, 4))
    data[0, 2] = -0.25
    data[2] = np.nan
    filtered = kalman_log_likelihood(
        system, data, regime_indices=[1, 2, 2, 2, 2], log_likelihood_start=1
    )
    smoothed = smooth_kalman_result(system, filtered)
    assert smoothed.smoothed_states is not None
    assert np.isfinite(filtered.log_likelihood)
    assert np.all(np.isfinite(smoothed.smoothed_states))
    np.testing.assert_array_equal(smoothed.smoothed_states[-1], filtered.filtered_states[-1])


def test_gensys2_horizon_and_time_varying_policies():
    terminal = gensys(policy_canonical()).transition
    zlb = policy_canonical(0.0, -0.25)
    path = gensys2(zlb, terminal, periods=2)
    mixed = gensys2((policy_canonical(0.0, -0.5), zlb), terminal)
    assert path[-1] is terminal
    assert path[0].CCC[2] == -0.25
    assert mixed[0].CCC[2] == -0.5
    np.testing.assert_array_equal(path[1].TTT, mixed[1].TTT)
    assert not np.array_equal(path[0].TTT, path[1].TTT)
    assert gensys2(zlb, terminal, periods=0) == (terminal,)


def test_model1002_temporary_policy_snapshot_uses_gensys2_and_augmentation():
    class ZLBModel1002(Model1002):
        def equilibrium_matrices(self):
            canonical = super().equilibrium_matrices()
            row = self.indexes.equilibrium_conditions["eq_mp"] - 1
            state = self.indexes.endogenous_states["R_t"] - 1
            canonical.Gamma0[row] = 0.0
            canonical.Gamma0[row, state] = 1.0
            canonical.Gamma1[row] = 0.0
            canonical.Psi[row] = 0.0
            canonical.C[row] = 0.025 - self.numeric_value("Rstarn")
            return canonical

    zlb, permanent = ZLBModel1002("ss104"), Model1002("ss104")
    system = compute_system(
        zlb,
        regime_models=(zlb, permanent),
        temporary_regimes=((1,),),
        tvis=True,
        information_sets=((1, 2), (2,)),
    )
    state = zlb.indexes.endogenous_states["R_t"] - 1
    np.testing.assert_allclose(system[1, "TTT"][state], 0.0, atol=1e-10)
    np.testing.assert_allclose(system[1, "RRR"][state], 0.0, atol=1e-10)
    assert system[1, "CCC"][state] == pytest.approx(0.025 - zlb.numeric_value("Rstarn"))
    assert system[1, "TTT"].shape[0] > len(zlb.indexes.endogenous_states)
    data = np.repeat(system[1, "DD"][None], 3, axis=0)
    data[:, 13:] = np.nan
    data[:, zlb.observables["obs_nominalrate"] - 1] = np.nan
    result = kalman_log_likelihood(system, data, regime_indices=[1, 2, 2])
    assert np.isfinite(result.log_likelihood)


def test_all_matrices_vary_without_reset_and_smoother_uses_next_regime():
    system = varied_systems()
    data = np.array([[0.5, -0.2], [np.nan, 0.3], [0.2, np.nan], [np.nan, np.nan], [-0.4, 0.6]])
    indices = [1, 1, 2, 2, 2]
    result = kalman_log_likelihood(system, data, regime_indices=indices, log_likelihood_start=2)
    # Source-independent Gaussian conditioning of the stacked latent path.
    mean0 = np.linalg.solve(np.eye(2) - system[1, "TTT"], system[1, "CCC"])
    t0 = system[1, "TTT"]
    q0 = system[1, "RRR"] @ system[1, "QQ"] @ system[1, "RRR"].T
    from scipy.linalg import solve_discrete_lyapunov

    prior_cov = solve_discrete_lyapunov(t0, q0)
    means = []
    blocks = []
    for period, reg in enumerate(indices):
        t, r, c, q = [system[reg, name] for name in ("TTT", "RRR", "CCC", "QQ")]
        means.append(t @ (mean0 if period == 0 else means[-1]) + c)
        row = [t @ blocks[-1][j] for j in range(period)] if period else []
        row.append(t @ (prior_cov if period == 0 else blocks[-1][-1]) @ t.T + r @ q @ r.T)
        for j in range(period):
            blocks[j].append(row[j].T)
        blocks.append(row)
    cov = np.block(blocks)
    mean = np.concatenate(means)
    z = np.zeros((10, 10))
    e = np.zeros((10, 10))
    d = np.zeros(10)
    for i, reg in enumerate(indices):
        z[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = system[reg, "ZZ"]
        e[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = system[reg, "EE"]
        d[2 * i : 2 * i + 2] = system[reg, "DD"]
    mask = ~np.isnan(data.ravel())
    z = z[mask]
    e = e[np.ix_(mask, mask)]
    gain = np.linalg.solve(z @ cov @ z.T + e, z @ cov).T
    expected_mean = mean + gain @ (data.ravel()[mask] - d[mask] - z @ mean)
    expected_cov = cov - gain @ z @ cov
    disturbance = smooth_kalman_result(system, result, data=data)
    np.testing.assert_allclose(disturbance.smoothed_states, expected_mean.reshape(5, 2), atol=1e-12)
    for i in range(5):
        np.testing.assert_allclose(
            disturbance.smoothed_covariances[i],
            expected_cov[2 * i : 2 * i + 2, 2 * i : 2 * i + 2],
            atol=1e-12,
        )
    smoothed = smooth_kalman_result(system, result)
    assert smoothed.smoothed_states is not None
    assert smoothed.smoothed_covariances is not None
    np.testing.assert_allclose(smoothed.smoothed_states, expected_mean.reshape(5, 2), atol=1e-12)
    for i in range(5):
        np.testing.assert_allclose(
            smoothed.smoothed_covariances[i],
            expected_cov[2 * i : 2 * i + 2, 2 * i : 2 * i + 2],
            atol=1e-12,
        )
    assert result.log_likelihood == result.log_likelihood_by_period[2:].sum()
    s = system.systems[1]
    np.testing.assert_allclose(
        result.predicted_states[2], s.transition.TTT @ result.filtered_states[1] + s.transition.CCC
    )
    np.testing.assert_allclose(
        result.predicted_covariances[2],
        s.transition.TTT @ result.filtered_covariances[1] @ s.transition.TTT.T
        + s.transition.RRR @ s.measurement.QQ @ s.transition.RRR.T,
    )
    observed = observables_from_states(system, result.filtered_states, regime_indices=indices)
    np.testing.assert_allclose(
        observed[2], s.measurement.ZZ @ result.filtered_states[2] + s.measurement.DD
    )


def test_information_sets_use_ordered_products_and_affine_constants():
    ts = tuple(s.transition for s in varied_systems().systems)
    myopic = k_periods_ahead_expectations(ts, 3, reg=1, information_set=(1,))
    informed = k_periods_ahead_expectations(ts, 3, reg=1, information_set=(1, 2))
    np.testing.assert_allclose(myopic[0], np.linalg.matrix_power(ts[0].TTT, 3))
    np.testing.assert_allclose(informed[0], np.linalg.matrix_power(ts[1].TTT, 3))
    assert not np.array_equal(myopic[0], informed[0])
    # Add a third, noncommuting transition; order must be T3*T2, not T2*T3.
    ts += (Transition(np.array([[0.1, 0.0], [0.3, 0.4]]), np.eye(2), np.array([0.4, -0.3])),)
    power, constant = k_periods_ahead_expectations(ts, 2, information_set=(1, 2, 3))
    np.testing.assert_allclose(power, ts[2].TTT @ ts[1].TTT)
    np.testing.assert_allclose(constant, ts[2].TTT @ ts[1].CCC + ts[2].CCC)
    sums = k_periods_ahead_expected_sums(ts, 2, information_set=(1, 2, 3))
    np.testing.assert_allclose(sums[0], ts[1].TTT + power)
    np.testing.assert_allclose(sums[1], ts[1].CCC + constant)


def test_tvis_selects_entire_belief_scenario_for_measurements():
    def expected_measurement(reg, transition, transitions, information_set):
        power, constant = k_periods_ahead_expectations(
            transitions, 1, reg=reg, information_set=information_set
        )
        return Measurement(power, constant, np.eye(1), np.eye(1))

    scenarios = (
        (scalar_canonical(0.1), scalar_canonical(0.2)),
        (scalar_canonical(0.3), scalar_canonical(0.8)),
    )
    system = compute_tvis_system(
        scenarios, expected_measurement, select_system=(1, 2), information_sets=((1, 2), (2,))
    )
    assert system[1, "TTT"][0, 0] == 0.1
    assert system[2, "TTT"][0, 0] == 0.8
    # Regime 1 believes scenario 1's future is 0.2, although realized regime 2 uses 0.8.
    assert system[1, "ZZ"][0, 0] == 0.2


def test_tvis_scenarios_can_have_different_temporary_policy_durations():
    zlb, permanent = policy_canonical(0.0, -0.25), policy_canonical()
    system = compute_tvis_system(
        ((zlb, zlb, permanent), (zlb, permanent, permanent)),
        state_measurement,
        select_system=(1, 2, 2),
        information_sets=((1, 2, 3), (2, 3), (3,)),
        scenario_temporary_regimes=(((1, 2),), ((1,),)),
    )
    assert system[1, "CCC"][2] == -0.25
    np.testing.assert_array_equal(system[2, "TTT"], gensys(permanent).transition.TTT)


def test_model_measurement_and_pseudo_rows_respect_information_sets():
    first, second = Model1002("ss104"), Model1002("ss104")
    from dataclasses import replace

    second.parameters["psi1"] = replace(second.parameters["psi1"], value=2.0)
    myopic = compute_system(first, regime_models=(first, second))
    informed = compute_system(
        first, regime_models=(first, second), tvis=True, information_sets=((1, 2), (2,))
    )
    assert isinstance(informed, RegimeSwitchingSystem)
    assert isinstance(myopic, RegimeSwitchingSystem)
    rate_row = first.observables["obs_exp_nominalrate1"] - 1
    rate_state = first.indexes.endogenous_states["R_t"] - 1
    expected = informed[2, "TTT"][rate_state].copy()
    expected[first.indexes.endogenous_states_augmented["e_exp_rm1"] - 1] = 1.0
    np.testing.assert_array_equal(informed[1, "ZZ"][rate_row], expected)
    assert not np.array_equal(informed[1, "ZZ"][rate_row], myopic[1, "ZZ"][rate_row])
    long_row = first.observables["obs_longrate"] - 1
    assert not np.array_equal(informed[1, "ZZ"][long_row], myopic[1, "ZZ"][long_row])
    assert not np.array_equal(informed[1, "ZZ_pseudo"], myopic[1, "ZZ_pseudo"])


def test_regime_dates_and_revisiting_regimes():
    dates = {1: "2020-Q1", 2: "2020-07-01", 3: "2021-Q1"}
    np.testing.assert_array_equal(
        regime_indices_from_dates(dates, ["2020-Q1", "2020-Q3", "2020-Q4", "2021-Q2"]), [1, 2, 2, 3]
    )
    with pytest.raises(ValueError, match="precede"):
        regime_indices_from_dates(dates, ["2019-Q4"])
    result = kalman_log_likelihood(varied_systems(), np.zeros((3, 2)), regime_indices=[2, 1, 2])
    assert np.isfinite(result.log_likelihood)
    smooth_kalman_result(varied_systems(), result)


@pytest.mark.parametrize("indices", [[0], [3], [1.0], [True], [1, 2]])
def test_regime_filter_rejects_invalid_paths(indices):
    with pytest.raises(ValueError):
        kalman_log_likelihood(varied_systems(), np.zeros((1, 2)), regime_indices=indices)


@pytest.mark.parametrize("blocks", [((2,),), ((1,), (1,)), ((2, 1),), ((1, 3),), ((1,), (2,))])
def test_solver_rejects_invalid_temporary_blocks(blocks):
    with pytest.raises(ValueError):
        solve_regime_switching((policy_canonical(), policy_canonical()), temporary_regimes=blocks)


def test_invalid_information_and_smoother_paths_fail_explicitly():
    with pytest.raises(ValueError):
        compute_regime_system((scalar_canonical(0.2),), state_measurement, information_sets=((2,),))
    with pytest.raises(ValueError, match="tvis"):
        compute_system(Model1002(), tvis=True)
    with pytest.raises(ValueError, match="regime_indices"):
        kalman_log_likelihood(varied_systems(), np.zeros((2, 2)))
    result = kalman_log_likelihood(varied_systems(), np.zeros((2, 2)), regime_indices=[1, 2])
    with pytest.raises(ValueError, match="match"):
        smooth_kalman_result(varied_systems(), result, regime_indices=[1, 1])
    with pytest.raises(ValueError, match="Regime-filtered"):
        smooth_kalman_result(varied_systems().systems[0], result)
    invalid = policy_canonical()
    invalid.Gamma0[3, 1] = 2.0
    with pytest.raises(ValueError, match="unit forecast"):
        gensys_to_predictable_form(invalid)


def test_m1_small_julia_oracle():
    oracle = json.loads((Path(__file__).parent / "fixtures" / "ss104_m1_oracle.json").read_text())
    permanent = gensys(policy_canonical()).transition
    path = gensys2(policy_canonical(0.0, -0.25), permanent, periods=2)
    for i, t in enumerate(path):
        for name in ("TTT", "RRR", "CCC"):
            np.testing.assert_allclose(
                getattr(t, name), oracle["gensys2"][i][name], atol=1e-10, rtol=1e-10
            )
    data = np.array([[0.5, -0.2], [np.nan, 0.3], [0.2, np.nan], [np.nan, np.nan], [-0.4, 0.6]])
    filtered = kalman_log_likelihood(varied_systems(), data, regime_indices=[1, 1, 2, 2, 2])
    for name in (
        "log_likelihood_by_period",
        "predicted_states",
        "predicted_covariances",
        "filtered_states",
        "filtered_covariances",
    ):
        np.testing.assert_allclose(
            getattr(filtered, name), oracle["kalman"][name], atol=1e-10, rtol=1e-10
        )
    smoothed = smooth_kalman_result(varied_systems(), filtered)
    assert smoothed.smoothed_states is not None
    np.testing.assert_allclose(
        smoothed.smoothed_states, oracle["kalman"]["smoothed_states"], atol=1e-10, rtol=1e-10
    )
    ts = tuple(s.transition for s in varied_systems().systems)
    for key, helper in (
        ("expectations", k_periods_ahead_expectations),
        ("expected_sums", k_periods_ahead_expected_sums),
    ):
        power, constant = helper(ts, 3, information_set=(1, 2))
        np.testing.assert_allclose(power, oracle[key]["TTT"], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(constant, oracle[key]["CCC"], atol=1e-10, rtol=1e-10)
