"""Native conditional disturbance attribution for dated regime-switching systems.

Uses the same Durbin--Koopman recursion as the native smoother and the explicitly
supplied effective Q_t (including dated zero-variance restrictions). Initial,
deterministic, measurement-gap and numerical-remainder terms stay separate.
This is a conditional structural-model attribution, not causal identification.
"""

from __future__ import annotations

import numpy as np

from nydsge.kalman import KalmanResult, _default_initial_state_covariance
from nydsge.regimes import RegimeSwitchingSystem


def regime_disturbance_attribution(
    system: RegimeSwitchingSystem,
    filtered: KalmanResult,
    smoothed_states: np.ndarray,
    observations: np.ndarray,
    regime_indices: np.ndarray,
    *,
    shock_covariances: np.ndarray,
    process_covariances: np.ndarray | None = None,
    initial_state: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """E[eps_t|Y] = Q_t R_t' r_t; verify against native smoothed states before attribution."""
    observations = np.asarray(observations, dtype=np.float64)
    smoothed_states = np.asarray(smoothed_states, dtype=np.float64)
    if smoothed_states.ndim != 2 or not np.isfinite(smoothed_states).all():
        raise ValueError("Smoothed states must be a finite periods x states array.")
    count, n = smoothed_states.shape
    if not count:
        raise ValueError("Attribution requires at least one filtered period.")
    if filtered.regime_indices is None or not np.array_equal(
        filtered.regime_indices, regime_indices
    ):
        raise ValueError("Attribution regimes must match the filter's regime_indices.")
    if observations.shape != (count, system.systems[0].measurement.ZZ.shape[0]):
        raise ValueError("Observations must match filtered periods and observables.")
    if np.isinf(observations).any():
        raise ValueError("Missing observations must be NaN, never infinite.")
    if filtered.predicted_states.shape != smoothed_states.shape:
        raise ValueError("Filter and smoother state shapes must agree.")
    if filtered.predicted_covariances.shape != (count, n, n):
        raise ValueError("Filter covariance shape must match smoothed states.")
    path = system.period_systems(regime_indices, count)
    j = path[0].measurement.QQ.shape[0]
    shock_covariances = np.asarray(shock_covariances, dtype=np.float64)
    if shock_covariances.shape != (count, j, j) or not np.isfinite(shock_covariances).all():
        raise ValueError("Shock covariances must be finite periods x shocks x shocks.")
    effective_process = np.array(
        [
            active.transition.RRR @ shock_covariances[t] @ active.transition.RRR.T
            for t, active in enumerate(path)
        ]
    )
    if process_covariances is not None:
        process_covariances = np.asarray(process_covariances, dtype=np.float64)
        if process_covariances.shape != (count, n, n) or not np.isfinite(process_covariances).all():
            raise ValueError("Process covariances must be finite periods x states x states.")
        if not np.allclose(process_covariances, effective_process, atol=1e-12, rtol=1e-12):
            raise ValueError(
                "Process covariance is inconsistent with the supplied shock covariance."
            )
    for covariance in shock_covariances:
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12):
            raise ValueError("Shock covariance must be symmetric.")
        if np.linalg.eigvalsh(covariance)[0] < -1e-12:
            raise ValueError("Shock covariance must be positive semidefinite.")
    m0, p0 = _default_initial_state_covariance(path[0], process_covariance=effective_process[0])
    m0 = m0 if initial_state is None else np.asarray(initial_state, dtype=np.float64)
    p0 = p0 if initial_covariance is None else np.asarray(initial_covariance, dtype=np.float64)
    if m0.shape != (n,) or p0.shape != (n, n):
        raise ValueError("Initial mean/covariance must match the state dimension.")
    if not np.isfinite(m0).all() or not np.isfinite(p0).all():
        raise ValueError("Initial mean/covariance must be finite.")
    first = path[0].transition
    if not np.allclose(
        filtered.predicted_states[0], first.TTT @ m0 + first.CCC, atol=1e-10, rtol=1e-10
    ):
        raise ValueError("Initial state does not match the filter initialization.")
    if not np.allclose(
        filtered.predicted_covariances[0],
        first.TTT @ p0 @ first.TTT.T + effective_process[0],
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError("Initial/process covariance does not match the filter initialization.")
    innovation_condition = np.zeros(count)
    r = np.zeros(n)
    information_vectors = np.zeros_like(smoothed_states)
    shocks = np.zeros((count, j))
    for t in range(count - 1, -1, -1):
        active = path[t]
        transition = path[min(t + 1, count - 1)].transition.TTT
        mask = np.isfinite(observations[t])
        z = active.measurement.ZZ[mask]
        e = active.measurement.EE[np.ix_(mask, mask)]
        p = filtered.predicted_covariances[t]
        if mask.any():
            f = z @ p @ z.T + e
            innovation_condition[t] = np.linalg.cond(f)
            innovation = observations[t, mask] - z @ filtered.predicted_states[t]
            innovation -= active.measurement.DD[mask]
            gain = transition @ p @ np.linalg.solve(f, z).T
            r = z.T @ np.linalg.solve(f, innovation) + (transition - gain @ z).T @ r
        else:
            r = transition.T @ r
        information_vectors[t] = r
        shocks[t] = shock_covariances[t] @ active.transition.RRR.T @ r
    implied = filtered.predicted_states + np.einsum(
        "tij,tj->ti", filtered.predicted_covariances, information_vectors
    )
    np.testing.assert_allclose(implied, smoothed_states, atol=1e-8, rtol=1e-10)
    initial = m0 + p0 @ path[0].transition.TTT.T @ information_vectors[0]
    deterministic = np.zeros(n)
    contributions = np.zeros((n, j))
    obs_initial, obs_deterministic, obs_contributions = [], [], []
    state_initial, state_deterministic, state_contributions = [], [], []
    for t, active in enumerate(path):
        a, b, c = active.transition.TTT, active.transition.RRR, active.transition.CCC
        initial = a @ initial
        deterministic = a @ deterministic + c
        contributions = a @ contributions + b * shocks[t][None, :]
        state_initial.append(initial.copy())
        state_deterministic.append(deterministic.copy())
        state_contributions.append(contributions.copy())
        obs_initial.append(active.measurement.ZZ @ initial)
        obs_deterministic.append(active.measurement.ZZ @ deterministic + active.measurement.DD)
        obs_contributions.append(active.measurement.ZZ @ contributions)
    fitted = np.array(
        [s.measurement.ZZ @ smoothed_states[t] + s.measurement.DD for t, s in enumerate(path)]
    )
    parts = np.asarray(obs_contributions)
    initial_obs, deterministic_obs = np.asarray(obs_initial), np.asarray(obs_deterministic)
    remainder = fitted - (parts.sum(axis=2) + initial_obs + deterministic_obs)
    state_reconstruction = (
        np.asarray(state_contributions).sum(axis=2)
        + np.asarray(state_initial)
        + np.asarray(state_deterministic)
    )
    # Never allocate numerical inconsistency to a structural innovation.
    # This remains an independent comparison against the supplied native smoother.
    return {
        "state_remainder": smoothed_states - state_reconstruction,
        "innovation_condition": innovation_condition,
        "smoother_information_error": implied - smoothed_states,
        "contributions": parts,
        "initial": initial_obs,
        "deterministic": deterministic_obs,
        "fitted": fitted,
        "shocks": shocks,
        "remainder": remainder,
        "measurement_gap": observations - fitted,
    }
