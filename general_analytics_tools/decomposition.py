"""Diagnostic multi-regime disturbance attribution using the native filter/smoother.

This is not a replacement for nydsge's single-regime historical_decomposition API.
Every contribution, initial condition, deterministic term and remainder is retained.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nydsge.kalman import KalmanResult, _default_initial_state_covariance
from nydsge.regimes import RegimeSwitchingSystem

HD_TOLERANCE = 1e-8  # Annualized log percentage points, never loosened by configuration.


def disturbance_attribution(
    system: RegimeSwitchingSystem,
    filtered: KalmanResult,
    smoothed_states: np.ndarray,
    observations: np.ndarray,
    regime_indices: np.ndarray,
    *,
    shock_covariances: np.ndarray,
    process_covariances: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """E[eps_t|Y] = Q_t R_t' r_t; verify against native smoothed states before attribution."""
    count, n = smoothed_states.shape
    path = system.period_systems(regime_indices, count)
    j = path[0].measurement.QQ.shape[0]
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
    m0, p0 = _default_initial_state_covariance(
        path[0], process_covariance=None if process_covariances is None else process_covariances[0]
    )
    initial = m0 + p0 @ path[0].transition.TTT.T @ information_vectors[0]
    deterministic = np.zeros(n)
    contributions = np.zeros((n, j))
    obs_initial, obs_deterministic, obs_contributions = [], [], []
    for t, active in enumerate(path):
        a, b, c = active.transition.TTT, active.transition.RRR, active.transition.CCC
        initial = a @ initial
        deterministic = a @ deterministic + c
        contributions = a @ contributions + b * shocks[t][None, :]
        obs_initial.append(active.measurement.ZZ @ initial)
        obs_deterministic.append(active.measurement.ZZ @ deterministic + active.measurement.DD)
        obs_contributions.append(active.measurement.ZZ @ contributions)
    fitted = np.array(
        [s.measurement.ZZ @ smoothed_states[t] + s.measurement.DD for t, s in enumerate(path)]
    )
    parts = np.asarray(obs_contributions)
    initial_obs, deterministic_obs = np.asarray(obs_initial), np.asarray(obs_deterministic)
    remainder = fitted - (parts.sum(axis=2) + initial_obs + deterministic_obs)
    return {
        "contributions": parts,
        "initial": initial_obs,
        "deterministic": deterministic_obs,
        "fitted": fitted,
        "shocks": shocks,
        "remainder": remainder,
        "measurement_gap": observations - fitted,
    }


def shock_groups(shocks: list[str]) -> dict[str, list[int]]:
    groups = {
        "Policy residual": [s for s in shocks if s.startswith(("rm_sh", "rm_ait_sh"))],
        "Price / wage markups": ["lambda_f_sh", "lambda_w_sh"],
        "Inflation target": ["pi_star_sh"],
        "Private demand": ["b_sh"],
        "Productivity": ["ztil_sh", "zp_sh"],
        "Investment / financial": ["mu_sh", "sigma_omega_sh", "mu_e_sh", "gamma_sh"],
        "Government spending": ["g_sh"],
        "Measurement / gap initialization": [
            "lr_sh",
            "tfp_sh",
            "gdpdef_sh",
            "corepce_sh",
            "gdp_sh",
            "gdi_sh",
            "meas_pi_sh",
            "pgap_sh",
            "ygap_sh",
            "condgdp_sh",
            "condcorepce_sh",
            *[f"exp_rm_sh{i}" for i in range(1, 7)],
        ],
    }
    indices = {
        label: [i for i, name in enumerate(shocks) if name in names]
        for label, names in groups.items()
    }
    used = {i for ids in indices.values() for i in ids}
    indices["COVID / other innovations"] = [i for i in range(len(shocks)) if i not in used]
    return indices


def export_decomposition(
    arrays: dict[str, np.ndarray], dates: pd.PeriodIndex, names: list[str], shocks: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation = {}
    for name in ("obs_corepce", "obs_gdp"):
        i = names.index(name)
        components = {
            label: 4 * arrays["contributions"][:, i, ids].sum(axis=1)
            for label, ids in shock_groups(shocks).items()
        }
        components.update(
            {
                "Initial state": 4 * arrays["initial"][:, i],
                "Deterministic terms": 4 * arrays["deterministic"][:, i],
                "Numerical remainder": 4 * arrays["remainder"][:, i],
            }
        )
        error = float(np.max(np.abs(components["Numerical remainder"])))
        validation[name] = {
            "max_reconstruction_error_annual_log_pp": error,
            "tolerance": HD_TOLERANCE,
            "status": "reconstruction_pass" if error <= HD_TOLERANCE else "diagnostic",
        }
        for label, values in components.items():
            rows.extend(
                {
                    "quarter": str(date),
                    "variable": name,
                    "component": label,
                    "contribution": float(value),
                }
                for date, value in zip(dates, values, strict=True)
            )
    return pd.DataFrame(rows), validation
