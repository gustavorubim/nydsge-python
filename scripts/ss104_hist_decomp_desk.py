#!/usr/bin/env python3
"""Desk-local ss104 historical shock decomposition at DSGE.jl ss104 defaults.

Untracked desk helper (not part of the upstream package; not Julia-validated).
Uses the same inputs, semi conditioning and filter as scripts/forecast_ss104_nyfed_params.py:
history through 2026Q2 plus the semi-conditioned 2026Q3 row. Smoothed structural shocks are
recovered per period as eps_t = Q_t R_t' (R_t Q_t R_t')^+ u_t, u_t = s_t - C_t - T_t s_{t-1};
contributions are propagated through the dated regime transitions. Initial conditions+trend is
the zero-shock path from the smoothed first state. Any unreconciled piece is reported as
"Observed-model residual". Output units: obs_* in model units x4 (annualized log-percent).

  uv run python scripts/ss104_hist_decomp_desk.py --out-dir outputs/<run>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_ss104_nyfed_params import forecast_history  # noqa: E402
from ss104_local import build_local_model  # noqa: E402

from nydsge.kalman import (  # noqa: E402
    kalman_log_likelihood,
    model_process_covariances,
    smooth_kalman_result,
)
from nydsge.regimes import RegimeSwitchingSystem, regime_indices_from_dates  # noqa: E402
from nydsge.solve import compute_system  # noqa: E402
from nydsge.ss104_workflow import prepare_ss104_observations  # noqa: E402

VARIABLES = ("obs_corepce", "obs_gdp", "obs_nominalrate")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tail-quarters", type=int, default=24)
    args = p.parse_args()
    model = build_local_model()  # code defaults; no saved estimate applied
    system = compute_system(model)
    if not isinstance(system, RegimeSwitchingSystem):
        raise TypeError("ss104 historical decomposition requires a RegimeSwitchingSystem.")
    frame = prepare_ss104_observations(model, forecast_history(model))
    start = pd.Period(model.get_setting("date_forecast_start"), freq="Q")
    keep = set(model.get_setting("cond_semi_names")) | {
        f"obs_exp_nominalrate{i}" for i in range(1, 7)
    }
    frame.loc[frame.index >= start, [k for k in model.observables if k not in keep]] = np.nan
    data = frame.to_numpy(dtype=float)
    idx = regime_indices_from_dates(model.get_setting("regime_dates"), frame.index)
    T = len(frame)
    pc = model_process_covariances(model, system, T, start_date=frame.index[0])
    filt = kalman_log_likelihood(
        system, data, regime_indices=idx, log_likelihood_start=2, process_covariances=pc
    )
    S = smooth_kalman_result(system, filt, data=data).smoothed_states
    if S is None:
        raise RuntimeError("Historical decomposition smoother did not return states.")
    path = system.period_systems(idx, T)
    names = list(model.indexes.exogenous_shocks)
    n, k = S.shape[1], len(names)
    base = np.zeros((T, n))
    contrib = np.zeros((T, k, n))
    eps = np.zeros((T, k))
    base[0] = S[0]
    max_shock_err = 0.0
    for t in range(1, T):
        tr = path[t].transition
        R = tr.RRR
        Q = path[t].measurement.QQ.copy()
        if pc is not None and not np.allclose(R @ Q @ R.T, pc[t], atol=1e-12):
            # Pre-ZLB periods: anticipated policy shocks have zero variance (Julia convention).
            for j, s in enumerate(names):
                if s.startswith("rm_shl"):
                    Q[j, :] = 0.0
                    Q[:, j] = 0.0
            if not np.allclose(R @ Q @ R.T, pc[t], atol=1e-10):
                raise ValueError(f"Cannot reproduce process covariance at period {t}")
        u = S[t] - tr.CCC - tr.TTT @ S[t - 1]
        eps[t] = Q @ R.T @ np.linalg.pinv(R @ Q @ R.T, rcond=1e-12) @ u
        max_shock_err = max(max_shock_err, float(np.max(np.abs(R @ eps[t] - u))))
        base[t] = tr.CCC + tr.TTT @ base[t - 1]
        contrib[t] = (tr.TTT @ contrib[t - 1].T).T + (R * eps[t]).T
    rows = []
    obs_names = list(model.observables)
    tail = frame.index[-args.tail_quarters :]
    max_recon = 0.0
    for t in range(T - args.tail_quarters, T):
        m = path[t].measurement
        for v in VARIABLES:
            i = obs_names.index(v)
            sm = float(m.ZZ[i] @ S[t] + m.DD[i])
            b = float(m.ZZ[i] @ base[t] + m.DD[i])
            c = contrib[t] @ m.ZZ[i]
            resid = sm - b - c.sum()
            max_recon = max(max_recon, abs(resid))
            q = str(frame.index[t])
            rows.append((q, v, "IC+trend", 4 * b, 4 * sm))
            rows.extend((q, v, s, 4 * float(x), 4 * sm) for s, x in zip(names, c, strict=True))
            rows.append((q, v, "Observed-model residual", 4 * resid, 4 * sm))
    out = pd.DataFrame(rows, columns=["date", "variable", "shock", "contribution", "smoothed"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for v in VARIABLES:
        out[out.variable == v].drop(columns="variable").to_csv(
            args.out_dir / f"hist_decomp_per_shock_{v.removeprefix('obs_')}.csv", index=False
        )
    print(
        f"tail {tail[0]}-{tail[-1]}; shock-recovery max err {max_shock_err:.2e}; "
        f"obs reconciliation max err {max_recon:.2e}"
    )


if __name__ == "__main__":
    main()
