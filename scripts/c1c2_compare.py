"""Before/after metrics for docs/c1_c2_fixes.md (forecast, decomposition, long-run inflation).

BEFORE: desk-run observables + the desk run's 8-parameter refresh, n_ant = 6.
AFTER:  public-data observables + full C1/C2 mode, n_ant = 0, 2026Q3 semi-conditioning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1c2_common import build_model, build_observables, qidx  # noqa: E402

from nydsge.data import reverse_transform_observables, reverse_transform_pseudo_observables
from nydsge.economy import _restrict_conditioning_quarters
from nydsge.forecast import forecast_one, historical_decomposition
from nydsge.models import Model1002
from nydsge.solve import compute_system

DESK_DIR = Path(os.environ.get("NYDSGE_DESK_DIR", "/workspace/nydsge/outputs/desk_fomc_20260916"))
COND_OBS = ("obs_nominalrate", "obs_spread", "obs_longrate", "obs_longinflation")


def metrics(
    model: Model1002, obs: pd.DataFrame, *, report_quarter: str = "2026-Q2", horizon: int = 8
) -> dict:
    system = compute_system(model)
    fc = forecast_one(
        model,
        input_type="mode",
        cond_type="none",
        output_vars=["histstates", "histobs", "histpseudo", "forecastobs", "forecastpseudo"],
        check_empty_columns=False,
        horizon=horizon + 4,
        data=obs,
        history_method="smoothed",
    )
    names = list(model.observables)
    f_obs = reverse_transform_observables(model, np.asarray(fc.observables))
    h_obs = reverse_transform_observables(model, np.asarray(fc.history_observables))
    pnames = list(model.pseudo_observable_mappings)
    h_pseudo = reverse_transform_pseudo_observables(
        model, np.asarray(fc.history_pseudo_observables)
    )
    hist_dates = list(obs["date"])[-h_obs.shape[0] :]
    fstart = model.get_setting("date_forecast_start")
    fdates = [
        f"{(qidx(fstart) + i - 1) // 4}-Q{(qidx(fstart) + i - 1) % 4 + 1}"
        for i in range(f_obs.shape[0])
    ]
    # smoothed in-sample values of conditioned quarters (e.g. 2026-Q3) are reported as h1
    smoothed_obs = reverse_transform_observables(
        model, np.asarray(fc.history_states) @ system.measurement.ZZ.T + system.measurement.DD
    )
    path = {}
    for i, d in enumerate(hist_dates):
        if qidx(d) > qidx(report_quarter):
            path[d] = {
                n: float(smoothed_obs[i, names.index(n)])
                for n in ("obs_corepce", "obs_nominalrate", "obs_gdp")
            }
    for i, d in enumerate(fdates):
        path[d] = {
            n: float(f_obs[i, names.index(n)])
            for n in ("obs_corepce", "obs_nominalrate", "obs_gdp")
        }
    path = dict(list(path.items())[:horizon])
    ri = hist_dates.index(report_quarter)
    lri = float(h_pseudo[ri, pnames.index("LongRunInflation")])

    hd = historical_decomposition(model, data=obs, check_empty_columns=False)
    shocks = list(model.indexes.exogenous_shocks)
    T = list(obs["date"])[-hd.observed.shape[0] :].index(report_quarter)
    oi = names.index("obs_corepce")
    sm = hd.smoothed_observables[T, oi]
    scale = reverse_transform_observables(model, hd.smoothed_observables)[T, oi] / sm
    pce = {s: float(hd.observable_contributions[T, oi, j] * scale) for j, s in enumerate(shocks)}
    zzp = system.pseudo_measurement.ZZ_pseudo
    og_row = pnames.index("OutputGap")
    og = {s: float(zzp[og_row] @ hd.state_contributions[T, :, j]) for j, s in enumerate(shocks)}

    def pick(d: dict) -> dict:
        ant = [s for s in shocks if s.startswith("rm_shl")]
        return {
            "pi_star_sh": d["pi_star_sh"],
            "rm_sh": d["rm_sh"],
            "rm_shl1-6": float(sum(d[s] for s in ant)) if ant else 0.0,
            "lambda_f_sh": d["lambda_f_sh"],
            "lambda_w_sh": d["lambda_w_sh"],
            "b_sh": d["b_sh"],
            "g_sh": d["g_sh"],
            "ztil_sh+zp_sh": d["ztil_sh"] + d["zp_sh"],
            "financial(mu,sigma_omega,mu_e,gamma)": d["mu_sh"]
            + d["sigma_omega_sh"]
            + d["mu_e_sh"]
            + d["gamma_sh"],
            "measurement(corepce_sh etc.)": sum(
                d[s] for s in ("lr_sh", "tfp_sh", "gdpdef_sh", "corepce_sh", "gdp_sh", "gdi_sh")
            ),
        }

    og_base = float(
        zzp[og_row] @ hd.state_baseline[T] + system.pseudo_measurement.DD_pseudo[og_row]
    )
    return {
        "forecast": path,
        "long_run_inflation_2026Q2": lri,
        "smoothed_output_gap_2026Q2": float(h_pseudo[ri, pnames.index("OutputGap")]),
        "smoothed_corepce_2026Q2": float(
            reverse_transform_observables(model, hd.smoothed_observables)[T, oi]
        ),
        "corepce_decomp_2026Q2": pick(pce),
        "corepce_initial_and_trend": float(hd.observable_baseline[T, oi] * scale),
        "outputgap_decomp_2026Q2": pick(og),
        "outputgap_initial": og_base,
        "reconciliation": float(hd.reconciliation_max_abs_error),
    }


def before() -> dict:
    out = DESK_DIR  # read-only
    meta = json.loads((out / "run_metadata.json").read_text())
    model = Model1002("ss10", settings={"data_vintage": "2026q2", "date_forecast_start": "2026-Q3"})
    for k, v in meta["model"]["refresh"]["parameters"].items():
        model.parameters[k] = replace(model.parameters[k], value=float(v))
    obs = pd.read_csv(out / "observables.csv")
    obs = obs[(obs.date >= "1964-Q1") & (obs.date <= "2026-Q2")].reset_index(drop=True)
    return metrics(model, obs)


def before_new_vintage() -> dict:
    """Desk parameters and data scope on the 2026-10-06 FRED vintage (isolates revisions)."""

    out = DESK_DIR  # read-only
    meta = json.loads((out / "run_metadata.json").read_text())
    model = Model1002("ss10", settings={"data_vintage": "2026q2", "date_forecast_start": "2026-Q3"})
    for k, v in meta["model"]["refresh"]["parameters"].items():
        model.parameters[k] = replace(model.parameters[k], value=float(v))
    obs = build_observables(model, with_public=False, start="1964-Q1")
    return metrics(model, obs)


def after(mode_path: str, *, condition: bool = True) -> dict:
    payload = json.loads(Path(mode_path).read_text())
    fstart = "2026-Q4" if condition else "2026-Q3"
    model = build_model(payload["n_mon_anticipated_shocks"], date_forecast_start=fstart)
    for k in payload["estimated_parameters"]:
        model.parameters[k] = replace(
            model.parameters[k], value=float(payload["parameter_values"][k])
        )
    obs = build_observables(model, end="2026-Q3" if condition else "2026-Q2")
    masked = payload.get("mask_covid_quarters") or []
    if masked:
        cols = [c for c in obs.columns if c.startswith("obs_")]
        obs.loc[obs["date"].isin(masked), cols] = np.nan
    if condition:
        obs = _restrict_conditioning_quarters(obs, model_end_date="2026-Q2", keep=COND_OBS)
    return metrics(model, obs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-condition", action="store_true")
    ap.add_argument("--new-vintage", action="store_true")
    a = ap.parse_args()
    if a.mode is not None:
        res = after(a.mode, condition=not a.no_condition)
    else:
        res = before_new_vintage() if a.new_vintage else before()
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
