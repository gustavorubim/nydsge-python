"""IRF summaries for docs/nyfed_comparison.md (BEFORE desk parameters vs AFTER C1/C2 mode).

All observables are reported in annualized percentage points (x4 for quarterly-rate
observables); obs_hours is a log level (percent deviation)."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1c2_common import build_model  # noqa: E402

from nydsge.models import Model1002
from nydsge.solve import compute_system

H = 41
DESK_DIR = Path(os.environ.get("NYDSGE_DESK_DIR", "/workspace/nydsge/outputs/desk_fomc_20260916"))


def irf(system, shock_index: int, size: float, *, pseudo: bool = False) -> np.ndarray:
    ttt, rrr = system.transition.TTT, system.transition.RRR
    zz = system.pseudo_measurement.ZZ_pseudo if pseudo else system.measurement.ZZ
    x = rrr[:, shock_index] * size
    out = []
    for _ in range(H):
        out.append(zz @ x)
        x = ttt @ x
    return np.asarray(out)


def summarize(model: Model1002) -> dict:
    s = compute_system(model)
    obs = list(model.observables)
    pse = list(model.pseudo_observable_mappings)
    shocks = list(model.indexes.exogenous_shocks)
    sd = np.sqrt(np.diag(s.measurement.QQ))
    o = {n: obs.index(n) for n in obs}
    res: dict = {}

    def pk(path: np.ndarray, sign: float = 1.0) -> tuple[float, int]:
        i = int(np.argmax(sign * path))
        return float(path[i]), i

    # Monetary policy: unanticipated shock scaled to -50 bp annualized FFR on impact
    j = shocks.index("rm_sh")
    unit = irf(s, j, 1.0)
    size = -0.50 / (4 * unit[0, o["obs_nominalrate"]])
    y = irf(s, j, size)
    ffr = 4 * y[:, o["obs_nominalrate"]]
    half = int(np.argmax(np.abs(ffr) < 0.5 * abs(ffr[0])))
    hours_pk, hours_t = pk(y[:, o["obs_hours"]])
    pce_pk, pce_t = pk(4 * y[:, o["obs_corepce"]])
    res["rm_sh_-50bp"] = {
        "size_in_sd": float(size / sd[j]),
        "ffr_impact_pp": float(ffr[0]),
        "ffr_half_life_q": half,
        "ffr_q_to_within_10pct": int(np.argmax(np.abs(ffr) < 0.1 * abs(ffr[0]))),
        "gdp_growth_impact_pp_ann": float(4 * y[0, o["obs_gdp"]]),
        "hours_peak_pct": hours_pk,
        "hours_peak_q": hours_t,
        "corepce_peak_pp_ann": pce_pk,
        "corepce_peak_q": pce_t,
    }
    # 1 sd unanticipated policy shock, and the +25 bp scaling
    res["rm_sh_1sd_ffr_impact_pp"] = float(4 * unit[0, o["obs_nominalrate"]] * sd[j])
    if "rm_shl4" in shocks:
        j4 = shocks.index("rm_shl4")
        u4 = irf(s, j4, 1.0)
        scale4 = (
            -0.50 / (4 * u4[4, o["obs_nominalrate"]]) if u4[4, o["obs_nominalrate"]] != 0 else 0
        )
        y4 = irf(s, j4, scale4)
        res["rm_shl4_-50bp_at_h4"] = {
            "ffr_impact_pp_h0": float(4 * y4[0, o["obs_nominalrate"]]),
            "gdp_growth_impact_pp_ann": float(4 * y4[0, o["obs_gdp"]]),
            "hours_peak_pct": pk(y4[:, o["obs_hours"]])[0],
        }
    # Spread shock 1 sd
    j = shocks.index("sigma_omega_sh")
    y = irf(s, j, sd[j])
    sp = 4 * y[:, o["obs_spread"]]
    res["sigma_omega_1sd"] = {
        "spread_impact_bp": float(100 * sp[0]),
        "spread_peak_bp": float(100 * sp.max()),
        "peak_q": int(sp.argmax()),
    }
    # pi* shock 1 sd
    j = shocks.index("pi_star_sh")
    y = irf(s, j, sd[j])
    yp = irf(s, j, sd[j], pseudo=True)
    pce = 4 * y[:, o["obs_corepce"]]
    lri = 4 * yp[:, pse.index("LongRunInflation")] if "LongRunInflation" in pse else None
    res["pi_star_1sd"] = {
        "corepce_h0_h4_h20_pp_ann": [float(pce[0]), float(pce[4]), float(pce[20])],
        "corepce_peak_pp_ann": pk(pce)[0],
        "corepce_peak_q": pk(pce)[1],
        "ffr_h0_h4_pp": [
            float(4 * y[0, o["obs_nominalrate"]]),
            float(4 * y[4, o["obs_nominalrate"]]),
        ],
        "output_gap_peak": pk(yp[:, pse.index("OutputGap")])[0],
        "output_gap_peak_q": pk(yp[:, pse.index("OutputGap")])[1],
        "pi_over_pistar_h10": float(pce[10] / (4 * yp[10, pse.index("LongRunInflation")]))
        if lri is not None
        else None,
    }
    # price markup 1 sd
    j = shocks.index("lambda_f_sh")
    y = irf(s, j, sd[j])
    pce = 4 * y[:, o["obs_corepce"]]
    res["lambda_f_1sd"] = {
        "corepce_impact_pp_ann": float(pce[0]),
        "corepce_peak_q": int(np.argmax(pce)),
        "corepce_h4_pp_ann": float(pce[4]),
        "gdp_growth_impact_pp_ann": float(4 * y[0, o["obs_gdp"]]),
        "ffr_peak_pp": pk(4 * y[:, o["obs_nominalrate"]])[0],
        "ffr_peak_q": pk(4 * y[:, o["obs_nominalrate"]])[1],
    }
    return res


def before_model() -> Model1002:
    meta = json.loads((DESK_DIR / "run_metadata.json").read_text())
    m = Model1002("ss10", settings={"date_forecast_start": "2026-Q3"})
    for k, v in meta["model"]["refresh"]["parameters"].items():
        m.parameters[k] = replace(m.parameters[k], value=float(v))
    return m


def after_model(mode_path: str) -> Model1002:
    payload = json.loads(Path(mode_path).read_text())
    m = build_model(payload["n_mon_anticipated_shocks"])
    for k in payload["estimated_parameters"]:
        m.parameters[k] = replace(m.parameters[k], value=float(payload["parameter_values"][k]))
    return m


if __name__ == "__main__":
    out = {
        "before_desk": summarize(before_model()),
        "dsgejl_defaults": summarize(Model1002("ss10")),
    }
    if len(sys.argv) > 1:
        out["after"] = summarize(after_model(sys.argv[1]))
    print(json.dumps(out, indent=1))
