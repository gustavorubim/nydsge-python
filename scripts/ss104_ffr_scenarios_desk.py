#!/usr/bin/env python3
"""Desk-local ss104 fed-funds scenarios at DSGE.jl ss104 defaults (untracked helper).

Same model, data vintage, semi conditioning and growth conventions as
scripts/forecast_ss104_nyfed_params.py. Baseline = zero future shocks from the
2026Q3 smoothed state. Scenarios add policy innovations from 2026Q4; the system is
linear, so deviations are exact differences from baseline.

Policy shock choice: in ss104 regimes >= 8 DSGE.jl zeroes rm_sh's loading in the
policy rule (remove_rm_t_shocks = 8); 2026Q4+ is regime 29 (flexible AIT, full
credibility). The live unanticipated policy shock is rm_ait_sh and the anticipated
(forward-guidance) shocks are rm_ait_shl1..6. rm_sh is run too, to document that it is inert.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_ss104_mode import annualized_growth  # noqa: E402
from forecast_ss104_nyfed_params import forecast_history, population_growth  # noqa: E402
from ss104_local import build_local_model  # noqa: E402

from nydsge.solve import compute_system  # noqa: E402
from nydsge.ss104_workflow import forecast_ss104  # noqa: E402

H = 40  # incl. conditional 2026Q3 -> 39 future quarters from 2026Q4
END = "2028Q4"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--peg-level", type=float, default=3.375)
    p.add_argument("--peg-end", default="2028Q4")
    a = p.parse_args()
    model = build_local_model()
    system = compute_system(model)
    history = forecast_history(model)
    shocks_names = list(model.indexes.exogenous_shocks)
    nfut = H - 1
    fut_dates = pd.period_range("2026Q4", periods=nfut, freq="Q")
    pop = population_growth(model)
    obs_names, ps_names = list(model.observables), list(model.pseudo_observables)

    def run(eps):
        r = forecast_ss104(
            model, system=system, data=history, cond_type="semi", horizon=H, shocks=eps
        )
        o = pd.DataFrame(r.observables, index=fut_dates, columns=obs_names)
        q = pd.DataFrame(r.pseudo_observables, index=fut_dates, columns=ps_names)
        return pd.DataFrame(
            {
                "core_pce": annualized_growth(q.PseudoCorePCE),
                "gdp_deflator": annualized_growth(o.obs_gdpdeflator),
                "ffr": 4 * q.NominalFFR,
                "gdp": annualized_growth(q.PseudoGDP + pop),
                "output_gap": q.OutputGap,
            }
        )

    def unit(name, t0=0):
        e = np.zeros((nfut, len(shocks_names)))
        e[t0, shocks_names.index(name)] = 1.0
        return e

    base = run(None)
    zero = np.zeros((nfut, len(shocks_names)))
    # Per-unit FFR impact responses (linear), used to scale shocks to a target FFR move.
    irf = {n: run(unit(n)) - base for n in ("rm_sh", "rm_ait_sh")}
    ffr_unit_impact = float(irf["rm_ait_sh"].ffr.iloc[0])
    rm_sh_max_effect = float(irf["rm_sh"].abs().to_numpy().max())
    scen = {}
    for bp in (-50, -25, 25, 50):
        e = zero.copy()
        e[0, shocks_names.index("rm_ait_sh")] = (bp / 100) / ffr_unit_impact
        scen[f"unant_{bp:+d}bp_2026Q4"] = (run(e), e[0, shocks_names.index("rm_ait_sh")])
    # Forward guidance: rm_ait_shl1 realised in 2026Q4 = policy-rate innovation dated 2027Q1,
    # announced in 2026Q4. Scale so the 2027Q1 FFR deviation is +25bp.
    fg_unit = run(unit("rm_ait_shl1")) - base
    fg_scale = 0.25 / float(fg_unit.ffr.iloc[1])
    e = zero.copy()
    e[0, shocks_names.index("rm_ait_shl1")] = fg_scale
    scen["fg_+25bp_2027Q1_announced_2026Q4"] = (run(e), fg_scale)
    # Peg: sequence of unanticipated rm_ait_sh surprises (one per quarter) solved recursively so
    # FFR equals the peg level each quarter 2026Q4..peg_end; free (rule) afterwards.
    npeg = len(pd.period_range("2026Q4", a.peg_end, freq="Q"))
    j = shocks_names.index("rm_ait_sh")
    e = zero.copy()
    for t in range(npeg):
        cur = run(e)
        resp = (run(e + unit("rm_ait_sh", t)) - cur).ffr.iloc[t]
        e[t, j] += (a.peg_level - cur.ffr.iloc[t]) / resp
    peg = run(e)
    scen[f"peg_{a.peg_level}_2026Q4-{a.peg_end}"] = (peg, float(np.abs(e[:npeg, j]).max()))
    rows = []
    for name, (lv, size) in [("baseline", (base, 0.0)), *scen.items()]:
        for d in lv.loc[:END].index:
            for v in lv.columns:
                rows.append(
                    {
                        "scenario": name,
                        "quarter": str(d),
                        "variable": v,
                        "level": lv.loc[d, v],
                        "baseline": base.loc[d, v],
                        "deviation": lv.loc[d, v] - base.loc[d, v],
                        "shock_size_model_units": size,
                    }
                )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out_dir / "scenarios_levels_deviations.csv", index=False)
    (a.out_dir / "notes.txt").write_text(
        f"rm_ait_sh FFR impact per unit (pp ann.): {ffr_unit_impact:.6f}\n"
        f"rm_sh max abs effect on any reported variable: {rm_sh_max_effect:.3e}\n"
        f"rm_ait_shl1 scale for +25bp 2027Q1: {fg_scale:.6f}; "
        f"its 2026Q4 FFR dev {fg_unit.ffr.iloc[0] * fg_scale:.4f}\n"
        f"peg max |shock| (model units): {float(np.abs(e[:npeg, j]).max()):.4f}; "
        f"peg shocks: {np.round(e[:npeg, j], 4).tolist()}\n"
    )
    print((a.out_dir / "notes.txt").read_text())


if __name__ == "__main__":
    main()
