"""Compare package forecasts with the NY Fed DSGE September 2026 release (docs/nyfed_comparison.md).

Usage: uv run python scripts/c1c2_nyfed_compare.py <run_dir> [<run_dir> ...]

NY Fed files (data/public/nyfed_dsge_20260918/) were downloaded from
https://www.newyorkfed.org/medialibrary/research/interactives/data/dsge/ on 2026-10-06.
Package GDP is per-capita (obs_gdp); aggregate GDP adds the terminal HP-trend population
growth, held constant over the horizon (DSGE.jl instead uses a population forecast).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nydsge.data import prepare_population_data
from nydsge.models import Model1002

ROOT = Path(__file__).resolve().parents[1]
NYFED = ROOT / "data/public/nyfed_dsge_20260918"


def _q(label: str) -> str:
    d = pd.Timestamp(label)
    return f"{d.year}-Q{(d.month - 1) // 3 + 1}"


def nyfed() -> dict:
    pce = pd.read_csv(
        NYFED / "nyfed_Quarter-to-quarter_percentage_change_annualized_PseudoCorePCE.csv"
    )
    gdp = pd.read_csv(NYFED / "nyfed_Quarter-to-quarter_percentage_change_annualized_PseudoGDP.csv")
    rstar = pd.read_csv(NYFED / "nyfed_Percent_NaturalRate.csv")
    table = pd.read_csv(NYFED / "nyfed_dsgeTableData.csv", index_col=0)

    def path(frame: pd.DataFrame) -> dict:
        out = {}
        for _, row in frame.iterrows():
            v = row["mean_forecast_new"] if pd.notna(row["mean_forecast_new"]) else np.nan
            out[_q(row["dates"])] = float(v)
        return out

    return {
        "corepce_qq": path(pce),
        "gdp_qq": path(gdp),
        "rstar": path(rstar.rename(columns={"mean_forecast": "mean_forecast_new"})),
        "table": table.to_dict(),
    }


def q4q4(qq: dict, year: int) -> float:
    vals = [qq.get(f"{year}-Q{i}") for i in range(1, 5)]
    if any(v is None or not np.isfinite(v) for v in vals):
        return float("nan")
    return float(100 * (np.prod([(1 + v / 100) ** 0.25 for v in vals]) - 1))


def run_paths(run: Path) -> dict:
    hist = pd.read_csv(run / "history_observables_reporting_units.csv")
    base = pd.read_csv(run / "baseline_observables_wide.csv")
    hp = pd.read_csv(run / "history_pseudo_observables_reporting_units.csv")
    bp = pd.read_csv(run / "baseline_pseudo_observables_wide.csv")
    levels = pd.read_csv(run / "fred_levels.csv")
    levels = levels.loc[levels["CNP16OV"].notna()].reset_index(drop=True)
    pop = prepare_population_data(Model1002("ss10"), levels)
    pop_growth_ann = float(4 * pop["dlfiltered_population_recorded"].iloc[-1])
    meta = json.loads((run / "run_metadata.json").read_text())
    data_end = meta["configuration"].get("model_end_date")
    qq_pce, qq_gdp, rstar = {}, {}, {}
    for frame in (hist, base):
        for _, r in frame.iterrows():
            if pd.notna(r.get("obs_corepce")):
                qq_pce[r["date"]] = float(r["obs_corepce"])
            if pd.notna(r.get("obs_gdp")):
                pc = float(r["obs_gdp"])
                qq_gdp[r["date"]] = float(
                    400 * (np.exp(np.log1p(pc / 400) + pop_growth_ann / 400) - 1)
                )
    # conditioned quarters (data on rates/spreads only) carry the smoothed model value
    decomp_path = run / "historical_decomposition_grouped.csv"
    if decomp_path.exists():
        dec = pd.read_csv(decomp_path)
        dec = dec.loc[dec["variable_kind"] == "observable"].drop_duplicates(["date", "variable"])
        for _, r in dec.iterrows():
            if r["variable"] == "obs_corepce" and r["date"] not in qq_pce:
                qq_pce[r["date"]] = float(r["smoothed"])
            if r["variable"] == "obs_gdp" and r["date"] not in qq_gdp:
                pc = float(r["smoothed"])
                qq_gdp[r["date"]] = float(
                    400 * (np.exp(np.log1p(pc / 400) + pop_growth_ann / 400) - 1)
                )
    for frame in (hp, bp):
        for _, r in frame.iterrows():
            rstar[r["date"]] = float(r["NaturalRate"])
    return {
        "corepce_qq": qq_pce,
        "gdp_qq": qq_gdp,
        "rstar": rstar,
        "pop_growth_ann": pop_growth_ann,
        "data_end": data_end,
    }


def main() -> None:
    ny = nyfed()
    out = {"nyfed": {}, "runs": {}}
    for year in (2026, 2027, 2028):
        out["nyfed"][year] = {
            "corepce_q4q4_table": float(ny["table"][str(year)]["Core PCE inflation (Q4/Q4)"]),
            "gdp_q4q4_table": float(ny["table"][str(year)]["GDP growth (Q4/Q4)"]),
            "rstar_q4_table": float(ny["table"][str(year)]["Real natural rate of interest (Q4)"]),
            "corepce_q4q4_from_quarterly": q4q4(
                {**{k: v for k, v in ny["corepce_qq"].items()}, **{}}, year
            ),
        }
    out["nyfed_corepce_qq"] = {k: v for k, v in ny["corepce_qq"].items() if k >= "2026-Q3"}
    out["nyfed_gdp_qq"] = {k: v for k, v in ny["gdp_qq"].items() if k >= "2026-Q3"}
    for arg in sys.argv[1:]:
        p = run_paths(Path(arg))
        res = {"pop_growth_ann": p["pop_growth_ann"]}
        for year in (2026, 2027, 2028):
            res[year] = {
                "corepce_q4q4": q4q4(p["corepce_qq"], year),
                "gdp_q4q4_aggregate_approx": q4q4(p["gdp_qq"], year),
                "rstar_q4": p["rstar"].get(f"{year}-Q4"),
            }
        res["corepce_qq"] = {
            k: v for k, v in p["corepce_qq"].items() if "2026-Q3" <= k <= "2028-Q4"
        }
        res["gdp_qq_aggregate_approx"] = {
            k: v for k, v in p["gdp_qq"].items() if "2026-Q3" <= k <= "2028-Q4"
        }
        out["runs"][arg] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
