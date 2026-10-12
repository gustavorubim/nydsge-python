"""Offline 40-quarter ss104 modal forecast and cached NY Fed comparison tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ss104_local import (  # noqa: E402
    AS_OF,
    FRED_SNAPSHOT,
    ROOT,
    build_local_data,
    build_local_model,
    input_provenance,
)

from nydsge.data import prepare_population_data  # noqa: E402
from nydsge.kalman import pseudo_observables_from_states  # noqa: E402
from nydsge.regimes import regime_indices_from_dates  # noqa: E402
from nydsge.solve import compute_system  # noqa: E402
from nydsge.ss104_estimation import ss104_parameter_catalog, update_ss104_parameters  # noqa: E402
from nydsge.ss104_workflow import forecast_ss104  # noqa: E402

BENCHMARK = ROOT / "data/public/nyfed_dsge_20260918"


def annualized_growth(quarterly_log_growth):
    """Exact annual compounding of quarterly log-percent growth."""
    return 100 * np.expm1(4 * np.asarray(quarterly_log_growth) / 100)


def benchmark_path(variable):
    path = BENCHMARK / f"nyfed_Quarter-to-quarter_percentage_change_annualized_{variable}.csv"
    data = pd.read_csv(path)
    return pd.Series(
        data.mean_forecast_new.to_numpy(),
        index=pd.PeriodIndex(pd.to_datetime(data.dates), freq="Q"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=Path, default=Path("data/estimates/m1002_ss104_mode.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ss104_m7"))
    parser.add_argument("--tables-dir", type=Path, default=Path("data/estimates"))
    args = parser.parse_args()
    mode = json.loads(args.mode.read_text(encoding="utf-8"))
    if mode["status"] == "running" or mode["subspec"] != "ss104":
        raise ValueError("Require a finished ss104 mode-search artifact.")
    if mode["input_sha256"] != input_provenance():
        raise ValueError("Local inputs differ from the estimation artifact.")
    model = build_local_model()
    catalog = ss104_parameter_catalog(model)
    if list(catalog) != mode["estimated_parameters"]:
        raise ValueError("Mode parameter regime coordinates do not match ss104.")
    update_ss104_parameters(model, catalog, mode["estimation_values"])
    # Reusing the estimation history avoids revising its HP population filter
    # merely because a partial conditional quarter exists in the snapshot.
    history = pd.concat(
        (
            build_local_data(model, end="2026-Q2"),
            build_local_data(model).loc[[pd.Period("2026Q3", freq="Q")]],
        )
    )
    system = compute_system(model)
    result = forecast_ss104(model, system=system, data=history, cond_type="semi", horizon=40)
    dates = pd.period_range("2026Q3", periods=40, freq="Q")
    cond_regimes = regime_indices_from_dates(model.get_setting("regime_dates"), dates[:1])
    cond_pseudo = pseudo_observables_from_states(
        system, result.conditional_states, regime_indices=cond_regimes
    )
    pseudo = pd.DataFrame(
        np.vstack((cond_pseudo, result.pseudo_observables)),
        index=dates,
        columns=list(model.pseudo_observables),
    )
    obs = pd.DataFrame(
        np.vstack((result.conditional_observables, result.observables)),
        index=dates,
        columns=list(model.observables),
    )
    levels = pd.read_csv(FRED_SNAPSHOT)
    quarters = pd.PeriodIndex(levels.date, freq="Q")
    levels = levels.loc[
        levels.CNP16OV.notna()
        & (quarters >= pd.Period("1959Q2", freq="Q"))
        & (quarters <= pd.Period("2026Q2", freq="Q"))
    ].reset_index(drop=True)
    population = prepare_population_data(model, levels)
    pop_quarter = float(population.dlfiltered_population_recorded.iloc[-1])
    comparison = pd.DataFrame(
        {
            "gdp_ss104": annualized_growth(pseudo.PseudoGDP + pop_quarter),
            "gdp_nyfed": benchmark_path("PseudoGDP").reindex(dates),
            "corepce_ss104": annualized_growth(pseudo.PseudoCorePCE),
            "corepce_nyfed": benchmark_path("PseudoCorePCE").reindex(dates),
            "ffr_ss104": 4 * pseudo.NominalFFR,
            "ffr_nyfed": np.nan,
            "output_gap_ss104": pseudo.OutputGap,
            "output_gap_nyfed": np.nan,
        },
        index=dates,
    )
    quarterly_gdp = pd.concat((history.loc[:"2026Q2", "obs_gdp"], pseudo.PseudoGDP)) + pop_quarter
    quarterly_pce = pd.concat((history.loc[:"2026Q2", "obs_corepce"], pseudo.PseudoCorePCE))
    published = pd.read_csv(BENCHMARK / "nyfed_dsgeTableData.csv", index_col=0)
    annual = []
    for year in range(2026, 2036):
        quarters = pd.period_range(f"{year}Q1", f"{year}Q4", freq="Q")
        row = {"year": year}
        for name, series, published_row in (
            ("gdp", quarterly_gdp, "GDP growth (Q4/Q4)"),
            ("corepce", quarterly_pce, "Core PCE inflation (Q4/Q4)"),
        ):
            row[f"{name}_ss104"] = float(100 * np.expm1(series.reindex(quarters).sum() / 100))
            row[f"{name}_nyfed"] = (
                float(published.loc[published_row, str(year)]) if str(year) in published else None
            )
        row["ffr_ss104_q4"] = float(comparison.loc[quarters[-1], "ffr_ss104"])
        row["output_gap_ss104_q4"] = float(comparison.loc[quarters[-1], "output_gap_ss104"])
        annual.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / "transformed_observations.csv", index_label="date")
    obs.to_csv(args.output_dir / "forecast_observables_model_units.csv", index_label="date")
    pseudo.to_csv(args.output_dir / "forecast_pseudo_model_units.csv", index_label="date")
    states = np.vstack((result.conditional_states, result.states))
    np.savez_compressed(
        args.output_dir / "forecast_arrays.npz",
        states=states,
        observables=obs.to_numpy(),
        pseudo_observables=pseudo.to_numpy(),
    )
    comparison.to_csv(args.tables_dir / "ss104_forecast_quarterly.csv", index_label="quarter")
    pd.DataFrame(annual).to_csv(args.tables_dir / "ss104_forecast_annual.csv", index=False)
    metadata = {
        "model": "m1002 ss104",
        "mode": str(args.mode),
        "mode_status": mode["status"],
        "as_of": AS_OF,
        "history_end": "2026Q2",
        "conditional_quarter": "2026Q3",
        "cond_type": "semi",
        "conditional_inputs": {
            k: float(history.loc["2026Q3", k])
            for k in (
                *model.get_setting("cond_semi_names"),
                *(f"obs_exp_nominalrate{i}" for i in range(1, 7)),
            )
            if pd.notna(history.loc["2026Q3", k])
        },
        "forecast_start": str(dates[0]),
        "forecast_end": str(dates[-1]),
        "horizon_including_conditional": 40,
        "future_quarters": len(result.states),
        "log_likelihood_including_condition": result.log_likelihood,
        "population_log_growth_annual_pp": 4 * pop_quarter,
        "population_method": "Terminal local HP trend held constant; aggregate GDP approximation",
        "growth_units": "exact compounded annual percent; Q4/Q4 from summed quarterly log growth",
        "ffr_units": "annual percentage points (4 times quarterly rate)",
        "gap_units": "log percentage points, y_t - y_f_t; not the cumulative AIT ygap_t",
        "input_sha256": input_provenance(),
        "mcmc_runs": 0,
    }
    (args.output_dir / "forecast_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(pd.DataFrame(annual).head(4).to_string(index=False))


if __name__ == "__main__":
    main()
