"""Offline ss104 forecast at published-code defaults; no estimation or sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast_ss104_mode import BENCHMARK, annualized_growth, benchmark_path  # noqa: E402
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
from nydsge.ss104_workflow import forecast_ss104  # noqa: E402

PARAMETER_SOURCE = (
    "DSGE.jl ss104 defaults (the published code's calibration, not the FRBNY production posterior)"
)
DSGE_COMMIT = "c09dfe881f8809f65b8c0f08a761ec6cc456a6fb"
SSR_COMMIT = "794fb02ff45a62537d8d5d037c1d12b3aa7e4574"
OUTPUT = ROOT / "outputs/ss104_nyfed_params"
TABLES = ROOT / "data/estimates"


def forecast_history(model):
    """Preserve the modal run's Q2 population filter and append its partial Q3."""
    return pd.concat(
        (
            build_local_data(model, end="2026-Q2"),
            build_local_data(model).loc[[pd.Period("2026Q3", freq="Q")]],
        )
    )


def semi_conditioned_history(model, history):
    """Export the actual filter input, including the same semi-conditioning mask."""
    frame = history.copy()
    keep = set(model.get_setting("cond_semi_names")) | {
        f"obs_exp_nominalrate{i}" for i in range(1, 7)
    }
    frame.loc[frame.index >= "2026Q3", [k for k in model.observables if k not in keep]] = np.nan
    return frame


def population_growth(model):
    levels = pd.read_csv(FRED_SNAPSHOT)
    quarters = pd.PeriodIndex(levels.date, freq="Q")
    levels = levels.loc[
        levels.CNP16OV.notna()
        & (quarters >= pd.Period("1959Q2", freq="Q"))
        & (quarters <= pd.Period("2026Q2", freq="Q"))
    ].reset_index(drop=True)
    return float(prepare_population_data(model, levels).dlfiltered_population_recorded.iloc[-1])


def forecast_tables(pseudo, history, pop_quarter):
    """Apply the existing modal forecast's growth and population conventions."""
    dates = pd.period_range("2026Q3", periods=40, freq="Q")
    if not pseudo.index.equals(dates) or not np.isfinite(pseudo.to_numpy()).all():
        raise ValueError("Require 40 finite forecast quarters including conditional 2026Q3.")
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
    quarterly_gdp = pd.concat((history.loc[:"2026Q2", "obs_gdp"], pseudo.PseudoGDP))
    quarterly_gdp += pop_quarter
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
            values = series.reindex(quarters)
            if not np.isfinite(values).all():
                raise ValueError(f"Incomplete {name} growth in {year}; cannot calculate Q4/Q4.")
            row[f"{name}_ss104"] = float(100 * np.expm1(values.sum() / 100))
            row[f"{name}_nyfed"] = (
                float(published.loc[published_row, str(year)]) if str(year) in published else None
            )
        row["ffr_ss104_q4"] = float(comparison.loc[quarters[-1], "ffr_ss104"])
        row["output_gap_ss104_q4"] = float(comparison.loc[quarters[-1], "output_gap_ss104"])
        annual.append(row)
    return comparison, pd.DataFrame(annual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    args = parser.parse_args()
    # The requested documentation was absent when this parameter source was selected.
    # Do not silently continue using defaults if a posterior document is later supplied.
    if (ROOT / "docs/DSGE_Model_Documentation_1002.pdf").exists():
        raise ValueError("Review the newly supplied model PDF before selecting its parameters.")
    model = build_local_model()
    history = forecast_history(model)
    effective_history = semi_conditioned_history(model, history)
    system = compute_system(model)
    result = forecast_ss104(model, system=system, data=history, cond_type="semi", horizon=40)
    dates = pd.period_range("2026Q3", periods=40, freq="Q")
    cond_pseudo = pseudo_observables_from_states(
        system,
        result.conditional_states,
        regime_indices=regime_indices_from_dates(model.get_setting("regime_dates"), dates[:1]),
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
    pop_quarter = population_growth(model)
    quarterly, annual = forecast_tables(pseudo, history, pop_quarter)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / "transformed_observations.csv", index_label="date")
    effective_path = args.output_dir / "conditioned_observations.csv"
    effective_history.to_csv(effective_path, index_label="date")
    obs.to_csv(args.output_dir / "forecast_observables_model_units.csv", index_label="date")
    pseudo.to_csv(args.output_dir / "forecast_pseudo_model_units.csv", index_label="date")
    np.savez_compressed(
        args.output_dir / "forecast_arrays.npz",
        states=np.vstack((result.conditional_states, result.states)),
        observables=obs.to_numpy(),
        pseudo_observables=pseudo.to_numpy(),
    )
    quarterly.to_csv(
        args.tables_dir / "ss104_nyfed_params_forecast_quarterly.csv", index_label="quarter"
    )
    annual.to_csv(args.tables_dir / "ss104_nyfed_params_forecast_annual.csv", index=False)
    metadata = {
        "model": "m1002 ss104",
        "parameter_source": PARAMETER_SOURCE,
        "model_document": "docs/DSGE_Model_Documentation_1002.pdf absent from port checkout",
        "upstream_model_document_audit": {
            "path": "docs/DSGE_Model_Documentation_1002.pdf in pinned DSGE.jl source clone",
            "sha256": "66af4ff49fee147abc426eb71ace8a272454a618a7dce7ab26e7ba208e5a79a4",
            "printed_date": "2021-03-03",
            "posterior_parameter_table": False,
            "evidence": "Section 4, page 14; Appendix D, pages 26-28: prior distributions only",
        },
        "dsge_commit": DSGE_COMMIT,
        "statespace_commit": SSR_COMMIT,
        "as_of": AS_OF,
        "data_vintage": "261006",
        "history_start": str(history.index[0]),
        "mainsample_start": "1960Q1",
        "history_end": "2026Q2",
        "conditional_quarter": "2026Q3",
        "cond_type": "semi",
        "conditional_inputs": effective_history.loc["2026Q3"].dropna().to_dict(),
        "forecast_start": str(dates[0]),
        "forecast_end": str(dates[-1]),
        "horizon_including_conditional": 40,
        "future_quarters": len(result.states),
        "log_likelihood_including_condition": result.log_likelihood,
        "population_log_growth_quarterly_pp": pop_quarter,
        "population_log_growth_annual_pp": 4 * pop_quarter,
        "population_method": "Terminal local HP trend held constant; aggregate GDP approximation",
        "growth_units": "exact compounded annual percent; Q4/Q4 from summed quarterly log growth",
        "ffr_units": "annual percentage points (4 times quarterly rate)",
        "input_sha256": input_provenance(),
        "conditioned_observations_sha256": hashlib.sha256(effective_path.read_bytes()).hexdigest(),
        "history_regimes": regime_indices_from_dates(
            model.get_setting("regime_dates"), history.index
        ).tolist(),
        "parameters_by_regime": {
            str(r): {k: p.value for k, p in model.regime_snapshot(r).parameters.items()}
            for r in range(1, 30)
        },
        "estimation_runs": 0,
        "mcmc_runs": 0,
    }
    text = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    (args.output_dir / "forecast_metadata.json").write_text(text, encoding="utf-8")
    (args.tables_dir / "ss104_nyfed_params_forecast_metadata.json").write_text(
        text, encoding="utf-8"
    )
    print(PARAMETER_SOURCE)
    print(annual.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
