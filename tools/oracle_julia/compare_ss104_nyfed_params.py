"""Export Julia paths and compare the same-input fixed-parameter local forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from forecast_ss104_nyfed_params import OUTPUT, TABLES, forecast_tables  # noqa: E402


def compare(oracle, *, input_dir=OUTPUT, tables_dir=TABLES, tolerance=1e-5):
    meta = json.loads((input_dir / "forecast_metadata.json").read_text(encoding="utf-8"))
    effective_path = input_dir / "conditioned_observations.csv"
    data_hash = hashlib.sha256(effective_path.read_bytes()).hexdigest()
    history = pd.read_csv(input_dir / "transformed_observations.csv", index_col="date")
    history.index = pd.PeriodIndex(history.index, freq="Q")
    effective = pd.read_csv(effective_path, index_col="date")
    effective.index = pd.PeriodIndex(effective.index, freq="Q")
    ours = pd.read_csv(input_dir / "forecast_pseudo_model_units.csv", index_col="date")
    ours.index = pd.PeriodIndex(ours.index, freq="Q")
    with h5py.File(oracle) as f:
        julia_meta = json.loads(f["metadata/json"].asstr()[()])
        for key in ("dsge_commit", "statespace_commit", "parameter_source", "cond_type"):
            if julia_meta[key] != meta[key]:
                raise ValueError(f"Julia and Python {key} differ")
        if (
            data_hash != meta["conditioned_observations_sha256"]
            or data_hash != julia_meta["conditioned_observations_sha256"]
            or julia_meta["parameter_max_abs_error"] > 1e-12
        ):
            raise ValueError("Julia and Python input data or parameters differ")
        names = list(f["names/observables"].asstr()[...])
        if set(names) != set(effective.columns):
            raise ValueError("Observable names differ")
        if list(f["workflow/dates"].asstr()[...]) != list(effective.index.astype(str)):
            raise ValueError("History dates differ")
        np.testing.assert_allclose(
            f["workflow/data"][...], effective[names].to_numpy(), atol=1e-12, rtol=0
        )
        np.testing.assert_array_equal(f["workflow/regimes"][...], meta["history_regimes"])
        pseudo_names = list(f["names/pseudo_observables"].asstr()[...])
        pseudo_names = [{"pgap": "pgap_t", "ygap": "ygap_t"}.get(k, k) for k in pseudo_names]
        if set(pseudo_names) != set(ours.columns):
            raise ValueError("Pseudo-observable names differ")
        julia = pd.DataFrame(f["forecast/pseudo"][...], index=ours.index, columns=pseudo_names)
        julia_loglik = float(f["workflow/loglik"][...].sum())
    pop = meta["population_log_growth_quarterly_pp"]
    py_quarterly, py_annual = forecast_tables(ours, history, pop)
    jl_quarterly, jl_annual = forecast_tables(julia, history, pop)
    # A comparison must concern the committed Python tables, not stale paths.
    for kind, computed in (("quarterly", py_quarterly), ("annual", py_annual)):
        saved = pd.read_csv(tables_dir / f"ss104_nyfed_params_forecast_{kind}.csv")
        actual = computed.reset_index(drop=True)
        np.testing.assert_allclose(
            saved.drop(columns="quarter", errors="ignore"), actual, atol=1e-12, rtol=0
        )
    quarterly_errors = {
        k: float(np.max(np.abs(py_quarterly[f"{k}_ss104"] - jl_quarterly[f"{k}_ss104"])))
        for k in ("gdp", "corepce", "ffr")
    }
    annual_errors = {
        k: float(np.max(np.abs(py_annual[f"{k}_ss104"] - jl_annual[f"{k}_ss104"])))
        for k in ("gdp", "corepce")
    }
    model_unit_errors = {
        k: float(np.max(np.abs(ours[k] - julia[k])))
        for k in ("PseudoGDP", "PseudoCorePCE", "NominalFFR")
    }
    passed = all(
        np.isfinite(v) and v <= tolerance
        for v in (*quarterly_errors.values(), *annual_errors.values())
    )
    report = {
        "passed": passed,
        "tolerance_percentage_points": tolerance,
        "quarterly_max_abs_pp": quarterly_errors,
        "annual_q4_q4_max_abs_pp": annual_errors,
        "model_units_max_abs": model_unit_errors,
        "log_likelihood": {
            "python": meta["log_likelihood_including_condition"],
            "julia": julia_loglik,
            "max_abs_error": abs(meta["log_likelihood_including_condition"] - julia_loglik),
        },
        "julia": julia_meta,
        "oracle_sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
        "forecast_quarters": len(julia),
        "annual_years": py_annual.year.tolist(),
        "input_sha256": meta["input_sha256"],
    }
    jl_quarterly.rename(columns=lambda k: k.replace("_ss104", "_julia")).to_csv(
        tables_dir / "ss104_nyfed_params_julia_forecast_quarterly.csv", index_label="quarter"
    )
    jl_annual.rename(columns=lambda k: k.replace("_ss104", "_julia")).to_csv(
        tables_dir / "ss104_nyfed_params_julia_forecast_annual.csv", index=False
    )
    (tables_dir / "ss104_nyfed_params_parity.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    julia.to_csv(input_dir / "julia_forecast_pseudo_model_units.csv", index_label="date")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--input-dir", type=Path, default=OUTPUT)
    parser.add_argument("--tables-dir", type=Path, default=TABLES)
    args = parser.parse_args()
    report = compare(args.oracle, input_dir=args.input_dir, tables_dir=args.tables_dir)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Same-input forecast parity failed; see saved error report.")


if __name__ == "__main__":
    main()
