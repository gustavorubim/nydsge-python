"""Bounded, offline native ss104 smoke on committed public snapshots.

This is a retrospective numerical workflow check, not a fresh-release forecast
or an empirical forecast score. It never loads Julia or oracle results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from ss104_local import AS_OF, build_local_data, build_local_model, input_provenance

from nydsge.estimate import estimate
from nydsge.forecast import compute_meansbands, forecast_one


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=32, choices=range(2, 257))
    parser.add_argument("--horizon", type=int, default=8, choices=range(1, 13))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = build_local_model()
    historical = build_local_data(model, end="2026-Q2")
    history = pd.concat((historical, build_local_data(model).loc[[pd.Period("2026Q3")]]))
    kwargs: dict[str, Any] = dict(
        data=history, cond_type="semi", horizon=args.horizon + 1, output_vars=["forecastobs"]
    )
    baseline = forecast_one(model, "mode", **kwargs)
    zero = forecast_one(
        model,
        "full",
        **kwargs,
        shock_samples=np.zeros((2, args.horizon, len(model.indexes.exogenous_shocks))),
    )
    np.testing.assert_array_equal(zero.observable_samples, np.stack([baseline.observables] * 2))
    bands = compute_meansbands(model, "full", **kwargs, draws=args.draws, seed=20261015)
    repeat = compute_meansbands(model, "full", **kwargs, draws=args.draws, seed=20261015)
    np.testing.assert_array_equal(bands.mean, repeat.mean)
    value = model.parameters["alpha"].value
    sensitivity = forecast_one(
        model,
        "full",
        **kwargs,
        parameter_names=["alpha"],
        parameter_draws=np.array([[value - 0.001], [value + 0.001]]),
        parameter_draw_provenance="Two-point alpha sensitivity grid; not posterior draws",
    )
    point = estimate(model, historical)
    assert point.optimization is None and np.isfinite(point.log_posterior)
    assert sensitivity.observable_samples is not None
    np.savez_compressed(
        args.output_dir / "native_arrays.npz",
        baseline=baseline.observables,
        mean=bands.mean,
        lower=bands.lower,
        upper=bands.upper,
        sensitivity=sensitivity.observable_samples,
    )
    pd.DataFrame(
        bands.mean,
        columns=list(model.observables),
        index=pd.period_range("2026Q4", periods=args.horizon, freq="Q"),
    ).to_csv(args.output_dir / "forecast_mean_model_units.csv", index_label="quarter")
    root = Path(__file__).resolve().parents[1]
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = {
        "status": "passed",
        "code_sha": code_sha,
        "code_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip(),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "input_sha256": input_provenance(),
        "source_vintage": "committed 2026-10-06 public snapshots",
        "as_of": AS_OF,
        "availability": "Retrospective; original survey publication dates unverified",
        "draws": args.draws,
        "horizon": args.horizon,
        "seed": 20261015,
        "uncertainty_sources": bands.uncertainty_sources,
        "parameter_draw_provenance": sensitivity.parameter_draw_provenance,
        "zero_shock_and_seeded_rerun": "array-exact",
        "log_posterior": point.log_posterior,
        "units": (
            "Native quarterly model units: growth/inflation log percent; "
            "nominal rate quarterly percent"
        ),
        "sources": [
            "https://fred.stlouisfed.org/",
            "https://www.newyorkfed.org/markets/market-intelligence/survey-of-market-expectations",
        ],
        "limitations": (
            "Small-draw tails are illustrative; QQ includes measurement-state "
            "innovations, not extra EE noise or latent-state draws. "
            "Public defaults are not a posterior. No out-of-sample score."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
