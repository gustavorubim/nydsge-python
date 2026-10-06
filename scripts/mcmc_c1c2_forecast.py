"""Forecasts with parameter uncertainty from the C1/C2 MCMC draws (docs/mcmc_c1c2.md).

Usage::

    uv run python scripts/mcmc_c1c2_forecast.py --draws /workspace/nydsge-mcmc/run1/pooled.npz \
        --n-param 1000 --n-shock 20 --n-decomp 300 --workers 8 \
        --out /workspace/nydsge-mcmc/run1/forecast.json

Setup matches configs/quarterly_economy_c1c2.json: COVID-masked smoother, 2026Q3 semi-
conditioned on FFR/spread/10y/SPF10, forecast from 2026Q4.  For every parameter draw:
the RTS-smoothed terminal (2026Q3) state, the zero-shock forecast, and ``n-shock``
simulated future-shock paths (N(0, QQ)).  Smoothed-state uncertainty is not sampled.
GDP is converted from per-capita to aggregate growth by adding the constant terminal
HP-trend population growth used in docs/nyfed_comparison.md.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1c2_common import ROOT, build_model, build_observables  # noqa: E402
from c1c2_compare import COND_OBS, metrics  # noqa: E402

from nydsge.data import reverse_transform_observables  # noqa: E402
from nydsge.economy import _restrict_conditioning_quarters  # noqa: E402
from nydsge.forecast import forecast_linear_system_samples, forecast_one  # noqa: E402
from nydsge.solve import compute_system  # noqa: E402

MASK = ("2020-Q2", "2020-Q3", "2020-Q4")
POP_GROWTH_ANN = 1.0552487467627714  # terminal HP-trend population growth, /tmp/econ_after run
H = 9  # 2026Q4 .. 2028Q4
FDATES = [f"{2026 + (3 + i) // 4}-Q{(3 + i) % 4 + 1}" for i in range(H)]
VARS = ("obs_corepce", "obs_nominalrate", "obs_gdp")
_S: dict = {}


def _init() -> None:
    model = build_model(0, date_forecast_start="2026-Q4")
    obs = build_observables(model, end="2026-Q3")
    cols = [c for c in obs.columns if c.startswith("obs_")]
    obs.loc[obs["date"].isin(MASK), cols] = np.nan
    obs = _restrict_conditioning_quarters(obs, model_end_date="2026-Q2", keep=COND_OBS)
    hist = reverse_transform_observables(model, obs[list(model.observables)].to_numpy(float))
    names = list(model.observables)
    dates = list(obs["date"])
    _S.update(
        model=model,
        obs=obs,
        original=dict(model.parameters),
        names=names,
        hist={
            d: {v: float(hist[dates.index(d), names.index(v)]) for v in VARS}
            for d in ("2026-Q1", "2026-Q2")
        },
    )


def _set(values: dict[str, float]) -> None:
    model = _S["model"]
    model.parameters.update(_S["original"])
    for k, v in values.items():
        model.parameters[k] = replace(model.parameters[k], value=float(v))


def _aggregate(pc: np.ndarray) -> np.ndarray:
    return 400 * (np.exp(np.log1p(pc / 400) + POP_GROWTH_ANN / 400) - 1)


def _q4q4(qq: np.ndarray) -> np.ndarray:
    return 100 * (np.prod((1 + qq / 100) ** 0.25, axis=-1) - 1)


def one_draw(task: dict) -> dict:
    _set(task["values"])
    model, obs, names = _S["model"], _S["obs"], _S["names"]
    system = compute_system(model)
    fc = forecast_one(
        model,
        input_type="mode",
        cond_type="none",
        output_vars=["histstates", "forecastobs"],
        check_empty_columns=False,
        horizon=H,
        data=obs,
        history_method="smoothed",
    )
    s_t = np.asarray(fc.history_states)[-1]
    q3 = reverse_transform_observables(
        model, (system.measurement.ZZ @ s_t + system.measurement.DD)[None, :]
    )[0]
    mean = reverse_transform_observables(model, np.asarray(fc.observables))
    sims = forecast_linear_system_samples(
        system, s_t, horizon=H, draws=task["n_shock"], seed=task["seed"]
    ).observable_samples
    sims = reverse_transform_observables(model, sims.reshape(-1, sims.shape[-1])).reshape(
        sims.shape
    )
    idx = {v: names.index(v) for v in VARS}
    hist = _S["hist"]

    def stats_for(paths: np.ndarray) -> dict:
        # paths: (n, H, nobs) in reporting units, forecast quarters 2026Q4..2028Q4
        pce = paths[:, :, idx["obs_corepce"]]
        ffr = paths[:, :, idx["obs_nominalrate"]]
        gdp = _aggregate(paths[:, :, idx["obs_gdp"]])
        n = paths.shape[0]
        pce26 = np.column_stack(
            [
                np.full(n, hist["2026-Q1"]["obs_corepce"]),
                np.full(n, hist["2026-Q2"]["obs_corepce"]),
                np.full(n, q3[idx["obs_corepce"]]),
                pce[:, 0],
            ]
        )
        gdp26 = np.column_stack(
            [
                _aggregate(np.full(n, hist["2026-Q1"]["obs_gdp"])),
                _aggregate(np.full(n, hist["2026-Q2"]["obs_gdp"])),
                _aggregate(np.full(n, q3[idx["obs_gdp"]])),
                gdp[:, 0],
            ]
        )
        return {
            # h1 = 2026Q3 (smoothed, conditioned), h2.. = 2026Q4..
            "corepce_h1_h8": np.column_stack([np.full(n, q3[idx["obs_corepce"]]), pce[:, :7]]),
            "ffr_h1_h8": np.column_stack([np.full(n, q3[idx["obs_nominalrate"]]), ffr[:, :7]]),
            "corepce_q4q4": np.column_stack([_q4q4(pce26), _q4q4(pce[:, 1:5]), _q4q4(pce[:, 5:9])]),
            "gdp_q4q4": np.column_stack([_q4q4(gdp26), _q4q4(gdp[:, 1:5]), _q4q4(gdp[:, 5:9])]),
        }

    out = {
        "mean": {k: v[0].tolist() for k, v in stats_for(mean[None]).items()},
        "sims": {k: v.tolist() for k, v in stats_for(sims).items()},
    }
    if task.get("decomp"):
        m = metrics(model, obs)
        out["decomp"] = {
            "corepce": m["corepce_decomp_2026Q2"],
            "corepce_initial": m["corepce_initial_and_trend"],
            "smoothed_corepce_2026Q2": m["smoothed_corepce_2026Q2"],
            "long_run_inflation_2026Q2": m["long_run_inflation_2026Q2"],
            "outputgap": m["outputgap_decomp_2026Q2"],
            "output_gap_2026Q2": m["smoothed_output_gap_2026Q2"],
        }
    return out


def _bands(x: np.ndarray) -> dict:
    q = np.quantile(x, [0.05, 0.16, 0.5, 0.84, 0.95], axis=0)
    return {
        "mean": np.mean(x, axis=0).tolist(),
        "q05": q[0].tolist(),
        "q16": q[1].tolist(),
        "median": q[2].tolist(),
        "q84": q[3].tolist(),
        "q95": q[4].tolist(),
    }


def summarize(results: list[dict]) -> dict:
    keys = results[0]["mean"].keys()
    out: dict = {}
    for k in keys:
        means = np.asarray([r["mean"][k] for r in results])
        sims = np.concatenate([np.asarray(r["sims"][k]) for r in results])
        out[k] = {
            "parameter_only (zero future shocks)": _bands(means),
            "parameter_and_shock": _bands(sims),
        }
    dec = [r["decomp"] for r in results if "decomp" in r]
    if dec:
        out["decomposition_2026Q2"] = {"n_draws": len(dec)}
        for group in ("corepce", "outputgap"):
            for comp in dec[0][group]:
                vals = np.asarray([d[group][comp] for d in dec])
                out["decomposition_2026Q2"][f"{group}:{comp}"] = _bands(vals)
        for key in (
            "corepce_initial",
            "smoothed_corepce_2026Q2",
            "long_run_inflation_2026Q2",
            "output_gap_2026Q2",
        ):
            out["decomposition_2026Q2"][key] = _bands(np.asarray([d[key] for d in dec]))
        shares = np.asarray(
            [
                [
                    d["corepce"]["pi_star_sh"],
                    d["corepce"]["lambda_f_sh"],
                    d["corepce"]["lambda_w_sh"],
                    d["corepce"]["rm_sh"] + d["corepce"]["rm_shl1-6"],
                ]
                for d in dec
            ]
        )
        dev = np.asarray([d["smoothed_corepce_2026Q2"] - d["corepce_initial"] for d in dec])
        frac = shares / dev[:, None]
        out["decomposition_2026Q2"]["share_of_shock_driven_deviation"] = {
            name: _bands(frac[:, i])
            for i, name in enumerate(["pi_star", "lambda_f", "lambda_w", "monetary"])
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", required=True)
    ap.add_argument("--n-param", type=int, default=1000)
    ap.add_argument("--n-shock", type=int, default=20)
    ap.add_argument("--n-decomp", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mode-shock-draws", type=int, default=20000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with np.load(args.draws) as z:
        names = [str(n) for n in z["parameter_names"]]
        draws = z["model_draws"][: args.n_param]
    mode = json.loads((ROOT / "data/estimates/m1002_ss10_c1c2_mode.json").read_text())
    mode_values = {k: float(mode["parameter_values"][k]) for k in mode["estimated_parameters"]}
    tasks = [
        {
            "values": dict(zip(names, map(float, row), strict=True)),
            "n_shock": args.n_shock,
            "seed": 10_000 + i,
            "decomp": i < args.n_decomp,
        }
        for i, row in enumerate(draws)
    ]
    with mp.get_context("fork").Pool(args.workers, initializer=_init) as pool:
        mode_res = pool.apply(
            one_draw,
            (
                {
                    "values": mode_values,
                    "n_shock": args.mode_shock_draws,
                    "seed": 7,
                    "decomp": True,
                },
            ),
        )
        results = pool.map(one_draw, tasks, chunksize=4)
    payload = {
        "n_param_draws": len(results),
        "n_shock_per_draw": args.n_shock,
        "dates_h1_h8": ["2026-Q3", *FDATES[:7]],
        "q4q4_years": [2026, 2027, 2028],
        "pop_growth_ann": POP_GROWTH_ANN,
        "mode": {
            "mean": mode_res["mean"],
            "shock_only_bands": {k: _bands(np.asarray(v)) for k, v in mode_res["sims"].items()},
            "decomp": mode_res.get("decomp"),
        },
        "posterior": summarize(results),
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["posterior"]["corepce_q4q4"], indent=1))


if __name__ == "__main__":
    main()
