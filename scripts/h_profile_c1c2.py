"""Re-optimize the C1/C2 posterior with parameters fixed or observables dropped.

Used in docs/mcmc_c1c2.md to check what drives the low habit-persistence estimate:

    uv run python scripts/h_profile_c1c2.py --fix h=0.73 --out /workspace/nydsge-mcmc/h073.json
    uv run python scripts/h_profile_c1c2.py --drop-obs obs_consumption --out ...
    uv run python scripts/h_profile_c1c2.py --likelihood-from 1985-Q1 --out ...

Same objective as scripts/estimate_c1c2_mode.py (COVID-masked by default, presample
1959Q3-Q4, log prior), BFGS with parallel central-difference gradients, started from the
masked mode.  Dropped observables are set to NaN for the whole sample; with
``--likelihood-from`` the filter still runs from 1959Q3 but only later periods enter the
likelihood.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1c2_common import (  # noqa: E402
    N_PRESAMPLE,
    PRESAMPLE_START,
    ROOT,
    build_model,
    build_observables,
    observation_matrix,
)

from nydsge.estimate import (  # noqa: E402
    _evaluate_log_posterior,
    _set_parameter_estimation_vector,
    estimation_parameter_names,
    parameter_estimation_vector,
)

MASK = ("2020-Q2", "2020-Q3", "2020-Q4")
_S: dict = {}


def _setup(cfg: dict) -> None:
    model = build_model(0)
    obs = build_observables(model)
    cols = [c for c in obs.columns if c.startswith("obs_")]
    if cfg["mask"]:
        obs.loc[obs["date"].isin(MASK), cols] = np.nan
    for name in cfg["drop_obs"]:
        obs[name] = np.nan
    data = observation_matrix(model, obs)
    dates = list(obs["date"])
    ll_start = dates.index(cfg["likelihood_from"]) if cfg["likelihood_from"] else N_PRESAMPLE
    start = json.loads(Path(cfg["start_json"]).read_text())["parameter_values"]
    for k, v in start.items():
        if k in model.parameters:
            model.parameters[k] = replace(model.parameters[k], value=float(v))
    for k, v in cfg["fix"].items():
        model.parameters[k] = replace(model.parameters[k], value=float(v))
    names = tuple(n for n in estimation_parameter_names(model) if n not in cfg["fix"])
    _S.update(
        model=model,
        data=data,
        names=names,
        original=dict(model.parameters),
        ll_start=ll_start,
        dates=dates,
    )


def _eval(x: np.ndarray) -> tuple[float, float, float]:
    model, original = _S["model"], _S["original"]
    try:
        _set_parameter_estimation_vector(model, original, _S["names"], np.asarray(x, float))
        lp, ll, lpr, _ = _evaluate_log_posterior(
            model, _S["data"], start_date=PRESAMPLE_START, log_likelihood_start=_S["ll_start"]
        )
    except Exception:
        return float("-inf"), float("-inf"), float("-inf")
    finally:
        model.parameters.update(original)
    return float(lp), float(ll), float(lpr)


def _neg(x: np.ndarray) -> float:
    lp = _eval(x)[0]
    return -lp if np.isfinite(lp) else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fix", action="append", default=[], help="name=value (model space)")
    ap.add_argument("--drop-obs", action="append", default=[])
    ap.add_argument("--likelihood-from", default=None)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--start-json", default=str(ROOT / "data/estimates/m1002_ss10_c1c2_mode.json"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--step", type=float, default=1e-4)
    args = ap.parse_args()
    cfg = {
        "fix": {k: float(v) for k, v in (s.split("=") for s in args.fix)},
        "drop_obs": args.drop_obs,
        "likelihood_from": args.likelihood_from,
        "mask": not args.no_mask,
        "start_json": args.start_json,
    }
    _setup(cfg)
    names = _S["names"]
    x0 = parameter_estimation_vector(_S["model"], names)
    pool = mp.get_context("fork").Pool(args.workers, initializer=_setup, initargs=(cfg,))
    n, h = len(names), args.step
    t0 = time.time()
    counts = {"g": 0}

    def grad(x: np.ndarray) -> np.ndarray:
        counts["g"] += 1
        pts = []
        for i in range(n):
            e = np.zeros(n)
            e[i] = h
            pts.extend([x + e, x - e])
        v = np.asarray(pool.map(_neg, pts, chunksize=max(1, len(pts) // (4 * args.workers))))
        g = (v[0::2] - v[1::2]) / (2 * h)
        return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    start = _eval(x0)
    res = minimize(
        _neg, x0, jac=grad, method="BFGS", options={"maxiter": args.maxiter, "gtol": 1e-3}
    )
    lp, ll, lpr = _eval(res.x)
    model, original = _S["model"], _S["original"]
    _set_parameter_estimation_vector(model, original, names, np.asarray(res.x, float))
    values = {k: float(p.value) for k, p in model.parameters.items()}
    model.parameters.update(original)
    g = grad(res.x)
    payload = {
        "config": cfg,
        "likelihood_start_date": _S["dates"][_S["ll_start"]],
        "n_free": n,
        "start": {"log_posterior": start[0], "log_likelihood": start[1], "log_prior": start[2]},
        "log_posterior": lp,
        "log_likelihood": ll,
        "log_prior": lpr,
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
        "gradient_evaluations": counts["g"],
        "grad_inf_norm": float(np.max(np.abs(g))),
        "elapsed_s": time.time() - t0,
        "parameter_values": values,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    pool.close()
    print(json.dumps({k: v for k, v in payload.items() if k != "parameter_values"}, indent=1))


if __name__ == "__main__":
    main()
