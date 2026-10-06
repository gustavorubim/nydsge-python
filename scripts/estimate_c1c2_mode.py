"""Posterior-mode search for Model1002 ss10 on the C1/C2 public-data sample.

Usage::

    uv run python scripts/estimate_c1c2_mode.py --out data/estimates/<name>.json \
        [--workers 8] [--maxiter 400] [--mask-covid] [--n-ant 0]

Objective: DSGE.jl-style log posterior (Kalman likelihood from mainsample start,
presample 1959Q3-Q4 used only to initialize, plus log prior) over all free
parameters, optimized in the repo's unbounded estimation space with BFGS and
parallel central-difference gradients.  A finite-difference Hessian at the
optimum is reported as a convergence diagnostic.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1c2_common import (  # noqa: E402
    N_PRESAMPLE,
    PRESAMPLE_START,
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

COVID_QUARTERS = ("2020-Q2", "2020-Q3", "2020-Q4")
_STATE: dict[str, object] = {}


def _setup(n_ant: int, mask_covid: bool) -> tuple[object, np.ndarray, tuple[str, ...], list[str]]:
    model = build_model(n_ant)
    obs = build_observables(model)
    if mask_covid:
        rows = obs["date"].isin(COVID_QUARTERS).to_numpy()
        cols = [c for c in obs.columns if c.startswith("obs_")]
        obs.loc[rows, cols] = np.nan
    data = observation_matrix(model, obs)
    names = estimation_parameter_names(model)
    return model, data, names, list(obs["date"])


def _init_worker(n_ant: int, mask_covid: bool) -> None:
    model, data, names, _ = _setup(n_ant, mask_covid)
    _STATE.update(model=model, data=data, names=names, original=dict(model.parameters))


def _neg_log_post(x: np.ndarray) -> float:
    model = _STATE["model"]
    original = _STATE["original"]
    try:
        _set_parameter_estimation_vector(model, original, _STATE["names"], np.asarray(x, float))
        lp, _, _, _ = _evaluate_log_posterior(
            model,
            _STATE["data"],
            start_date=PRESAMPLE_START,
            log_likelihood_start=N_PRESAMPLE,
        )
    except Exception:
        return float("inf")
    finally:
        model.parameters.update(original)
    return float(-lp) if np.isfinite(lp) else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--n-ant", type=int, default=0)
    parser.add_argument("--mask-covid", action="store_true")
    parser.add_argument("--step", type=float, default=1e-4)
    parser.add_argument("--time-limit-min", type=float, default=240.0)
    parser.add_argument("--start-json", default=None)
    args = parser.parse_args()

    model, data, names, dates = _setup(args.n_ant, args.mask_covid)
    x0 = parameter_estimation_vector(model, names)
    start_values = {n: float(model.parameters[n].value) for n in names}
    if args.start_json:
        start_model = json.loads(Path(args.start_json).read_text())["parameter_values"]
        for n in names:
            start_values[n] = float(start_model[n])
        from dataclasses import replace

        from nydsge.parameters import update_parameter_value  # noqa: F401

        for n in names:
            model.parameters[n] = replace(model.parameters[n], value=start_values[n])
        x0 = parameter_estimation_vector(model, names)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    pool = ctx.Pool(args.workers, initializer=_init_worker, initargs=(args.n_ant, args.mask_covid))
    _init_worker(args.n_ant, args.mask_covid)
    n = len(names)
    h = args.step
    history: list[dict[str, float]] = []
    t0 = time.time()
    counters = {"f": 0, "g": 0}

    def f(x: np.ndarray) -> float:
        counters["f"] += 1
        return _neg_log_post(x)

    def grad(x: np.ndarray) -> np.ndarray:
        counters["g"] += 1
        points = []
        for i in range(n):
            e = np.zeros(n)
            e[i] = h
            points.extend([x + e, x - e])
        values = np.asarray(
            pool.map(_neg_log_post, points, chunksize=max(1, len(points) // (4 * args.workers)))
        )
        fp, fm = values[0::2], values[1::2]
        f0 = None
        g = np.empty(n)
        for i in range(n):
            if np.isfinite(fp[i]) and np.isfinite(fm[i]):
                g[i] = (fp[i] - fm[i]) / (2 * h)
            else:
                if f0 is None:
                    f0 = _neg_log_post(x)
                if np.isfinite(fp[i]):
                    g[i] = (fp[i] - f0) / h
                elif np.isfinite(fm[i]):
                    g[i] = (f0 - fm[i]) / h
                else:
                    g[i] = 0.0  # both neighbours indeterminate/invalid: no local information
        return np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

    best = {"x": x0.copy(), "f": f(x0)}
    f_start = best["f"]
    print(f"start -logpost {f_start:.4f}  n_params {n}", flush=True)

    class TimeUp(Exception):
        pass

    def checkpoint(xk: np.ndarray, status: str) -> None:
        fk = _neg_log_post(xk)
        if fk <= best["f"]:
            best.update(x=xk.copy(), f=fk)
        history.append(
            {"iter": len(history) + 1, "neg_log_post": fk, "elapsed_s": time.time() - t0}
        )
        print(
            f"iter {len(history):4d} -logpost {fk:.4f} elapsed {time.time() - t0:7.1f}s", flush=True
        )
        _write(
            out,
            model,
            names,
            best["x"],
            best["f"],
            f_start,
            start_values,
            history,
            counters,
            status,
            args,
            dates,
        )
        if (time.time() - t0) / 60.0 > args.time_limit_min:
            raise TimeUp

    status = "running"
    rounds = []
    x = x0.copy()
    try:
        for rnd in range(3):  # BFGS restarts reset the inverse-Hessian approximation
            res = minimize(
                f,
                x,
                jac=grad,
                method="BFGS",
                options={"maxiter": args.maxiter, "gtol": 1e-3},
                callback=lambda xk: checkpoint(xk, "running"),
            )
            rounds.append(
                {
                    "round": rnd,
                    "fun": float(res.fun),
                    "nit": int(res.nit),
                    "success": bool(res.success),
                    "message": str(res.message),
                }
            )
            if res.fun <= best["f"]:
                best.update(x=np.asarray(res.x).copy(), f=float(res.fun))
            improvement = float(np.abs(f(x) - best["f"]))
            x = best["x"].copy()
            print(f"round {rnd}: {res.message} f={res.fun:.4f} nit={res.nit}", flush=True)
            if improvement < 1e-2:
                break
        status = "converged" if rounds and rounds[-1]["success"] else "stopped"
    except TimeUp:
        status = "time_limit"
    g = grad(best["x"])
    # Hessian diagonal-and-full via central differences of the parallel gradient
    hess = np.empty((n, n))
    hstep = 1e-3
    for i in range(n):
        e = np.zeros(n)
        e[i] = hstep
        hess[i] = (grad(best["x"] + e) - grad(best["x"] - e)) / (2 * hstep)
    hess = 0.5 * (hess + hess.T)
    eig = np.linalg.eigvalsh(hess)
    diagnostics = {
        "grad_inf_norm": float(np.max(np.abs(g))),
        "grad_largest": sorted(zip(names, map(float, g), strict=True), key=lambda t: -abs(t[1]))[
            :8
        ],
        "hessian_min_eig": float(eig[0]),
        "hessian_max_eig": float(eig[-1]),
        "hessian_n_negative_eig": int(np.sum(eig < 0)),
        "bfgs_rounds": rounds,
    }
    try:
        cov = np.linalg.inv(hess)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        diagnostics["se_estimation_space"] = dict(zip(names, map(float, se), strict=True))
    except np.linalg.LinAlgError:
        pass
    _write(
        out,
        model,
        names,
        best["x"],
        best["f"],
        f_start,
        start_values,
        history,
        counters,
        status,
        args,
        dates,
        diagnostics=diagnostics,
    )
    np.save(out.with_suffix(".hessian.npy"), hess)
    pool.close()
    print(
        json.dumps(
            {
                "status": status,
                **{k: v for k, v in diagnostics.items() if k != "se_estimation_space"},
            },
            indent=1,
        ),
        flush=True,
    )


def _write(
    out,
    model,
    names,
    x,
    fbest,
    fstart,
    start_values,
    history,
    counters,
    status,
    args,
    dates,
    diagnostics=None,
) -> None:
    original = dict(model.parameters)
    _set_parameter_estimation_vector(model, original, names, np.asarray(x, float))
    lp, ll, lprior, _ = _evaluate_log_posterior(
        model, _STATE["data"], start_date=PRESAMPLE_START, log_likelihood_start=N_PRESAMPLE
    )
    values = {k: float(p.value) for k, p in model.parameters.items()}
    model.parameters.update(original)
    payload = {
        "status": status,
        "spec": "m1002",
        "subspec": "ss10",
        "n_mon_anticipated_shocks": args.n_ant,
        "mask_covid_quarters": list(COVID_QUARTERS) if args.mask_covid else [],
        "sample": {
            "presample_start": PRESAMPLE_START,
            "likelihood_start": dates[N_PRESAMPLE],
            "end": dates[-1],
            "periods": len(dates),
        },
        "optimizer": (
            f"scipy BFGS, parallel central-difference gradient (step {args.step:g}), "
            "up to 3 restarts"
        ),
        "log_posterior_start": -fstart,
        "log_posterior": float(lp),
        "log_likelihood": float(ll),
        "log_prior": float(lprior),
        "function_evaluations_serial": counters["f"],
        "gradient_evaluations": counters["g"],
        "estimated_parameters": list(names),
        "start_values": start_values,
        "parameter_values": values,
        "history": history,
        "diagnostics": diagnostics,
    }
    out.write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
