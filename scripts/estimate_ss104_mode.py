"""One offline ss104 BFGS mode search, using the existing C1/C2 central-gradient path.

Windows spawn workers evaluate the same native likelihood independently. Atomic
checkpoints preserve the best finite posterior and report optimizer termination
honestly. No sampler, numerical Hessian, network access or ss10 warm start.
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
from ss104_local import build_local_data, build_local_model, input_provenance  # noqa: E402

from nydsge.ss104_estimation import (  # noqa: E402
    evaluate_ss104_posterior,
    ss104_estimation_bounds,
    ss104_estimation_vector,
    ss104_parameter_catalog,
    ss104_projected_gradient,
    update_ss104_parameters,
)

_STATE = {}


def _init_worker():
    model = build_local_model()
    _STATE.update(
        model=model,
        data=build_local_data(model, end="2026-Q2"),
        catalog=ss104_parameter_catalog(model),
    )


def _objective(values):
    try:
        update_ss104_parameters(_STATE["model"], _STATE["catalog"], values)
        lp, _, _, _ = evaluate_ss104_posterior(_STATE["model"], _STATE["data"])
        return -float(lp) if np.isfinite(lp) else float("inf")
    except (ValueError, np.linalg.LinAlgError, OverflowError, ZeroDivisionError):
        return float("inf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/estimates/m1002_ss104_mode.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--time-limit-min", type=float, default=120)
    parser.add_argument("--step", type=float, default=1e-4)
    parser.add_argument(
        "--resume", action="store_true", help="Continue this same default-start search"
    )
    parser.add_argument("--bounded", action="store_true", help="Bounded BFGS at an active bound")
    args = parser.parse_args()
    if args.workers < 1 or args.maxiter < 1 or args.step <= 0 or args.time_limit_min <= 0:
        parser.error("workers/maxiter/step/time limit must be positive")
    _init_worker()
    model, data, catalog = (_STATE[k] for k in ("model", "data", "catalog"))
    start = ss104_estimation_vector(catalog)
    t0 = time.monotonic()
    best = {"x": start.copy(), "f": _objective(start)}
    if not np.isfinite(best["f"]):
        raise ValueError("The ss104 default posterior is nonfinite on local data.")
    f_start = best["f"]
    counters = {"serial_objectives": 1, "gradients": 0, "gradient_objectives": 0}
    history = []
    provenance = input_provenance()
    rounds = []
    if args.resume:
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        if previous["status"] == "running":
            raise ValueError("Cannot resume a running search.")
        if previous["input_sha256"] != provenance or previous["estimated_parameters"] != list(
            catalog
        ):
            raise ValueError(
                "Resume inputs or parameter coordinates differ from the original search."
            )
        if not np.isclose(previous["log_posterior_start"], -f_start, atol=1e-6, rtol=0):
            raise ValueError("Resume must preserve the original ss104-default starting point.")
        start = np.array(previous["estimation_values"])
        best = {"x": start.copy(), "f": _objective(start)}
        if not np.isfinite(best["f"]):
            raise ValueError("The saved ss104 candidate is not a finite posterior point.")
        t0 -= previous["elapsed_seconds"]
        history = previous["history"]
        counters = previous["counters"]
        rounds = previous.get("optimizer_rounds", [previous["diagnostics"]])
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write(status, diagnostics=None):
        update_ss104_parameters(model, catalog, best["x"])
        lp, ll, prior, _ = evaluate_ss104_posterior(model, data)
        report = {
            "status": status,
            "spec": model.spec,
            "subspec": model.subspec,
            "optimizer": (
                f"scipy BFGS; central-difference step {args.step}; gtol 0.001; "
                "up to 3 Hessian-reset rounds plus bounded BFGS if needed, "
                "in one default-start search"
            ),
            "starting_point": "DSGE.jl ss104 defaults; prior-bearing free parameter regimes",
            "sample": {
                "presample_start": str(data.index[0]),
                "likelihood_start": "1960Q1",
                "end": str(data.index[-1]),
                "periods": len(data),
                "presample_periods": 2,
            },
            "log_posterior_start": -f_start,
            "log_posterior": lp,
            "log_likelihood": ll,
            "log_prior": prior,
            "estimated_parameters": list(catalog),
            "held_free_without_prior": [
                name if regime is None else f"{name}@{regime}"
                for name, baseline in model.parameters.items()
                for regime, p in model.get_setting("parameter_regimes")
                .get(name, {None: baseline})
                .items()
                if not p.fixed and p.prior is None
            ],
            "start_values": {k: p.value for k, p in catalog.items()},
            "estimation_values": best["x"].tolist(),
            "parameter_values": {k: p.value for k, p in model.parameters.items()},
            "parameter_regime_values": {
                k: {r: p.value for r, p in ps.items()}
                for k, ps in model.get_setting("parameter_regimes").items()
            },
            "data_nonmissing_counts": data.notna().sum().to_dict(),
            "input_sha256": provenance,
            "elapsed_seconds": time.monotonic() - t0,
            "counters": counters,
            "history": history,
            "diagnostics": diagnostics,
            "optimizer_rounds": rounds,
            "mcmc": {
                "runs": 0,
                "reason": (
                    "Mode only: no validated ss104 proposal curvature or convergence "
                    "within the optional one-hour budget."
                ),
            },
        }
        temporary = args.out.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(args.out)

    class TimeUp(Exception):
        pass

    def check_time():
        if time.monotonic() - t0 >= 60 * args.time_limit_min:
            raise TimeUp

    def objective(x):
        check_time()
        counters["serial_objectives"] += 1
        value = _objective(x)
        if value < best["f"]:
            best.update(x=x.copy(), f=value)
        return value

    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:

        def gradient(x):
            check_time()
            counters["gradients"] += 1
            # Wolfe search may request a gradient at an infeasible trial.
            # The existing C1/C2 driver treats that as no local information.
            f0 = _objective(x)
            if not np.isfinite(f0):
                return np.zeros_like(x)
            h = args.step
            points = []
            for i in range(len(x)):
                delta = np.zeros_like(x)
                delta[i] = h
                points.extend((x + delta, x - delta))
            values = np.array(pool.map(_objective, points, chunksize=1))
            counters["gradient_objectives"] += len(points)
            plus, minus = values[::2], values[1::2]
            g = (plus - minus) / (2 * h)
            invalid = ~np.isfinite(g)
            if invalid.any():
                for i in np.flatnonzero(invalid):
                    if np.isfinite(plus[i]):
                        g[i] = (plus[i] - f0) / h
                    elif np.isfinite(minus[i]):
                        g[i] = (f0 - minus[i]) / h
                    else:
                        g[i] = 0.0
            return g

        def checkpoint(x):
            history.append(
                {
                    "iteration": len(history) + 1,
                    "negative_log_posterior": _objective(x),
                    "elapsed_seconds": time.monotonic() - t0,
                }
            )
            print(json.dumps(history[-1]), flush=True)
            write("running")
            check_time()

        print(f"start log posterior {-f_start:.6f}; {len(catalog)} coordinates", flush=True)
        write("running")
        try:
            converged = False
            diagnostics = rounds[-1] if rounds else None
            round_limit = len(rounds) + 1 if args.bounded else 3
            for round_number in range(len(rounds), round_limit):
                print(f"BFGS round {round_number + 1}", flush=True)
                result = minimize(
                    objective,
                    best["x"].copy(),
                    method="L-BFGS-B" if args.bounded else "BFGS",
                    jac=gradient,
                    bounds=ss104_estimation_bounds(catalog) if args.bounded else None,
                    options=(
                        {
                            "maxiter": args.maxiter,
                            "gtol": 1e-3,
                            "ftol": 1e-15,
                            "maxls": 40,
                            "maxcor": 20,
                        }
                        if args.bounded
                        else {"maxiter": args.maxiter, "gtol": 1e-3}
                    ),
                    callback=checkpoint,
                )
                diagnostics = {
                    "round": round_number + 1,
                    "method": "L-BFGS-B" if args.bounded else "BFGS",
                    "success": bool(result.success),
                    "message": str(result.message),
                    "iterations": int(result.nit),
                    "gradient_inf_norm": float(np.max(np.abs(result.jac))),
                    "negative_log_posterior": best["f"],
                    "inverse_hessian_kind": "BFGS approximation; not a posterior covariance",
                }
                rounds.append(diagnostics)
                print(json.dumps(diagnostics), flush=True)
                write("running", diagnostics)
                if result.success and np.isfinite(result.fun):
                    # Line-search probes can improve the saved candidate slightly.
                    # Certify stationarity at the actual saved point, not a prior iterate.
                    gbest = gradient(best["x"])
                    projected = ss104_projected_gradient(catalog, best["x"], gbest)
                    diagnostics["saved_point_raw_gradient_inf_norm"] = float(np.max(np.abs(gbest)))
                    diagnostics["saved_point_gradient_inf_norm"] = float(np.max(np.abs(projected)))
                    converged = diagnostics["saved_point_gradient_inf_norm"] <= 1e-3
                    if converged:
                        break
            write("converged" if converged else "stopped", diagnostics)
        except TimeUp:
            write("time_limit", {"success": False, "message": "Configured mode-search time limit"})
        except Exception as error:
            write("failed", {"success": False, "message": str(error)})
            raise
        print(args.out.read_text(encoding="utf-8")[:500], flush=True)


if __name__ == "__main__":
    main()
