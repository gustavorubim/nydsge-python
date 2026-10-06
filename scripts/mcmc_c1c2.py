"""Random-walk Metropolis-Hastings for the C1/C2 Model1002 ss10 posterior (docs/mcmc_c1c2.md).

Usage::

    uv run python scripts/mcmc_c1c2.py --out-dir /workspace/nydsge-mcmc/run1 \
        --chains 8 --scale 0.30 --hours 3.2 [--overdispersion 2.0] [--n-mode-starts 2]

Each chain runs in its own process and calls the repo sampler
``nydsge.estimate.metropolis_hastings`` in chunks (the chain state carries over between
chunks), so partial results are checkpointed to ``chain_<k>.npz`` while it runs.
Target: COVID-masked (2020Q2-Q4) C1/C2 sample, presample 1959Q3-Q4, likelihood from
1960Q1, log prior, plus the estimation-space Jacobian (``jacobian=True``).
Proposal: N(0, scale^2 * H^-1), H the finite-difference Hessian at the BFGS mode in
estimation space.  Chains start at the mode or at mode + overdispersion * N(0, H^-1).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1c2_common import N_PRESAMPLE, PRESAMPLE_START, ROOT  # noqa: E402
from estimate_c1c2_mode import _setup  # noqa: E402

from nydsge.estimate import (  # noqa: E402
    _log_posterior_for_estimation_values,
    _set_parameter_estimation_vector,
    metropolis_hastings,
    parameter_estimation_vector,
)

MODE_JSON = ROOT / "data/estimates/m1002_ss10_c1c2_mode_maskcovid.json"
HESSIAN_NPY = ROOT / "data/estimates/m1002_ss10_c1c2_mode_maskcovid.hessian.npy"


def load_problem() -> tuple[object, np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    """Model, masked data, parameter names, mode (estimation space), proposal covariance."""

    model, data, names, _ = _setup(0, True)
    mode = json.loads(MODE_JSON.read_text())
    if tuple(mode["estimated_parameters"]) != tuple(names):
        msg = "mode parameter order does not match the model"
        raise ValueError(msg)
    for name, value in mode["parameter_values"].items():
        if name in model.parameters:
            model.parameters[name] = replace(model.parameters[name], value=float(value))
    x_mode = parameter_estimation_vector(model, names)
    hessian = np.load(HESSIAN_NPY)
    hessian = 0.5 * (hessian + hessian.T)
    covariance = np.linalg.inv(hessian)
    covariance = 0.5 * (covariance + covariance.T)
    np.linalg.cholesky(covariance)
    return model, data, names, x_mode, covariance


def log_target(model: object, data: np.ndarray, names: tuple[str, ...], x: np.ndarray) -> float:
    original = dict(model.parameters)  # type: ignore[attr-defined]
    value = _log_posterior_for_estimation_values(
        model,  # type: ignore[arg-type]
        data,
        original,
        names,
        x,
        start_date=PRESAMPLE_START,
        log_likelihood_start=N_PRESAMPLE,
        jacobian=True,
    )
    model.parameters.update(original)  # type: ignore[attr-defined]
    return value


def start_points(
    x_mode: np.ndarray,
    covariance: np.ndarray,
    *,
    chains: int,
    n_mode_starts: int,
    overdispersion: float,
    seed: int,
    model: object,
    data: np.ndarray,
    names: tuple[str, ...],
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(covariance)
    points: list[np.ndarray] = []
    for k in range(chains):
        if k < n_mode_starts:
            points.append(x_mode.copy())
            continue
        for _ in range(200):
            candidate = x_mode + overdispersion * chol @ rng.standard_normal(x_mode.size)
            if np.isfinite(log_target(model, data, names, candidate)):
                points.append(candidate)
                break
        else:
            msg = "could not find a finite overdispersed start"
            raise RuntimeError(msg)
    return points


def run_chain(task: dict) -> dict:
    model, data, names, _, covariance = load_problem()
    original = dict(model.parameters)
    _set_parameter_estimation_vector(model, original, names, np.asarray(task["start"]))
    out = Path(task["out"])
    seeds = np.random.SeedSequence(task["seed"]).spawn(100000)
    deadline = task["deadline"]
    max_draws = task["max_draws"]
    chunk = task["chunk"]
    draws: list[np.ndarray] = []
    logpost: list[np.ndarray] = []
    accepted = 0
    proposals = 0
    started = time.time()
    k = 0
    while sum(len(d) for d in draws) < max_draws and time.time() < deadline:
        n = min(chunk, max_draws - sum(len(d) for d in draws))
        result = metropolis_hastings(
            model,
            data,
            parameter_names=names,
            draws=n,
            burnin=0,
            proposal_covariance=covariance,
            proposal_scale=task["scale"],
            seed=int(seeds[k].generate_state(1)[0]),
            start_date=PRESAMPLE_START,
            log_likelihood_start=N_PRESAMPLE,
            jacobian=True,
        )
        k += 1
        draws.append(result.estimation_draws)
        logpost.append(result.log_posterior)
        accepted += int(round(result.acceptance_rate * n))
        proposals += n
        np.savez(
            out,
            estimation_draws=np.concatenate(draws),
            log_target=np.concatenate(logpost),  # model-space log posterior (no Jacobian)
            start=np.asarray(task["start"]),
            scale=task["scale"],
            accepted=accepted,
            proposals=proposals,
            elapsed=time.time() - started,
        )
        with open(str(out) + ".progress", "w") as handle:
            handle.write(
                f"draws {proposals} accept {accepted / proposals:.3f} "
                f"last_logpost {result.log_posterior[-1]:.2f} "
                f"elapsed {time.time() - started:.0f}s\n"
            )
    return {"chain": task["chain"], "proposals": proposals, "accepted": accepted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--max-draws", type=int, default=10**7)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--overdispersion", type=float, default=2.0)
    parser.add_argument("--n-mode-starts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument(
        "--scales", default=None, help="comma-separated per-chain scales (pilot tuning)"
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, data, names, x_mode, covariance = load_problem()
    lp_mode = log_target(model, data, names, x_mode)
    starts = start_points(
        x_mode,
        covariance,
        chains=args.chains,
        n_mode_starts=args.n_mode_starts,
        overdispersion=args.overdispersion,
        seed=args.seed,
        model=model,
        data=data,
        names=names,
    )
    scales = (
        [float(s) for s in args.scales.split(",")] if args.scales else [args.scale] * args.chains
    )
    deadline = time.time() + args.hours * 3600.0
    meta = {
        "parameter_names": list(names),
        "mode_estimation_values": x_mode.tolist(),
        "log_target_at_mode": lp_mode,
        "start_log_targets": [log_target(model, data, names, s) for s in starts],
        "scales": scales,
        "chains": args.chains,
        "n_mode_starts": args.n_mode_starts,
        "overdispersion": args.overdispersion,
        "seed": args.seed,
        "hours": args.hours,
        "mode_json": str(MODE_JSON.relative_to(ROOT)),
        "hessian": str(HESSIAN_NPY.relative_to(ROOT)),
        "target": "log posterior (masked 2020Q2-Q4) + log|Jacobian| in estimation space",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    tasks = [
        {
            "chain": k,
            "start": starts[k].tolist(),
            "scale": scales[k],
            "seed": args.seed + 1000 * (k + 1),
            "deadline": deadline,
            "max_draws": args.max_draws,
            "chunk": args.chunk,
            "out": str(out_dir / f"chain_{k}.npz"),
        }
        for k in range(args.chains)
    ]
    with mp.get_context("fork").Pool(args.chains) as pool:
        results = pool.map(run_chain, tasks)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
