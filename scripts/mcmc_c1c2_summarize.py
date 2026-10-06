"""Convergence diagnostics and posterior summaries for scripts/mcmc_c1c2.py output.

Usage::

    uv run python scripts/mcmc_c1c2_summarize.py --run-dir /workspace/nydsge-mcmc/run1 \
        --burn-frac 0.5 --out-csv data/estimates/mcmc_c1c2_summary.csv \
        --out-json /workspace/nydsge-mcmc/run1/summary.json [--thin-out pooled.npz]

Diagnostics follow Vehtari, Gelman, Simpson, Carpenter and Buerkner (2021): rank-normalized
split R-hat (max of bulk and folded) and bulk/tail effective sample sizes, computed
across chains after discarding the first ``burn-frac`` of every chain (all chains are cut
to the same length).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1c2_common import build_model  # noqa: E402

from nydsge.estimate import _model_values_for_estimation_vector  # noqa: E402


def _autocov(x: np.ndarray) -> np.ndarray:
    n = x.size
    centered = x - x.mean()
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(centered, size)
    acov = np.fft.irfft(f * np.conj(f), size)[:n] / n
    return acov


def ess(chains: np.ndarray) -> float:
    """Multi-chain ESS (Stan / Geyer initial monotone sequence). chains: (M, N)."""

    m, n = chains.shape
    if n < 4:
        return float("nan")
    acov = np.stack([_autocov(c) for c in chains])
    chain_mean = chains.mean(axis=1)
    w = acov[:, 0].mean() * n / (n - 1)
    b = n * chain_mean.var(ddof=1) if m > 1 else 0.0
    var_plus = (n - 1) / n * w + b / n
    if var_plus <= 0:
        return float("nan")
    rho = 1.0 - (w - acov.mean(axis=0) * n / (n - 1)) / var_plus
    rho[0] = 1.0
    pairs = []
    t = 0
    while t + 1 < n:
        p = rho[t] + rho[t + 1]
        if p < 0:
            break
        pairs.append(p)
        t += 2
    pairs_arr = np.minimum.accumulate(np.asarray(pairs)) if pairs else np.asarray([1.0])
    tau = -1.0 + 2.0 * pairs_arr.sum()
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def _split(chains: np.ndarray) -> np.ndarray:
    m, n = chains.shape
    half = n // 2
    return np.concatenate([chains[:, :half], chains[:, n - half :]], axis=0)


def _rank_normalize(chains: np.ndarray) -> np.ndarray:
    flat = chains.ravel()
    ranks = stats.rankdata(flat, method="average")
    z = stats.norm.ppf((ranks - 0.375) / (flat.size + 0.25))
    return z.reshape(chains.shape)


def _rhat_basic(chains: np.ndarray) -> float:
    m, n = chains.shape
    w = chains.var(axis=1, ddof=1).mean()
    b = n * chains.mean(axis=1).var(ddof=1)
    if w <= 0:
        return float("nan")
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def rhat(chains: np.ndarray) -> float:
    split = _split(chains)
    bulk = _rhat_basic(_rank_normalize(split))
    folded = np.abs(split - np.median(split))
    tail = _rhat_basic(_rank_normalize(folded))
    return float(max(bulk, tail))


def ess_bulk(chains: np.ndarray) -> float:
    return ess(_rank_normalize(_split(chains)))


def ess_tail(chains: np.ndarray) -> float:
    split = _split(chains)
    out = []
    for q in (0.05, 0.95):
        cut = np.quantile(split, q)
        out.append(ess((split <= cut).astype(float)))
    return float(min(out))


def load_chains(run_dir: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((run_dir / "meta.json").read_text())
    chains = []
    for k in range(meta["chains"]):
        path = run_dir / f"chain_{k}.npz"
        if path.exists():
            with np.load(path) as z:
                chains.append({key: z[key] for key in z.files})
    return meta, chains


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--burn-frac", type=float, default=0.5)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--thin-out", default=None, help="pooled thinned model-space draws (.npz)")
    ap.add_argument("--n-thin", type=int, default=2000)
    ap.add_argument("--chains", default=None, help="comma-separated subset of chains")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    meta, chains = load_chains(run_dir)
    keep = [int(c) for c in args.chains.split(",")] if args.chains else list(range(len(chains)))
    chains = [chains[k] for k in keep]
    names = tuple(meta["parameter_names"])
    n_min = min(c["estimation_draws"].shape[0] for c in chains)
    burn = int(args.burn_frac * n_min)
    model = build_model(0)
    original = dict(model.parameters)

    def to_model(x: np.ndarray) -> np.ndarray:
        """Vectorized estimation -> model space (same formulas as transform_to_model_space)."""

        out = np.empty_like(x)
        for j, name in enumerate(names):
            par = original[name]
            col = x[:, j]
            if par.transform in ("identity", "untransformed"):
                out[:, j] = col
            elif par.transform == "exponential":
                out[:, j] = np.exp(col)
            elif par.value_bounds is None:
                out[:, j] = col * col
            else:
                lo, hi = par.value_bounds
                out[:, j] = lo + (hi - lo) / (1.0 + np.exp(-col))
        check = _model_values_for_estimation_vector(original, names, x[0])
        np.testing.assert_allclose(out[0], check, rtol=1e-12, atol=1e-12)
        return out

    est = np.stack([c["estimation_draws"][burn:n_min] for c in chains])  # (M, N, P)
    mod = np.stack([to_model(c) for c in est])
    lt = np.stack([c["log_target"][burn:n_min] for c in chains])
    m, n, p = mod.shape
    mode_values = json.loads((Path(__file__).resolve().parents[1] / meta["mode_json"]).read_text())[
        "parameter_values"
    ]
    rows = []
    for j, name in enumerate(names):
        x = mod[:, :, j]
        q = np.quantile(x, [0.05, 0.16, 0.5, 0.84, 0.95])
        per_chain_median = np.median(x, axis=1)
        prior = original[name].prior
        rows.append(
            {
                "parameter": name,
                "mode": float(mode_values[name]),
                "start_dsgejl": float(original[name].value),
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)),
                "q05": float(q[0]),
                "q16": float(q[1]),
                "median": float(q[2]),
                "q84": float(q[3]),
                "q95": float(q[4]),
                "rhat": rhat(x),
                "ess_bulk": ess_bulk(x),
                "ess_tail": ess_tail(x),
                "chain_median_min": float(per_chain_median.min()),
                "chain_median_max": float(per_chain_median.max()),
                "prior": "" if prior is None else prior.name,
                "prior_mean": None if prior is None else prior.mean,
                "prior_std": None if prior is None else prior.std,
                "prior_nu": None if prior is None else prior.nu,
                "prior_tau": None if prior is None else prior.tau,
            }
        )
    lt_summary = {
        "rhat": rhat(lt),
        "ess_bulk": ess_bulk(lt),
        "chain_means": lt.mean(axis=1).tolist(),
    }
    acceptance = [
        {
            "chain": keep[i],
            "proposals": int(c["proposals"]),
            "accepted": int(c["accepted"]),
            "rate": float(c["accepted"] / c["proposals"]),
            "post_burn_rate": float(
                np.mean(np.any(np.diff(c["estimation_draws"][burn:n_min], axis=0) != 0, axis=1))
            ),
            "start_log_target": float(meta["start_log_targets"][keep[i]]),
            "draws": int(c["estimation_draws"].shape[0]),
            "elapsed_s": float(c["elapsed"]),
        }
        for i, c in enumerate(chains)
    ]
    summary = {
        "run_dir": str(run_dir),
        "chains_used": keep,
        "draws_per_chain_used": n,
        "burn_in_per_chain": burn,
        "n_min": n_min,
        "scale": meta["scales"],
        "acceptance": acceptance,
        # chain_<k>.npz "log_target" holds the recorded model-space log posterior
        "log_posterior": lt_summary,
        "max_rhat": float(max(r["rhat"] for r in rows)),
        "min_ess_bulk": float(min(r["ess_bulk"] for r in rows)),
        "n_rhat_gt_1_01": int(sum(r["rhat"] > 1.01 for r in rows)),
        "n_rhat_gt_1_05": int(sum(r["rhat"] > 1.05 for r in rows)),
        "n_rhat_gt_1_1": int(sum(r["rhat"] > 1.1 for r in rows)),
        "parameters": rows,
    }
    # posterior correlations of h with everything else
    pooled = mod.reshape(-1, p)
    corr = np.corrcoef(pooled.T)
    hj = names.index("h")
    order = np.argsort(-np.abs(corr[hj]))
    summary["h_correlations"] = [(names[i], float(corr[hj, i])) for i in order[1:13]]
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=1))
    if args.out_csv:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.out_csv, index=False, float_format="%.6g")
    if args.thin_out:
        rng = np.random.default_rng(0)
        idx = rng.choice(pooled.shape[0], size=min(args.n_thin, pooled.shape[0]), replace=False)
        np.savez(
            args.thin_out,
            parameter_names=np.asarray(names),
            model_draws=pooled[idx],
            estimation_draws=est.reshape(-1, p)[idx],
        )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("parameters",)}, indent=1))


if __name__ == "__main__":
    main()
