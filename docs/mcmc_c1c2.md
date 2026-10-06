# Short MCMC on the C1/C2 masked configuration

> **Status: INTERIM.** The production run (8 chains, 3.2 h wall clock, started 12:41 ET on
> 2026-10-06) was still running when this file was written. Every number below comes from
> the chains read at **14:41 ET**, about 61,000 draws per chain, with the first 50% of each
> chain discarded. Final diagnostics, forecast bands, decomposition bands and the h-driver
> checks will replace this section when the run completes.

## Setup

**Target.**
- Configuration: `configs/quarterly_economy_c1c2.json`. 2020Q2–Q4 observables are masked, the presample is 1959Q3–Q4, and the likelihood runs from 1960Q1 to 2026Q2.
- All 57 free parameters are sampled. `n_mon_anticipated_shocks = 0`.
- The density is the log likelihood plus the log prior. Sampling happens in the repo's unbounded estimation space, so the target also includes log|dθ/dx|.

**Sampler.**
- It is the repo's `nydsge.estimate.metropolis_hastings` (random walk), driven by `scripts/mcmc_c1c2.py`.
- Each chain runs in its own process. The driver calls the sampler in chunks of 1,000 draws and checkpoints to `chain_<k>.npz` after every chunk. The chain state carries over between chunks.

**Proposal.**
- The proposal is N(0, c²·H⁻¹), where H is the finite-difference Hessian at the BFGS mode, in estimation space (`data/estimates/m1002_ss10_c1c2_mode_maskcovid.hessian.npy`).
- I tuned the scale c with a pilot: 8 chains started at the mode, about 1,250–1,500 draws each.

| c | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.50 | 0.60 |
|---|---|---|---|---|---|---|---|---|
| Acceptance | 0.95 | 0.89 | 0.81 | 0.72 | 0.59 | 0.50 | 0.30 | 0.11 |

  Production uses **c = 0.5**.

**Chains.**
- There are 8 chains: 2 start at the mode, and 6 start at overdispersed points, mode + 2·N(0, H⁻¹).
- The overdispersed starts have log targets of −1228 to −1279. The log target at the mode is −1126.17 (log posterior −1039.15 plus log Jacobian −87.02).
- The draw files live in `/workspace/nydsge-mcmc/run1/` and are not committed.

## Sampler fix (in this PR)

`metropolis_hastings` proposes random-walk steps in the transformed (unbounded) space, but it
used to evaluate the *model-space* log posterior without the change-of-variables term. Its draws
therefore came from p(θ(x))·|dx/dθ| rather than from the posterior, which biases every
transformed parameter: logistic-bounded ones such as h and the ζ's, and exponential ones such as
the σ's.

What changed:
- `jacobian=True` (now the default) adds Σᵢ log|dθᵢ/dxᵢ|. DSGE.jl's sampler works in model space, so there the term is unnecessary; with the Jacobian, the transformed-space sampler targets the same posterior.
- The new helpers are `parameters.log_abs_jacobian_to_model_space` (stable logistic form) and `estimate.log_abs_jacobian`.
- The recorded `log_posterior` draws stay model-space (no Jacobian), so they remain comparable with `estimate(...).log_posterior`.
- `log_likelihood_start` is now passed through, so presample periods can be excluded from the likelihood as in the mode search.
- `jacobian=False` reproduces the old behaviour.
- Tests are in `tests/test_mh_jacobian.py`: a numerical-derivative check for each transform, target arithmetic, presample forwarding, and chain continuation across calls.

## Interim convergence diagnostics (14:41 ET, not final)

Statistics: rank-normalized split R-hat (the larger of bulk and folded) and bulk ESS, pooled
across the 8 chains (Vehtari et al. 2021, `scripts/mcmc_c1c2_summarize.py`). Each chain keeps
30,500 draws after burn-in.

- **Acceptance per chain:** 0.252–0.267 over the whole run, 0.248–0.264 after burn-in.
- **R-hat:** max 1.065 (σ_ztil 1.065, ρ_ztil 1.064, ρ_z_p 1.053). 3 of 57 parameters are above 1.05, none above 1.1, and 35 are above 1.01.
- **Bulk ESS:** minimum 94, in the technology block.
- **Log posterior:** R-hat 1.007, ESS 788. The eight chain means are −1054.4 to −1055.5, so the overdispersed and mode starts agree on the log posterior.

**Verdict at this point:** *not yet converged* by the usual R-hat < 1.01 standard. The policy,
nominal-rigidity, habit and financial parameters are close (R-hat 1.004–1.021). The technology
processes (ztil, z_p) mix slowly. Bands at this stage should not be treated as reliable.

## Interim posterior, key parameters (median, 5–95%)

| Parameter | Mode | Median | 5% | 95% | R-hat | ESS |
|---|---:|---:|---:|---:|---:|---:|
| ψ1 | 1.414 | 1.467 | 1.207 | 1.759 | 1.013 | 727 |
| ψ2 | 0.0872 | 0.0821 | 0.0567 | 0.1149 | 1.014 | 767 |
| ρ_R | 0.769 | 0.765 | 0.701 | 0.815 | 1.009 | 683 |
| ζ_p | 0.943 | 0.941 | 0.914 | 0.960 | 1.011 | 423 |
| ι_p | 0.333 | 0.317 | 0.162 | 0.498 | 1.004 | 792 |
| ζ_w | 0.907 | 0.912 | 0.886 | 0.932 | 1.015 | 677 |
| σ_c | 1.488 | 1.485 | 1.271 | 1.728 | 1.021 | 651 |
| h | 0.302 | 0.329 | 0.262 | 0.401 | 1.013 | 872 |
| σ_π* | 0.0341 | 0.0343 | 0.0273 | 0.0421 | 1.012 | 757 |
| ρ_λf | 0.894 | 0.894 | 0.828 | 0.939 | 1.011 | 433 |
| σ_λf | 0.0620 | 0.0605 | 0.0459 | 0.0770 | 1.007 | 798 |
| σ_r_m | 0.227 | 0.233 | 0.212 | 0.258 | 1.016 | 709 |
| SP* (`spr`) | 1.759 | 1.756 | 1.632 | 1.889 | 1.005 | 973 |
| ζ_sp,b | 0.0517 | 0.0522 | 0.0459 | 0.0590 | 1.008 | 977 |
| ρ_σω | 0.997 | 0.995 | 0.980 | 0.999 | 1.011 | 1101 |
| σ_σω | 0.0340 | 0.0354 | 0.0285 | 0.0429 | 1.011 | 784 |

`F(ω̄)` and γ* are fixed.

## Interim habit persistence h

| | Value |
|---|---|
| Posterior median | **0.329** |
| 68% interval | [0.288, 0.372] |
| 5–95% interval | [0.262, 0.401] |
| Spread of the 8 chain medians | 0.323–0.338 |
| Prior Beta(14, 6), i.e. mean 0.7, sd 0.1 | median 0.707; 5–95% [0.524, 0.853]; P(h < 0.4) = 0.003 |
| BFGS mode | 0.302 |
| DSGE.jl starting value | 0.535 |
| SR647 posterior mean | 0.73 |

The posterior sits almost entirely where the prior has 0.3% mass, so low h is a feature of
the likelihood, not a prior or optimizer artifact. The largest posterior correlations of h
are with:
- σ_c: −0.67
- ψ2: −0.36
- ρ_rm: +0.34
- σ_b: −0.34

Checks of what drives low h (profiles with h fixed at 0.535 / 0.73, dropping consumption or
investment, a post-1985 likelihood) are pending; the script is `scripts/h_profile_c1c2.py`.

## Commands to finish (after the run ends at about 15:55 ET)

```bash
cd /workspace/nydsge-fix
R=/workspace/nydsge-mcmc/run1
uv run python scripts/mcmc_c1c2_summarize.py --run-dir $R --burn-frac 0.5 \
  --out-json $R/summary.json --out-csv data/estimates/mcmc_c1c2_summary.csv \
  --thin-out $R/pooled.npz --n-thin 2000
uv run python scripts/mcmc_c1c2_forecast.py --draws $R/pooled.npz --n-param 1000 \
  --n-shock 20 --n-decomp 300 --workers 8 --out $R/forecast.json
# h drivers (each is a BFGS re-optimization from the masked mode; run 2 at a time with 4 workers)
uv run python scripts/h_profile_c1c2.py --fix h=0.535 --workers 4 --out $R/h_fix0535.json
uv run python scripts/h_profile_c1c2.py --fix h=0.73  --workers 4 --out $R/h_fix073.json
uv run python scripts/h_profile_c1c2.py --drop-obs obs_consumption --workers 4 --out $R/h_nocons.json
uv run python scripts/h_profile_c1c2.py --drop-obs obs_investment  --workers 4 --out $R/h_noinv.json
uv run python scripts/h_profile_c1c2.py --likelihood-from 1985-Q1  --workers 4 --out $R/h_post85.json
```

How to read the outputs:
- **Forecast.** `forecast.json` holds draw-averaged bands for core PCE h1–h8, FFR h1–h8, and core PCE / GDP Q4/Q4 2026–28. It has two variants: parameter-only (zero future shocks) and parameter + future shocks. It also has the mode reference (mode mean plus shock-only bands), and the 2026Q2 core PCE and output-gap decomposition with posterior bands.
- **Validation.** At the mode, the forecast script reproduces the package exactly: core PCE 2.99, 2.73, … and Q4/Q4 3.22 / 2.26 / 1.82.
- **What the bands omit.** Smoothed-state uncertainty is not sampled, and 2026Q3 is a smoothed point for each draw.
