# Short MCMC on the C1/C2 masked configuration

Final results. The run took 8 chains × about 108k random-walk MH draws, from 12:41 to 15:55 ET on
2026-10-06. The post-processing pipeline (`/workspace/nydsge-mcmc/finish.sh`) finished at
16:10 ET.

## Summary

| Question | Answer |
|---|---|
| Converged? | **Acceptably, by conventional thresholds, for this length of run.** Max R-hat is 1.023, with 0 of 57 parameters above 1.05 and 9 above 1.01. Min bulk ESS is 286 and min tail ESS is 406. Mode and overdispersed starts agree. The slowest parameters are the technology processes (ρ_ztil, σ_ztil, ρ_z_p). |
| Habit persistence h | Median **0.326**, 68% [0.285, 0.368], 5–95% [0.259, 0.395]. That is far below the prior (median 0.71; P(h < 0.4) = 0.003) and below SR647's 0.73. **This comes from the likelihood.** Fixing h at 0.73 costs 18.5 log-likelihood points after re-optimising everything else. Low h survives dropping consumption, dropping investment, and a post-1985 likelihood. |
| Forecast with parameter uncertainty | Core PCE Q4/Q4 2026 / 2027 / 2028: 3.20 / 2.23 / 1.80 (posterior mean), against 3.22 / 2.26 / 1.82 at the mode and 3.3 / 2.1 / 1.8 from the NY Fed. Parameter uncertainty is small next to shock uncertainty. |

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
- Every chain ran for about 3.2 h of wall clock and ended at 108,000–109,000 draws.
- The draw files live in `/workspace/nydsge-mcmc/run1/` and are not committed. The parameter summary is committed as `data/estimates/mcmc_c1c2_summary.csv`.

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

## Convergence diagnostics (final)

Statistics: rank-normalized split R-hat (the larger of bulk and folded), bulk ESS and tail ESS
(the smaller of the 5% and 95% quantile ESS), pooled across 8 chains (Vehtari et al. 2021,
`scripts/mcmc_c1c2_summarize.py`). The first 50% of each chain is discarded as burn-in, leaving
54,000 draws per chain and 432,000 pooled.

**Acceptance and starts:**

| Chain | Start | Start log target | Draws | Acceptance (all) | Acceptance (post burn-in) |
|---|---|---:|---:|---:|---:|
| 0 | mode | −1126.2 | 108,000 | 0.263 | 0.260 |
| 1 | mode | −1126.2 | 108,000 | 0.263 | 0.263 |
| 2 | overdispersed | −1264.7 | 109,000 | 0.254 | 0.254 |
| 3 | overdispersed | −1278.9 | 109,000 | 0.254 | 0.250 |
| 4 | overdispersed | −1228.3 | 108,000 | 0.253 | 0.256 |
| 5 | overdispersed | −1243.4 | 108,000 | 0.255 | 0.255 |
| 6 | overdispersed | −1264.0 | 108,000 | 0.263 | 0.258 |
| 7 | overdispersed | −1275.9 | 108,000 | 0.256 | 0.252 |

Acceptance is 0.25–0.26, slightly below the 0.30 the pilot suggested. It is close to the
0.234 asymptotic optimum for random-walk MH, and inside the target range in spirit.

**R-hat and ESS:**
- Max R-hat is **1.023**. 0 parameters are above 1.05 and **9 of 57** are above 1.01: ρ_ztil 1.023, σ_ztil 1.023, ρ_z_p 1.019, ρ_λf 1.017, σ_z_p 1.016, σ_gdp 1.014, ζ_p 1.012, η_λf 1.012, ρ_gdp 1.011.
- Min bulk ESS is **286** (ρ_ztil), then σ_ztil 289, ρ_z_p 325, σ_z_p 386, ρ_λf 546. Min tail ESS is 406.

**Agreement across starts.** The log posterior has R-hat 1.009 and bulk ESS 1402. The eight
chain means are −1054.57 to −1055.71. The 8 chain medians for each key parameter are tight;
for h they span 0.321–0.329.

**Interim vs final.** At the interim read (14:41 ET, 30.5k post-burn-in draws per chain), max
R-hat was 1.065 and 3 parameters were above 1.05. Every one of these numbers improved with the
longer run.

**Verdict.** R-hat < 1.01 holds for 48 of 57 parameters, and all are < 1.025. By the stricter
modern standard, the technology block (ztil, z_p) and ρ_λf are borderline; their
95%-interval endpoints are the least reliable. The policy, rigidity, habit, financial, π* and
σ_r_m parameters all have R-hat ≤ 1.012 and ESS ≥ 750. Their medians and 5–95% bands are
usable, with the caveats below.

## Habit persistence h

| | Value |
|---|---|
| Posterior median | **0.326** |
| Posterior mean | 0.326 |
| 68% interval | [0.285, 0.368] |
| 5–95% interval | [0.259, 0.395] |
| R-hat / bulk ESS | 1.004 / 1479 |
| Prior Beta(14, 6), mean 0.7, sd 0.1 | median 0.707; 5–95% [0.524, 0.853]; P(h < 0.4) = 0.003 |
| BFGS mode (masked) | 0.302 |
| DSGE.jl starting value | 0.535 |
| SR647 posterior mean (Table 1) | 0.730 |
| Full-sample mode (COVID not masked; `c1_c2_fixes.md`) | 0.10 |

**Likelihood feature or prior/mode artifact?** It is a likelihood feature.
- The posterior median is slightly above the mode (0.326 vs 0.302), as expected when the Jacobian shifts mass inward from a bound, but the whole posterior sits where the prior puts 0.3% mass.
- The prior pulls toward 0.7, so the data has to work *against* the prior to land here.
- h's strongest posterior correlations:

| Parameter | Correlation with h |
|---|---:|
| σ_c | −0.64 |
| σ_b | −0.36 |
| ψ2 | −0.32 |
| ρ_rm | +0.28 |
| σ_μ | −0.25 |
| σ_z_p | +0.20 |

  The main trade-off is habit against intertemporal substitution: lower h goes with higher σ_c, and lower σ_c with higher h.

**Driver checks** (`scripts/h_profile_c1c2.py`). Each row is a full BFGS re-optimisation of all
other parameters from the masked mode. They are conditional modes, not marginal posteriors.
The reference is the masked mode: log posterior −1039.15 (log likelihood −937.56, log prior
−101.59), h = 0.302, σ_c = 1.488.

| Check | h at optimum | σ_c | Log posterior | Log likelihood | Δ log lik vs ref | Converged |
|---|---:|---:|---:|---:|---:|---|
| h fixed at 0.535 | 0.535 | 1.029 | −1049.41 | −945.17 | −7.61 | yes, 74 iterations, grad 6e-4 |
| h fixed at 0.73 | 0.73 | 0.552 | −1070.62 | −956.07 | **−18.51** | yes, 74 iterations, grad 4e-4 |
| `obs_consumption` dropped | **0.309** | 1.104 | −787.95 | −696.80 | n/a (different data) | yes, 76 iterations |
| `obs_investment` dropped | **0.354** | 1.311 | −538.38 | −436.32 | n/a | yes, 77 iterations |
| Likelihood from 1985Q1 only (filter still runs from 1959Q3) | **0.358** | 1.272 | −235.59 | −138.77 | n/a | yes, 62 iterations |

Reading the checks:
- **The likelihood clearly prefers low h.**
  - Forcing SR647's 0.73 costs 18.5 log-likelihood points (a likelihood-ratio statistic of 37 for one restriction). The log posterior falls by 31.5, because σ_c must collapse to 0.55, against its N(1.5, 0.37) prior.
  - DSGE.jl's 0.535 costs 7.6 log-likelihood points.
- **It is not driven by any one observable or by the early sample.**
  - Without the consumption series, h is still 0.31.
  - Without investment it is 0.35.
  - With only post-1985 data in the likelihood it is 0.36.
- **COVID pushes h even lower.** Unmasked, the mode is h = 0.10. Masking 2020Q2–Q4 is what lifts it to about 0.3.
- **Most likely interpretation.** The extended sample (to 2026Q2, with the new SPF and 10y observables and no anticipated policy shocks) favours a high-σ_c, low-habit configuration for consumption dynamics. SR647's 0.73 came from a different model vintage and a sample ending around 2008.
- **Not tested here:** which *combination* of observables drives h, and the role of the missing anticipated shocks. Treat consumption-response IRFs from this posterior with care.

## Posterior: key parameters (median, 68% and 5–95%)

| Parameter | Prior | Mode | Median | 16–84% | 5–95% | R-hat | ESS |
|---|---|---:|---:|---|---|---:|---:|
| ψ1 (inflation) | N(1.5, 0.25) | 1.414 | 1.458 | 1.302–1.632 | 1.212–1.755 | 1.006 | 1491 |
| ψ2 (gap) | N(0.12, 0.05) | 0.0872 | 0.0831 | 0.0670–0.1011 | 0.0573–0.1146 | 1.004 | 1374 |
| ψ3 (gap growth) | N(0.12, 0.05) | 0.256 | 0.271 | 0.245–0.298 | 0.229–0.318 | 1.004 | 1363 |
| ρ_R | B(0.75, 0.10) | 0.769 | 0.767 | 0.730–0.798 | 0.702–0.816 | 1.003 | 1297 |
| ζ_p | B(0.5, 0.1) | 0.943 | 0.942 | 0.927–0.953 | 0.914–0.959 | 1.012 | 756 |
| ι_p | B(0.5, 0.15) | 0.333 | 0.316 | 0.224–0.421 | 0.170–0.489 | 1.004 | 1613 |
| ζ_w | B(0.5, 0.1) | 0.907 | 0.912 | 0.898–0.923 | 0.888–0.930 | 1.005 | 1342 |
| σ_c | N(1.5, 0.37) | 1.488 | 1.491 | 1.361–1.630 | 1.275–1.724 | 1.005 | 1431 |
| h | B(0.7, 0.1) | 0.302 | 0.326 | 0.285–0.368 | 0.259–0.395 | 1.004 | 1479 |
| σ_π* | RIG(ν=6, τ=0.03) | 0.0341 | 0.0344 | 0.0301–0.0390 | 0.0274–0.0423 | 1.007 | 1419 |
| ρ_λf | B(0.5, 0.2) | 0.894 | 0.893 | 0.857–0.922 | 0.828–0.939 | 1.017 | 546 |
| σ_λf | RIG(ν=2, τ=0.1) | 0.0620 | 0.0608 | 0.0517–0.0704 | 0.0463–0.0771 | 1.005 | 1534 |
| η_λf | B(0.5, 0.2) | 0.783 | 0.761 | 0.673–0.830 | 0.602–0.863 | 1.012 | 998 |
| ρ_λw | B(0.5, 0.2) | 0.508 | 0.451 | 0.276–0.628 | 0.186–0.726 | 1.010 | 905 |
| σ_λw | RIG(ν=2, τ=0.1) | 0.366 | 0.368 | 0.345–0.394 | 0.331–0.411 | 1.004 | 1331 |
| σ_r_m | RIG(ν=2, τ=0.1) | 0.227 | 0.233 | 0.220–0.248 | 0.212–0.258 | 1.003 | 1332 |
| ρ_rm | B(0.5, 0.2) | 0.155 | 0.174 | 0.116–0.239 | 0.080–0.284 | 1.005 | 1169 |
| **Financial frictions** | | | | | | | |
| SP* (`spr`, ann. %) | G(2.0, 0.1) | 1.759 | 1.755 | 1.678–1.835 | 1.628–1.887 | 1.003 | 1676 |
| ζ_sp,b | B(0.05, 0.005) | 0.0517 | 0.0521 | 0.0484–0.0562 | 0.0462–0.0590 | 1.006 | 1894 |
| ρ_σω | B(0.75, 0.15) | 0.997 | 0.995 | 0.988–0.998 | 0.981–0.999 | 1.003 | 2185 |
| σ_σω | RIG(ν=4, τ=0.05) | 0.0340 | 0.0347 | 0.0307–0.0391 | 0.0283–0.0421 | 1.006 | 1696 |
| F(ω̄), γ* | fixed | | | | | | |
| **Other** | | | | | | | |
| S'' (`Spp`) | N(4, 1.5) | 2.083 | 1.971 | 1.525–2.536 | 1.289–2.941 | 1.004 | 844 |
| α | N(0.3, 0.05) | 0.218 | 0.209 | 0.195–0.224 | 0.186–0.234 | 1.003 | 1326 |

Prior notation: B = beta_alt(mean, sd), G = gamma_alt(mean, sd), RIG = root-inverse-gamma.

The posterior medians are close to the BFGS mode for every key parameter; the largest relative
shifts are in h, ρ_λw, ι_w and S''. The two notable features both survive:
- low habit;
- very sticky prices (ζ_p 0.94).

## Forecasts with parameter uncertainty vs mode vs NY Fed (Sep 2026)

**Method** (`scripts/mcmc_c1c2_forecast.py`):
- 1,000 parameter draws, sampled uniformly from the 432k pooled post-burn-in draws (2,000 thinned, then the first 1,000).
- For each draw, the COVID-masked RTS smoother with 2026Q3 semi-conditioning (DFF, spread, 10y, SPF10) gives the 2026Q3 state. From there the script computes the zero-shock forecast and 20 simulated future-shock paths, N(0, QQ).
- "Parameter only" = the distribution of the zero-shock paths across draws. "Parameter + shock" = all 20,000 simulated paths.
- "Mode" = the masked mode with 20,000 shock paths.
- GDP is converted from per capita to aggregate by adding a constant 1.055% population growth (see `nyfed_comparison.md`).
- At the mode the script reproduces the economy package exactly.

**Core PCE (% ann.)** h1 = 2026Q3, which is a smoothed value given Q3 rates data, so it carries parameter uncertainty only.

| Quarter | Mode | Posterior mean | Param-only 5–95% | Param + shock 5–95% | Param + shock 16–84% | Mode, shock-only 5–95% |
|---|---:|---:|---|---|---|---|
| h1 2026Q3 | 2.99 | 2.96 | 2.78–3.14 | 2.78–3.14 | 2.85–3.07 | 2.99 (point) |
| h4 2027Q2 | 2.32 | 2.30 | 2.06–2.49 | 0.52–4.06 | 1.24–3.36 | 0.61–4.04 |
| h8 2028Q2 | 1.85 | 1.82 | 1.58–2.03 | −0.32–3.98 | 0.53–3.13 | −0.24–3.96 |

**FFR (%)** h1 is the actual 2026Q3 DFF average and is held fixed.

| | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mode | 3.67 | 3.81 | 3.85 | 3.83 | 3.78 | 3.71 | 3.65 | 3.59 |
| Posterior mean | 3.67 | 3.85 | 3.92 | 3.92 | 3.88 | 3.83 | 3.77 | 3.71 |
| Param-only 5% | 3.67 | 3.76 | 3.76 | 3.71 | 3.62 | 3.54 | 3.47 | 3.41 |
| Param-only 95% | 3.67 | 3.93 | 4.06 | 4.10 | 4.10 | 4.07 | 4.04 | 4.01 |
| Param + shock 5% | 3.67 | 2.63 | 2.24 | 1.91 | 1.61 | 1.36 | 1.16 | 0.95 |
| Param + shock 95% | 3.67 | 5.05 | 5.58 | 5.92 | 6.13 | 6.29 | 6.36 | 6.46 |
| Param + shock 16% | 3.67 | 3.11 | 2.90 | 2.71 | 2.51 | 2.35 | 2.18 | 2.04 |
| Param + shock 84% | 3.67 | 4.58 | 4.93 | 5.14 | 5.26 | 5.32 | 5.34 | 5.39 |
| Mode, shock-only 5% | 3.67 | 2.59 | 2.18 | 1.84 | 1.54 | 1.27 | 1.07 | 0.91 |
| Mode, shock-only 95% | 3.67 | 4.99 | 5.52 | 5.80 | 6.01 | 6.16 | 6.24 | 6.32 |

The NY Fed does not publish an FFR path, so there is no comparison.

**Q4/Q4 (%)** The NY Fed figures, with their 68% bands, are from the Liberty Street post of
Sep 18, 2026: https://libertystreeteconomics.newyorkfed.org/2026/09/the-new-york-fed-dsge-model-forecast-september-2026/

| | 2026 | 2027 | 2028 |
|---|---:|---:|---:|
| **Core PCE, NY Fed Sep 2026** (68% band) | **3.3** (3.0, 3.7) | **2.1** (1.0, 3.2) | **1.8** (0.6, 3.0) |
| Core PCE, mode | 3.22 | 2.26 | 1.82 |
| Core PCE, posterior mean (median) | 3.20 (3.20) | 2.23 (2.24) | 1.80 (1.79) |
| Core PCE, param-only 5–95% | 3.12–3.29 | 2.00–2.42 | 1.57–1.99 |
| Core PCE, param + shock 16–84% | 3.00–3.40 | 1.32–3.13 | 0.64–2.96 |
| Core PCE, param + shock 5–95% | 2.86–3.54 | 0.72–3.73 | −0.11–3.73 |
| Core PCE, mode shock-only 16–84% | 3.02–3.41 | 1.39–3.14 | 0.69–2.97 |
| **GDP, NY Fed Sep 2026** (68% band) | **1.2** (−0.8, 3.1) | **0.1** (−5.2, 5.5) | **0.5** (−4.9, 5.9) |
| GDP, mode | 2.48 | 2.48 | 2.73 |
| GDP, posterior mean (median) | 2.37 (2.38) | 2.45 (2.45) | 2.77 (2.75) |
| GDP, param-only 5–95% | 2.15–2.58 | 1.89–3.00 | 2.25–3.28 |
| GDP, param + shock 16–84% | 1.55–3.19 | 0.16–4.75 | 0.42–5.11 |
| GDP, param + shock 5–95% | 1.00–3.72 | −1.33–6.22 | −1.11–6.72 |
| GDP, mode shock-only 16–84% | 1.62–3.31 | 0.18–4.71 | 0.36–5.08 |

Reading the forecasts:
- **Parameter uncertainty is second-order.** The parameter-only 90% band for core PCE is about ±0.2 pp at h4 and h8. Adding it to shock uncertainty barely widens the bands: 2027 core PCE 16–84% is 1.32–3.13, against 1.39–3.14 with shocks alone.
- **The posterior mean is very close to the mode.** Core PCE is 0.02–0.03 pp lower. The FFR path is 0.04–0.12 pp higher.
- **Core PCE vs the NY Fed.** The NY Fed means (3.3 / 2.1 / 1.8) all sit inside our parameter + shock 68% bands, and their 2027–28 means are within 0.15 pp of ours. The 2026 difference is mostly data vintage: the NY Fed uses pre-revision 2026Q1–Q2 core PCE (see `nyfed_comparison.md`).
- **GDP vs the NY Fed.** Their means (1.2 / 0.1 / 0.5) are at or below the bottom of our parameter + shock 68% bands (2026: 1.55–3.19; 2027: 0.16–4.75; 2028: 0.42–5.11), and *outside* the parameter-only 90% bands. Parameter uncertainty does not account for the GDP gap. The gap is a specification and conditioning difference: no anticipated policy shocks or SME path, no SPF Q3 GDP nowcast, and no COVID/AIT blocks.
- **Band widths are comparable to the NY Fed's.** For 2027 core PCE, our 68% band is 1.32–3.13 against their 1.0–3.2.

## 2026Q2 core PCE decomposition with posterior bands

Computed over the first 300 of the 1,000 parameter draws. Contributions are in pp of
annualised core PCE, from smoothed shocks (`scripts/c1c2_compare.py` logic). The 2026Q2
observation is 3.27 in every draw.

| Driver | Mode | Posterior median | 5–95% |
|---|---:|---:|---|
| Initial conditions + steady state | 2.00 | 2.01 | 1.96–2.04 |
| π* (inflation target) | 0.52 | **0.40** | −0.05–0.75 |
| λ_f (price markup) | 2.56 | **2.42** | 2.07–2.79 |
| λ_w (wage markup) | −0.25 | −0.32 | −0.54 to −0.17 |
| Monetary (`rm_sh`; no anticipated shocks) | −0.09 | −0.04 | −0.16–0.10 |
| b (risk premium) | −0.15 | −0.15 | −0.33 to −0.02 |
| g (government) | −0.57 | −0.56 | −0.82 to −0.36 |
| Technology (ztil + zp) | −0.43 | −0.09 | −0.49–0.01 |
| Financial (μ, σ_ω, μ_e, γ) | 0.13 | 0.11 | 0.05–0.20 |
| Measurement | −0.45 | −0.44 | −0.56 to −0.32 |

| Other 2026Q2 quantity | Mode | Posterior median | 5–95% |
|---|---:|---:|---|
| Smoothed long-run inflation | 2.33 | 2.26 | 2.00–2.44 |
| Output gap | −5.30 | −4.63 | −5.95 to −3.24 |

**Shares** of the shock-driven deviation (observed minus initial conditions/steady state, about
1.27 pp), as posterior medians with 5–95% bands:

| Driver | Median share | 5–95% |
|---|---:|---|
| π* | 0.31 | −0.04–0.57 |
| λ_f | 1.91 | 1.62–2.24 |
| λ_w | −0.25 | −0.42 to −0.14 |
| Monetary | −0.03 | −0.12–0.08 |

The λ_f share exceeds 1 because the other contributions are negative.

**Conclusion of the C1/C2 fix holds under parameter uncertainty.** Price markups, not π*,
dominate 2026 core inflation. π*'s contribution has a 95% upper bound of 0.75 pp, against
2.46 pp in the pre-fix desk run. The output-gap decomposition is noisier: the `rm_sh`
contribution has a median of 0.79 and a 5–95% band of −0.30–2.30.

## Caveats

- **Short run.** About 108k draws per chain, 432k post-burn-in in total, with R-hat up to 1.023 in the technology block. The bands for ρ_ztil, σ_ztil, ρ_z_p, σ_z_p and ρ_λf are the least reliable. A longer run, or a blocked or adaptive sampler, would tighten them.
- **Single proposal covariance.** All chains use the inverse Hessian at the mode, scaled by 0.5. The proposal was not adapted during the run.
- **What the bands include.** Forecast bands draw parameters and future shocks. They hold the smoothed 2026Q3 state fixed per draw (no simulation smoother), and h1 = 2026Q3 is a smoothed point per draw. Because they omit state uncertainty, the bands are somewhat too narrow, mainly at short horizons.
- **Decomposition draws.** The decomposition uses 300 of the 1,000 forecast draws.
- **h checks are conditional modes.** They are not marginal likelihoods or Bayes factors.
- **Specification differences remain** (see `c1_c2_fixes.md` and `nyfed_comparison.md`):
  - ss10, with no COVID shocks or flexible average-inflation targeting;
  - `n_mon_anticipated_shocks = 0`;
  - per-capita GDP converted to aggregate with constant population growth;
  - a post-revision data vintage;
  - the COVID-masking choice.
- **Sampler fix.** The repo sampler's Jacobian fix (see "Sampler fix" above) changes the target for any earlier MH results produced with this code. Those were biased for transformed parameters.

## Reproduce

```bash
uv run python scripts/mcmc_c1c2.py --out-dir /workspace/nydsge-mcmc/run1 --chains 8 \
  --n-mode-starts 2 --overdispersion 2.0 --scale 0.5 --hours 3.2 --chunk 1000
/workspace/nydsge-mcmc/finish.sh   # summarize, forecast, five h_profile runs
```

The finish steps are the commands in `finish.sh`:
- `scripts/mcmc_c1c2_summarize.py --burn-frac 0.5 --thin-out pooled.npz --n-thin 2000`;
- `scripts/mcmc_c1c2_forecast.py --n-param 1000 --n-shock 20 --n-decomp 300`;
- `scripts/h_profile_c1c2.py` with `--fix h=0.535`, `--fix h=0.73`, `--drop-obs obs_consumption`, `--drop-obs obs_investment` and `--likelihood-from 1985-Q1`.

## Appendix: all 57 estimated parameters

Columns: median and 5–95% from 432k pooled post-burn-in draws. The full CSV, with the mean,
sd, 16/84% quantiles, tail ESS, per-chain median range and DSGE.jl start values, is
`data/estimates/mcmc_c1c2_summary.csv`.

| Parameter | Prior | Mode | Median | 5% | 95% | R-hat | Bulk ESS |
|---|---|---:|---:|---:|---:|---:|---:|
| `alpha` | normal(0.3, 0.05) | 0.2182 | 0.2093 | 0.186 | 0.2339 | 1.003 | 1326 |
| `zeta_p` | beta_alt(0.5, 0.1) | 0.9431 | 0.9418 | 0.9136 | 0.9587 | 1.012 | 756 |
| `iota_p` | beta_alt(0.5, 0.15) | 0.3325 | 0.3162 | 0.1703 | 0.489 | 1.004 | 1613 |
| `Phi` | normal(1.25, 0.12) | 1.095 | 1.082 | 1.025 | 1.141 | 1.007 | 1500 |
| `Spp` | normal(4, 1.5) | 2.083 | 1.971 | 1.289 | 2.941 | 1.004 | 844 |
| `h` | beta_alt(0.7, 0.1) | 0.3023 | 0.326 | 0.2591 | 0.3951 | 1.004 | 1479 |
| `ppsi` | beta_alt(0.5, 0.15) | 0.4829 | 0.4682 | 0.3595 | 0.5868 | 1.005 | 1510 |
| `nu_l` | normal(2, 0.75) | 3.191 | 3.035 | 2.222 | 3.941 | 1.004 | 1555 |
| `zeta_w` | beta_alt(0.5, 0.1) | 0.9074 | 0.9118 | 0.8879 | 0.9303 | 1.005 | 1342 |
| `iota_w` | beta_alt(0.5, 0.15) | 0.406 | 0.3743 | 0.1948 | 0.5846 | 1.003 | 1465 |
| `beta` | gamma_alt(0.25, 0.1) | 0.09117 | 0.09811 | 0.04826 | 0.1665 | 1.003 | 1831 |
| `psi1` | normal(1.5, 0.25) | 1.414 | 1.458 | 1.212 | 1.755 | 1.006 | 1491 |
| `psi2` | normal(0.12, 0.05) | 0.08716 | 0.08308 | 0.0573 | 0.1146 | 1.004 | 1374 |
| `psi3` | normal(0.12, 0.05) | 0.256 | 0.2709 | 0.2293 | 0.3177 | 1.004 | 1363 |
| `sigma_c` | normal(1.5, 0.37) | 1.488 | 1.491 | 1.275 | 1.724 | 1.005 | 1431 |
| `rho` | beta_alt(0.75, 0.1) | 0.7688 | 0.7665 | 0.702 | 0.8159 | 1.003 | 1297 |
| `spr` | gamma_alt(2, 0.1) | 1.759 | 1.755 | 1.628 | 1.887 | 1.003 | 1676 |
| `zeta_spb` | beta_alt(0.05, 0.005) | 0.05172 | 0.05211 | 0.04615 | 0.05901 | 1.006 | 1894 |
| `gamma` | normal(0.4, 0.1) | 0.3686 | 0.3582 | 0.2998 | 0.415 | 1.005 | 1558 |
| `Lmean` | normal(-45, 5) | -47.03 | -47.19 | -48.53 | -45.93 | 1.003 | 1378 |
| `rho_g` | beta_alt(0.5, 0.2) | 0.9814 | 0.9802 | 0.9705 | 0.9885 | 1.010 | 1159 |
| `rho_b` | beta_alt(0.5, 0.2) | 0.9481 | 0.9464 | 0.9333 | 0.9565 | 1.008 | 1509 |
| `rho_mu` | beta_alt(0.5, 0.2) | 0.7571 | 0.7712 | 0.7103 | 0.8269 | 1.006 | 1060 |
| `rho_ztil` | beta_alt(0.5, 0.2) | 0.9681 | 0.9022 | 0.7487 | 0.9717 | 1.023 | 286 |
| `rho_lambda_f` | beta_alt(0.5, 0.2) | 0.8938 | 0.8931 | 0.8275 | 0.9386 | 1.017 | 546 |
| `rho_lambda_w` | beta_alt(0.5, 0.2) | 0.5079 | 0.4514 | 0.1863 | 0.7259 | 1.010 | 905 |
| `rho_rm` | beta_alt(0.5, 0.2) | 0.1547 | 0.1744 | 0.08034 | 0.2839 | 1.005 | 1169 |
| `rho_sigma_w` | beta_alt(0.75, 0.15) | 0.9974 | 0.9946 | 0.9807 | 0.9993 | 1.003 | 2185 |
| `rho_lr` | beta_alt(0.5, 0.2) | 0.9638 | 0.9618 | 0.9367 | 0.9819 | 1.006 | 1740 |
| `rho_z_p` | beta_alt(0.5, 0.2) | 0.7584 | 0.568 | 0.4065 | 0.7848 | 1.019 | 325 |
| `rho_tfp` | beta_alt(0.5, 0.2) | 0.2013 | 0.2248 | 0.1207 | 0.3321 | 1.004 | 1620 |
| `rho_gdpdef` | beta_alt(0.5, 0.2) | 0.441 | 0.4554 | 0.3306 | 0.573 | 1.008 | 1392 |
| `rho_corepce` | beta_alt(0.5, 0.2) | 0.1384 | 0.1679 | 0.05279 | 0.3379 | 1.008 | 1779 |
| `rho_gdp` | normal(0, 0.2) | 0.8864 | 0.07549 | -0.1165 | 0.2906 | 1.011 | 775 |
| `rho_gdi` | normal(0, 0.2) | 0.5412 | 0.9247 | 0.8843 | 0.9638 | 1.004 | 1094 |
| `rho_gdpvar` | normal(0, 0.4) | -0.2977 | -0.1639 | -0.7356 | 0.4505 | 1.005 | 1477 |
| `sigma_g` | RIG(ν=2, τ=0.1) | 2.272 | 2.196 | 2.023 | 2.391 | 1.006 | 1573 |
| `sigma_b` | RIG(ν=2, τ=0.1) | 0.02961 | 0.02836 | 0.02468 | 0.0326 | 1.004 | 1534 |
| `sigma_mu` | RIG(ν=2, τ=0.1) | 0.5485 | 0.5644 | 0.4763 | 0.7092 | 1.005 | 845 |
| `sigma_ztil` | RIG(ν=2, τ=0.1) | 0.5266 | 0.4435 | 0.3422 | 0.5425 | 1.023 | 289 |
| `sigma_lambda_f` | RIG(ν=2, τ=0.1) | 0.06197 | 0.06076 | 0.04628 | 0.0771 | 1.005 | 1534 |
| `sigma_lambda_w` | RIG(ν=2, τ=0.1) | 0.3658 | 0.3684 | 0.3312 | 0.4106 | 1.004 | 1331 |
| `sigma_r_m` | RIG(ν=2, τ=0.1) | 0.2269 | 0.2331 | 0.2122 | 0.2581 | 1.003 | 1332 |
| `sigma_sigma_omega` | RIG(ν=4, τ=0.05) | 0.03396 | 0.03474 | 0.02829 | 0.04209 | 1.006 | 1696 |
| `sigma_pi_star` | RIG(ν=6, τ=0.03) | 0.03414 | 0.03443 | 0.02743 | 0.04229 | 1.007 | 1419 |
| `sigma_lr` | RIG(ν=2, τ=0.75) | 0.1149 | 0.1161 | 0.1072 | 0.1258 | 1.005 | 1731 |
| `sigma_z_p` | RIG(ν=2, τ=0.1) | 0.2139 | 0.4036 | 0.1961 | 0.5486 | 1.016 | 386 |
| `sigma_tfp` | RIG(ν=2, τ=0.1) | 0.6522 | 0.6765 | 0.6154 | 0.7437 | 1.002 | 1777 |
| `sigma_gdpdef` | RIG(ν=2, τ=0.1) | 0.1843 | 0.1853 | 0.1692 | 0.2028 | 1.005 | 1523 |
| `sigma_corepce` | RIG(ν=2, τ=0.1) | 0.1171 | 0.1191 | 0.103 | 0.1364 | 1.003 | 1586 |
| `sigma_gdp` | RIG(ν=2, τ=0.1) | 0.3587 | 0.2561 | 0.2209 | 0.2981 | 1.014 | 595 |
| `sigma_gdi` | RIG(ν=2, τ=0.1) | 0.2351 | 0.3051 | 0.2735 | 0.3388 | 1.005 | 1562 |
| `eta_gz` | beta_alt(0.5, 0.2) | 0.408 | 0.4879 | 0.1752 | 0.8162 | 1.006 | 1003 |
| `eta_lambda_f` | beta_alt(0.5, 0.2) | 0.7825 | 0.7607 | 0.6022 | 0.8632 | 1.012 | 998 |
| `eta_lambda_w` | beta_alt(0.5, 0.2) | 0.5064 | 0.4434 | 0.2029 | 0.7118 | 1.009 | 1011 |
| `Gamma_gdpdef` | normal(1, 2) | 1.049 | 1.05 | 0.9766 | 1.125 | 1.005 | 1407 |
| `delta_gdpdef` | normal(0, 2) | 0.01009 | 0.01068 | -0.03341 | 0.05441 | 1.009 | 1295 |
