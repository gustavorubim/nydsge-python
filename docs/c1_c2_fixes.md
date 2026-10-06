# C1/C2 fixes: missing observables, anticipated policy shocks, re-estimated mode

Branch `fix/c1-c2-observables`. Reproduce with `configs/quarterly_economy_c1c2.json`.
All new behaviour is opt-in. With the legacy config (`configs/quarterly_economy.json`),
the package reproduces `main`.

| Review item | Status on this branch |
|---|---|
| C1: four observable blocks missing (10y SPF inflation, 10y rate, TFP, OIS 1–6) | Three wired from free public sources. OIS is not wired; no defensible free proxy exists (see §2). |
| C2: anticipated policy shocks unidentified | `n_mon_anticipated_shocks = 0`, stated explicitly. The model drops `obs_nominalrate1..6`, `rm_shl1..6`, and the `rm_tl*` states. |
| H2: parameters not an estimated mode | Posterior mode over all 57 free parameters on the extended sample, with presample. |
| H3: +25 bp scenario double-counts the September hike | The 2026Q3 FFR is conditioned on the FRED DFF average (3.667%). Scenario deviations now start in 2026Q4, on top of the realised hike. |
| M1: raw-unit minimum-norm unemployment conditioning | Opt-in QQ^½ standardisation: `conditional_shock_weighting: "qq"`. |

## 1. Observables wired

`src/nydsge/public_sources.py` holds the loaders and parsers. The committed cache is
`data/public/nonfred_sources_20261006.csv`, built on 2026-10-06. The FRED snapshot for the
same day is `data/public/fred_levels_20261006.csv`. The tests read only these committed CSVs
and never touch the network. To refresh the data, call `build_public_sources()` and write the
result to a new CSV.

| Observable | DSGE.jl mnemonic | Free source used | Transform (same as DSGE.jl `m1002` `observables.jl`) | Span in cache |
|---|---|---|---|---|
| `obs_longinflation` | `ASACX10` | Philadelphia Fed SPF `INFCPI10YR` ([inflation.xlsx](https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/historical-data/inflation.xlsx)) | `(x − 0.5)/4`: quarterly, CPI→PCE wedge of 0.5 pp | 1979Q4–2026Q3 (164 obs) |
| `obs_longrate` | `FYCCZA` | Fed Board GSW yield curve `SVENY10` ([feds200628.csv](https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv)), quarterly mean of daily values | `x/4` | 1971Q3–2026Q3 (221 obs) |
| `obs_tfp` | `TFPKQ`, `TFPJQ` | SF Fed / Fernald `dtfp`, `alpha` ([quarterly_tfp.xlsx](https://www.frbsf.org/wp-content/uploads/quarterly_tfp.xlsx)) | `(dtfp − mean)/(4(1 − alpha))`. The mean is taken over [presample start − 1Q, mainsample end]. | 1959Q1–2026Q2 (270 obs) |

### Notes on the sources

**ASACX10 before 1991Q4.**
- The Haver series that DSGE.jl uses starts in 1979Q4, with semiannual Blue Chip/Livingston values. The Philadelphia Fed spreadsheet starts only in 1991Q4.
- The 24 pre-1991Q4 values (1979Q4 6.90 … 1991Q1 4.05) are copied verbatim from DSGE.jl's public reference input [`test/reference/input_data/raw/dlx_160812.csv`](https://raw.githubusercontent.com/FRBNY-DSGE/DSGE.jl/main/test/reference/input_data/raw/dlx_160812.csv).
- Quarters between those values are left missing (NaN), exactly as in DSGE.jl. The Kalman filter skips the missing entries.
- Before 1979Q4 the observable is NaN.
- From 1991Q4 on, the reference file matches the SPF series to rounding (max abs diff 0.005 pp).
- `tests/test_public_sources.py` checks reference points 1985Q4, 1991Q4, 2000Q1, and 2008Q4.

**Long rate: SVENY10, not GS10.**
- The task said "GS10 quarterly average /4". DSGE.jl's `FYCCZA` is in fact the Gürkaynak–Sack–Wright 10-year zero-coupon yield, not the GS10 constant-maturity par yield. DSGE.jl's `docs/src/frbny_data.md` names `FYCCZA` as the 10y zero-coupon yield.
- The cache matches `dlx_160812.csv` at the test points: 1971Q3 6.2371, 1985Q4 9.9393, 1991Q4 7.6687, 2000Q1 6.5776, 2008Q4 3.8926.
- So I used `SVENY10`. GS10 stays in the FRED file for reference.
- Over 1971Q3–2026Q3, SVENY10 − GS10 averages +4.5 bp with a 24 bp standard deviation. By period:

| Period | Mean SVENY10 − GS10 | Max abs difference |
|---|---:|---:|
| 1971–1989 | −16 bp | 99 bp |
| 1990–2007 | +19 bp | — |
| 2008–2019 | +15 bp | — |
| 2020–2026 | +4 bp | 9 bp |

- The 2026Q3 values are 4.787 (SVENY10) and 4.757 (GS10).

**TFP demeaning window (bug fix in `data.py`).**
- `_fernald_tfp` previously demeaned over every non-missing quarter in the levels frame.
- It now uses DSGE.jl's window: `date_presample_start − 1Q` through `date_forecast_start − 1Q`.
- Without date settings it falls back to the old behaviour, so the existing V&V tests are unchanged.

### Sample

| Item | Setting |
|---|---|
| Presample | 1959Q3–1959Q4, two periods, as in DSGE.jl |
| Likelihood start | 1960Q1 |
| Sample end | 2026Q2 (268 periods) |
| Levels | Loaded from 1959Q2 (`difference_lead_quarter: true`), so 1959Q3 growth rates exist |

FRED's `AWHNONAG` starts in 1964, so hours are NaN in 1959–1963. The legacy config started in 1964Q1 with no presample.

## 2. OIS 1–6 and C2: `n_mon_anticipated_shocks = 0`

DSGE.jl identifies the six anticipated policy shocks from OIS-implied expected FFR at horizons
1–6 quarters. That data is internal Board data, from 2008Q4 onward. I assessed the free
candidates:

| Candidate | Verdict |
|---|---|
| CME fed funds futures | Historical strips are not freely downloadable. Rejected. |
| T-bill / 1–2y CMT yields (FRED) | Contaminated by the bill–OIS basis (large and time-varying in 2008–09 and 2020) and by term premia. Using them as expected FFR would attribute basis and term-premium moves to forward guidance. Rejected. |
| NY Fed SME/SPD (Survey of Market Expectations / Primary Dealers) | Published as per-meeting summaries, not as a consistent machine-readable quarterly 1–6Q expected-FFR series back to 2008Q4. Rejected for now. The NY Fed uses SME FFR expectations as 2026Q3 conditioning information (Liberty Street, 2026-09-18). |
| SPF `TBILL` forecasts | Only about four quarterly horizons. T-bill, not FFR. Survey sluggishness. DSGE.jl docs mention SPF and Blue Chip as possible alternatives. This is the best future candidate, but not defensible as a drop-in OIS replacement. Not used. |

Decision: no OIS proxy is wired. The model is run with `n_mon_anticipated_shocks = 0`:
- 13 observables, 18 shocks, 78 states.
- `σ_r_m1..20` are fixed at 0.

Consequence: the ZLB period (2009–15, 2020–21) is absorbed by the unanticipated policy shock
`rm_sh` and by other shocks. Forward guidance cannot be separately identified. The legacy
setup kept six anticipated shocks with no data on them; their smoothed values were driven by
priors and cross-equation restrictions alone.

## 3. Re-estimation (posterior mode)

| Item | Detail |
|---|---|
| Script | `scripts/estimate_c1c2_mode.py` |
| Objective | DSGE.jl-style log posterior: Kalman likelihood from 1960Q1 with the 1959Q3–Q4 presample, plus the Model1002 priors. Optimised in the repo's unbounded estimation space. |
| Optimizer | scipy BFGS. Gradients are central differences (step 1e-4), with 114 likelihood evaluations per gradient, computed in parallel on a fork pool (8 workers). Up to 3 restarts. |
| Start values | DSGE.jl Model1002 defaults. These are the published `m1002` parameter values that `main` uses, apart from the 8 shock parameters `main` refreshes. |
| Free parameters | 57, all estimated. None remains at its starting value. `σ_b` moved least (0.0292 → 0.0296); its gradient is ≈ 0 at the mode. |
| Fixed (DSGE.jl fixed list, plus OIS) | `δ, Υ, λ_w, π*, ε_p, ε_w, Fω, γ*, g*, ρ_μe, ρ_γ, ρ_π* (=0.99), me_level, σ_μe, σ_γ, σ_r_m1..20 (=0, no OIS), Iendoα, γ_gdi, δ_gdi` |
| MCMC | None. No posterior was sampled, so forecast bands reflect future-shock uncertainty only, not parameter uncertainty. |

Two modes were estimated:

| | COVID-masked (recommended) | Full sample |
|---|---|---|
| File | `data/estimates/m1002_ss10_c1c2_mode.json` (copy of `..._maskcovid.json`) | `data/estimates/m1002_ss10_c1c2_mode_full.json` |
| Sample treatment | 2020Q2–2020Q4 observables set to NaN | all quarters |
| Log posterior at start | −5885.10 | −6591.55 |
| Log posterior at mode | **−1039.15** (log-lik −937.56, log prior −101.59) | **−1342.02** (log-lik −1224.36, log prior −117.65) |
| BFGS iterations / gradient evals / wall time | 113 / 244 / 455 s | 220 / 353 / 848 s |
| Gradient ∞-norm at mode | 3.5e-4 | 8.3e-4 |
| Hessian eigenvalues (finite-difference) | min 0.455, max 1.67e4, 0 negative | min 0.325, max 6.05e4, 0 negative |
| Restart check | Restarted from the full-sample mode: converged to the same −1039.1507 | Restarted from the masked mode: converged to the same −1342.0160 |

The two log posteriors are not comparable with each other, because the masked run has fewer
observations. The Hessian is positive definite in both cases. That plus identical optima from
two different starting points is good evidence of a local mode in each case. It is not proof of
a global mode, and no MCMC was run.

**Why mask 2020Q2–Q4.** The full-sample mode is visibly distorted by the COVID quarters:
- Habit `h` collapses to 0.10.
- `σ_c` rises to 1.69.
- A −50 bp policy shock raises GDP growth by 3.3 pp on impact and hours by 1.27%, about 2.5× the SR647 benchmark.

The NY Fed handles COVID with dedicated COVID shocks and measurement treatment, which ss10
lacks. Masking those three quarters is the simplest defensible substitute. `masked_quarters` in
the config applies the same mask in the package's smoother. The full-sample variant is kept as
`configs/quarterly_economy_c1c2_fullsample.json` for sensitivity.

### Key parameter movements

| Parameter | Start (DSGE.jl default) | Masked mode | Full-sample mode |
|---|---:|---:|---:|
| ψ1 (inflation response) | 1.37 | 1.41 | 1.13 |
| ψ2 (output gap response) | 0.039 | 0.087 | 0.091 |
| ψ3 (gap growth response) | 0.246 | 0.256 | 0.223 |
| ρ_R (rate smoothing) | 0.713 | 0.769 | 0.799 |
| ζ_p (Calvo prices) | 0.894 | 0.943 | 0.970 |
| ι_p (price indexation) | 0.186 | 0.332 | 0.354 |
| ζ_w (Calvo wages) | 0.929 | 0.907 | 0.928 |
| σ_π* | 0.0269 | 0.0341 | 0.0411 |
| ρ_π* | 0.99 (fixed) | 0.99 | 0.99 |
| ρ_λf | 0.883 | 0.894 | 0.849 |
| σ_λf | 0.131 | 0.062 | 0.074 |
| η_λf (MA) | 0.789 | 0.782 | 0.725 |
| ρ_λw | 0.388 | 0.508 | 0.570 |
| σ_λw | 0.386 | 0.366 | 0.407 |
| σ_r_m | 0.238 | 0.227 | 0.272 |
| h (habit) | 0.535 | 0.302 | 0.10 |
| σ_c | 0.872 | 1.49 | 1.69 |
| α | 0.160 | 0.218 | 0.287 |
| ρ_lr / σ_lr (10y measurement) | 0.694 / 0.177 | 0.964 / 0.115 | 0.961 / 0.115 |
| σ_tfp (TFP measurement) | 0.939 | 0.652 | 0.699 |
| σ_gdp / σ_gdi | 0.10 / 0.10 | 0.359 / 0.235 | 0.415 / 0.172 |

The full 57-parameter table is in the appendix. Standard errors in estimation space are stored
in each mode JSON under `diagnostics.se_estimation_space`. They come from the inverse
finite-difference Hessian and are not posterior standard deviations.

How to read the movements:
- **Policy rule.** It became more inertial (ρ_R 0.71 → 0.77) and twice as responsive to the gap (ψ2 0.04 → 0.09).
- **Prices.** They became stickier (ζ_p 0.89 → 0.94) and more indexed (ι_p 0.19 → 0.33).
- **Price markups.** Their volatility halved (σ_λf 0.131 → 0.062).
- **π\* and long-run expectations.** σ_π* rose, but it now has to fit the SPF 10y expectation. That pins π* close to the survey: smoothed long-run inflation is 2.33%, versus 2.94% before.

## 4. H3: conditioning on the actual 2026Q3 FFR

FRED `DFF` (series in the FRED snapshot) shows the FOMC move effective 2026-09-17: from 3.63%
(3.50–3.75 range) to 3.88% (3.75–4.00). The 2026Q3 DFF average is 3.667%; 2026Q4 to date is 3.88%.

The new config sets `conditioning_end_date: 2026-Q3` with conditioning observables:
- `obs_nominalrate`
- `obs_spread`
- `obs_longrate`
- `obs_longinflation`

This mirrors DSGE.jl's semi-conditional forecast (`cond_semi_names` = spread, nominal rate, long
rate), plus the 2026Q3 SPF 10y. The parameters are still estimated on data through 2026Q2. The
smoother uses the partial 2026Q3 row, and the forecast starts in 2026Q4.

Policy-path scenarios (e.g. `tightening_25bp`) are now deviations from 2026Q4 onward, relative
to a baseline that already contains the September hike. "+25 bp" therefore means an additional
hike, not a re-application of the realised one.

The baseline FFR path changes from 4.25, 4.59, 4.75, … (desk) to 3.67 (actual Q3), 3.81, 3.85, 3.83, …

## 5. M1: QQ^½-standardised minimum-norm conditioning

`solve_shocks_for_observable_targets(..., shock_scale=σ)` solves the minimum-norm problem in
z = ε/σ (equivalently, with a QQ⁻¹ weighting) and maps back with ε = σ·z. Zero-σ shocks are dropped.
The economy package uses it when `conditional_shock_weighting: "qq"`.

Raw-unit minimum norm penalises a 1-unit innovation equally for every shock, whatever its scale.
So it loads on shocks with large σ, measured in their own units. The QQ version picks the
smallest set of shocks in standard deviations.

Unemployment-conditioned scenarios, AFTER mode:

| Target | Raw: max shock (sd) | Raw: core PCE min dev (pp) | Raw: FFR min dev (pp) | QQ: max shock (sd) | QQ: core PCE min dev (pp) | QQ: FFR min dev (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 5% | 0.99 | −0.40 | −1.04 | 0.23 | −0.11 | −0.37 |
| 7% | 3.68 | −1.49 | −3.88 | 0.87 | −0.40 | −1.38 |
| 10% | 7.72 | −3.12 | −8.13 | 1.82 | −0.84 | −2.90 |

GDP growth paths are nearly identical between the two (5%: −1.15 vs −1.21 pp). The raw solution
gets there with roughly 4× larger shocks, in standard deviations, that are concentrated in
policy and markups.

## 6. BEFORE vs AFTER

Column definitions:
- **Desk:** `/workspace/nydsge/outputs/desk_fomc_20260916`. Data vintage around 2026-09-16, Econer's working tree, `n_ant = 6`, 8-parameter refresh.
- **Main @ new vintage:** HEAD `3004965` with `configs/quarterly_economy.json` on the 2026-10-06 FRED snapshot.
- **AFTER:** `configs/quarterly_economy_c1c2.json`. Masked mode, Q3-conditioned, `n_ant = 0`.
- **AFTER-full:** the full-sample mode, same conditioning.

Quarters are calendar quarters; h1 is 2026Q3. In AFTER, 2026Q3 is the smoothed value given the
Q3 conditioning data, not an unconditional forecast.

| Variable | Desk | Main @ new vintage | AFTER (masked) | AFTER-full |
|---|---:|---:|---:|---:|
| Core PCE h4 (2027Q2, % ann.) | 2.68 | 2.48 | **2.32** | 2.06 |
| Core PCE h8 (2028Q2) | 2.57 | 2.42 | **1.85** | 1.61 |
| Core PCE path h1–h8 | 3.03 2.86 2.75 2.68 2.63 2.60 2.58 2.57 | 2.78 2.62 2.53 2.48 2.44 2.42 2.42 2.42 | 2.99 2.73 2.51 2.32 2.17 2.04 1.94 1.85 | 2.85 2.53 2.27 2.06 1.90 1.78 1.68 1.61 |
| FFR h1–h8 (%) | 4.25 4.59 4.75 4.80 4.77 4.71 4.60 4.50 | 4.29 4.63 4.79 4.83 4.79 4.70 4.56 4.45 | 3.67* 3.81 3.85 3.83 3.78 3.71 3.65 3.59 | 3.67* 3.74 3.73 3.67 3.60 3.54 3.48 3.44 |
| Smoothed long-run inflation, 2026Q2 (π*, % ann.) | 2.94 | 2.94 | **2.33** | 2.42 |
| Output gap, 2026Q2 | −4.35 | −4.83 | −5.30 | −5.31 |
| Core PCE observed, 2026Q2 | 3.52 | 3.27 | 3.27 | 3.27 |

\* Actual 2026Q3 DFF average (conditioned).

The 2026Q2 core PCE observation itself was revised from 3.52 to 3.27 in the September 2026 NIPA
annual update. That explains about 0.2–0.25 pp of the near-term difference between Desk and the
other columns. AFTER without the Q3 conditioning gives:
- core PCE: 3.15, 2.88, 2.64, 2.45, 2.29, 2.15, 2.04, 1.94;
- FFR: 3.96, 4.07, 4.07, 4.01, 3.93, 3.85, 3.76, 3.69.

### 2026Q2 historical decomposition (smoothed-shock contributions)

Units: core PCE in pp (annualised, deviation from steady state); output gap in % of output.

| Driver | Core PCE: Desk | Core PCE: Main @ new vintage | Core PCE: AFTER | Core PCE: AFTER-full | Output gap: Desk | Output gap: Main @ new vintage | Output gap: AFTER | Output gap: AFTER-full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| π* (inflation target) | **2.46** | 2.45 | **0.51** | 0.53 | 1.96 | 1.95 | 0.52 | 0.95 |
| Monetary: unanticipated `rm_sh` | −0.02 | −0.02 | −0.09 | −0.20 | −0.01 | −0.04 | 0.18 | −0.02 |
| Monetary: anticipated `rm_shl1–6` | −0.16 | −0.20 | 0 (none) | 0 (none) | **−2.16** | −2.84 | 0 (none) | 0 (none) |
| λ_f price markup | 2.09 | 2.06 | **2.56** | 2.30 | −2.40 | −2.60 | −3.42 | −2.48 |
| λ_w wage markup | −0.94 | −0.93 | −0.25 | −0.25 | 0.45 | 0.48 | −0.09 | −0.24 |
| b (risk premium) | −0.59 | −0.65 | −0.15 | | −1.02 | −1.17 | −0.73 | −1.95 |
| g (government) | −0.47 | −0.44 | −0.57 | | −0.53 | −0.50 | −1.02 | −2.18 |
| Technology (ztil, zp) | −0.13 | −0.28 | −0.43 | | −1.78 | −1.60 | −1.74 | |
| Financial (μ, σ_ω, μ_e) | −0.09 | −0.11 | 0.13 | | 1.17 | 1.51 | 1.01 | 2.00 |
| Measurement | −0.60 | −0.58 | −0.45 | | — | — | — | — |
| Initial conditions + steady state | 1.97 | 1.97 | 2.00 | | | | | |

Notes on the decomposition table:
- Main @ new vintage shows the package's grouped totals: monetary −0.22 / −2.88 and markups 1.13 / −2.12 for core PCE / output gap. The λ_f / λ_w and rm_sh / rm_shl splits in that column come from `scripts/c1c2_compare.py` at the same parameters.
- Blank cells were not computed.
- AFTER reconciles to the smoothed observable to 1.7e-7.

What changed economically:
1. **π\* was the dominant driver of 2026 core inflation in the desk run (2.46 pp). It is now 0.51 pp.**
   - Without the SPF 10y observable, π* was a free low-frequency residual that soaked up persistent inflation. That is why the desk's long-run inflation was 2.94%.
   - With SPF 10y wired, π* is pinned near survey expectations (SPF CPI10 ≈ 2.3 → PCE-equivalent ≈ 1.8 + model wedge). Smoothed long-run inflation falls to 2.33%.
   - The persistent component is now attributed to price markups (λ_f 2.56 pp), which mean-revert. Hence the faster disinflation: h8 core PCE 1.85 vs 2.57.
2. **Anticipated monetary shocks no longer exist.** In the desk run, `rm_shl` contributed −2.16 pp to the output gap with no OIS data to discipline it. In AFTER, the monetary contribution to the gap is +0.18 (`rm_sh` only).
3. **λ_w no longer offsets λ_f** (−0.94 → −0.25). Part of the desk's markup split was compensating for the π* misattribution.

## 7. Caveats

- **Spread splice (L1).** It persists. FRED only exposes the last ~3 years of ICE BofA history (`BAMLC8A0C15PYEY` from 2023-10), so the spread observable still splices BAA−GS10 with ICE.
- **Population.** The HP filter endpoint problem is unchanged. GDP growth in the package is per capita; see `docs/nyfed_comparison.md` for the aggregate approximation.
- **No MCMC.** Bands are future-shock only. The mode is a local optimum confirmed from two starts.
- **ss10 is not the current NY Fed model.** The current NY Fed model has COVID shocks, flexible average-inflation targeting, and possibly other changes. See `docs/nyfed_comparison.md`.
- **No anticipated shocks.** The ZLB and forward-guidance episodes load on `rm_sh` and other shocks.
- **Data vintage.** The 2026-10-06 snapshot reflects the September 2026 NIPA annual revision. The desk run does not.
- **COVID mask.** It is a judgment call. The full-sample mode is provided for comparison, and its forecasts are lower for inflation and FFR.
- **TFP.** Unadjusted Fernald `dtfp`, as for ss10. The utilization-adjusted series is only for ss15/16.

## Appendix: all 57 estimated parameters
| Parameter | Start (DSGE.jl default) | Mode, COVID-masked (recommended) | Mode, full sample |
|---|---:|---:|---:|
| `alpha` | 0.1596 | 0.2182 | 0.2828 |
| `zeta_p` | 0.894 | 0.9431 | 0.973 |
| `iota_p` | 0.1865 | 0.3325 | 0.3445 |
| `Phi` | 1.107 | 1.095 | 0.957 |
| `Spp` | 2.731 | 2.083 | 1.351 |
| `h` | 0.5347 | 0.3023 | 0.1001 |
| `ppsi` | 0.6862 | 0.4829 | 0.3925 |
| `nu_l` | 2.598 | 3.191 | 2.806 |
| `zeta_w` | 0.9291 | 0.9074 | 0.9294 |
| `iota_w` | 0.2992 | 0.406 | 0.3825 |
| `beta` | 0.1402 | 0.09117 | 0.1137 |
| `psi1` | 1.368 | 1.414 | 1.14 |
| `psi2` | 0.0388 | 0.08716 | 0.09054 |
| `psi3` | 0.2464 | 0.256 | 0.2261 |
| `sigma_c` | 0.8719 | 1.488 | 1.677 |
| `rho` | 0.7126 | 0.7688 | 0.7964 |
| `spr` | 1.744 | 1.759 | 1.812 |
| `zeta_spb` | 0.0559 | 0.05172 | 0.05241 |
| `gamma` | 0.3673 | 0.3686 | 0.3707 |
| `Lmean` | -45.94 | -47.03 | -47.84 |
| `rho_g` | 0.9863 | 0.9814 | 0.9591 |
| `rho_b` | 0.941 | 0.9481 | 0.956 |
| `rho_mu` | 0.8735 | 0.7571 | 0.7224 |
| `rho_ztil` | 0.9446 | 0.9681 | 0.9156 |
| `rho_lambda_f` | 0.8827 | 0.8938 | 0.8498 |
| `rho_lambda_w` | 0.3884 | 0.5079 | 0.5588 |
| `rho_rm` | 0.2135 | 0.1547 | 0.2226 |
| `rho_sigma_w` | 0.9898 | 0.9974 | 0.9997 |
| `rho_lr` | 0.6936 | 0.9638 | 0.9612 |
| `rho_z_p` | 0.891 | 0.7584 | 0.5309 |
| `rho_tfp` | 0.1953 | 0.2013 | 0.1193 |
| `rho_gdpdef` | 0.5379 | 0.441 | 0.4284 |
| `rho_corepce` | 0.232 | 0.1384 | 0.1149 |
| `rho_gdp` | 0 | 0.8864 | 0.8638 |
| `rho_gdi` | 0 | 0.5412 | 0.1481 |
| `rho_gdpvar` | 0 | -0.2977 | -0.1257 |
| `sigma_g` | 2.523 | 2.272 | 2.545 |
| `sigma_b` | 0.0292 | 0.02961 | 0.03857 |
| `sigma_mu` | 0.4559 | 0.5485 | 0.7111 |
| `sigma_ztil` | 0.6742 | 0.5266 | 0.4369 |
| `sigma_lambda_f` | 0.1314 | 0.06197 | 0.07447 |
| `sigma_lambda_w` | 0.3864 | 0.3658 | 0.4079 |
| `sigma_r_m` | 0.238 | 0.2269 | 0.2731 |
| `sigma_sigma_omega` | 0.0428 | 0.03396 | 0.03547 |
| `sigma_pi_star` | 0.0269 | 0.03414 | 0.04081 |
| `sigma_lr` | 0.1766 | 0.1149 | 0.1148 |
| `sigma_z_p` | 0.1662 | 0.2139 | 0.3707 |
| `sigma_tfp` | 0.9391 | 0.6522 | 0.6931 |
| `sigma_gdpdef` | 0.1575 | 0.1843 | 0.1859 |
| `sigma_corepce` | 0.0999 | 0.1171 | 0.1074 |
| `sigma_gdp` | 0.1 | 0.3587 | 0.4112 |
| `sigma_gdi` | 0.1 | 0.2351 | 0.1832 |
| `eta_gz` | 0.84 | 0.408 | 0.4258 |
| `eta_lambda_f` | 0.7892 | 0.7825 | 0.7237 |
| `eta_lambda_w` | 0.4226 | 0.5064 | 0.5469 |
| `Gamma_gdpdef` | 1.035 | 1.049 | 1.051 |
| `delta_gdpdef` | 0.0181 | 0.01009 | 0.00943 |
