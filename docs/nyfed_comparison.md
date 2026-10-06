# Comparison with published NY Fed DSGE material

Every number below was either taken from the cited public source (URL given) or computed in
this repo by the named script. Repo runs are compared in three versions:
- **old**: the desk run `outputs/desk_fomc_20260916` (Econer's tree) and `main @ 3004965` on the 2026-10-06 FRED snapshot;
- **fixed**: `configs/quarterly_economy_c1c2.json`, using the COVID-masked mode;
- **fixed-full**: the full-sample mode.

## Sources

| Source | URL | Used for |
|---|---|---|
| Liberty Street Economics, "The New York Fed DSGE Model Forecast—September 2026" (Del Negro, Dogra, Elbarmi, Lee, Pham, Wright; Sep 18, 2026) | https://libertystreeteconomics.newyorkfed.org/2026/09/the-new-york-fed-dsge-model-forecast-september-2026/ | Q4/Q4 forecast table (Sep and Jun 2026), conditioning information, narrative |
| NY Fed DSGE model page | https://www.newyorkfed.org/research/policy/dsge | Model Q&A, chart data |
| NY Fed DSGE chart data (CSV), downloaded 2026-10-06 to `data/public/nyfed_dsge_20260918/` | `https://www.newyorkfed.org/medialibrary/research/interactives/data/dsge/<file>.csv`, e.g. [nyfed_dsgeTableData.csv](https://www.newyorkfed.org/medialibrary/research/interactives/data/dsge/nyfed_dsgeTableData.csv) | Quarterly core PCE / GDP paths, r* |
| Del Negro, Eusepi, Giannoni, Sbordone, Tambalotti, Cocci, Hasegawa, Linder, "The FRBNY DSGE Model", Staff Report 647 (2013) | https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr647.pdf | IRF benchmarks (text, Section 4 / Figs. 8–10), Table 1 posterior |
| DSGE.jl Model1002 documentation (PDF) | https://github.com/FRBNY-DSGE/DSGE.jl/blob/main/docs/DSGE_Model_Documentation_1002.pdf | Equations and priors only. **No IRFs and no posterior estimates are published in it.** |
| DSGE.jl data documentation | https://raw.githubusercontent.com/FRBNY-DSGE/DSGE.jl/main/docs/src/frbny_data.md | Observable mnemonics |
| DSGE.jl `m1002.jl` source | https://github.com/FRBNY-DSGE/DSGE.jl/blob/main/src/models/representative/m1002/m1002.jl | COVID and AIT (pgap/ygap) model blocks |

DSGE.jl reference outputs: the repo's V&V suite (`docs/translation_validation_2026q2.md`) already
shows parity with DSGE.jl at the default parameters. The repo's IRFs at those parameters are
therefore DSGE.jl's IRFs. The DSGE.jl IRF test fixture (`test_irfdsge.jld2`) is a small synthetic
system, not Model1002, so it gives no additional benchmark.

## Specification differences

These must be kept in mind before reading any gap below as an error.

| | NY Fed production model (Sep 2026) | This repo |
|---|---|---|
| Model | Current FRBNY DSGE: a Model1002-family model with COVID-19 shocks and a flexible average-inflation-targeting policy rule. DSGE.jl's public `m1002.jl` has the corresponding hooks: COVID states and shocks (`g_covid_sh`, `λ_f_covid_sh`, `σ_ω_covid_sh`, … at lines ~201–250) and price/output-gap states `pgap_t`/`ygap_t` (`add_pgap`, `add_altpolicy_pgap`, lines ~280–297). The exact production settings are not published. | Model1002 **ss10** (DSGE.jl subspec), without the COVID or pgap/ygap blocks, and with the standard Taylor-type rule |
| Anticipated policy shocks | Yes, identified from market FFR expectations | **Fixed: 0** (no free OIS data, see `c1_c2_fixes.md` §2). Old: 6 with no data. |
| Estimation | Full Bayesian posterior (SMC/MCMC), with parameter uncertainty in the bands | **Fixed: posterior mode only**, with bands from future shocks only. Old: DSGE.jl default parameters plus an 8-parameter refresh. |
| 2026Q3 conditioning | SPF nowcasts of GDP and core PCE, SPF short-run inflation expectations, SME FFR expectations, 10y Treasury, Baa (Liberty Street) | **Fixed: actual 2026Q3 FFR (DFF), spread, 10y GSW yield, SPF 10y CPI.** No GDP or core PCE nowcast. Old: none. |
| Data vintage | Data released through 2026Q2, before the Sep-2026 NIPA annual revision (core PCE 2026Q1/Q2 history in their CSV: 4.42 / 3.57) | 2026-10-06 FRED, post-revision (3.88 / 3.27). Desk: 4.35 / 3.52. |
| GDP | Aggregate real GDP growth | Per-capita `obs_gdp`. Here it is converted to aggregate by adding terminal HP-trend population growth (1.06–1.07% ann.), held constant. |
| Federal funds rate path | **Not published** | Reported. No comparison is possible. |

## Forecast comparison (Q4/Q4, %)

Computed by `scripts/c1c2_nyfed_compare.py`. The NY Fed figures are from the Liberty Street
table, which agrees with the CSVs.

| | 2026 | 2027 | 2028 | 2029 |
|---|---:|---:|---:|---:|
| **Core PCE**, NY Fed Sep 2026 (68% band) | 3.3 (3.0, 3.7) | 2.1 (1.0, 3.2) | 1.8 (0.6, 3.0) | 1.8 (0.5, 3.1) |
| Core PCE, NY Fed Jun 2026 | 3.1 | 1.8 | 1.6 | 1.7 |
| Core PCE, old: desk | 3.44 | 2.67 | 2.58 | — |
| Core PCE, old: main @ new vintage | 3.14 | 2.47 | 2.44 | — |
| Core PCE, **fixed** | 3.22 | **2.26** | **1.82** | — |
| Core PCE, fixed-full | 3.13 | 2.00 | 1.60 | — |
| **GDP growth**, NY Fed Sep 2026 (68% band) | 1.2 (−0.8, 3.1) | 0.1 (−5.2, 5.5) | 0.5 (−4.9, 5.9) | 1.2 (−4.4, 6.8) |
| GDP, old: desk (aggregate approx.) | 1.72 | 2.49 | 2.90 | — |
| GDP, old: main @ new vintage | 2.76 | 2.66 | 2.98 | — |
| GDP, **fixed** | 2.48 | 2.48 | 2.73 | — |
| GDP, fixed-full | 2.57 | 2.66 | 2.87 | — |
| **Short-run r\*** (Q4), NY Fed Sep 2026 | 1.9 | 1.8 | 1.4 | 1.2 |
| r*, old: desk | 0.67 | 1.19 | 1.17 | — |
| r*, old: main @ new vintage | 0.75 | 1.26 | 1.19 | — |
| r*, **fixed** | 0.98 | 1.85 | 1.85 | — |
| r*, fixed-full | 1.53 | 2.24 | 2.17 | — |

**Vintage adjustment for 2026.** Each run's 2026 Q4/Q4 uses its own Q1–Q2 history. If the NY Fed's
pre-revision Q1–Q2 core PCE history is used instead, 2026 core PCE becomes:

| Run | 2026 core PCE (NY Fed history) | NY Fed |
|---|---:|---:|
| Fixed | 3.43 | 3.32 (recomputed from its CSV) |
| Desk | 3.47 | 3.32 |

The forecast-only comparison for 2026Q3–Q4 core PCE (% ann.):

| Run | 2026Q3 | 2026Q4 |
|---|---:|---:|
| NY Fed | 2.76 | 2.53 |
| Fixed | 2.99 | 2.73 |
| Desk | 3.03 | 2.86 |

### Quarterly core PCE path (% ann.)

| Quarter | NY Fed | Desk (old) | Main @ new vintage (old) | Fixed | Fixed minus NY Fed |
|---|---:|---:|---:|---:|---:|
| 2026Q3 | 2.76 | 3.03 | 2.78 | 2.99 | +0.23 |
| 2026Q4 | 2.53 | 2.86 | 2.62 | 2.73 | +0.20 |
| 2027Q1 | 2.31 | 2.75 | 2.53 | 2.51 | +0.20 |
| 2027Q2 | 2.12 | 2.68 | 2.48 | 2.32 | +0.20 |
| 2027Q3 | 2.00 | 2.63 | 2.44 | 2.17 | +0.17 |
| 2027Q4 | 1.91 | 2.60 | 2.42 | 2.04 | +0.13 |
| 2028Q1 | 1.85 | 2.58 | 2.42 | 1.94 | +0.08 |
| 2028Q2 | 1.80 | 2.57 | 2.42 | 1.85 | +0.05 |
| 2028Q3 | 1.78 | 2.58 | 2.44 | 1.78 | 0.00 |
| 2028Q4 | 1.77 | 2.59 | 2.46 | 1.72 | −0.05 |

### Gap analysis

**Inflation: the fix closes most of the gap.**
- Gaps against the NY Fed 2027 / 2028 Q4/Q4 core PCE:

| Run | Gap 2027 (pp) | Gap 2028 (pp) |
|---|---:|---:|
| Old: desk | +0.57 | +0.78 |
| Old: main @ new vintage | +0.37 | +0.64 |
| Fixed | +0.16 | +0.02 |

- In the old setup, inflation flattened at about 2.4–2.6% because smoothed π* was 2.94%. Without the SPF 10y observable, π* was a free low-frequency residual.
- With SPF 10y wired, π* falls to 2.33%. The fixed path converges toward the NY Fed path. It is about 0.2 pp higher through 2027Q2, the same by 2028Q3, and its endpoint is anchored near 1.7–1.8%.
- The remaining near-term gap is plausibly explained by three things:
  - The NY Fed conditions on the SPF Q3 core PCE nowcast; this repo does not.
  - Different TFP treatment: the NY Fed writes that lower TFP growth *raises* inflation in their forecast.
  - Spec differences: flexible AIT and the COVID shocks.

**GDP: still a large gap.**
- The NY Fed model is pessimistic: 0.1% in 2027 and 0.5% in 2028. Its post attributes this to more restrictive expected policy (from SME expectations, i.e. anticipated shocks) and to lower TFP growth.
- The repo is at 2.5–2.9% in every version. Note that per-capita growth plus about 1.06% population growth is mechanically 1 pp above a per-capita reading.
- Without anticipated policy shocks and without SME data, this repo cannot replicate the NY Fed's "more restrictive than expected" channel. That is the most likely single source of the 2027–28 GDP gap, about 2.4 pp.
- The residual is attributable to unmodelled COVID/AIT features and to the per-capita/aggregate approximation.

**Short-run r\*.**
- The fixed version moves toward the NY Fed in 2027: 1.85 vs 1.8 (old 1.19).
- It stays below the NY Fed in 2026: 0.98 vs 1.9.
- It overshoots in 2028: 1.85 vs 1.4.
- r* is a model-specific latent object, so these gaps should be read loosely.

## Impulse-response comparison

Computed with `scripts/c1c2_irfs.py`. The "old" column uses DSGE.jl default parameters; the desk
refresh does not change these shocks' IRFs. Published benchmarks come from SR647 (2013, the
predecessor model, posterior mean). Its text gives approximate numbers; its figures were not
digitised.

| Shock / response | SR647 published (text) | Old (DSGE.jl defaults) | Fixed (masked mode) | Fixed-full |
|---|---|---:|---:|---:|
| −50 bp unanticipated policy: hours peak | "humped shaped … roughly 0.5% at the peak" | 0.47% at q2 | 0.54% at q2 | 1.27% |
| −50 bp: output growth on impact | "+0.7% on impact", back to steady state in ≈1 yr | +1.29 pp ann. | +1.55 pp ann. | +3.33 pp ann. |
| −50 bp: FFR persistence | "reabsorbed over the course of two years" | half-life 3q; within 10% of impact after 11q | half-life 2q; within 10% after 5q | half-life 3q; within 10% after 5q |
| −50 bp: core PCE peak | "relatively modest, but fairly persistent increase" | +0.023 pp | +0.025 pp | — |
| Size of a 1 sd `rm_sh` on FFR impact | σ_r = 0.139 (SR647 Table 1, quarterly) | 0.566 pp ann. | 0.486 pp ann. | 0.414 pp ann. |
| 1 sd spread (σ_ω) shock: spread impact | "roughly 35 basis points" | 18.6 bp | 14.8 bp | 15.3 bp |
| Anticipated (4q-ahead) −50 bp | "leads to higher interest rates now" | FFR +0.42 pp on impact, GDP growth +1.51 pp | n/a (no anticipated shocks) | n/a |
| 1 sd π* shock: core PCE h0 / h4 / h20 | (no π* shock in SR647) | 0.21 / 0.25 / 0.22 | 0.149 / 0.222 / 0.205 | — |
| 1 sd π*: output gap peak | — | 0.22 at q10 | 0.235 at q12 | — |
| 1 sd π*: π response / π* response at h10 | — | 2.50 | 1.77 | 1.45 |
| 1 sd λ_f (price markup): core PCE impact / h4 | — | 0.89 / 0.18 | 0.59 / 0.27 | 0.67 / — |
| 1 sd λ_f: GDP growth impact | — | −0.51 pp | −0.05 pp | +0.26 pp |
| 1 sd λ_f: FFR peak | — | 0.23 at q1 | 0.23 at q2 | — |

Reading the IRFs:
- **Monetary policy.**
  - The fixed mode's hours response (0.54% at q2, hump-shaped) matches SR647's ≈0.5%.
  - The impact on output growth is about 2× SR647's 0.7%. The comparison is loose: SR647 states "output growth higher by 0.7%" without saying whether that is annualised. The repo reports pp annualised; if SR647's figure is quarterly, the comparison flips.
  - The fixed FFR response decays faster than SR647's "two years". ρ_R is 0.77, against SR647's posterior 0.80, but there is no anticipated-shock channel.
  - The full-sample mode's +3.3 pp output response and 1.27% hours are clearly out of line. That is part of the reason the masked mode is recommended.
- **Spread shock.** All versions give a smaller impact (15–19 bp) than SR647's ≈35 bp. SR647's model is the 2013 vintage with a different spread measure and sample.
- **π\*.** Re-estimation cuts the π*-driven overshoot (π/π* at h10: 2.50 → 1.77), because ψ2, ρ_R and ζ_p changed. The π* process is pinned by the SPF observable.
- **Price markup.** The impact on inflation is smaller (0.89 → 0.59) but more persistent at h4 (0.18 → 0.27): higher ι_p and ζ_p. The output cost on impact is almost zero.
- **No published NY Fed IRFs exist for Model1002 ss10 or the current production model.**
  - The DSGE.jl m1002 documentation PDF has none.
  - Liberty Street posts show forecasts and shock decompositions, not IRFs.
  - So the only quantitative IRF benchmark is SR647's text, and it describes a predecessor model.

## Parameter comparison with SR647 (posterior mean, Table 1)

Shown for orientation only. SR647 uses a different model vintage, different priors, and a sample
ending around 2008.

| | SR647 post. mean | Fixed (masked) | Old (DSGE.jl default) |
|---|---:|---:|---:|
| ρ_R | 0.800 | 0.769 | 0.713 |
| ζ_p | 0.878 | 0.943 | 0.894 |
| ζ_w | 0.902 | 0.907 | 0.929 |
| h | 0.730 | 0.302 | 0.535 |
| ρ_λf | 0.500 | 0.894 | 0.883 |
| σ_λf | 0.084 | 0.062 | 0.131 |
| σ_r (σ_r_m) | 0.139 | 0.227 | 0.238 |

The fixed mode's low habit persistence (0.30 vs 0.73 in SR647) is the most notable departure.
It is plausibly related to the extended post-2008 sample and the absence of anticipated shocks.
It is worth checking with MCMC before relying on consumption dynamics.
