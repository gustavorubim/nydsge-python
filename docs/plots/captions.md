# Plot takeaways

- **forecast_gdp_pce_annual.png**: 2027 GDP/PCE: ss104 defaults 1.12/1.97%; saved estimate 1.25/2.50%; NY Fed 0.10/2.10%. Defaults are code calibration; estimate convergence uncertified.
- **forecast_ffr_outputgap.png**: 2027Q4 FFR: ss10 3.71%, ss104 3.62%; gaps -4.28/-4.44 log pp. ss104 best saved estimate; convergence uncertified.
- **regime_timeline.png**: 29 regimes; 5 actual ZLB quarters; AIT credibility rises from 0.000 (2020Q4) to 1.000 (2026Q4), fixed rather than estimated.
- **covid_shocks.png**: 2020Q2 innovations: phi 15.77, ziid 2.05, biidc -1.52 model units. ss104 best saved estimate; convergence uncertified.

**enhancement_contributions.png — skipped:** M1–M5 are integrated infrastructure and model blocks with no supported independent toggle API. Defining and validating five comparable counterfactual models is outside a cheap plotting replay; no ablation or re-estimation was run.

## Numbers plotted (Q4/Q4, %)

| Year | GDP ss10 | GDP ss104 estimate | GDP ss104 defaults | GDP NY Fed | PCE ss10 | PCE ss104 estimate | PCE ss104 defaults | PCE NY Fed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 2.513 | 2.395 | 3.609 | 1.200 | 3.258 | 3.353 | 3.063 | 3.300 |
| 2027 | 2.520 | 1.248 | 1.123 | 0.100 | 2.281 | 2.505 | 1.968 | 2.100 |
| 2028 | 2.774 | 1.765 | 1.251 | 0.500 | 1.834 | 2.060 | 1.717 | 1.800 |
| 2029 | 2.877 | 2.551 | 2.045 | 1.200 | 1.641 | 1.834 | 1.739 | 1.800 |

## Sources and interpretation

- ss10 baseline: recommended COVID-masked C1/C2 saved estimate, replayed with the existing `c1c2_compare.after` recipe; not the earlier desk/default run. No optimizer or sampler is called.
- ss104 annual/quarterly paths: committed `data/estimates/ss104_forecast_*.csv`; NY Fed annual figures: cached September 18, 2026 `nyfed_dsgeTableData.csv`.
- ss104 (NY Fed/default params): `ss104_nyfed_params_forecast_annual.csv`; DSGE.jl ss104 defaults (the published code's calibration, not the FRBNY production posterior). Same local data, semi conditioning and growth conventions as the saved estimate.
- GDP adds constant terminal HP population log growth; annual GDP and PCE use 100 × expm1(sum of four quarterly log-percent growth rates / 100). Existing C1/C2 docs compound their approximate annualized reporting rates, so their rounded annual numbers differ slightly.
- 2026 uses actual Q1–Q2 and model Q3–Q4. ss10 conditions Q3 on rates, spread and long-run inflation; ss104 also uses available SME rate expectations. The specification, parameters and conditioning all differ; this is not a causal ablation.
- FFR = 4 × quarterly NominalFFR; output gap = y − flexible output in log percentage points, not cumulative AIT ygap. Quarterly panel: 2026Q3–2029Q2 (12 quarters).
- Regimes/credibility come from the current ss104 settings. Actual temporary ZLB: regimes 5–9. Hatched extension represents the latest alternative survey belief (ZLB through regime 18), not the realized rate path; it is separate from credibility.
- COVID shocks: full-history smoothing through 2026Q2 at the saved ss104 estimate, preserving missing data and dated covariance regimes. Covariance-weighted Durbin–Koopman structural innovations are in native model units; one-quarter anticipated innovations are dated when the news arrives. Zero estimates follow zero shock variance, not missing-data imputation.
- Historical likelihood replay: -1153.812337; COVID transition reconstruction max error: 6.68e-07.
- ss104 optimizer status is time_limit; saved projected gradient norm 23.4596; convergence uncertified. These are point paths with no uncertainty bands.
