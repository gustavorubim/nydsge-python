# ss104 analytics upgrade validation — 2026-10-15 UTC

The toolkit reproduces the committed public-default reference forecast, including
the survey-on/off rate gap. It adds explicit expected-rate measurements, verified
held and announced policy paths, separate price/wage markups, Fed funds HD and
cyclical HD. The operational desk adapter was not located on Gus and was not
replaced or certified. Oct 28 / Dec 9 desk operations remain on their existing setup.

## Identity and environment

- Worktree: `C:\Users\gusta\Documents\Projects\nydsge-python-analytics-upgrades`.
- Branch: `agent/analytics-survey-scenarios`; base main
  `cb4b028d70cba79c58131ef6cb89c6ec438506fb` (PR 5).
- Isolated `.venv`, CPython 3.14.0; locked NumPy 2.4.6, SciPy 1.18.0,
  pandas 3.0.3, matplotlib 3.11.0; CPU float64, one math thread for bounded runs.
- Native Model1002 `ss104`, public code defaults, 29 dated regimes. No estimation,
  sampling, GPU, Julia runtime, new credentials or paid services.
- Existing live `nydsge-econer` checkout and looping project were not modified.

## Inputs and measurement

Real committed files: `data/public/fred_levels_20261006.csv`,
`data/public/nonfred_sources_20261006.csv`, and
`data/public/ss104_survey_ffr_local.csv`. No new observations or interpolated rates
were created. Level-vintage label is 261006. Local capture of these committed bytes
was `2026-10-15T00:10:13.697633+00:00`; this is **not** independently verified original
upstream retrieval. The archive manifest explicitly records that distinction.
Source URLs: <https://fred.stlouisfed.org/> and
<https://www.newyorkfed.org/markets/market-intelligence/survey-of-market-expectations>.

Survey formation is 2026-07-15, with conservative availability 2026-10-12. This
run explicitly uses retrospective `analysis_vintage` as of 2026-10-12 UTC,
including the committed historical 2024 survey records. It is not a July or
historical point-in-time information set. Raw workbooks and their actual release
dates were not independently retrieved; workbook source names/hashes remain in
the committed CSV. The native horizon-one row targets the next quarter; same-quarter
survey targets are not shifted. Missing horizons remain missing.

Input lead quarter is 1959Q2. It enters differences and population HP filtering,
then is removed before the filter sample (1959Q3 onward). Likelihood starts
1960Q1. Historical transforms are frozen through 2026Q2, matching the reference;
2026Q3 semi-conditioning uses spread, effective FFR, long rate and available six
expected-rate rows. Other Q3 measurements remain latent. Forecasts start 2026Q4,
12 quarters through 2029Q3. GDP/investment are per capita annualized log growth;
core PCE/deflator are annualized log inflation; FFR is annual percent; hours are
100-log levels. These are not aggregate compounded SAAR or year-over-year rates.
Underlying native source/SA conventions are retained; population is NSA with HP1600.

## Executed validation and results

- Full suite: **615 passed, 4 skipped**, 84.35 s. Skips are existing optional cases.
- `ruff format --check .`, `ruff check .`, `ty check .`: passed.
- Renderer correction recheck: **14 passed**; six actual figures visually inspected.
- `nydsge doctor --json`: native Windows and NumPy CPU available.
- `nydsge vv runtime-purity`: passed, 34 files checked.
- Native system validation, smoother agreement at 1e-8, forecast agreement at
  1e-10 and surprise impact normalization at 1e-10 passed.
- Independent execution of `forecast_ss104_nyfed_params.forecast_history` plus
  native `forecast_ss104(..., cond_type="semi", horizon=13)` agrees across all six
  baseline plotted variables: max absolute difference **7.11e-15** plotted units.
- At 2027Q3, FFR is **3.637036% with expectations**, **4.471315% without**;
  difference **−0.834279 pp**. This is same-model/input/window ablation.
- A +25 bp impact surprise lowers annualized log core PCE by −0.020865 pp in
  2026Q4, −0.024001 in 2027Q1, −0.023471 in Q2 and −0.022136 in Q3.
- Direct synthetic regressions cover units/horizon alignment, publication and
  formation no-lookahead, OIS mapping, duplicate rejection, lead-quarter/native
  transformation, zero innovation, absolute hold, announcement timing, unreachable
  guidance, split markup groups and renderer diagnostics/integrity.

The three full-sample HD gates **fail and remain diagnostic**: core PCE
`2.701876766764144e-8`, GDP `2.861036918488935e-6`, FFR
`3.036392568045443e-6`, against unchanged `1e-8` plotted-unit threshold. These are
this run's errors, not the user's different run or a universal status. Initial,
deterministic, measurement-gap and numerical remainder terms remain exported.
Cyclical plots subtract initial/deterministic terms but retain remainders and the
same diagnostic status. No pass is obtained by changing the accounting baseline.

## Policy paths and interpretation

The final smoke targets **3.6673913043478263%** (hold the observed Q3 effective
rate) versus **3.9173913043478263%** (25 bp higher), in every quarter 2026Q4–2027Q3.
These are sequential unanticipated innovations, not an announced permanent rule;
after the constraint window, innovations are zero and rates evolve endogenously.
The separate guidance case announces in 2026Q4 a +25 bp deviation from baseline
in 2027Q1. Native anticipated monetary news enters at announcement, so output and
inflation can move before the target quarter. All actual constraints passed at
1e-8 annual pp (max reported error zero in the smoke). Exact primitives and rate
targets are in `manifest.json`. Default anticipated-shock variances are zero;
the guidance case is explicitly off prior support, with no SD/probability claim.

The rate rise works through the model's sticky-price IS/expectations and policy
mapping; systematic policy feedback remains distinct from the primitive policy
residual. The scenario compares conditional model paths, not measured effects of
an intraday FOMC surprise. Price/wage markup bars are separate; CPI component
accounting was not used as structural shock evidence.

## Identification caveat

This is an illustrative linear representative-agent DSGE at public code defaults,
not an estimated posterior or Fed staff forecast. Matching native reference
numbers validates execution, not empirical calibration or causal shock
identification. Demand/cost-push labels depend on model restrictions. Core PCE
measurement innovations are not structural supply shocks; government spending
cannot identify taxes/transfers. Survey target rates and OIS quotes proxy noisy
expected effective FFR under an explicit mapping, without a new term-premium or
target/effective-spread equation. No state, parameter or predictive uncertainty
bands are claimed. Distribution/credit channels beyond those represented cannot
be inferred from these outputs.

Decision-relevant risks: vintage/publication uncertainty, revisions and HP
endpoints; sparse expected-rate horizons; public-default sensitivity; information
and credibility assumptions; omitted energy/distribution channels; and diagnostic
HD residuals. Operational desk parity and fresh official-input release validation
remain separate work. No current-release recommendation is made.

## Local artifacts and rerun

Final computation: `outputs/upgrade_final_smoke/` (status completed); final
reviewed figures: `outputs/upgrade_deliverable_figures/`.
Reference comparison: `outputs/upgrade_reference_smoke/reference_comparison.json`.
Full tests: `outputs/final_tests.xml`.

The computation archive retains exact resolved `config.json`, source/input/code
hashes, frozen levels, historical/endpoint/observed measurements, expected-rate
observations, on/off ablation, forecasts, scenario constraints/innovations, exact
regime parameters, full/cyclical HD CSVs, totals and raw decomposition arrays.
Figure receipts include archive-manifest and renderer hashes plus captions.

```powershell
uv sync --locked --extra plot
uv run nydsge doctor --json
uv run nydsge vv runtime-purity
uv run python -m general_analytics_tools run --config inputs/reference_final.json --output-dir outputs/my_unique_rerun --timeout-seconds 180
uv run python -m general_analytics_tools plot --run-dir outputs/my_unique_rerun
uv run ruff format --check .
uv run ruff check .
uv run ty check .
uv run python -m pytest -o addopts='' -q --tb=short --basetemp outputs/my_unique_test_tmp
```

Never reuse an existing output directory. `inputs/reference_final.json` and
`inputs/reference_sources.json` are ignored local files, available on Gus; use the
archived resolved config for inspection. The public toolkit README documents the
portable input/config contract. Desk adapters and their operational configs were
not migrated. There are zero hosted Actions workflows; validation here is local,
not a claim of hosted CI. Publication state and final commit are reported separately.
