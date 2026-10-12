# M7 ss104 estimation and NY Fed forecast comparison

**EXPERIMENTAL estimated forecast.**

**Port correctness: MET** — the public ss104 translation passes the documented
Julia parity gates. **Estimated-forecast readiness: NOT MET** — the M7 mode is
unconverged and the local vintage has the data gaps listed below. Port parity
does not certify this estimated forecast for production use.

**Convergence exception:** the single default-start estimation reached its
configured cumulative runtime limit. The independently checked projected
gradient is **23.4596**, above the **0.001** target. The tables therefore use
the best finite estimate, **not a certified posterior mode**.

This is an offline estimation and forecast of the public ss104 model on
the repository's existing local vintage. It is not a reconstruction of the
FRBNY posterior or September 2026 production release. All computations are native
Python; the Julia oracle is used only for independent validation.

## Estimation and local inputs

The model starts at **DSGE.jl ss104 defaults**, including the dated calibration
from [M6](ss104_m6.md). The existing SciPy BFGS path and parallel central
differences are used in the repo's unbounded coordinates (step 1e-4, gradient
tolerance 1e-3), with up to three Hessian-reset rounds in the same search as in
the existing C1/C2 driver. A precision-loss stop is resumed at the best saved
point, preserving the original default start, input hashes, iteration history
and cumulative runtime. The model-space objective is Kalman log likelihood plus the
joint prior. If an explicit model bound stops unconstrained BFGS, the same
candidate continues with SciPy's bounded BFGS (`L-BFGS-B`) and unchanged bounds,
parameters and data. Convergence is checked at the saved point with the
projected gradient: an outward derivative at an active bound is excluded,
while any feasible inward derivative remains. No parameter bound or prior is
relaxed. In this run the initial blocker was `Phi` approaching its lower bound
of 1 (Phi=1.0000040075 after the unconstrained rounds).
The objective has no sampling transform Jacobian. All **96 free parameter-regime
coordinates with defined priors** are estimated independently, following the
existing estimator's selection rule. Five free-flagged coordinates without a
defined prior are held at their defaults: `rho_ait_rm`, `rho_meas_pi@2`,
`sigma_ait_rm@2`, `sigma_condgdp@2` and `sigma_condcorepce@2` (the last two are
conditioning-only and absent from the historical likelihood). The posterior
objective is conditional on those calibrations, not a joint estimate of all 101 free flags.
A label such as `sigma_phi@2` denotes
parameter regime 2, not model regime 2. Baseline priors are not counted again
when a parameter has an explicit regime table; fixed entries remain fixed.

The local inputs are `data/public/fred_levels_20261006.csv` and
`data/public/nonfred_sources_20261006.csv`, transformed with the existing
`scripts/c1c2_common.py` pipeline. Presample **1959Q3-Q4** initializes the filter;
**1960Q1-2026Q2** contributes to the likelihood (266 scored quarters, 268 total).
COVID observations are retained. The M6 one-quarter gap initialization and
2020Q4-2021Q4 nominal-rate/OIS masks are applied identically in fitting and
forecasting. Source files are hashed in the mode artifact.

Two real local workbooks, `outputs/sep-2024-data.xlsx` and
`outputs/jul-2026-data.xlsx`, supply the Combined-panel median FFR point paths.
Their selected records and workbook hashes are preserved in
[`ss104_survey_ffr_local.csv`](../data/public/ss104_survey_ffr_local.csv).
The July source is the [official SME workbook](https://www.newyorkfed.org/medialibrary/media/markets/survey/2026/jul-2026-data.xlsx).
The date **2026-10-12** in `available_date` is a conservative local availability
bound observed during this run, **not an assertion of the original publication
date**. The run is as of that bound and does not certify historical real-time
availability. Selection never carries a survey formed in a later quarter into
an earlier quarter. Annual rate fractions are multiplied by 25 to obtain
quarterly percentage points.

The completed search has status **time_limit** after **97 accepted iterations**, taking **120.81 minutes**. Log posterior improves from **-6487.849291** to **-1249.959509** (log likelihood **-1153.812337**, log prior **-96.147172**). Termination: `Configured mode-search time limit`. Final projected gradient infinity norm at the saved point: **23.4596**. This is the best finite posterior candidate from the completed search, not a certified converged mode.

The 120-minute cumulative optimizer guard was configured for this final
estimation; it is not a system-imposed runtime limit. The complete default-start
trajectory and all three unconstrained stops plus the bounded continuation are
retained in the mode artifact. A fresh derivative check has no unknown
coordinates; only `Phi` needs a one-sided derivative at its lower bound. The
largest remaining projected derivative is `psi2` (-23.4596). Further optimizer
work is required before treating this candidate as a converged mode.

No MCMC was run. There is no validated ss104 proposal Hessian, and this
96-coordinate regime solve does not provide evidence that a useful posterior
sample and convergence checks would finish within the optional one-hour budget.
Forecasts have no parameter-uncertainty bands. The optimizer's status and final
gradient are recorded explicitly; fixed-default M6 parity does not validate a
new estimated posterior.

The repeated policy-path solve now caches steady-state calculations on their
14 structural inputs and computes only the policy measurements actually used by
the realized belief path. All transition paths are retained for expectations.
The cache invalidates on structural parameter updates; model regimes, equations,
and the numerical parity gates are unchanged.

## Missing files and series

Missing observations stay **NaN**, with no fixture, zero, interpolation or
vendor substitute. The mode artifact records nonmissing counts for all 28 rows.
The following would be needed for a fuller vintage-matched run:

| Input | Local coverage and exact requirement |
|---|---|
| SME/SPD expected FFR, horizons 1-6 | Only **2024Q3** is available in estimation, and **2026Q3** in conditioning. Each of `obs_exp_nominalrate1` through `obs_exp_nominalrate6` has one estimation observation. Supply the latest eligible survey for every missing quarter **2021Q4-2026Q2 except 2024Q3**, with original publication dates. Extend `data/public/ss104_survey_ffr_local.csv` with `survey_date,available_date,target_quarter,rate` records in quarterly percentage points. Official long workbooks need `survey_release_date,panel_type,question_tag,aggregation,horizon_date,aggregation_value`; use `fftr_pathofmodes`, `pctl50`, Combined. Older PDF-only vintages need normalized real records. A production-matched 2026Q3 path also requires the exact survey vintage used by FRBNY. |
| Historical OIS anticipated FFR, horizons 1-6 | All six `obs_nominalrate1`-`obs_nominalrate6` columns are absent. The raw-source loader recognizes `data/public/ois_261006.csv` (or `ois.csv`) with `date,ant1,ant2,ant3,ant4,ant5,ant6`, already in model quarterly rate units. Internal historical OIS inputs are not available locally. The M7 snapshot helper currently reads the two fixed public snapshots; adding an OIS file also requires merging it into that local pipeline. |
| Anticipated GDP | `obs_gdp1` has no observations. Need `data/public/medianantgdp_261006.csv` (or `medianantgdp.csv`) with `date,antgdp1`, in model per-capita quarterly log-percent growth, plus the underlying SPF/production nowcast vintage and population conversion. The identity raw transform does not perform an annual/aggregate conversion. Merge the real series into the local pipeline before fitting. |
| pgap/ygap initialization | No external history file is required by default ss104. The source defines **obs_pgap=-.125, obs_ygap=-12 in 2020Q2 only**; each has exactly one observation. Other quarters intentionally stay NaN. `ygap_t` is the cumulative AIT gap; the comparison's output gap is `OutputGap = y_t-y_f_t`. |
| Full 2026Q3 GDP/core-PCE conditioning | No vintage-matched nowcast data file is present. Need `data/public/ss104_conditioning_2026q3.csv` with dated `obs_gdp,obs_corepce` in model quarterly units and release provenance, then explicitly attach it to the forecast data. The partial-quarter PCE level in the FRED snapshot is not treated as a full-quarter nowcast. This run uses ss104 **semi** conditioning. The production release also names short-run SPF inflation expectations; their unpublished model wiring is outside M6 certification. |
| Historical spread splice | The cached BofA `BAMLC8A0C15PYEY` coverage starts in 2023Q4; matching the reference splice needs its **2016Q4-2023Q3** observations in the same local levels snapshot. Existing BAA coverage is used by the established transform until BofA becomes available. |
| Vintage-matched population forecast | The local terminal HP trend is held constant. A release match needs the actual forecast of quarterly `CNP16OV` population levels (thousands of persons), for example `data/public/population_forecast_261006.csv` with `date,CNP16OV`, and explicit wiring into the population preparation/forecast conversion. No staff population forecast is present in this run. |
| Published forecast comparators | Cached NY Fed GDP/core-PCE quarterly paths and annual tables exist. The earlier comparison records that the FFR path is not published. There is no cached published output-gap path; matching it needs the release's numerical output-gap export. Neither a real natural rate nor a cumulative AIT gap is substituted. |

## Forecast convention and published benchmark

The forecast horizon is **2026Q3-2036Q2, 40 quarters**: one conditional quarter
plus 39 future quarters. Semi conditioning uses the local 2026Q3 nominal rate,
spread and long rate, and all six July SME expected-rate observations. It uses
the source's `cond_semi_names`; the 2026Q3 long-inflation and partial-quarter
GDP/core-PCE inputs are excluded. All future innovations are zero at the mode.

GDP/core PCE use the ss104 `PseudoGDP`/`PseudoCorePCE` rows. Quarterly log-percent
growth `g` is reported as **100*(exp(4*g/100)-1)**, exactly compounded annual
growth. GDP adds the terminal local HP population trend, held constant, so it
is an aggregate approximation. Its terminal trend uses the same through-2026Q2
population window as estimation. Appending the Q3 condition does not refilter
the fitted history. Q4/Q4 is **100*(exp(sum(g_Q1..Q4)/100)-1)**;
2026 uses this run's observed Q1-Q2 history and forecast Q3-Q4. FFR is four
times the quarterly nominal rate; output gap is in log percentage points.
The earlier comparison used the repo's simple annualization
`400*(exp(g/100)-1)` and fourth-root aggregation. Its *published NY Fed numbers*
are reused here, with the exact-compounding convention stated separately.
This matches the pinned DSGE.jl source's `src/data/transformations.jl:260-262`
(`100*(exp(g/100)^4-1)`) and `loggrowthtopct_4q` at lines 462-479. The older
generic Python reporting transform is not used for these M7 comparison tables.

No network lookup was made. The benchmark is the **September 18, 2026** release
already recorded in [nyfed_comparison.md](nyfed_comparison.md), with cached
CSVs in [`data/public/nyfed_dsge_20260918`](../data/public/nyfed_dsge_20260918).
The original [Liberty Street forecast](https://libertystreeteconomics.newyorkfed.org/2026/09/the-new-york-fed-dsge-model-forecast-september-2026/)
reports GDP Q4/Q4 **1.2/0.1/0.5/1.2%** and core PCE
**3.3/2.1/1.8/1.8%** for 2026-2029. The benchmark is a published posterior
forecast at a different vintage, conditioning set and population path.
Numerical closeness to that release is not a port-parity gate.

### Annual comparison

GDP/core PCE are Q4/Q4 percent; FFR/output gap are Q4 values.

| Year | GDP ss104 | GDP NY Fed | Core PCE ss104 | Core PCE NY Fed | FFR ss104 | FFR NY Fed | Output gap ss104 | Output gap NY Fed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 2.39 | 1.20 | 3.35 | 3.30 | 3.61 | — | -3.80 | — |
| 2027 | 1.25 | 0.10 | 2.50 | 2.10 | 3.62 | — | -4.44 | — |
| 2028 | 1.76 | 0.50 | 2.06 | 1.80 | 4.06 | — | -4.64 | — |
| 2029 | 2.55 | 1.20 | 1.83 | 1.80 | 4.26 | — | -4.20 | — |
| 2030 | 3.00 | — | 1.74 | — | 4.28 | — | -3.44 | — |
| 2031 | 3.15 | — | 1.72 | — | 4.25 | — | -2.62 | — |
| 2032 | 3.15 | — | 1.74 | — | 4.22 | — | -1.87 | — |
| 2033 | 3.08 | — | 1.78 | — | 4.18 | — | -1.23 | — |
| 2034 | 2.99 | — | 1.83 | — | 4.14 | — | -0.71 | — |
| 2035 | 2.90 | — | 1.88 | — | 4.10 | — | -0.31 | — |

### Full quarterly horizon

GDP/core PCE are compounded annual percent; FFR is annual percentage points. 2026Q3 is conditional.

| Quarter | GDP ss104 | GDP NY Fed | Core PCE ss104 | Core PCE NY Fed | FFR ss104 | FFR NY Fed | Output gap ss104 | Output gap NY Fed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026Q3 | 3.16 | 1.52 | 3.21 | 2.76 | 3.67 | — | -3.70 | — |
| 2026Q4 | 1.71 | -0.16 | 2.96 | 2.53 | 3.61 | — | -3.80 | — |
| 2027Q1 | 1.28 | 0.16 | 2.74 | 2.31 | 3.63 | — | -3.97 | — |
| 2027Q2 | 1.17 | 0.29 | 2.57 | 2.12 | 3.63 | — | -4.14 | — |
| 2027Q3 | 1.21 | 0.27 | 2.42 | 2.00 | 3.64 | — | -4.30 | — |
| 2027Q4 | 1.33 | 0.38 | 2.29 | 1.91 | 3.62 | — | -4.44 | — |
| 2028Q1 | 1.47 | 0.44 | 2.18 | 1.85 | 3.66 | — | -4.55 | — |
| 2028Q2 | 1.64 | 0.51 | 2.09 | 1.80 | 3.82 | — | -4.63 | — |
| 2028Q3 | 1.86 | 0.79 | 2.01 | 1.78 | 3.95 | — | -4.66 | — |
| 2028Q4 | 2.09 | 1.09 | 1.95 | 1.77 | 4.06 | — | -4.64 | — |
| 2029Q1 | 2.30 | 1.17 | 1.89 | 1.78 | 4.14 | — | -4.58 | — |
| 2029Q2 | 2.48 | 1.36 | 1.85 | 1.78 | 4.19 | — | -4.48 | — |
| 2029Q3 | 2.64 | 1.48 | 1.81 | 1.80 | 4.23 | — | -4.35 | — |
| 2029Q4 | 2.78 | 1.57 | 1.78 | 1.81 | 4.26 | — | -4.20 | — |
| 2030Q1 | 2.89 | 1.79 | 1.76 | 1.84 | 4.27 | — | -4.03 | — |
| 2030Q2 | 2.97 | — | 1.74 | — | 4.28 | — | -3.84 | — |
| 2030Q3 | 3.04 | — | 1.73 | — | 4.28 | — | -3.64 | — |
| 2030Q4 | 3.09 | — | 1.72 | — | 4.28 | — | -3.44 | — |
| 2031Q1 | 3.12 | — | 1.72 | — | 4.27 | — | -3.23 | — |
| 2031Q2 | 3.15 | — | 1.71 | — | 4.27 | — | -3.03 | — |
| 2031Q3 | 3.16 | — | 1.72 | — | 4.26 | — | -2.82 | — |
| 2031Q4 | 3.17 | — | 1.72 | — | 4.25 | — | -2.62 | — |
| 2032Q1 | 3.16 | — | 1.72 | — | 4.24 | — | -2.42 | — |
| 2032Q2 | 3.16 | — | 1.73 | — | 4.23 | — | -2.23 | — |
| 2032Q3 | 3.14 | — | 1.74 | — | 4.23 | — | -2.04 | — |
| 2032Q4 | 3.13 | — | 1.75 | — | 4.22 | — | -1.87 | — |
| 2033Q1 | 3.11 | — | 1.76 | — | 4.21 | — | -1.69 | — |
| 2033Q2 | 3.09 | — | 1.77 | — | 4.20 | — | -1.53 | — |
| 2033Q3 | 3.07 | — | 1.78 | — | 4.19 | — | -1.38 | — |
| 2033Q4 | 3.05 | — | 1.80 | — | 4.18 | — | -1.23 | — |
| 2034Q1 | 3.02 | — | 1.81 | — | 4.17 | — | -1.09 | — |
| 2034Q2 | 3.00 | — | 1.82 | — | 4.16 | — | -0.96 | — |
| 2034Q3 | 2.98 | — | 1.84 | — | 4.15 | — | -0.83 | — |
| 2034Q4 | 2.96 | — | 1.85 | — | 4.14 | — | -0.71 | — |
| 2035Q1 | 2.93 | — | 1.86 | — | 4.13 | — | -0.60 | — |
| 2035Q2 | 2.91 | — | 1.87 | — | 4.12 | — | -0.50 | — |
| 2035Q3 | 2.89 | — | 1.89 | — | 4.11 | — | -0.40 | — |
| 2035Q4 | 2.87 | — | 1.90 | — | 4.10 | — | -0.31 | — |
| 2036Q1 | 2.85 | — | 1.91 | — | 4.09 | — | -0.23 | — |
| 2036Q2 | 2.83 | — | 1.92 | — | 4.08 | — | -0.15 | — |

The quarterly table includes the complete forecast horizon at the saved estimate. A dash denotes an
unavailable cached published value, not zero. NY Fed quarterly GDP/core PCE
values end at 2030Q1 in the cached export. FFR and output gap are reported for
ss104, with published comparators explicitly unavailable.

Against the cached September release, GDP Q4/Q4 differences for **2026/2027/2028/2029** are +1.19/+1.15/+1.26/+1.35 percentage points; core PCE differences are +0.05/+0.40/+0.26/+0.03. These are forecast-level differences. A different posterior, post-revision data, sparse survey history, semi versus full nowcast conditioning and the population approximation all differ. This single run does not identify the causal contribution of each difference or establish release reproduction. The annual population log-growth approximation is **1.069890 pp**.

## M6 parity and independent machine oracle

The original committed M6 oracle remains
[`tests/fixtures/ss104_default_oracle.h5`](../tests/fixtures/ss104_default_oracle.h5),
with per-regime results in [ss104_parity.json](ss104_parity.json).
It validates the public fixed-default 29-regime model and controlled
history/conditional replay, not estimation on this public snapshot.

An independent export generated on another machine is at
`C:\Users\gusta\master\ss104_oracle_box.h5`; its supplied comparison is
`C:\Users\gusta\master\ss104-parity-box.json`, preserved as
[ss104_parity_box.json](ss104_parity_box.json).
It uses **Julia 1.12.7**, DSGE.jl
`c09dfe881f8809f65b8c0f08a761ec6cc456a6fb` and StateSpaceRoutines.jl
`794fb02ff45a62537d8d5d037c1d12b3aa7e4574`, with the same fixed-default scope.
SHA-256 of the independent HDF5:
`98ed2a7233a2794870ba42a0b7280c55a06fd38617f50d3ba7711413a22f3c5f`.
SHA-256 of the supplied comparison JSON:
`56214cc56af09ca92ebcef69e3946ba12f23398911dedee58839347a35da7d0d`.

| Maximum absolute error | Original M6 oracle | Independent machine, supplied comparison |
|---|---:|---:|
| TTT / RRR / CCC | 2.12594e-11 / 1.43530e-11 / 3.62377e-12 | 9.34719e-12 / 4.76064e-12 / 9.94760e-13 |
| ZZ / DD | 4.21352e-12 / 6.91003e-13 | 2.01617e-12 / 1.17240e-13 |
| QQ / EE / parameter values | 0 / 0 / 0 | 0 / 0 / 0 |
| History likelihood | 1.40127e-7 | **3.71096e-7** |
| Conditional likelihood | 1.35343e-7 | 3.79209e-7 |
| Unconditional forecast observations | 3.37272e-8 | **6.59366e-8** |
| Conditional forecast observations | 6.82894e-8 | **1.53845e-7** |
| Smoothed history states | 1.69636e-6 | **5.18873e-6** |

The independent history log likelihood is Julia **-27.82574143616523** versus
Python **-27.82574180726159** in the supplied report. Matrices, likelihoods
and unconditional observations meet the original M6 thresholds. However,
the independent **conditional observation error exceeds 1e-7** and its
**smoother error exceeds 5e-6** slightly. These are explicitly recorded as
cross-machine gate exceptions; original tests and tolerances were not relaxed.
A fresh M7 comparison against that HDF5 reproduces the same conclusion
(`outputs/ss104-parity-box-m7.json`: likelihood 3.71104e-7, forecast observations
6.59386e-8, conditional observations 1.53848e-7, smoother 5.14556e-6).

## Artifacts, reproduction and checks

- Committed mode, starting/estimated values, prior/likelihood split, counts,
  input hashes, accepted-iteration history and convergence status:
  [`m1002_ss104_mode.json`](../data/estimates/m1002_ss104_mode.json).
- Committed full-horizon quarterly and annual tables:
  [`ss104_forecast_quarterly.csv`](../data/estimates/ss104_forecast_quarterly.csv),
  [`ss104_forecast_annual.csv`](../data/estimates/ss104_forecast_annual.csv).
- Local generated bundle: `outputs/ss104_m7/` contains transformed data,
  model-unit observable/pseudo paths, lossless arrays and forecast metadata.
  Optimizer log: `outputs/ss104-m7-estimation.log`.

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe scripts/estimate_ss104_mode.py --workers 12 --maxiter 400 --time-limit-min 120
# Continue an interrupted/precision-stopped search without changing its inputs:
.\.venv\Scripts\python.exe scripts/estimate_ss104_mode.py --workers 12 --maxiter 400 --time-limit-min 120 --resume
# Continue at an explicit parameter bound, preserving the same estimation:
.\.venv\Scripts\python.exe scripts/estimate_ss104_mode.py --workers 12 --maxiter 400 --time-limit-min 120 --resume --bounded
.\.venv\Scripts\python.exe scripts/forecast_ss104_mode.py
.\.venv\Scripts\python.exe tools/oracle_julia/compare_ss104.py C:/Users/gusta/master/ss104_oracle_box.h5 --out outputs/ss104-parity-box-m7.json
.\.venv\Scripts\python.exe -m pytest tests/test_ss104_estimation.py tests/test_ss104.py tests/test_regimes.py tests/test_model1002.py tests/test_survey_ffr.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
```

Validation: **105 targeted tests passed**; full suite **554 passed, 4 optional-backend skips**. Ruff lint and format checks passed (80 Python files). `git diff --check` passed. The native forecast/export command also passed against a checkpoint and the completed estimate. No MCMC was run.

Work and local commits remain on **ss104-port**. Main stays at
`6716227be0ed573cd45f6f04fd38cb15a5807c37`; this M7 session issues no push or force-push.
