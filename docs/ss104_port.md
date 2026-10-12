# ss104 port checklist

Port of NY Fed DSGE.jl Model1002 subspec ss104 into nydsge-python. Spec: docs/ss104_fidelity_scan.md.
Rules: branch ss104-port only, no force-push, main untouched, no long MCMC before M7.

- [x] M1: regime-switching solve (gensys per regime + gensys2 for temporary ZLB/alt policy), regime-aware Kalman filter/smoother, time-varying information sets
- [x] M2: COVID shocks with regime volatilities
- [x] M3: AR(1) inflation measurement error
- [x] M4: flexible AIT, pgap/ygap, time-varying credibility 2020Q4-2026Q4
- [x] M5: SME/SPD expected FFR observables 1-6Q
- [x] M6: full ss104 parity + conditional forecast
- [x] M7: EXPERIMENTAL estimation artifacts + NY Fed comparison (mode unconverged)
- [ ] M7 estimated-forecast readiness: mode convergence and data gaps unresolved

## Acceptance

- **Port correctness: MET** — the documented Julia parity gates validate the
  public ss104 translation, including regime matrices and controlled replay.
- **Estimated-forecast readiness: NOT MET** — the M7 estimated forecast is
  **EXPERIMENTAL**. Its mode is unconverged, and historical OIS, survey and
  anticipated-GDP inputs have data gaps. Passing port parity does not close
  these estimation and input requirements.

## Progress log

### 2026-10-12 — PR #4 review fixes and final validation

- **Steady-state restoration (`bb1d09b`):** forecast cleanup invalidates the
  last draw's cache key after restoring parameters and steady-state values.
  One parameterized regression covers fresh `sigma_g+0.01` and initialized
  repeated `alpha+0.01` draws for ss10 and ss104, comparing states, observables,
  pseudo-observables and subsequent solves to independently recomputed
  fresh-model results at absolute tolerance 1e-12. The ss104 draw path uses
  its single-regime Taylor configuration, then re-enables all 29 regimes for
  the reuse check. Before the fix, both NaN Gamma0 and repeated-draw drift
  were reproduced; all four cases now pass.
- **Interrupted search recovery (`3f75584`):** checkpoints record the owning
  PID. `--resume` recovers a running checkpoint when its process has exited;
  `--resume --force-resume` explicitly overrides live/unknown PID protection
  after stopping any existing search. Synthetic-checkpoint cases verify live
  and exited processes, legacy files without a PID, and forced recovery.
  Input hashes, parameter coordinates and the original default starting
  posterior remain mandatory even when forced; history/counters/time persist.
- **Non-default Julia parity (`0cdcbba`):** three deterministic bounded
  perturbations of all 96 free prior-bearing coordinates are committed as
  compact fixtures (~3.3 MB each), generated with Julia 1.8.5 and the pinned
  source overlay. All 492 retained datasets per fixture were checked exactly
  against the raw exports. All 29 regime matrices, parameter tables,
  likelihoods, filter/smoother means and forecast paths are tested.
  Covariances retain periods 0, 2, 4 and final in each compact fixture; the
  default fixture tests complete histories, and complete non-default histories
  were also validated against the raw exports. Explicit covariance gates:
  absolute <2e-4; relative to max oracle entry <1e-6 filtered / <5e-10 predicted.
  Matrix maximum is 2.53806e-11; parameters/QQ/EE match exactly. New vector
  workflow budgets are documented in the
  [oracle recipe](../tools/oracle_julia/README.md#ss104-non-default-parameter-vectors):
  2e-6 states, 1e-5 smoother states, 5e-7 forecast observables, 1e-6 pseudo-observables.
  Original fixed-default gates remain unchanged; some non-default workflow
  errors exceed those tighter default limits (observable maximum 3.11695e-7,
  smoother maximum 8.93469e-6).
- **Forecast labels (`dfbe5a3`):** README, this checklist and the M7 comparison
  label the estimated forecast **EXPERIMENTAL**. Port correctness is met;
  estimated-forecast readiness is not met because the mode is unconverged and
  data gaps remain. A regression checks these claims against the saved M7 run.

Final validation: **571 passed, 4 skipped (575 collected)** in **72.91 seconds**
with `python -m pytest --tb=short` and `OPENBLAS_NUM_THREADS=1`.
`ruff check .` passed; `ruff format --check .` passed (**86 Python files**).
`git diff --check` passed. All work stays on `ss104-port`; main remains
`6716227be0ed573cd45f6f04fd38cb15a5807c37`; no force-push or merge.

### 2026-10-12 — M7 offline estimation and published forecast comparison

Completed one final BFGS trajectory from ss104 defaults on the existing
2026-10-06 local FRED/non-FRED snapshots: 96 prior-bearing parameter-regime
coordinates, presample 1959Q3-Q4, likelihood 1960Q1-2026Q2, COVID observations
retained. Five priorless coordinates stay at defaults; missing OIS, survey
history and anticipated GDP remain NaN. Two real local survey workbooks
supply 2024Q3 and 2026Q3; source-defined pgap/ygap initialization is preserved.

Search status **time_limit**, **97 iterations**,
**120.81 minutes**; log posterior
**-6487.849291 -> -1249.959509**
(likelihood **-1153.812337**, prior **-96.147172**).
Best finite search candidate; optimizer convergence is not certified.
No MCMC was run under the optional one-hour sampling budget.

Produced the 40-quarter 2026Q3-2036Q2 ss104 path at the best finite estimate, with one semi
conditional quarter using rates/spreads/long rates and the available six
SME expectations. Committed mode and quarterly/annual comparison CSVs are
under `data/estimates/`; generated arrays/data/metadata are under
`outputs/ss104_m7/`. [ss104_forecast_comparison.md](ss104_forecast_comparison.md)
reuses the cached NY Fed September 2026 forecast, reports GDP/core PCE/FFR/
output gap over the complete horizon, and names each missing file/series.

The independent machine oracle `C:\Users\gusta\master\ss104_oracle_box.h5`
and `ss104-parity-box.json` are cited with hashes. Its likelihood/forecast-observable
errors are **3.71096e-7 / 6.59366e-8**. Conditional-observation and smoother
cross-machine exceptions (**1.53845e-7 / 5.18873e-6**) are disclosed; M6 gates
are unchanged. Original M6 oracle tests still pass.

Validation: **105 targeted passed; 554 full-suite passed, 4 optional skips**.
Ruff lint/format and diff checks passed. Implementation/optimizer commits: `a0e5363`, `6661453`.
All work stays on `ss104-port`; main is unchanged; no push was issued by this M7 session.

### 2026-10-12 — M6 full fixed-default parity and conditional forecasts

Completed native solve/filter/Durbin-Koopman smoother and full/semi conditional
mean forecasts with **84/110 states, 48 shocks, 28 observables, 26 pseudo-observables,
130 parameters and 29 regimes**. The Julia 1.8.5 source exporter and lossless HDF5
fixture cover every regime, all parameter values, a controlled history and
unconditional/conditional forecast replay. The initial Julia dependency blocker
was resolved by bootstrapping pinned package types and overlaying the clean
c09dfe8 reference methods; no reference checkout or main branch was changed.

Parity maxima over 29 regimes: **TTT 2.12594e-11, RRR 1.43530e-11, CCC 3.62377e-12;
ZZ 4.21352e-12, DD 6.91003e-13; QQ/EE/parameter values 0**. Pseudo ZZ/DD:
**2.55795e-13 / 3.88578e-14**. History loglik Julia/Python:
**-27.82574194739678 / -27.825741807269697** (error **1.40127e-7**).
Conditional loglik: **-24.886624714156007 / -24.886624578812913**
(error **1.35343e-7**). Filter/smoother mean errors: **4.86429e-7 / 1.69636e-6**.
Unconditional forecast state/observable errors: **2.32131e-7 / 3.37272e-8**;
conditional: **2.38242e-7 / 6.82894e-8**. All are asserted in tests with
documented float64 tolerances. Source quirks (temporary-ZLB equation freezing,
gap laws throughout history and skipped ZLB DD adjustment) are preserved.

Targeted validation: **100 tests passed**. Final validation: **549 passed,
4 optional-backend skips**; `ruff check .` and `ruff format --check .` passed
(75 Python files formatted). `git diff --check` passed. API, source recipe and boundaries:
[ss104_m6.md](ss104_m6.md); all per-regime numbers:
[ss104_parity.json](ss104_parity.json). This is public fixed-default/controlled-
replay parity, not reproduction of an estimated FRBNY release. M7 is unchecked;
no ss104 estimation or MCMC was used to build the artifacts, and nothing was pushed.

### 2026-10-12 — M5 expected FFR and survey data

Completed six anticipated Taylor/AIT shocks, six expected-FFR measurement errors
(sigma .0375 + .00625*h after regime 9), and the anticipated GDP observable.
The expected-FFR rows use the imperfect-awareness transition path with M1 TVIS;
the older OIS rows use credibility-weighted perfect-policy paths, as in the source.
AIT anticipated innovations are active from regime 9 through the forecast-start
regime (28 for 2026Q3), then zero in regime 29. Full dimensions are now
**84 endogenous / 110 augmented / 48 shocks / 28 observables / 29 regimes**.

`nydsge.survey_ffr` loads the official SME/SPD long-table workbooks, selects
`pctl50` of `fftr_pathofmodes` by panel, takes the last target date per quarter,
and attaches horizons 1-6 to transformed quarterly histories. Annual fractions
are multiplied by **25** to get quarterly percentage points. Publication dates
must be explicit because the workbook's `survey_release_date` dates the questions.
Latest-survey selection respects publication cutoffs; absent horizons and
pre-2021Q4 quarters remain NaN. Older PDF-only vintages can enter via normalized
records; internal OIS history is not reconstructed.

Validation: **97 targeted tests passed**. Expected-FFR ZZ/QQ calibration maximum
error **0**; the real July-2026 SME path matches **.9075/.9075/.9075/.9075/.9075/.8450**
in quarterly units (max error <1e-15). Trimmed source fixture:
`tests/fixtures/sme_jul2026_ffr.csv`, from the
[official July workbook](https://www.newyorkfed.org/medialibrary/media/markets/survey/2026/jul-2026-data.xlsx).
Full-model Julia oracle export is running under workspace-local Julia 1.8.5;
M6 will record the actual cross-language errors.

### 2026-10-12 — M4 flexible AIT and imperfect awareness

Added pgap/ygap, AIT shock plus six anticipated AIT shocks, phi_pi=4,
phi_y=3, rho_smooth=.9 and half-life 10 (rho=.9330329915368074).
The GDP-gap equation uses lagged-output coefficient **-1** as in the source.
The initialization API inserts **-.125/-12** in 2020Q2 only, with sigma 20
initialization innovations in that regime. Actual ZLB regimes 5-9 precede AIT;
credibility increases from 0 to 1 over regimes 5-29 (2020Q4-2026Q4).
M1's predictable-form/gensys2 machinery solves ten perfect-policy paths;
imperfect-awareness equations use their weighted future transitions. Historical
SPD beliefs switch among liftoff regimes 19, 16, 15, 11 and 10. Forward measurements
use complete belief paths, with the actual policy's TVIS shortened at regime 10.
Taylor and pi-star innovations are removed at regimes 8 and 5 respectively.

Validation: **94 targeted tests passed**. All non-expectational TTT/RRR/CCC
equation residuals: maximum **2.487e-14**; credibility/gap calibration maximum
error **0**; all five ZLB nominal rates equal **.025 quarterly**.
Full default-parameter Julia parity remains M6's gate. The lean Julia setup
also found ModelConstructors/FillArrays incompatible with Julia 1.11/1.12;
a workspace-local Julia 1.8.5 is being used with the existing pinned manifest.

### 2026-10-12 — M3 inflation measurement error

Added augmented `e_meas_pi_t` and its lag with rho .232 after regime 1,
unit innovation loading, and **current minus lag** loadings on both inflation
observables. ss104 uses sigma .0999 in model regimes 2-10 and zero otherwise;
the legacy core-PCE error has sigma zero in 2-10 and .0999 outside that interval.
Its persistence remains .232, including while its innovations are disabled.
This follows `augment_states.jl`, `measurement.jl:161-172`, and the ss104 override
of `add_meas_pi!` / `rm_iid_pce_meas_err!` (the GDP-deflator error remains active).

Validation: **93 targeted tests passed**. AR/lag/RRR/ZZ/QQ calibration maximum
error **0** across 29 regimes; independent four-quarter impulse path error
**<1e-15**. Full Julia parity remains the M6 gate; the M2 dependency blocker
does not block the native implementation.

### 2026-10-12 — M2 COVID equations and volatility regimes

Added phi/Ephi, ziid and biidc states, their one-quarter anticipated states,
Euler/MRS/productivity loadings, proportional contemporaneous innovations and
the independent anticipated biidc innovation. `Model1002("ss104")` now builds
29 dated parameter snapshots; `compute_system` uses M1's regime system.
The ss64 volatility schedule includes COVID sigmas 400/5/4 in 2020Q1-Q3,
4/.05/.04 in Q4, zero thereafter, proportional multipliers 2 in Q1,
anticipated biidc sigma 4 in Q3-Q4, standard shock regimes and zero permanent
productivity variance in Q1-Q2. Parameter priors/fixed flags travel with snapshots.

Validation: **92 targeted tests passed** (ss104, regimes, Model1002).
Numerical gates: COVID canonical transition residual **2.303e-15** over all
29 regimes; calibrated QQ maximum error **0**. These are equation/calibration
checks, not full Julia model parity (reserved for M6). Julia 1.12.6 full-package
setup currently fails with `Unsatisfiable requirements ... SortingAlgorithms`:
SMC's StatsBase/Distributions constraints conflict with Julia-compatible versions;
the registry download also fails in the restricted network. Evidence:
`outputs/ss104-julia-setup.log`. M3-M5 can proceed using the cloned sources.

### 2026-10-12 — M1 regime infrastructure

Implemented permanent-regime canonical solves, DSGE.jl gensys2 backward recursion
for temporary ZLB/alternative policies, a validated regime system containing all
seven matrices, regime-aware Kalman filtering/RTS smoothing, dated regime-path
mapping, and time-varying information horizons/scenario selection. Model1002
forward-looking measurement and pseudo-measurement rows use the selected regime
path. The existing single-regime path is preserved: 68 exported arrays match the
pre-M1 branch head exactly across ss10/ss104 and 0/6 anticipated shocks.

Parity harness API: `nydsge.regimes.{gensys_to_predictable_form, gensys2,
solve_regime_switching, compute_regime_system, compute_model_regime_system,
compute_tvis_system, k_periods_ahead_expectations,
k_periods_ahead_expected_sums, regime_indices_from_dates, RegimeSwitchingSystem}`;
`nydsge.solve.compute_system(..., regime_models=..., temporary_regimes=...,
information_sets=..., tvis=True)`; and
`nydsge.kalman.{kalman_log_likelihood, smooth_kalman_result,
observables_from_states, pseudo_observables_from_states}` with `regime_indices`.
Contracts, source-file/line mapping and Julia oracle recipe:
[ss104_m1.md](ss104_m1.md).

Tests include exact one-regime solver/filter/smoother equality, two-regime
temporary ZLB, Model1002 temporary-policy augmentation, independent Gaussian
smoothing, and a pinned Julia kernel oracle at 1e-10. Remaining gaps are M2-M5's
ss104-specific equations, parameter regimes and credibility calibration, plus
M6's estimator/conditional-forecast orchestration and full ss104 parity. M1
certifies the infrastructure; the ss104 model specification remains incomplete.
No long MCMC or push was performed.

Validation: full `pytest` **539 passed, 4 optional-backend skips**; `ruff check .`
and `ruff format --check .` passed. All changed Python files/tests pass `ty`.
Repo-wide `ty` still reports the same 35 pre-existing script diagnostics reproduced
in an untouched archive of `dc2aaee8`; they are outside M1.
