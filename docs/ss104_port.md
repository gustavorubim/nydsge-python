# ss104 port checklist

Port of NY Fed DSGE.jl Model1002 subspec ss104 into nydsge-python. Spec: docs/ss104_fidelity_scan.md.
Rules: branch ss104-port only, no force-push, main untouched, no long MCMC before M7.

- [x] M1: regime-switching solve (gensys per regime + gensys2 for temporary ZLB/alt policy), regime-aware Kalman filter/smoother, time-varying information sets
- [x] M2: COVID shocks with regime volatilities
- [x] M3: AR(1) inflation measurement error
- [x] M4: flexible AIT, pgap/ygap, time-varying credibility 2020Q4-2026Q4
- [ ] M5: SME/SPD expected FFR observables 1-6Q
- [ ] M6: full ss104 parity + conditional forecast
- [ ] M7: final estimation + NY Fed forecast comparison doc

## Progress log

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
