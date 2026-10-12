# ss104 port checklist

Port of NY Fed DSGE.jl Model1002 subspec ss104 into nydsge-python. Spec: docs/ss104_fidelity_scan.md.
Rules: branch ss104-port only, no force-push, main untouched, no long MCMC before M7.

- [x] M1: regime-switching solve (gensys per regime + gensys2 for temporary ZLB/alt policy), regime-aware Kalman filter/smoother, time-varying information sets
- [ ] M2: COVID shocks with regime volatilities
- [ ] M3: AR(1) inflation measurement error
- [ ] M4: flexible AIT, pgap/ygap, time-varying credibility 2020Q4-2026Q4
- [ ] M5: SME/SPD expected FFR observables 1-6Q
- [ ] M6: full ss104 parity + conditional forecast
- [ ] M7: final estimation + NY Fed forecast comparison doc

## Progress log

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
