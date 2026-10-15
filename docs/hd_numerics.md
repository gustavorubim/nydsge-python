# Multi-regime historical attribution: numerical contract and open blockers

This implementation is native Python/NumPy/SciPy. Julia is not used to compute
attributions or run the checks below. A pinned, previously generated independent
Julia oracle is used only as read-only comparison evidence.

## Native API and additive identity

`nydsge.decomposition.regime_disturbance_attribution` exposes dated-regime
conditional disturbance means, including singular and zero shock variances,
missing observations, nonzero transition/measurement intercepts, and explicitly
supplied nondefault filter initial mean/covariance. The analytics toolkit calls
this native API. The original single-regime historical-decomposition API is not
silently extended or reinterpreted.

For each observation, the API retains

```
fitted = sum(shock contributions) + initial + deterministic + numerical remainder
observed = fitted + measurement gap
```

It computes `E[epsilon_t | Y] = Q_t R_t' r_t` using the native
Durbin--Koopman information recursion, propagating each conditional disturbance
through the actual dated transition matrices. Effective Q must match the
filter's process covariance, including dated zero-variance restrictions. The
initial mean/covariance must match the filter initialization. Invalid shapes,
infinities, negative shock variance, inconsistent regimes and mismatched process
covariances fail explicitly. A supplied initial distribution is not replaced by
the unconditional stationary distribution.

The output now also includes state-space reconstruction remainders, innovation
covariance condition numbers, and disagreement with the supplied smoother's
information identity. The exported validation identifies the worst quarter.
These quantities are diagnostics, not extra economic shocks. The numerical
remainder remains a separate plotted component. Annualization happens only at
the export boundary: factor four for annualized log growth and annual rate pp.
The strict maximum absolute reconstruction gate remains **1e-8 in plotted units**,
across the entire sample rather than just the displayed chart window.

Independent joint-Gaussian tests construct the complete latent prior and
observation matrix without a Kalman recursion. They verify disturbance means,
smoothed states and additive identities to 2e-14 with changing regimes,
nonzero initial means, correlated initial uncertainty, nonzero intercepts,
rank-deficient QQ, individual missing observations and an entirely missing
quarter. A single-period test checks initial/terminal indexing. No innovations
are allocated to zero prior-support directions.

## Frozen public-input diagnosis (2026-10-15)

A bounded fresh Python execution reused the immutable public-input observation
archive `upgrade_final_smoke/observed_observations.csv` from the preceding
analytics validation: 1959Q3--2026Q3, public default ss104, actual quarter-specific
measurement masks, and the original dated process-covariance restrictions.
The retained filter/smoother algorithm produces these maximum reconstruction
errors:

| Observable | Maximum error (plotted units) | Gate |
|---|---:|---|
| Core PCE | 2.70210854e-8 annualized log pp | diagnostic |
| GDP | 2.86124712e-6 annualized log pp | diagnostic |
| Fed funds | 3.03666520e-6 annual rate pp | diagnostic |

The effective process covariance equals `R Q R'` exactly in this experiment, so
an omitted dated shock-variance mask does not explain this run's remainder.
The maximum predicted covariance magnitude is 657130.05. Innovation covariance
condition numbers increase from 2524.77 in 2019Q4 to 5.94697e6 in 2020Q1,
4.33398e6 in 2020Q2 and 3.50294e6 in 2020Q3. Predicted-covariance asymmetry reaches
1.16415e-10. Small upstream covariance errors can therefore be amplified by the
backward conditional-mean calculation; simple summation roundoff alone has not
been established as the cause.

Diagnostic alternatives were evaluated without adopting them:

- A Joseph covariance update still gave GDP 3.04174e-6, core PCE 2.79640e-8 and
  FFR 3.16567e-6; it changed the native smoothed state by up to 1.57108e-5.
  It did not fix the problem.
- An algebraically reordered backward information recursion changed states by
  up to 7.67189e-6. Reordering arithmetic is not sufficient evidence of improved
  reference fidelity and was not substituted into production.
- Projecting transition discrepancies into supported shock directions leaves a
  2.80127e-6 state-space residual outside that support. Forcing such a discrepancy
  into shocks would not be a conditional-Gaussian attribution. No correction or
  balancing residual was assigned to economic shocks.

**Open blocker:** real-input multi-regime reconstruction has not passed the
strict gate. The new native interface and diagnostic visibility do not certify
this sample's HD or resolve the native smoother's numerical conditioning.
A sound next step is an independently validated square-root/invariant-preserving
filter-smoother investigation, with the existing Julia gates unchanged.

## Independent cross-machine oracle recheck

The existing independent file `ss104_oracle_box.h5` was read without Julia,
regeneration or modification. SHA256:
`98ed2a7233a2794870ba42a0b7280c55a06fd38617f50d3ba7711413a22f3c5f`.
Its reference lineage remains DSGE.jl
`c09dfe881f8809f65b8c0f08a761ec6cc456a6fb` and StateSpaceRoutines.jl
`794fb02ff45a62537d8d5d037c1d12b3aa7e4574`.

| Surface | Fresh maximum absolute error | Existing default budget | Result |
|---|---:|---:|---|
| Transition TTT | 9.59943e-12 | 5e-10 | pass |
| Parameters / QQ / EE | 0 | exact | pass |
| History likelihood | 3.71104e-7 | 1e-6 | pass |
| Conditional likelihood | 3.79217e-7 | 1e-6 | pass |
| Unconditional forecast observables | 6.59386e-8 | 1e-7 | pass |
| Conditional forecast observables | 1.53848096e-7 | 1e-7 | fail |
| History smoothed states | 5.14555612e-6 | 5e-6 | fail |
| Conditional smoothed states | 5.09934190e-6 | 5e-6 | fail |

The history predicted covariance difference is 4.88521e-5 on scale 657144.60
(relative 7.43399e-11); the filtered covariance difference is 6.39082e-5 on
scale 230.30 (relative 2.77498e-7). This localizes amplification downstream from
very small matrix differences, but does not prove a unique source of the
cross-machine difference. The committed same-machine default and three
nondefault-vector fixtures remain separate evidence. Passing those cannot
supersede these independent exceptions.

Reproduce with an explicitly provided local oracle, without a Julia executable:

```powershell
uv run python tools/oracle_julia/compare_ss104.py PATH_TO_EXISTING_ORACLE.h5 --out outputs/independent_comparison.json
uv run pytest tests/test_hd_precision.py tests/test_general_analytics_tools.py
```

No private desk inputs, new oracle binaries or generated figures are committed.
