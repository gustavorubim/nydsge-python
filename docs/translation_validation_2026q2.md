# Model1002 ss10 Translation Validation — 2026-Q2 Audit

Audit date: 2026-07-30

## Reference contract

- Julia reference: `DSGE.jl` 1.3.0
- Julia package tree: `e746a4a5ab9c26d897239e722b0f19d4bb3bd77e`
- Python target: `Model1002`, subspec `ss10`, CPU `float64`
- Current-data sample: 1964-Q1 through 2026-Q2
- Forecast start: 2026-Q3
- Forecast horizon: 20 quarters

The audit uses the pinned Julia environment in `tools/oracle_julia/Manifest.toml`.
It does not compare against the private current New York Fed production model.

## Results

The current-data hard-target comparison passed all 59 required arrays under the
strict CPU tolerance profile (`atol=1e-10`, `rtol=1e-10`). Coverage includes:

- parameters, bounds, fixed flags, and steady state;
- observable and pseudo-observable metadata and transforms;
- canonical equilibrium matrices and solved transition/system matrices;
- Kalman filtered states, covariances, and period/total likelihoods;
- posterior likelihood and prior components;
- deterministic mode forecasts;
- stochastic forecasts driven by the same structural-shock draws; and
- forecast/history means and bands.

Selected worst differences:

| Surface | Maximum absolute difference |
| --- | ---: |
| Solved transition matrix (`system/TTT`) | 1.862e-12 |
| Shock-loading matrix (`system/RRR`) | 1.406e-12 |
| Deterministic forecast observables | 2.120e-12 |
| Shared-shock stochastic forecast observables | 2.423e-12 |
| Final filtered state | 1.941e-11 |
| Total log likelihood | 2.938e-09 |

The total-likelihood difference is small relative to its magnitude
(4.794e-13) and passes the declared mixed absolute/relative tolerance.

The committed smoke oracle also passes all 59 hard-target arrays without a
Julia runtime, and the committed Julia-derived financial-frictions fixture
passes both helper-surface arrays. The fixture loader now rejects duplicate
dataset keys instead of silently overwriting one oracle file with another.

The full Python repository suite completed successfully with 494 collected
tests; four platform-specific tests were skipped. `ruff` and `ty` both pass on
`src` and `tests`.

## Scenario-layer checks

The quarterly package adds tests that are not upstream Julia APIs:

- exact policy-rate paths are solved using only the unexpected monetary-policy
  shock (`rm_sh`);
- requested rate paths reconcile to numerical precision;
- structural scenario components add correctly across shock, start, duration,
  and decay;
- every Model1002 shock is assigned exactly once in the reporting taxonomy;
- `corepce_sh` is classified as a measurement innovation;
- grouped historical decompositions reconcile in model and reporting units;
- CPI summary, food, shelter, other-services, and goods accounting reconcile;
- stale CPI snapshots, date-grid defects, duplicate scenarios, and unknown
  shocks fail closed; and
- the complete quarterly package has a frozen-data end-to-end smoke test.

## Boundaries

This evidence supports high confidence in the public `ss10` equations,
measurement system, filtering, likelihood, and forecast translation. It does
not certify other DSGE.jl models, optional subspec branches outside the tested
contract, a private production parameterization, or the local targeted
optimizer result.

The targeted eight-parameter MAP refresh used by the quarterly package is
therefore labeled as a Python-local estimate. Its updated parameters can be
used for analysis, but optimizer parity should not be inferred from the
fixed-parameter Julia comparisons above.
