# Feature and numerical validation matrix

The goal is independent native Python implementation of the covered **New York
Fed Model1002** behavior. This is neither the Board's FRB/US model nor an official
Federal Reserve product. Julia is optional for offline oracle regeneration;
committed oracle fixtures are test inputs, never production-computation inputs.

## Reference identity and coverage

| Scope | Reference and evidence | Acceptance contract | Limits |
|---|---|---|---|
| ss10 original hard target | DSGE.jl 1.3.0; package tree `e746a4a5ab9c26d897239e722b0f19d4bb3bd77e`; vintage 181115, origin 2018Q4 | 59 surfaces, mixed `atol=rtol=1e-10`; horizon 40, 1,000 shared shock draws | Historical release fixture result, not every future model/configuration |
| ss10 2026Q2 translation audit | Same 1.3.0 pin; 1964Q1–2026Q2, 20 forecast quarters | 59 arrays, mixed `atol=rtol=1e-10` | Public calibration; Python-local optimizer refresh uncertified |
| ss104 canonical matrices/parameters | DSGE.jl `c09dfe881f8809f65b8c0f08a761ec6cc456a6fb`; StateSpaceRoutines.jl `794fb02ff45a62537d8d5d037c1d12b3aa7e4574` | Canonical/QQ/EE absolute `<=1e-12`; parameter values exact | 29 regimes, public defaults and three curated non-default vectors |
| ss104 solved/measurement matrices | Same pins, committed source-exported fixtures | TTT/RRR/CCC/ZZ/DD/pseudo matrices absolute `<=5e-10` | Controlled replay, not private production configuration |
| ss104 likelihood/filter/smoother | Same pins | Likelihood `<1e-6`; predicted/filtered state means `<2e-6`; smoother `<5e-6` default, `<1e-5` perturbed | Covariances absolute `<2e-4`, plus max-entry-normalized bounds: filtered `<1e-6`, predicted `<5e-10` |
| ss104 deterministic forecasts | Same pins | States `<1e-6` default / `<2e-6` perturbed; observables `<1e-7` / `<5e-7`; pseudo-observables `<1e-6` | Controlled unconditional/full conditional replay, 40-quarter horizon |
| ss104 independent-machine oracle | Same source pins, Julia 1.12.7 | Original gates retained | Reported conditional-observable `1.53845e-7` and smoother `5.18873e-6` exceed original `1e-7` / `5e-6`; not all-gates parity |
| Native policy paths | Toolkit target realization and timing tests | Annual-percent rate targets within `1e-8` percentage points | Held sequential surprises differ from announced guidance; not intraday FOMC identification |
| HD additive reconstruction | Toolkit strict plotted-unit gate | Maximum absolute error `<=1e-8`, with initial state, deterministic path and remainder visible | A passing run certifies that run only; never conceal a failing residual |
| Native ss104 estimation dispatch | Generic fixed posterior and selected-coordinate local optimization/Hessian | Native interface and model-preservation tests | No new optimizer/Hessian Julia parity; generic ss104 MH unsupported; M7 remains unconverged |
| Native parameter-draw forecasts/bands | Native Python interfaces and invariant tests | See current tests/API and run-specific uncertainty metadata | No independent Julia parameter-draw parity claim; supplied draws are not automatically posterior draws |
| Empirical forecast quality | No implemented rolling real-time scorecard | Requires dated out-of-sample evaluation | Source-code parity does not establish predictive superiority |

Exact executable fixture assertions: [`tests/test_ss104.py`](../tests/test_ss104.py).
Detailed results: [ss104 M6](ss104_m6.md),
[release recipe](release_fixture.md),
[2026Q2 audit](translation_validation_2026q2.md),
[port inventory](port_matrix.csv). Shared shock replay checks computational
agreement conditional on the same innovations; it does not validate independent
random-number streams or parameter uncertainty.

## Independence, uncertainty and fail-closed interpretation

The older ss10 oracle obtains transform/source metadata from
`tools/oracle_julia/python_metadata_ss10.json`. Equality there verifies a shared
contract, not independently translated measurement semantics. Native measurement
tests and source inspection are additional evidence; a fully independent raw-data
transformation oracle remains a priority.

Public defaults are a calibration, not an estimated posterior. M7 reached its
runtime limit and is unconverged. Future-shock bands at fixed parameters omit
parameter uncertainty. Bands over supplied parameter draws include only the
uncertainty represented by those draws, and include future innovations only if
explicitly simulated. They do not automatically include revisions, model
misspecification or sampling/optimizer convergence uncertainty.

No supported interface may silently use Julia, replace missing official data,
ignore an invalid draw, relabel an optimizer failure as convergence, or weaken a
parity gate. Unsupported operations must raise explicit errors. Numeric identity
and timing tests for newly added native features do not replace an independent
upstream oracle comparison.

## Daily-driver acceptance

Python-only hosted CPU CI exercises locked dependencies, committed fixtures and
native tests on Windows/Linux and Python 3.12/3.14. A workflow definition is not
evidence of a green run: report actual check conclusions for the reviewed commit.
The hosted fixture suite does not regenerate the larger historical release
fixture or retrieve current economic data. Never generalize a local numerical
pass to all BLAS versions, data vintages or parameter vectors.

For production research use, require reproducible input availability, conditioning
and parameter provenance; passing run-specific decomposition gates; explicit
uncertainty labels; and preserved output hashes. For empirical forecasting claims,
add real-time rolling-origin errors and coverage against simple benchmarks.

The `future_model_innovations` source draws from dated model QQ. This includes
measurement-state innovations where QQ assigns them positive variance, alongside
economic shocks. It does not draw additional observation noise from EE or the
conditional terminal-state covariance. Parameter rows re-estimate conditional
state means at each supplied vector; they do not sample latent-state uncertainty.
