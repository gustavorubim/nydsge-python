# Workflow correctness follow-up

Base: merged main `dd0503f253fae40a4a484efa0c7c0d6eef31fac7`.
This follow-up fixes independently reproduced workflow defects. It does not
change economic calibration, source fixtures, parity tolerances or operational
desk scripts. Julia remains optional offline oracle tooling.

## Review finding to regression mapping

| Finding | Correction | Focused acceptance evidence |
|---|---|---|
| Live economy source grid dropped the transformation lead and conditional quarters | Construct the live merge grid with the same `_levels_start`/`_levels_end` boundaries as the selected source rows and frozen snapshot path | Eight offline cases compare exact live/frozen frames across lead on/off, conditioning on/off, Q1 year boundary and Q2 starts; preserve actual lead levels, trailing missing conditional values and exclude extra future quarters |
| ss104 CLI rejected unavailable all-missing measurement channels | Permit native missing channels through CLI loading while retaining native schema/date validation | `tests/test_workflow_exports.py` executes actual ss104 estimation and forecast/meansbands exports with missing expected-rate channels; ss10 retains its default empty-column guard |
| Exported conditional means/bands dates and uncertainty could lose context | Label future quarters after the conditional window and preserve uncertainty provenance | The same file checks none/full/semi dates for ss10 and ss104, archive SHA-bound `filename.npz.metadata.json` sidecars, numeric-loader compatibility and replacement of stale metadata |
| ss10 parameter-draw histories/conditioning diverged from the point driver | Reuse the native point workflow for each supplied parameter row | `tests/test_forecast_contracts.py` checks same-parameter reductions and conditioning/initial-state behavior; this establishes internal agreement, not a new posterior or independent Julia result |

Additional regression mapping:

- `tests/test_workflow_data_contracts.py` checks duplicate/missing quarter rejection,
  calendar ordering and explicit date indices; rejects missing conditional rows or
  prior-origin history; optional FRED series no longer swallow API/malformed-payload
  failures as legitimate missing observations.
- `tests/test_forecast_contracts.py` checks rank-deficient HD against covariance
  weights and zero-variance innovations, while preserving unique full-column-rank
  recovery. No residual is reassigned to manufacture closure and no gate is relaxed.

Focused live-grid tests: **8 passed**. Export/CLI regression tests: **12 passed** after the independent-review follow-up.
Scoped Ruff lint and formatting passed.
Historical decomposition still accepts valid partial dated research windows;
only forecasting requires history to reach its declared origin.
Integrated validation: **687 passed, 4 skipped** in 104.26 seconds; full Ruff
format/check, ty and 37-file runtime-purity checks passed. The locked isolated
Windows environment and wheel build succeeded. Existing oracle fixtures and
numerical gates remain unchanged.

The bounded frozen-public-input pipeline completed and rendered six figures.
Strict plotted-unit HD errors were core PCE `8.56e-11`, GDP `1.61e-9`, and FFR
`1.26e-9`, all below `1e-8`. Native ss104 CLI conditional forecast and means/bands
exports also completed with two future rows following one conditional quarter.
Local generated evidence remains ignored under `outputs/`; source snapshot
vintage and availability limits remain in the run manifests.

## Limits

The source-fetch tests use clearly synthetic offline inputs and perform no live
retrieval. They establish identical sample boundaries and missing-value handling,
not official release availability, data freshness or forecast accuracy. Keep
real source snapshots, vintage, units, publication/retrieval times and conditioning
choices explicit in each production analysis archive.

A Python-only source-parity test is implementation evidence within its fixture
scope. It does not certify a private NY Fed production configuration or empirical
forecast quality. M7 remains unconverged. Retain run-specific HD checks, original
strict thresholds, uncertainty labels and all previously documented independent
machine exceptions.

## Independent-review follow-up

The review of the first pushed head exposed three neighboring contract gaps.
Seeded ss10 future-innovation ensembles now retain their uncertainty source
through the kernel, point wrapper, means/bands and CLI/archive exports. Full/semi
ss10 point and parameter-draw routes reject absent conditioning data before
solving or implicitly reading a local file. Unconditional no-data runs remain
valid. Finally, ss104 historical export dates come from the same native prepared
frame as the returned arrays, preserving supplied presample rows instead of
silently omitting a mismatched date axis. Actual CLI regressions cover both
forecast and historical-state bands exports starting in 1959Q3.
