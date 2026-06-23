# NY DSGE Python Porting Plan

## 1. Mission

Port the New York Fed DSGE workflow from Julia into native Python so the Python
package can construct, solve, estimate, forecast, validate, analyze, and
benchmark the model without any Julia runtime dependency.

The central goal is not just to make a similar Python model. The goal is to make
the Python implementation produce the same numerical results as the upstream
Julia implementation for the defined hard target, with repeatable validation
evidence that can be regenerated and inspected.

Julia remains useful during migration, but only as an offline oracle generator.
The runtime package under `src/nydsge/` must not call Julia, import Julia, shell
out to Julia, use WSL hooks, or depend on Julia-generated files to compute normal
results.

## 2. Source Of Truth And Local Contracts

Upstream behavioral source of truth:

- `FRBNY-DSGE/DSGE.jl`
- Model source area: `src/models/representative/m1002/`
- Solver source area: `src/solve/`
- Data source area: `src/data/`
- Estimation and forecast source areas: `src/estimate/` and forecast workflows

Python target:

- Runtime package: `src/nydsge/`
- Model implementation: `src/nydsge/models/m1002*.py`
- Data pipeline: `src/nydsge/data.py`
- Solver and state-space system: `src/nydsge/solve.py`
- Kalman and posterior: `src/nydsge/kalman.py`, `src/nydsge/estimate.py`
- Forecasts and means/bands: `src/nydsge/forecast.py`
- Runtime adapters: `src/nydsge/runtime.py`, `src/nydsge/backends.py`
- Validation and verification: `src/nydsge/vv.py`
- Benchmarks: `src/nydsge/bench.py`, `src/nydsge/benchmark_capture.py`,
  `src/nydsge/benchmark_compare.py`

Migration-only Julia tools:

- `tools/oracle_julia/export_model1002.jl`
- `tools/oracle_julia/benchmark_model1002.jl`
- `tools/oracle_julia/setup_env.jl`

Documentation and tracking:

- `README.md`
- `PORTING_PLAN.md`
- `docs/port_matrix.csv`
- `docs/benchmark_references.md`

## 3. Hard Target

The first release target is strict CPU float64 parity for upstream
`Model1002("ss10")`.

Required settings:

- `subspec=ss10`
- `data_vintage=181115`
- `date_forecast_start=2018-Q4`
- native NumPy/SciPy CPU float64 reference path
- no Julia runtime dependency

Required numerical surfaces:

- model settings
- parameter values, bounds, fixed flags, transforms, priors, descriptions, and
  scaling rules
- observable and pseudo-observable metadata
- steady state
- financial-frictions helper surfaces
- canonical matrices: `Gamma0`, `Gamma1`, `C`, `Psi`, `Pi`
- transition matrices: `TTT`, `RRR`, `CCC`
- measurement matrices: `ZZ`, `DD`, `QQ`, `EE`
- pseudo-measurement matrices
- transition augmentation
- solve existence and uniqueness status
- transformed 181115 observable data
- Kalman likelihood and per-period likelihood decomposition
- predicted and filtered state histories
- covariance histories
- current-parameter posterior decomposition
- mode forecast outputs
- full forecast draw outputs
- modal and full `histobs`
- modal and full `forecastobs`
- forecast pseudo-observables where supported
- deterministic means/bands outputs
- sampler metadata and replay diagnostics where deterministic replay is possible

Required user-facing outputs:

- command-line and Python API access to solve, estimate, forecast, and means/bands
- candidate fixture exports for every validation surface
- labeled comparison reports with mismatch coordinates
- benchmark reports with platform and runtime metadata
- forecast analysis artifacts when requested: all-macro forecast plots, impulse
  response plots, and historical decomposition plots

## 4. Non-Negotiable Constraints

Runtime purity:

- `src/nydsge/` must be native Python.
- Runtime code must not call Julia.
- Runtime code must not require WSL.
- Runtime code must not shell out to migration tools.
- Runtime code must not read oracle fixtures to compute production results.

Oracle isolation:

- Julia is allowed only under `tools/oracle_julia/`.
- Julia-generated fixtures are evidence, not runtime dependencies.
- Julia benchmark JSON files may be attached explicitly for benchmark comparison.
- Python must be able to run without Julia installed after fixtures are generated.

Numerical discipline:

- NumPy/SciPy float64 CPU is the release-blocking reference path.
- Accelerator backends are secondary and tolerance-based.
- Every parity result should preserve labels and dimensions.
- Orientation differences must be explicit and documented.
- No comparison should pass because a required oracle surface is missing.
- A fixture is not release-grade unless it covers the intended horizon, draw
  count, labels, and data vintage.

Platform discipline:

- Native Windows is supported.
- Native macOS is supported.
- Native Linux is supported.
- WSL is intentionally unsupported for this project.
- Torch CUDA is the Windows accelerator target.
- Torch MPS is the macOS accelerator target.
- Torch CUDA and JAX CUDA are Linux accelerator targets.

## 5. Current Implementation Inventory

This section records what has been implemented or partially implemented so the
plan is not just aspirational.

### 5.1 Project Scaffold

Implemented:

- `uv`/hatchling project scaffold.
- CLI entry point through `nydsge`.
- `ruff`, `ty`, and `pytest` configured.
- Source package under `src/nydsge`.
- Test suite under `tests`.
- Migration oracle tools under `tools/oracle_julia`.

Still needed:

- Make repo-wide lint and type checks resilient to generated output folders.
- Keep generated artifacts either ignored or moved under a documented artifact
  path that tooling does not scan by default.
- Update `docs/port_matrix.csv` status values from broad `started` labels to
  explicit final states.

### 5.2 Runtime And Backend Selection

Implemented:

- Explicit runtime config for backend, device, and dtype.
- Native platform detection.
- WSL rejection path.
- NumPy CPU reference path.
- Optional Torch and JAX backend adapters.
- MPS float32 guard.
- Backend status reporting in `nydsge doctor`.
- Runtime purity audit.

Still needed:

- Reconfirm all runtime-purity checks after future CLI/reporting work.
- Keep explicit skipped rows for unavailable accelerators.
- Validate active environments separately from historical benchmark environments.

### 5.3 Model Registry And Core Types

Implemented:

- Model dataclasses and shared core objects.
- Discoverable model registry.
- `Model1002` aliases under `m1002` and `model1002`.
- Settings, parameters, observables, pseudo-observables, transforms, and priors.

Still needed:

- Keep registry behavior stable as additional subspecs or models are added.
- Avoid expanding model scope until the `Model1002 ss10` hard target is complete.

### 5.4 Model1002 Index And Metadata Surfaces

Implemented:

- Base `ss10` state, shock, equation, observable, and pseudo-observable order.
- Anticipated-shock expansion.
- Expected-FFR SPD paths.
- Flexible-AIT initialization metadata.
- `add_pgap` and `add_ygap` variants for implemented policy-rule branches.
- `ss104` can run through the `ss10` pipeline where compatible.
- Expected-FFR fallback behavior when `expected_ffr` is empty.

Still needed:

- Broader newer-subspec branches remain out of scope or partially gated.
- More Julia oracle permutations are needed for optional branch combinations.

### 5.5 Parameter Surface

Implemented:

- Static `ss10` parameter table.
- Values, fixed flags, bounds, transforms, and model-space scaling.
- Base prior metadata.
- Parameter descriptions and categories.
- Regime metadata.
- Anticipated-policy shock eta loading.
- GDP-deflator priors.
- Conditional-observation measurement-error parameters.
- Candidate export and Julia comparison coverage for model setup and hard target.

Still needed:

- Confirm optional branch parameter activation with wider Julia oracle
  permutations.
- Preserve metadata coverage when new parameters or regimes are promoted.

### 5.6 Financial Frictions And Steady State

Implemented:

- BGG helper formulas.
- Fixed-point numerical helper tests.
- `ss10` steady-state formulas.
- Financial-frictions root solve.
- Strict Julia oracle comparison for steady-state and helper surfaces.

Still needed:

- Keep root-solve edge cases covered.
- Add release-grade fixture evidence for the full hard target, not only focused
  surfaces.

### 5.7 Observables And Pseudo-Observables

Implemented:

- Base observable order.
- Source mnemonics.
- Reporting transforms.
- Reverse transforms.
- `hours_first`, `first_observable`, and `last_observable` metadata.
- Forward-looking metadata.
- Anticipated GDP observables and transforms.
- Expected-FFR SPD observables.
- Flexible-AIT initialization pseudo-data observables.
- Default `ss10` pseudo-observable order and transforms.
- Optional pseudo-observables from flexible-AIT and `add_pgap`/`add_ygap`
  state variants.
- Regime-keyed activation with numeric and named regime keys.

Still needed:

- Extend oracle coverage for richer optional observable combinations.
- Keep every observable output labeled by date and metric.

### 5.8 Equilibrium Conditions

Implemented:

- Default `ss10` `Gamma0`, `Gamma1`, `C`, `Psi`, and `Pi` builder.
- Generated Python matrices from active upstream equations.
- Flexible-AIT initialization `pgap` and `ygap` shock equations.
- `add_pgap`/`add_ygap` policy-rule branch coverage for active aliases:
  `ngdp`, `ait`, `smooth_ait`, `smooth_ait_gdp`, `flexible_ait`, and `rw`.
- Named-baseline `set_pgap1` regime-selector handling.

Still needed:

- Capture broader Julia oracle fixture permutations for optional equation
  branches.
- Keep strict matrix parity as a release-blocking gate.

### 5.9 Measurement And Pseudo-Measurement

Implemented:

- Default `ss10` `ZZ`, `DD`, `QQ`, and `EE` builder.
- Base observables.
- Anticipated nominal-rate rows.
- Anticipated GDP rows.
- Expected-FFR SPD rows.
- Flexible-AIT initialization `pgap`/`ygap` rows.
- Conditional GDP, Core PCE, and anticipated-GDP measurement-error rows.
- Constant-transition expectation helpers.
- Regime-horizon mapping.
- Overlap handling that preserves union tags.
- Default `ss10` pseudo-measurement builder.
- Upstream 21-row pseudo-observable order.
- Strict full-compare pseudo-measurement parity for default surfaces.

Still needed:

- Richer optional observables remain partially covered.
- Wider Julia fixture permutations are needed for optional measurement branches.

### 5.10 State Augmentation

Implemented:

- Lag states.
- Expected-inflation state.
- Measurement-error AR states.
- Expected-FFR SPD measurement-error states.
- Conditional GDP, Core PCE, and anticipated-GDP measurement-error states.
- Optional augmentation for direct numeric-regime and named-regime activation.

Still needed:

- Broader Julia oracle fixture permutations for optional augmentation branches.

### 5.11 Solver

Implemented:

- Canonical and state-space matrix containers.
- Direct nonsingular `Gamma0` solve path.
- SciPy QZ/gensys reference path.
- Finite-value validation for canonical and state-space matrices.
- Transition existence and uniqueness fixture export.
- Strict Julia matrix comparison for canonical transition and required system
  matrices after orientation normalization.

Still needed:

- Keep full QZ/gensys behavior covered beyond the direct path.
- Ensure failures expose clear existence/uniqueness diagnostics.
- Decide which non-`ss10` solve paths are out of scope versus future targets.

### 5.12 Data Loading And Vintage Pipeline

Implemented:

- Disk CSV loading.
- Explicit CSV path handling.
- Matrix ordering.
- In-sample and out-of-sample filtering.
- `date_forecast_start` and `date_mainsample_start`.
- Model1002 level-to-observable transforms.
- Raw-level CSV data build.
- HP-filtered population helper columns.
- Per-capita divisor.
- Optional population forecast extension of the HP filter.
- Local source-file acquisition and merge.
- Observable metadata source requirements.
- Canonical source-root preparation.
- Required-quarter source coverage checks.
- Required-quarter value validation.
- Optional all-missing source columns.
- FRED graph CSV download.
- FRED observations API real-time vintage fetch.
- `.env` and `FRED_API_KEY` resolution.
- FRED JSON error normalization.
- ALFRED missing-series fill for optional series.
- Realtime duplicate collapse to latest realtime row.
- Explicit `vintage_dates` and output-type request support.
- FRED-backed Julia 181115 transformed observable CSV export and Python replay.
- Strict hard-target V&V through `--data-out` and `--data-in` workflow.

Still needed:

- Keep a documented source bundle contract for every non-FRED local input.
- Add repeatable end-to-end raw-source-to-hard-target release evidence.
- Distinguish current public FRED fetch behavior from point-vintage ALFRED/FRED
  behavior in validation reports.

### 5.13 Estimation

Implemented:

- Current-parameter posterior evaluation for `Model1002 ss10`.
- Runtime-aware state-space setup.
- DSGE-style stationary-prior Kalman likelihood.
- Per-period log-likelihood histories.
- Predicted and filtered states.
- Covariance histories.
- Prior summation aligned to Julia per-parameter prior contributions.
- Selected-parameter optimizer.
- Finite-difference Hessian.
- Saved mode and Hessian archives.
- Hessian-derived MH proposal covariance.
- Reference random-walk Metropolis-Hastings sampler in estimation space.
- NPZ sampler persistence.
- Archive schema validation.
- Parameter metadata preservation.
- Sampler diagnostics for acceptance windows, proposal covariance health,
  log-posterior range, and per-parameter effective sample sizes.
- Julia-style pre-ZLB anticipated-policy process covariance regime.
- Matching stationary initialization for real-vintage filtering.
- Fixed-parameter model-value replay semantics for Julia sampler proposal
  vectors.
- Strict transformed-CSV Kalman oracle smoke.
- Strict current-parameter posterior oracle smoke.
- Strict sampler posterior replay smoke with component traces.
- Sampler metadata parity and adaptation-control diagnostics.

Still needed:

- Broaden sampler parity beyond smoke fixtures.
- Validate longer chains and production draw counts.
- Keep deterministic replay paths separate from stochastic diagnostic checks.
- Confirm mode/Hessian/adaptation parity on more than minimal fixtures.

### 5.14 Forecasting

Implemented:

- Linear system forecast helper.
- `Model1002 ss10` mode forecast.
- `Model1002 ss10` full forecast.
- `cond_type` paths for none, semi, and full where implemented.
- Reference full shock-draw forecast sample arrays.
- Explicit shock path padding and truncation.
- Python and Julia orientation handling.
- 3D shock-sample full forecast archives.
- Sampler parameter-draw forecast arrays.
- Deterministic and sample-quantile means/bands.
- Observable and pseudo-observable reverse-transform export paths.
- Julia-style `histobs` data output semantics.
- Stationary Kalman prior filter-backed history states.
- Smoother-backed history states.
- Dated conditional-data horizon reduction.
- Sampler-backed history and conditional sample means/bands.
- Deterministic full-conditioning structural shock path solve.
- Conditional fixture exports.
- ZLB and market-implied FFR conditional target builder.
- CLI fixture export.
- Deterministic zero-shock mode forecast oracle export.
- Forecast-mode V&V profile.
- Transformed-CSV mode-history smoke parity.
- Deterministic zero-shock full forecast sample parity.
- Full-history transformed-CSV parity.
- Hard-target explicit zero-shock parity including posterior decomposition.
- FRED-backed real-vintage history/mode/full zero-shock forecast replay.

Still needed:

- Productize forecast analysis plots into a supported command or script.
- Replace approximate historical decomposition charts with true
  shock-contribution decomposition, or name them explicitly as approximations.
- Expand production-horizon and production-draw forecast validation.

### 5.15 Oracle Fixtures And V&V

Implemented:

- Python fixture comparator.
- Canonical solve candidate writer.
- Combined Model1002 parameter metadata, steady-state, transition-status,
  matrix, Kalman, posterior, forecast, and means/bands exports.
- Oracle-matching `data_vintage` and forecast-start settings.
- Manifest labels for parameters, metadata, steady state, equations, states,
  shocks, observables, pseudo-observables, dates, and draws.
- Kalman state and covariance history exports.
- Posterior prior/likelihood decomposition exports.
- Forecast history and pseudo-observable outputs.
- Sampler-draw forecast and history samples.
- Sampler-draw means/bands.
- Explicit shock-sample forecast candidate exports.
- Sampler and shock-sample provenance metadata.
- Conditional forecast labels.
- Custom fixture stems.
- Export-suite candidate parity bundle.
- Data-backed posterior artifacts.
- Hard-target oracle coverage checks.
- Optional oracle comparison with labeled mismatch coordinates in JSON and table
  reports.
- Profile-filtered compare.
- Case-insensitive fixture suffix loading.
- Julia 1.8 local oracle project.
- HDF5 fixture exports for metadata, steady state, transition status, system,
  Kalman, posterior, deterministic forecasts, and sampler smokes.
- HDF5 orientation normalization.
- Named strict, CPU-oracle, forecast, and accelerator tolerance profiles.
- Deterministic hard-target smoke input manifest export.
- Julia exporter reuse of loaded history for forecast seeding.
- Real Julia strict smoke passes across many core profiles.

Still needed:

- Maintain a clear distinction between smoke fixtures and release-grade fixtures.
- Add a release fixture recipe for full horizon and intended draw count.
- Keep missing-oracle detection strict so absent surfaces cannot silently pass.

### 5.16 Benchmarking

Implemented:

- Native forecast benchmark.
- Single Kalman benchmark.
- Batched Kalman benchmark.
- Python hard-target replay benchmark.
- Available runtime targets run.
- Unsupported CPU/CUDA/MPS/JAX targets report skipped.
- Backend outputs compare against NumPy CPU reference through V&V backend parity.
- Optional externally generated Julia benchmark JSON baseline attachment.
- Speedup ratios through `nydsge bench --baseline`.
- Versioned benchmark reference reports with command and platform metadata.
- Julia-side Model1002 forecast baseline producer.
- Windows CPU benchmark reports captured.
- Windows CUDA benchmark reports captured using `.venv313` with CUDA-enabled
  Torch.

Still needed:

- macOS MPS baseline capture.
- Linux CUDA/JAX baseline capture.
- Cross-machine benchmark summary curation.
- Reconcile active `.venv` runtime status with historical benchmark environments
  in documentation.

### 5.17 Forecast Analysis Artifacts

Implemented as ad hoc output:

- A generated run folder under `outputs/forecast_run_20260623-143100`.
- Forecast fixture for a selected period.
- All-observable macro forecast PNG.
- 19 per-metric impulse response PNGs.
- 19 per-metric stacked-bar decomposition-style PNGs.

Not yet productized:

- The plotting logic currently lives in an output-folder script.
- `matplotlib` is not part of the declared project dependencies.
- The stacked-bar decomposition is an IRF-accumulated approximation, not a
  confirmed true historical shock decomposition.
- There are no tests for forecast plotting outputs.
- There is no documented `nydsge` command for generating the complete report.

## 6. Verifiable Reward Gates

Each gate must have:

- command or script
- expected pass/fail condition
- artifact location
- tolerance profile, if numerical
- label coverage
- clear skip behavior for unavailable optional runtimes

### Reward 1: Repo Hygiene Passes

Commands:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check .
uv run pytest
```

Pass condition:

- Formatting passes.
- Lint passes.
- Type checking passes.
- Tests pass.
- Generated output folders do not break source validation.

Current status:

- Tests pass in the latest review: `468 passed, 4 skipped`.
- `ruff format --check .`, `ruff check .`, and `ty check .` all pass repo-wide
  (unscoped); generated `outputs/` artifacts are untracked and excluded from
  tooling.

### Reward 2: Runtime Purity Passes

Command:

```powershell
uv run nydsge vv runtime-purity --json
```

Pass condition:

- `src/nydsge/` contains no Julia runtime call path.
- No shell-out wrappers are used for production computation.
- No WSL/bash runtime hooks are used.
- Julia is isolated to `tools/oracle_julia/`.

Failure condition:

- Any production runtime command requires Julia.
- Any Python module calls a Julia executable for model results.

### Reward 3: Native Runtime Doctor Is Accurate

Command:

```powershell
uv run nydsge doctor --json
```

Pass condition:

- Native platform row is available on native Windows, macOS, or Linux.
- WSL is rejected.
- NumPy CPU reports available.
- Torch CPU/CUDA/MPS rows report available or skipped with a reason.
- JAX CPU/CUDA rows report available or skipped with a reason.
- Explicit backend requests fail cleanly when unavailable.

Artifacts:

- JSON doctor output.
- Benchmark reports should embed equivalent runtime status rows.

### Reward 4: Backend Parity Works For Available Backends

Command:

```powershell
uv run nydsge vv backend-parity --kernel all --horizon 40 --periods 40 --tolerance-profile accelerator --json
```

Pass condition:

- NumPy CPU reference runs.
- Available Torch/JAX targets compare to NumPy CPU within accelerator tolerance.
- Unavailable backends are skipped with explicit reasons.
- Requested unavailable backends fail clearly.

### Reward 5: Model Registry And Constructor Contract Pass

Commands:

```powershell
uv run nydsge models --json
uv run nydsge solve --json
```

Pass condition:

- `Model1002` is discoverable.
- `m1002` and `model1002` aliases resolve.
- Default `ss10` constructs.
- Compatible `ss104` path constructs where supported.
- Unsupported subspecs raise explicit `NotPortedError`.

### Reward 6: Parameter Surface Matches Julia

Commands:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10.h5
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile model-setup --tolerance-profile strict --json
```

Pass condition:

- Parameter values match.
- Fixed flags match.
- Bounds match.
- Scaled values match.
- Prior metadata matches.
- Descriptions and categories are covered in metadata fixtures.
- Labeled mismatch coordinates are emitted on failure.

### Reward 7: Observable Metadata Matches Julia

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile model-metadata --tolerance-profile strict --json
```

Pass condition:

- Observable names match.
- Pseudo-observable names match.
- Source metadata matches.
- Transform metadata matches.
- Reporting metadata matches.
- Date and metric labels are present where arrays are exported.

### Reward 8: Financial Frictions Match Julia

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile financial-frictions --tolerance-profile strict --json
```

Pass condition:

- Helper values match.
- Derivative helper surfaces match.
- Root-solve dependent values match.
- Failures report helper name and coordinate.

### Reward 9: Steady State Matches Julia

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile model-setup --tolerance-profile strict --json
```

Pass condition:

- Every steady-state scalar in the fixture matches within strict CPU tolerance.
- Labels identify the failing steady-state variable if a mismatch occurs.

### Reward 10: Canonical Matrix Parity Passes

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile matrix --tolerance-profile strict --json
```

Pass condition:

- `Gamma0` matches.
- `Gamma1` matches.
- `C` matches.
- `Psi` matches.
- `Pi` matches.
- Equation, state, and shock labels are present.
- Orientation handling is explicit and documented.

### Reward 11: Transition And Measurement Matrix Parity Passes

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile matrix --tolerance-profile strict --json
```

Pass condition:

- `TTT`, `RRR`, and `CCC` match.
- `ZZ`, `DD`, `QQ`, and `EE` match.
- Pseudo-measurement arrays match where included.
- Existence and uniqueness status matches.
- Labels identify dates, states, shocks, and observables as appropriate.

### Reward 12: Optional Equation Branches Have Oracle Coverage

Command pattern:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/<case>.h5 <case flags>
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate_<case> <matching flags> --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate_<case> --profile matrix --tolerance-profile strict --json
```

Pass condition:

- Flexible-AIT branches compare.
- Expected-FFR SPD branches compare.
- `add_pgap` branches compare.
- `add_ygap` branches compare.
- Conditional measurement-error branches compare.
- Named and numeric regime-key paths compare.

### Reward 13: Data CSV Loading Contract Passes

Command:

```powershell
uv run nydsge data build --input path\to\raw_levels.csv --output path\to\observables.csv
```

Pass condition:

- Output columns follow Model1002 observable order.
- Dates are parsed and filtered correctly.
- In-sample and out-of-sample windows honor model settings.
- Required columns fail loudly when absent.

### Reward 14: FRED And Vintage Source Preparation Passes

Commands:

```powershell
uv run nydsge data sources --source-root path\to\raw --vintage 181115 --json
uv run nydsge data prepare-sources --source-root path\to\raw --vintage 181115 --start-date 1959-Q3 --end-date 2018-Q3 --json
uv run nydsge data build-sources --source-root path\to\raw --output path\to\observables.csv --start-date 1959-Q3 --end-date 2018-Q3
```

Pass condition:

- Source requirements list required and optional mnemonics.
- FRED API key resolution works from CLI, environment, or `.env`.
- Optional missing ALFRED/FRED series are filled only when allowed.
- Required all-missing source columns fail.
- Duplicate realtime rows collapse deterministically.
- Point-vintage behavior is documented.

### Reward 15: Raw Data Smoke Gate Passes

Command:

```powershell
uv run nydsge vv raw-data-smoke --source-root path\to\raw --output-dir tests\fixtures\raw_data_smoke --start-date 1959-Q3 --end-date 2018-Q3 --horizon 40 --json
```

Pass condition:

- Raw source files transform into observables.
- Candidate fixture generation runs from transformed observables.
- Required coverage and required values are checked.
- Output includes provenance for preprocessing controls.

### Reward 16: Kalman Parity Passes

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile kalman --tolerance-profile strict --json
```

Pass condition:

- Total likelihood matches.
- Per-period likelihood components match.
- Predicted states match.
- Filtered states match.
- Predicted covariance histories match.
- Filtered covariance histories match.
- Date and state labels are present.

### Reward 17: Posterior Parity Passes

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile posterior --tolerance-profile strict --json
```

Pass condition:

- Log likelihood matches.
- Log prior matches.
- Log posterior matches.
- Per-parameter prior contributions match.
- Fixed-parameter replay semantics match.
- Transformed and model-space parameter semantics are validated.

### Reward 18: Optimizer And Hessian Outputs Are Reproducible

Command:

```powershell
uv run nydsge estimate --data path\to\observables.csv --optimize --parameters alpha,rho --maxiter 25 --hessian --mode-output outputs\mode.npz
```

Pass condition:

- Mode archive is written.
- Hessian archive is written when requested.
- Archive schema validates.
- Parameter metadata is preserved.
- Re-running with fixed settings produces comparable objective behavior.

### Reward 19: Metropolis-Hastings Sampler Diagnostics Pass

Commands:

```powershell
uv run nydsge estimate --data path\to\observables.csv --mode-input outputs\mode.npz --mh-draws 1000 --mh-burnin 100 --sampler-output outputs\sampler.npz
uv run nydsge vv sampler-diagnostics --sampler outputs\sampler.npz --windows 4 --json
```

Pass condition:

- Sampler archive schema validates.
- Acceptance windows are reported.
- Proposal covariance health is reported.
- Log-posterior range is reported.
- Per-parameter effective sample sizes are reported.
- Fixed mask and parameter labels are preserved.

### Reward 20: Julia Sampler Replay Smoke Passes

Command pattern:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_sampler.h5 --include-sampler true
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile sampler-posterior-replay --tolerance-profile strict --json
```

Pass condition:

- Proposal vectors match replay expectations.
- Previous vectors match replay expectations.
- Proposal posterior components match.
- Acceptance uniforms and log-acceptance values match where exported.
- Presample-aware replay behavior is covered.

### Reward 21: Mode Forecast Parity Passes

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile forecast-mode --tolerance-profile strict --json
```

Pass condition:

- Mode forecast states match.
- Mode forecast observables match.
- Mode forecast pseudo-observables match where present.
- Shock paths match.
- Forecast date labels match.

### Reward 22: Mode Forecast History Parity Passes

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile forecast-mode-history --tolerance-profile strict --json
```

Pass condition:

- Mode history states match.
- Mode `histobs` match.
- Filtered and smoothed history paths are covered where applicable.
- Missing history observations are represented consistently.

### Reward 23: Full Forecast Parity Passes

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile forecast-full --tolerance-profile strict --json
```

Pass condition:

- Full forecast state samples match.
- Full forecast observable samples match.
- Full forecast pseudo-observable samples match where present.
- Shock samples match.
- Draw labels and forecast date labels match.

### Reward 24: Full Forecast History Parity Passes

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile forecast-full-history --tolerance-profile strict --json
```

Pass condition:

- Full history state samples match.
- Full history observable samples match.
- Full history pseudo-observable samples match where present.
- Draw and date labels match.

### Reward 25: Means/Bands Parity Passes

Command:

```powershell
uv run nydsge meansbands --input-type full --draws 1000 --seed 123 --horizon 40 --json
```

Pass condition:

- Mean arrays are deterministic for fixed seed and draw inputs.
- Lower and upper bands are deterministic for fixed seed and draw inputs.
- `forecastobs`, `histobs`, and `forecastpseudo` sources are covered where
  supported.
- Candidate means/bands fixtures compare to oracle where available.

### Reward 26: Hard Target Combined Gate Passes

Command:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile hard-target --tolerance-profile strict --json
```

Pass condition:

- Setup surfaces pass.
- Metadata surfaces pass.
- Matrix surfaces pass.
- Kalman surfaces pass.
- Posterior surfaces pass.
- Forecast surfaces pass.
- History surfaces pass.
- Means/bands surfaces pass.
- No required hard-target surface is missing.

Current status:

- The short checked-in hard-target smoke comparison passes in the latest review.
- A full release-grade hard-target fixture still needs to be explicitly
  regenerated, documented, and compared at intended horizon/draw count.

### Reward 27: Real-Vintage Hard Target Replay Passes

Command:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_real_history.h5 --include-history true --include-kalman true --include-posterior true --include-forecast true --include-full-forecast true --full-draws 2 --data-out tests/fixtures/oracle/observables.csv --horizon 2
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --data tests/fixtures/oracle/observables.csv --shock-samples tests/fixtures/oracle/m1002_ss10_real_history.h5 --allow-empty-data-columns --horizon 2 --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile hard-target --tolerance-profile strict --json
```

Pass condition:

- Julia-exported transformed observables replay in Python.
- Python candidate outputs match the Julia oracle.
- Optional all-empty vintage columns are allowed only for the documented replay
  case.

### Reward 28: Conditional Forecasting Passes

Command examples:

```powershell
uv run nydsge forecast --cond-type semi --data path\to\observables_with_condition.csv --include-history --horizon 40 --json
uv run nydsge forecast --cond-type full --data path\to\observables_with_condition.csv --include-history --horizon 40 --json
```

Pass condition:

- Conditional-data horizon reduction is correct.
- Conditional shock paths solve deterministically.
- Conditional states and observables are exported.
- Conditional output labels identify dates and observables.
- Unsupported conditioning types raise explicit errors.

### Reward 29: ZLB And Market-Implied FFR Conditioning Passes

Command:

```powershell
uv run nydsge forecast --horizon 8 --zlb-rates "0.25,0.25,0.50,1.00" --json
```

Pass condition:

- ZLB target vector is built correctly.
- Forecast horizon adjusts correctly.
- Conditional outputs are exported with labels.
- Invalid target lengths or values fail clearly.

### Reward 30: Forecast Fixture Export Is Complete

Command:

```powershell
uv run nydsge vv export-forecast --output-dir outputs\forecast_run --forecast-start 2018-Q4 --horizon 40 --data path\to\observables.csv --include-history --history-method smoothed --json
```

Pass condition:

- Forecast states are exported.
- Forecast observables are exported.
- Forecast pseudo-observables are exported when requested.
- History states and observables are exported when requested.
- Manifest labels cover every array axis.
- Output can be reloaded by downstream plotting/reporting tools.

### Reward 31: Forecast Plotting Is Productized

Target command:

```powershell
uv run nydsge report forecast --input outputs\forecast_run\forecast_period.npz --manifest outputs\forecast_run\manifest.json --output-dir outputs\forecast_run\figures
```

Pass condition:

- Command is documented.
- Plotting dependencies are declared as an optional extra, for example
  `plot = ["matplotlib>=..."]`.
- All-observable macro forecast panel is generated.
- Per-observable forecast plots are generated if requested.
- Per-observable IRF plots are generated.
- Per-observable historical decomposition plots are generated.
- Output filenames are deterministic.
- The command exits nonzero on missing labels, missing arrays, or inconsistent
  dimensions.

### Reward 32: Impulse Response Outputs Are Numerically Defined

Target command:

```powershell
uv run nydsge report irf --model model1002 --subspec ss10 --horizon 40 --output-dir outputs\irf
```

Pass condition:

- IRFs are computed from the solved state-space system.
- Shock normalization is documented.
- One figure per observable is generated.
- Optional combined figures are generated.
- Underlying numeric IRF arrays are exported.
- Labels identify shock, horizon, and observable.

### Reward 33: Historical Decomposition Is True Decomposition

Target command:

```powershell
uv run nydsge report historical-decomposition --data path\to\observables.csv --forecast-start 2018-Q4 --output-dir outputs\historical_decomposition
```

Pass condition:

- Decomposition is based on filtered or smoothed shock/state contributions.
- Contributions reconcile to the target observable within documented residual
  tolerance.
- Stacked bars include major shock groups or individual shocks.
- Residual/initial-condition contribution is shown or explicitly accounted for.
- Every metric receives a chart.
- Numeric decomposition arrays are exported.
- Charts are not labeled historical decomposition if they are only IRF
  accumulations.

### Reward 34: Benchmark Capture Passes On Windows CPU

Command:

```powershell
uv run python scripts/capture_benchmarks.py --kernel all --horizon 40 --periods 40 --batches 8 --draws 2 --repeats 3 --capture-julia-baseline --label windows-cpu --julia-version 1.8 --output-dir reports\benchmarks
```

Pass condition:

- Local Python benchmark report is written.
- Julia forecast baseline is written if requested.
- Baseline comparison report is written.
- NumPy CPU rows run.
- Unsupported accelerator rows are skipped with reasons.

### Reward 35: Benchmark Capture Passes On Windows CUDA

Command:

```powershell
uv run python scripts/capture_benchmarks.py --kernel all --horizon 40 --periods 40 --batches 8 --draws 2 --repeats 3 --capture-julia-baseline --label windows-cuda --julia-version 1.8 --output-dir reports\benchmarks
```

Pass condition:

- Torch CUDA is available in the active environment.
- CUDA rows run.
- CUDA outputs remain within accelerator tolerance against NumPy CPU.
- Benchmark report records CUDA device/runtime status.

Current status:

- Windows CUDA reports exist from `.venv313`.
- Active `.venv` may still report Torch unavailable, so environment-specific
  status must be clear.

### Reward 36: Benchmark Capture Passes On macOS MPS

Command:

```powershell
uv run python scripts/capture_benchmarks.py --kernel all --horizon 40 --periods 40 --batches 8 --draws 2 --repeats 3 --label macos-mps --output-dir reports\benchmarks
```

Pass condition:

- Native macOS platform row is available.
- Torch MPS row is available.
- MPS runs with supported dtype behavior.
- MPS output compares to NumPy CPU within accelerator tolerance.

Current status:

- Pending real-machine capture.

### Reward 37: Benchmark Capture Passes On Linux CUDA/JAX

Command:

```powershell
uv run python scripts/capture_benchmarks.py --kernel all --horizon 40 --periods 40 --batches 8 --draws 2 --repeats 3 --label linux-cuda --output-dir reports\benchmarks
```

Pass condition:

- Native Linux platform row is available.
- Torch CUDA or JAX CUDA row is available.
- Available accelerator rows run.
- Outputs compare to NumPy CPU within accelerator tolerance.

Current status:

- Pending real-machine capture.

### Reward 38: Cross-Machine Benchmark Summary Is Curated

Command:

```powershell
uv run python scripts/compare_benchmark_reports.py --report reports\benchmarks\windows-cpu_all_2026-06-23_local.json --report reports\benchmarks\windows-cuda_all_2026-06-23_local.json --baseline-machine windows-cpu --baseline-backend numpy --output reports\benchmarks\cross_machine_summary.json
```

Pass condition:

- Reports have compatible command signatures unless `--no-strict` is used.
- Speedup ratios are emitted.
- Skipped rows remain visible.
- Platform metadata is preserved.

### Reward 39: CLI Error Surfaces Are Intentional

Command examples:

```powershell
uv run nydsge forecast --input-type bad
uv run nydsge forecast --cond-type bad
uv run nydsge estimate --backend torch --device cuda --data path\to\observables.csv
```

Pass condition:

- Unsupported input types fail clearly.
- Unsupported conditioning types fail clearly.
- Unavailable backends fail clearly when explicitly requested.
- Non-target subspecs raise `NotPortedError` with actionable source-area notes.

### Reward 40: Documentation Matches Reality

Commands:

```powershell
uv run nydsge doctor --json
uv run nydsge vv compare --profile hard-target --tolerance-profile strict --json
uv run pytest
```

Pass condition:

- `README.md` examples remain executable or clearly marked as examples requiring
  local paths.
- `docs/port_matrix.csv` statuses reflect actual support.
- `docs/benchmark_references.md` reflects captured and pending machine classes.
- `PORTING_PLAN.md` current status is updated after major validation passes.

## 7. Validation Ladder

Validation should move from cheap and local to expensive and release-grade.

### Level 0: Static Hygiene

Commands:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check .
```

Purpose:

- Catch syntax, style, import, and type errors before numerical validation.

Required before:

- Any commit that claims implementation completeness.

### Level 1: Unit And CLI Tests

Command:

```powershell
uv run pytest
```

Purpose:

- Validate local semantics, error handling, serialization, labels, and CLI
  contracts.

Current evidence:

- Latest review observed `458 passed, 4 skipped`.

### Level 2: Python Self-Consistency

Commands:

```powershell
uv run nydsge solve --json
uv run nydsge forecast --horizon 40 --json
uv run nydsge meansbands --horizon 40 --json
```

Purpose:

- Confirm the Python system computes coherent outputs without oracle comparison.

Warning:

- Passing Level 2 does not prove Julia parity.

### Level 3: Focused Oracle Profiles

Commands:

```powershell
uv run nydsge vv compare --profile model-setup --tolerance-profile strict --json
uv run nydsge vv compare --profile model-metadata --tolerance-profile strict --json
uv run nydsge vv compare --profile matrix --tolerance-profile strict --json
uv run nydsge vv compare --profile financial-frictions --tolerance-profile strict --json
uv run nydsge vv compare --profile kalman --tolerance-profile strict --json
uv run nydsge vv compare --profile posterior --tolerance-profile strict --json
uv run nydsge vv compare --profile forecast-mode --tolerance-profile strict --json
uv run nydsge vv compare --profile forecast-mode-history --tolerance-profile strict --json
uv run nydsge vv compare --profile forecast-full --tolerance-profile strict --json
uv run nydsge vv compare --profile forecast-full-history --tolerance-profile strict --json
```

Purpose:

- Isolate mismatches to one model surface.

Required evidence:

- JSON output showing pass/fail.
- Labeled maximum absolute and relative difference.
- Shape comparison.
- Coordinate labels for mismatch maxima.

### Level 4: Hard Target Smoke

Command:

```powershell
uv run nydsge vv compare --profile hard-target --tolerance-profile strict --json
```

Purpose:

- Confirm the combined hard-target surface passes on a short checked-in fixture.

Current evidence:

- Latest review observed the hard-target strict profile passing against the
  checked-in candidate/oracle fixture surface.

Warning:

- A short smoke fixture is not sufficient for release-grade parity.

### Level 5: Real-Vintage Replay

Commands:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_real_history.h5 --include-history true --include-kalman true --include-posterior true --include-forecast true --include-full-forecast true --full-draws 2 --data-out tests/fixtures/oracle/observables.csv --horizon 2
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --data tests/fixtures/oracle/observables.csv --shock-samples tests/fixtures/oracle/m1002_ss10_real_history.h5 --allow-empty-data-columns --horizon 2 --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile hard-target --tolerance-profile strict --json
```

Purpose:

- Validate data-backed replay rather than synthetic-only arrays.

Required before:

- Claiming 181115 vintage parity.

### Level 6: Release-Grade Hard Target

Target command pattern:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_release.h5 --include-history true --include-kalman true --include-posterior true --include-forecast true --include-full-forecast true --full-draws 1000 --data-out tests/fixtures/oracle/observables_release.csv --horizon 40
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate_release --data tests/fixtures/oracle/observables_release.csv --shock-samples tests/fixtures/oracle/m1002_ss10_release.h5 --allow-empty-data-columns --horizon 40 --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate_release --profile hard-target --tolerance-profile strict --json
```

Purpose:

- Validate the intended production horizon and draw count.

Notes:

- The exact draw count can be adjusted if artifact size becomes too large, but
  the release document must state the chosen count.
- If a smaller release fixture is chosen, the limitation must be explicit.

### Level 7: Raw-Source End-To-End Replay

Command pattern:

```powershell
uv run nydsge data prepare-sources --source-root path\to\raw --vintage 181115 --start-date 1959-Q3 --end-date 2018-Q3 --json
uv run nydsge data build-sources --source-root path\to\raw --output outputs\observables_181115.csv --start-date 1959-Q3 --end-date 2018-Q3
uv run nydsge vv export-suite --output-dir outputs\candidate_181115 --data outputs\observables_181115.csv --horizon 40 --json
uv run nydsge vv compare --oracle-dir tests\fixtures\oracle --candidate-dir outputs\candidate_181115 --profile hard-target --tolerance-profile strict --json
```

Purpose:

- Validate from source data to model outputs.

Required before:

- Claiming full data pipeline parity.

### Level 8: Accelerator Parity

Command:

```powershell
uv run nydsge vv backend-parity --kernel all --horizon 40 --periods 40 --tolerance-profile accelerator --json
```

Purpose:

- Validate native optional runtime outputs against NumPy CPU.

Required before:

- Claiming support for a specific accelerator backend.

### Level 9: Benchmarks

Command:

```powershell
uv run python scripts/capture_benchmarks.py --kernel all --horizon 40 --periods 40 --batches 8 --draws 2 --repeats 3 --capture-julia-baseline --label <machine-label> --julia-version 1.8 --output-dir reports\benchmarks
```

Purpose:

- Preserve runtime performance evidence and Julia baseline speedup comparisons.

Required before:

- Claiming performance improvement.

### Level 10: User-Facing Forecast Report

Target command pattern:

```powershell
uv run nydsge report forecast --data path\to\observables.csv --forecast-start 2018-Q4 --horizon 40 --output-dir outputs\forecast_report
```

Purpose:

- Confirm an end user can generate usable forecast analysis artifacts.

Required outputs:

- forecast arrays
- manifest
- all-observable macro forecast chart
- per-metric IRF charts
- per-metric historical decomposition charts
- numeric IRF arrays
- numeric decomposition arrays

## 8. Tolerance Profiles

Strict CPU tolerance:

- Target: deterministic CPU oracle and matrix parity.
- Typical value: `atol=1e-10`, `rtol=1e-10`.
- Used for setup, metadata, matrices, Kalman, posterior, and hard target.

Forecast tolerance:

- Target: forecast accumulation surfaces where small numerical accumulation
  differences may occur.
- Typical value: `atol=1e-8`, `rtol=1e-8`.
- Used only when strict CPU tolerance is too narrow for the forecast surface and
  the deviation is understood.

Accelerator tolerance:

- Target: Torch CUDA, Torch MPS, and JAX CUDA comparisons against NumPy CPU.
- Typical value: `atol=1e-5`, `rtol=1e-5`.
- Used only for optional accelerators.

Rules:

- Do not loosen tolerance to hide unexplained mismatches.
- Every tolerance profile must be named in the validation output.
- Every mismatch report must include shape, max absolute difference, max relative
  difference, and labels.

## 9. Artifact Policy

Source-controlled artifacts:

- `PORTING_PLAN.md`
- `README.md`
- `docs/port_matrix.csv`
- `docs/benchmark_references.md`
- small test fixtures that are intentionally versioned
- scripts and source code

Generated artifacts:

- large `.npz`, `.npy`, `.h5`, `.hdf5`, `.jld`, and `.jld2` files
- benchmark reports under `reports/benchmarks/` when intentionally curated
- output figures under `outputs/`
- local raw data bundles
- local candidate exports

Rules:

- Generated binary arrays should stay ignored unless intentionally curated.
- Generated scripts inside `outputs/` should not be treated as production code.
- If a generated script becomes important, promote it into `scripts/` or
  `src/nydsge/`.
- Output folders should not break repo-wide validation commands.
- Release evidence should include command, date, platform, environment, and
  artifact path.

## 10. Remaining Release-Blocking Work

Before claiming the initial port complete:

- Fix repo-wide lint hygiene.
- Decide whether `ruff` and `ty` should exclude `outputs/` by config or whether
  generated Python scripts should never be placed there.
- Regenerate a full hard-target fixture at intended horizon and draw count.
- Compare the full hard-target fixture under strict CPU tolerance.
- Document the release fixture recipe and expected runtime.
- Update `docs/port_matrix.csv` status values from `started` to explicit final
  states.
- Confirm all runtime CLI/API paths operate without Julia installed.
- Add a productized forecast report command or documented script for plots.
- Implement true historical decomposition if the chart is named historical
  decomposition.
- Declare plotting dependencies as optional extras if plotting becomes a product
  feature.
- Capture remaining macOS MPS and Linux CUDA/JAX baselines on real machines.
- Add or refresh end-to-end raw-data-to-hard-target evidence.

## 11. Remaining Numerical Validation Work

Additional validation to strengthen confidence:

- Wider optional-branch oracle fixture matrix.
- Flexible-AIT initialization permutations.
- Expected-FFR SPD permutations.
- Conditional observation measurement-error permutations.
- `add_pgap` and `add_ygap` regime permutations.
- Named and numeric regime-key permutations.
- Longer forecast horizons.
- Larger full-forecast draw counts.
- Longer sampler chains.
- Additional mode/Hessian/adaptation comparisons.
- Raw-source data replay against curated source bundles.
- Cross-machine backend parity summaries.

## 12. Forecast Reporting And Analysis Plan

The recent manual forecast run proved that arrays and charts can be generated,
but it is not yet a completed product feature.

Current manual output:

- `outputs/forecast_run_20260623-143100/forecast_period.npz`
- `outputs/forecast_run_20260623-143100/manifest.json`
- `outputs/forecast_run_20260623-143100/figures/macro_forecasts_all_observables.png`
- 19 IRF PNG files
- 19 stacked-bar decomposition-style PNG files

Target product behavior:

- `nydsge forecast` generates numerical forecast outputs.
- `nydsge report forecast` or equivalent generates charts.
- Reports read fixture arrays and manifest labels.
- Reports fail on missing labels or inconsistent dimensions.
- Figures are deterministic and organized by artifact type.
- Numeric chart inputs are exported alongside PNGs.

Required report outputs:

- all-observable macro forecast panel
- per-observable forecast plots
- per-observable IRF plots
- per-observable historical decomposition stacked bars
- manifest of generated files
- JSON summary of chart inputs and model settings

Historical decomposition requirement:

- True decomposition must reconcile shock, initial-condition, and residual
  contributions to the observable path.
- If the method is only IRF accumulation, the chart must be named accordingly and
  not presented as historical decomposition.

## 13. Scope Boundaries

In scope for the first completion claim:

- `Model1002 ss10`
- compatible `ss104` paths already routed through the `ss10` implementation
- CPU float64 hard-target parity
- data-backed 181115 replay
- mode and full forecasts
- modal and full histories
- means/bands
- posterior and Kalman decomposition
- runtime purity
- validation CLI
- benchmark capture and comparison workflow

Out of scope for the first completion claim unless explicitly promoted:

- other DSGE.jl models
- all Model1002 newer subspecs
- full production support for every optional branch combination
- making Julia unavailable for fixture generation
- guaranteeing accelerator parity on machines we have not captured
- GUI or notebook report frontends

## 14. Definition Of Done

The initial port is complete when all of the following are true:

- `src/nydsge/` computes the hard target without Julia installed.
- Runtime purity passes.
- Native platform doctor behaves correctly.
- CPU strict hard-target V&V passes against generated Julia oracle fixtures.
- The release hard-target fixture covers the chosen production horizon and draw
  count.
- Data-backed 181115 replay passes.
- Raw-source-to-observable workflow is documented and smoke-tested.
- Forecast, history, means/bands, Kalman, posterior, and sampler surfaces have
  labeled validation artifacts.
- Unit, CLI, lint, type, and format checks pass.
- `docs/port_matrix.csv` uses explicit final statuses.
- Accelerator reports are captured where machines are available and explicitly
  pending where they are not.
- User-facing forecast artifacts are generated by supported commands or
  documented scripts.
- Historical decomposition charts either implement true decomposition or are
  explicitly named as approximations.
- README examples match the supported workflow.

## 15. Stop Rules

Stop and mark the relevant gate blocked if:

- A required Julia oracle surface cannot be exported reproducibly.
- A comparison passes only because an oracle fixture is missing.
- A Python result matches only after unexplained transposes or unlabeled
  orientation hacks.
- A runtime path requires Julia for production computation.
- A backend silently falls back to CPU when an accelerator was explicitly
  requested.
- A data-source command fills required missing data silently.
- A forecast report labels IRF accumulation as historical decomposition.
- A benchmark report omits skipped accelerator rows.
- A validation command is run against stale fixtures while claiming current
  parity.

## 16. Current Status Snapshot

Snapshot date: 2026-06-23.

Observed during latest review:

- `uv run pytest`: `468 passed, 4 skipped`
- `uv run ruff format --check .`: passed (47 files)
- `uv run ruff check .`: passed (repo-wide, unscoped)
- `uv run ty check .`: passed (repo-wide, unscoped)
- `uv run nydsge vv compare --profile hard-target --tolerance-profile strict --json`: passed
- `uv run nydsge doctor --json`: native Windows and NumPy CPU available
- `uv run nydsge vv runtime-purity --json`: passed (26 files, no Julia)
- Active `.venv`: Torch and JAX unavailable
- Windows CUDA benchmark reports exist from `.venv313`
- Release-grade hard-target fixture validated: full `181115` history (235
  quarters), horizon 40, 1000 draws; all numerical surfaces pass strict `1e-10`.
  Recipe and the placeholder-metadata caveat are documented in
  `docs/release_fixture.md`.

Resolved since the prior snapshot:

- Repo hygiene: generated `outputs/` artifacts are untracked, gitignored, and
  excluded from ruff/ty; the `tests/test_model1002.py` unused-variable failure is
  fixed. Unscoped format/lint/type checks now pass.
- Productized forecast reporting: `nydsge report forecast|irf|historical-decomposition`
  emit labeled numeric arrays, JSON summaries, and (with the optional `plot`
  extra) deterministic figures.
- True historical decomposition: RTS-smoothed per-shock contributions reconcile
  to the smoothed observable path (not an IRF accumulation).
- `docs/port_matrix.csv` statuses promoted from `started` to explicit final
  states.

Interpretation:

- The Python port has strong parity evidence for the `Model1002 ss10` hard target
  at the release horizon and draw count.
- Remaining work is primarily cross-machine accelerator captures (macOS MPS,
  Linux CUDA/JAX), broader optional-branch oracle permutations, longer sampler
  chains, and closing the placeholder metadata transform/source oracle gap.

