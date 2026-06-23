# Julia Oracle Tools

This directory is reserved for migration-only scripts that run upstream
`DSGE.jl` to export Python-readable parity fixtures.

Rules:

- Scripts here may call Julia.
- The `nydsge` package runtime must never import Julia, shell out to Julia, or
  require Julia to be installed.
- Exported fixtures should use CSV, NPZ, HDF5, or JSON metadata so Python tests
  can run without Julia.

## Model1002 Matrix Fixtures

`export_model1002.jl` exports parameter, steady-state, canonical, transition,
transition existence/uniqueness status, state-space, and pseudo-measurement
matrix fixtures for the Python parity tests.

```powershell
juliaup add 1.8
julia +1.8 tools/oracle_julia/setup_env.jl
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10.h5
uv run nydsge vv export-matrices --output-dir tests/fixtures/candidate --data-vintage 181115 --forecast-start 2018-Q4
```

Run this only in a Julia environment where `DSGE.jl`, `ModelConstructors.jl`,
and `HDF5.jl` are available. The registered `DSGE v1.3.0` dependency graph is
not compatible with Julia 1.10/1.12 because of older `JLD2` bounds, so the
oracle project is bootstrapped and run with native Julia 1.8 via `juliaup`.
Compare exported fixtures with Python outputs via:

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile model-setup
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile matrix
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --tolerance-profile strict --json
```

## Model1002 Benchmark Baselines

`benchmark_model1002.jl` writes a Julia oracle JSON report that can be attached
to Python benchmark output with `nydsge bench --baseline`. It is still
migration-only: Python never calls this script at runtime.

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/benchmark_model1002.jl --out tests/fixtures/oracle/julia_benchmark_model1002.json --kernel forecast --horizon 40 --repeats 3
uv run nydsge bench --kernel forecast --horizon 40 --repeats 3 --baseline tests/fixtures/oracle/julia_benchmark_model1002.json --json
```

The current baseline producer times Julia's native deterministic forecast call
after one untimed warmup. The JSON report includes Julia version metadata,
elapsed samples, and a `results` array with `kernel`, `horizon`, `dtype`, and
`elapsed_seconds`, matching the Python baseline loader contract.

Pass `--include-financial-frictions true` to export the upstream BGG helper
formula surface used by the Model1002 steady-state calculation:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_financial.h5 --include-financial-frictions true
uv run nydsge vv export-financial-frictions --output-dir tests/fixtures/candidate --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile financial-frictions --tolerance-profile strict --json
```

Pass `--include-forecast true` to the Julia exporter to add deterministic
zero-shock mode forecast arrays in the same namespace used by Python
`export-suite`:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10.h5 --include-forecast true --horizon 40
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --horizon 40 --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile forecast-mode --tolerance-profile strict --json
```

Pass `--include-history true` to add mode `histobs` and deterministic histobs
means/bands. If Julia should build the transformed data itself, configure the
FRED API key expected by `FredData.jl`; the exporter can write the transformed
CSV with `--data-out` so Python can replay the exact same data. Otherwise pass a
prebuilt transformed observable CSV with `--data-in` and feed that same CSV to
Python `export-suite`:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10.h5 --include-history true --data-in path\to\observables.csv
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --data path\to\observables.csv --horizon 40 --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile forecast-mode-history --tolerance-profile strict --json
```

For real-vintage FRED/ALFRED parity, export the oracle and transformed CSV,
then replay the CSV on the Python side. Some 181115 optional series are all
missing in ALFRED; pass `--allow-empty-data-columns` to Python only for this
Julia replay case.

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_real_history.h5 --include-history true --include-kalman true --include-posterior true --include-forecast true --include-full-forecast true --full-draws 2 --data-out tests/fixtures/oracle/observables.csv --horizon 2
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --data tests/fixtures/oracle/observables.csv --shock-samples tests/fixtures/oracle/m1002_ss10_real_history.h5 --allow-empty-data-columns --horizon 2 --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile hard-target --tolerance-profile strict --json
```

Pass `--include-full-forecast true` to add deterministic full forecast sample
artifacts. Without `--shock-samples-in`, the Julia exporter builds repeated
zero-shock samples using `--full-draws`; this is the exact full-artifact smoke
gate and intentionally avoids cross-language RNG stream differences. If an
explicit HDF5 shock cube is supplied, it must contain `shock_samples` or
`forecast_full/shock_samples`.

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_full.h5 --include-full-forecast true --full-draws 2 --horizon 40
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --shock-samples path\to\zero_shock_samples.npz --horizon 40 --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile forecast-full --tolerance-profile strict --json
```

Add `--include-history true --data-in path\to\observables.csv` to the same
Julia export to include full `histobs`, full history sample arrays, and full
histobs means/bands:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_full_history.h5 --include-full-forecast true --full-draws 2 --include-history true --data-in path\to\observables.csv --horizon 40
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --data path\to\observables.csv --shock-samples path\to\zero_shock_samples.npz --horizon 40 --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile forecast-full-history --tolerance-profile strict --json
```

Pass `--include-posterior true --data-in path\to\observables.csv` to add the
current-parameter posterior decomposition. Feed the same transformed observable
CSV to Python `export-posterior`:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_posterior.h5 --include-posterior true --data-in path\to\observables.csv --horizon 40
uv run nydsge vv export-posterior --output-dir tests/fixtures/candidate --data path\to\observables.csv --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile posterior --tolerance-profile strict --json
```

The Python HDF5 loader reads Julia dataset paths directly, so an oracle dataset
such as `system/TTT` compares against the candidate key written from
`system.npz` as `system/TTT`.

## Python Candidate Suite

Once matching Julia oracle fixtures are generated, use the native candidate
suite command to rebuild the Python side of the hard parity target without
calling Julia:

```powershell
uv run nydsge vv export-hard-target-inputs --output-dir tests/fixtures/hard_target_smoke --json
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --data path\to\observables.csv --full-draws 1000 --seed 123 --horizon 40 --json
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --shock-samples path\to\shock_samples.npz --horizon 40 --json
uv run nydsge vv export-kalman --output-dir tests/fixtures/candidate --data path\to\observables.csv --json
uv run nydsge vv export-posterior --output-dir tests/fixtures/candidate --data path\to\observables.csv --json
uv run nydsge vv oracle-coverage --oracle-dir tests/fixtures/oracle --profile hard-target --json
uv run nydsge vv oracle-coverage --oracle-dir tests/fixtures/oracle --profile kalman --json
uv run nydsge vv oracle-coverage --oracle-dir tests/fixtures/oracle --profile posterior --json
uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --oracle-dir tests/fixtures/oracle --tolerance-profile forecast --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile kalman --tolerance-profile strict --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile posterior --tolerance-profile strict --json
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile hard-target --tolerance-profile strict --json
```

Fast sampler-oracle smoke fixtures can be generated without running the Julia
mode optimizer or Hessian calculation by supplying a tiny diagonal proposal:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_sampler_smoke.h5 --include-sampler true --sampler-draws 2 --sampler-burnin 0 --sampler-blocks 1 --sampler-param-blocks 1 --sampler-thin 1 --sampler-calculate-hessian false --sampler-reoptimize false --sampler-proposal-scale 1.0e-8 --sampler-seed 123 --data-out tests/fixtures/oracle/sampler_observables.csv
uv run nydsge vv oracle-coverage --oracle-dir tests/fixtures/oracle --profile sampler --json
uv run nydsge vv oracle-coverage --oracle-dir tests/fixtures/oracle --profile sampler-trace --json
uv run nydsge vv oracle-coverage --oracle-dir tests/fixtures/oracle --profile sampler-proposal-trace --json
uv run nydsge vv sampler-fixture-summary --sampler tests/fixtures/oracle --json
uv run nydsge vv sampler-proposal-trace-check --sampler tests/fixtures/oracle --json
uv run nydsge vv sampler-posterior-replay --sampler tests/fixtures/oracle --data tests/fixtures/oracle/sampler_observables.csv --allow-empty-data-columns --json
uv run nydsge vv sampler-compare --oracle-sampler tests/fixtures/oracle --candidate-sampler path\to\python_sampler.npz --windows 4 --json
```

The fixture summary command reports the HDF5 draw orientation, parameter labels,
the `sampler/fixed` mask in sampler parameter order, retained-draw covariance
health, sampler attributes, and retained-draw
`accepted`/`log_posterior` traces when the exporter runs through its traced
proposal-scale path. The base `sampler` coverage profile now requires
`sampler/fixed` alongside `sampler/mhparams` and `sampler/proposal_covariance`;
`sampler-trace` is the stricter sampler-reward gate. The
`sampler-proposal-trace` profile additionally requires retained proposal
vectors, previous-state vectors, proposal/previous log-posterior values, uniform
acceptance draws, and log-acceptance ratios. Those arrays, plus the sampler CSV
written by `--data-out`, are the deterministic inputs for Python posterior
replay checks.
The sampler proposal trace identity gate validates retained proposal/previous
traces, acceptance decisions, retained log posterior values, and retained
`mhparams` metadata:
`uv run nydsge vv sampler-proposal-trace-check --sampler tests/fixtures/oracle --json`.
The sampler posterior replay diagnostic evaluates those model-space Julia
proposal and previous vectors through the Python posterior path:
`uv run nydsge vv sampler-posterior-replay --sampler tests/fixtures/oracle --data tests/fixtures/oracle/sampler_observables.csv --allow-empty-data-columns --json`.
Replay masks `sampler/fixed` coordinates back to the Python model values before
evaluating posterior components, then compares log posterior, log likelihood,
and log prior traces separately.
The real sampler smoke currently reaches finite replay and reports the measured
posterior offset in JSON; exact absolute sampler-likelihood normalization and
full mode/Hessian/adaptation sampler parity remain pending.
The sampler compare command converts the traced Julia HDF5 fixture to the same
diagnostic schema used by Python sampler `.npz` archives, then compares retained
draws, accepted flags, log posterior traces, covariance, acceptance windows, and
per-parameter diagnostics under an explicit tolerance profile.

`export-suite` writes parameters, steady state, matrices, mode forecasts, mode
means/bands, optional posterior and history artifacts when `--data` is
supplied, and optional full-distribution artifacts, including forecastobs and
histobs draw arrays,
when `--full-draws`, `--shock-samples`, or `--sampler-draws` is supplied. If
`--oracle-dir` is present, it runs the same fixture comparator used by
`nydsge vv compare`.
`oracle-coverage` should pass before treating any Julia export directory as a
complete parity oracle for the hard target.
For Kalman parity, export the Julia oracle with `--include-kalman true` and
the same transformed observable CSV passed to Python `vv export-kalman`.
For posterior parity, export the Julia oracle with `--include-posterior true`
and the same transformed observable CSV passed to Python `vv export-posterior`.
For deterministic hard-target smoke tests, use the same transformed observable
CSV on both sides, include Julia posterior artifacts with
`--include-posterior true`, and pass explicit repeated zero-shock samples to
Python with `--shock-samples`; this avoids depending on Julia and Python RNG
stream parity. The Julia exporter reuses the loaded history frame to seed
history-backed forecasts, so `--data-out` and `--data-in` replay runs stay
forecast-consistent. `vv export-hard-target-inputs` creates the deterministic
observable CSV, zero-shock NPZ archive, and a manifest with the exact Julia
oracle, Python candidate, and strict comparison commands for this smoke.
