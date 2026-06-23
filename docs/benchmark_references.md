# Benchmark Reference Reports

`nydsge bench --output` writes a durable JSON report for local and cross-machine
benchmark curation. The Python runtime still does not call Julia; Julia baseline
JSON can be attached explicitly with `--baseline`.

Current workflow tools:

- `nydsge bench --output` writes local Python benchmark reports.
- `tools/oracle_julia/benchmark_model1002.jl` writes migration-only Julia
  forecast baselines for explicit `--baseline` comparisons.
- `scripts/capture_benchmarks.py` runs the local Python capture plus optional
  Julia baseline attachment.
- `scripts/compare_benchmark_reports.py` compares captured reports across
  machines.

Current real-machine baseline status:

- `windows-cpu`, `macos-cpu`, `linux-cpu`: captured when available from each target machine.
- `windows-cuda`: **pending** on real-machine baseline capture.
- `macos-mps`: **pending** on real-machine baseline capture.
- `linux-cuda`: **pending** on real-machine CUDA/JAX baseline capture.

Keep all pending real-machine baselines out of the repository source tree until they
are intentionally curated under `reports/benchmarks/`.

Example local capture:

```powershell
uv run nydsge bench --kernel all --horizon 40 --periods 40 --batches 8 --draws 2 --repeats 3 --output reports\benchmarks\windows_cpu.json
```

Example with a Julia oracle baseline:

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/benchmark_model1002.jl --out reports\benchmarks\julia_model1002_forecast.json --kernel forecast --horizon 40 --repeats 3
uv run nydsge bench --kernel forecast --horizon 40 --repeats 3 --baseline reports\benchmarks\julia_model1002_forecast.json --output reports\benchmarks\windows_cpu_vs_julia.json
```

Report schema:

- `schema_version`: versioned report shape.
- `created_utc`: capture timestamp.
- `command`: kernel, horizon, periods, batches, draws, repeats, dtype,
  pseudo-observable flag, and optional baseline path.
- `platform`: OS, release, machine, processor, and Python version.
- `results`: benchmark rows matching the normal `nydsge bench --json` output,
  including skipped native targets and optional baseline speedups.

Reference timing files should be collected separately for native Windows CPU,
Windows CUDA, macOS CPU/MPS, Linux CPU, and Linux CUDA/JAX machines. Unsupported
targets should remain in the report as explicit skipped rows.

Native benchmark labels:

- `windows-cpu`: native Windows with the NumPy CPU reference path.
- `windows-cuda`: native Windows with PyTorch CUDA available.
- `macos-cpu`: native macOS with the NumPy CPU reference path.
- `macos-mps`: native macOS with PyTorch MPS available.
- `linux-cpu`: native Linux with the NumPy CPU reference path.
- `linux-cuda`: native Linux with PyTorch CUDA or JAX CUDA available.

Do not capture Windows timings from WSL. `nydsge doctor` and benchmark reports
should show the `platform/native` row as available before collecting reference
timings; accelerator rows that are unavailable should stay in the report as
skipped rows.

### Cross-Machine Capture Protocol

For each native machine, capture:

1. A local Python timing report using the same CLI parameters.
2. Optionally, a Julia forecast baseline and second Python timing report with
   `--baseline` to compute speedups.

Use the repository script for the full capture flow:

```powershell
uv run python scripts/capture_benchmarks.py `
  --kernel all `
  --horizon 40 `
  --periods 40 `
  --batches 8 `
  --draws 2 `
  --repeats 3 `
  --capture-julia-baseline `
  --label windows-cpu `
  --output-dir reports\benchmarks
```

```powershell
uv run python scripts/capture_benchmarks.py `
  --kernel all `
  --horizon 40 `
  --periods 40 `
  --batches 8 `
  --draws 2 `
  --repeats 3 `
  --label linux-cuda `
  --julia-script tools\oracle_julia\benchmark_model1002.jl `
  --julia-version 1.8 `
  --output-dir reports\benchmarks
```

The script writes up to three JSON files per pass:

- `<label>_<kernel>_<date>_local.json` with native outputs.
- `<label>_julia_forecast_<date>.json` for the captured Julia baseline (if requested).
- `<label>_<kernel>_vs_julia_<date>.json` with baseline speedup fields when a
  baseline is available.

Benchmark report files include:

- `platform` metadata (OS, machine, processor, Python).
- `runtime_statuses` for all runtime backends/devices probed on that machine.
- `results` with ran/skipped/failed states and optional baseline speedup.

### Cross-Machine Comparison

After captures are collected, compare them with:

```powershell
uv run python scripts/compare_benchmark_reports.py `
  --report reports\benchmarks\windows-cpu_forecast_2026-06-22_local.json `
  --report reports\benchmarks\linux-cpu_forecast_2026-06-22_local.json `
  --baseline-machine linux-cpu `
  --baseline-backend numpy `
  --output reports\benchmarks\cross_machine_summary.json
```

Use `--kernel` to narrow to `forecast`, `kalman`, `kalman-batch`, or
`hard-target`, and `--no-strict` when signatures legitimately differ (for example,
different repeat counts during warmup sweeps).

The output payload is JSON with:

- `summary.signature`: canonical command signature used for comparison.
- `summary.rows`: flattened rows across machines.
- `speedups`: per `(kernel,backend,device,horizon,repeats,dtype)` entries with machine
  elapsed times and speedup ratios.

Compare only the currently captured machine set:

- `macos-cpu`
- `linux-cpu`
- `windows-cpu`

and retain these files under versioned `reports/benchmarks/` locations for
cross-machine comparisons.
