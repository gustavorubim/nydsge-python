# Benchmark Reference Reports

`nydsge bench --output` writes a durable JSON report for local and cross-machine
benchmark curation. The Python runtime still does not call Julia; Julia baseline
JSON can be attached explicitly with `--baseline`.

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
