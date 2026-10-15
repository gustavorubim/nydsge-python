# Repository agent guidance

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [PORTING_PLAN.md](PORTING_PLAN.md)
for the source-of-truth, CPU reference, parity and verification contracts.
Preserve unrelated work and use an isolated branch/worktree when needed.

For ad hoc macro forecasts, policy surprises, historical decompositions or
plots, use [general_analytics_tools](general_analytics_tools/README.md) and its
[scoped instructions](general_analytics_tools/AGENTS.md). Keep data-vintage,
measurement, uncertainty and reconstruction caveats visible. Do not silently
substitute observables, adopt unconverged estimates or relabel dated outputs.

For daily-driver changes, read [the feature/tolerance matrix](docs/feature_matrix.md)
and [reproducible workflow](docs/daily_driver.md). Distinguish independent oracle
parity from native invariant tests. Preserve source pins and existing tolerances;
never certify all ss104 surfaces from the ss10 hard-target result. Keep Julia
optional offline tooling only. Label supplied parameter draws and future-shock
uncertainty separately; reject invalid draws and retain model identity. Do not
adopt M7 as a validated posterior. Keep private inputs and generated runs ignored.
Hosted CI uses Python-only CPU tests, read-only permissions and no secrets.
