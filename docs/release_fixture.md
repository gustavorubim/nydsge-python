# Release-Grade Hard-Target Fixture Recipe

This document records the reproducible recipe for the release-grade hard-target
parity fixture (PORTING_PLAN.md Validation Ladder **Level 6**) and the validation
result captured on the reference Windows machine.

Unlike the short checked-in smoke fixture (horizon 12, 4 draws, 20 history
periods), the release fixture exercises the **production horizon and draw count**
against the full `181115` data vintage.

## Settings

| Setting | Value |
|---|---|
| Model / subspec | `Model1002` / `ss10` |
| Data vintage | `181115` (full history, 1959-Q3 .. 2018-Q3, 235 in-sample quarters) |
| Forecast start | `2018-Q4` |
| Forecast horizon | `40` quarters |
| Full-forecast draws | `1000` |
| Shock content | deterministic zero-shock draws shared oracle↔candidate (the accepted parity method; draws are identical by construction so the full-forecast sample machinery is validated at production shape `(1000, 40, ·)`) |
| Tolerance profile | `strict` (`atol = rtol = 1e-10`) |

> Draw count note (per PORTING_PLAN.md Level 6): the production draw count is
> `1000`. Because the parity method shares a deterministic zero-shock sample set
> between the Julia oracle and the Python candidate, the 1000 draws are identical
> and the comparison validates the full-forecast array pipeline at production
> dimensions rather than stochastic spread. Stochastic-spread parity is covered
> separately by the means/bands surfaces.

## Recipe

Requires Julia 1.8 with the instantiated `tools/oracle_julia` project and a
`FRED_API_KEY` (the Julia exporter loads the point-in-time `181115` vintage via
DSGE.jl and writes it to `--data-out`). Run from the repo root:

```bash
# 1. Julia oracle: full vintage + horizon-40 / 1000-draw forecast, history, posterior.
#    FRED_API_KEY must be exported (e.g. read from .env).
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl \
  --out outputs/release/oracle/m1002_ss10_release.h5 \
  --include-history true --include-kalman true --include-posterior true \
  --include-forecast true --include-full-forecast true --full-draws 1000 \
  --data-out outputs/release/observables_release.csv --horizon 40

# 2. Python candidate: replay against the Julia-exported vintage, reusing the
#    oracle shock samples so the full forecast is bit-for-bit comparable.
uv run nydsge vv export-suite --output-dir outputs/release/candidate \
  --data outputs/release/observables_release.csv \
  --shock-samples outputs/release/oracle/m1002_ss10_release.h5 \
  --allow-empty-data-columns --horizon 40 --json

# 3. Strict compare.
uv run nydsge vv compare --oracle-dir outputs/release/oracle \
  --candidate-dir outputs/release/candidate \
  --profile hard-target --tolerance-profile strict --json
```

`--allow-empty-data-columns` is required because the `181115` vintage carries no
observations for the anticipated/conditional columns (`obs_tfp`,
`obs_nominalrate1..6`); DSGE.jl emits "all missing" warnings for these, which is
expected.

## Expected runtime and artifact size (reference Windows machine)

- Julia export (1.8, project already instantiated): ~3 minutes.
- Python candidate export + compare: ~1 minute.
- `m1002_ss10_release.h5`: ~110 MB (gitignored under `outputs/`).

## Validation result (2026-06-23)

All numerical hard-target surfaces pass strict `1e-10`:

| Profile | Surfaces | Result |
|---|---|---|
| `parameters` | 4 | ✅ pass |
| `steady-state` | 1 | ✅ pass |
| `matrix` | 16 | ✅ pass |
| `posterior` | 6 | ✅ pass (worst `posterior/log_likelihood` abs `7.2e-10` on a ~1.5e3 value → within strict `rtol`) |
| `forecast-mode` | 5 | ✅ pass |
| `forecast-mode-history` | 4 | ✅ pass |
| `forecast-full` | 5 | ✅ pass |
| `forecast-full-history` | 5 | ✅ pass |

Forecast arrays validated at production shape: `forecast_full/state_samples`
`(1000, 40, 84)`, `forecast_full/observable_samples` `(1000, 40, 19)`,
`forecast_full/history_observable_samples` `(1000, 235, 19)`.

### Known limitation: placeholder metadata transform/source surfaces

The combined `hard-target` profile additionally lists four **metadata string**
surfaces inherited from `model-metadata`:

- `metadata/observable_forward_transforms`
- `metadata/observable_reverse_transforms`
- `metadata/observable_sources`
- `metadata/pseudo_observable_reverse_transforms`

These are **not validatable against the current Julia oracle**. The DSGE.jl
exporter writes placeholder values (every transform = `"identity"`, every source
empty), whereas the Python candidate exports the real reporting-transform names
(e.g. `loggrowthtopctannualizedpercapita`) and FRED source mnemonics. The
observable/pseudo-observable **names and order** are exported truthfully on both
sides and do compare cleanly; only the transform/source string *content* has no
real oracle. The short checked-in smoke fixture hides this because it omits these
attributes on both sides.

Consequently the release validation is reported as the union of the eight
numerical profiles above (all pass). Closing the metadata gap would require the
Julia exporter to emit real reporting-transform metadata matching the Python
convention, tracked as future work and noted in `docs/port_matrix.csv`.
