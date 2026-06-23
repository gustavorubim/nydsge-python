# Parity Goals: Same Answer From Python or Julia

This document is the **goal contract** for the port. Success means a user can
run the production `Model1002("ss10")` workflow in **either** native Python
(`nydsge`) **or** upstream Julia (`DSGE.jl`) with the locked settings below and
arrive at the **same numerical answers**, with reproducible evidence for every
claim.

Julia remains the offline oracle generator during migration. The Python runtime
must not call Julia. Parity is proven by exported fixtures and labeled compare
reports, not by visual inspection.

Related docs:

- `PORTING_PLAN.md` — full migration inventory and architecture
- `docs/release_fixture.md` — release-grade fixture recipe (Level 6)
- `docs/port_matrix.csv` — per-area status tracking

---

## North-Star Goal

> **Given the locked hard-target configuration, every required numerical surface
> produced by `nydsge` matches the Julia oracle within strict CPU float64
> tolerance, and every required user-facing command yields equivalent labeled
> outputs.**

### Locked configuration (the answer we must match)

| Setting | Value |
|---|---|
| Model | `Model1002` |
| Subspec | `ss10` (compatible `ss104` routed through `ss10` pipeline) |
| Data vintage | `181115` |
| Forecast start | `2018-Q4` |
| In-sample end | `2018-Q3` (235 quarters) |
| Forecast horizon | `40` quarters |
| Full-forecast draws | `1000` |
| Reference dtype | float64 CPU (NumPy/SciPy) |
| Tolerance | `strict`: `atol = rtol = 1e-10` |

### What “same answer” includes

All labeled float64 arrays and scalars on these surfaces:

1. Parameters, steady state, financial-frictions helpers
2. Canonical and state-space matrices (`Gamma*`, `TTT/RRR/CCC`, `ZZ/DD/QQ/EE`, pseudo-measurement)
3. Observable and pseudo-observable **metadata** (names, order, transforms, sources)
4. Transformed `181115` observable data
5. Kalman filter: likelihood, per-period components, predicted/filtered states and covariances
6. Current-parameter posterior decomposition
7. Mode forecast and mode `histobs`
8. Full forecast and full `histobs` sample arrays
9. Deterministic means/bands from those forecasts
10. Sampler replay surfaces where deterministic replay is possible
11. Forecast analysis numeric outputs (IRFs, historical decomposition arrays)

### What is explicitly out of scope for this goal

- Other DSGE.jl models
- Model1002 subspecs beyond `ss10` / compatible `ss104`
- Every optional branch combination (flexible-AIT permutations, etc.) unless a reward gate is added
- Accelerator parity (CUDA/MPS/JAX) — separate tolerance profile
- Performance benchmarks (documented separately in `docs/benchmark_references.md`)

---

## How To Use This Document

Work through rewards **in order**. Each reward has:

- **Commands** — copy-paste runnable checks
- **Pass** — objective condition (exit code + JSON field or test count)
- **Artifact** — where evidence lives
- **Status** — snapshot as of 2026-06-23 (update when a gate closes)

A reward is **closed** only when its pass condition is met on the **release**
fixture (`outputs/release/`) or a checked-in smoke fixture where noted.

**Stop rule:** Never mark a reward passed if a required oracle surface is missing
and the compare silently skipped it. Run `vv oracle-coverage` before trusting
`vv compare`.

---

## Phase 0 — Trust The Tooling

### Reward 0: Repo hygiene

**Commands:**

```powershell
uv run ruff format --check .
uv run ruff check src tests scripts tools docs README.md pyproject.toml
uv run ty check src tests scripts tools
uv run pytest -ra
```

**Pass:** All four commands exit 0; pytest reports 0 failures.

**Artifact:** CI log or local terminal output.

**Status:** ✅ **PASS** (468 passed, 4 skipped — Torch/JAX optional)

---

### Reward 1: Python runtime is Julia-free

**Command:**

```powershell
uv run nydsge vv runtime-purity --json
```

**Pass:** `"passed": true`, `"findings": []`.

**Artifact:** JSON stdout.

**Status:** ✅ **PASS**

---

### Reward 2: Compare gate cannot silently skip missing surfaces

**Requirement:** When `--profile <name>` is passed to `vv compare`, the command
must fail if any array required by that profile is missing from the **oracle** or
**candidate** (status `missing_oracle` or `missing_candidate`), not only when
overlapping arrays mismatch.

**Command (after implementation):**

```powershell
uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile hard-target --tolerance-profile strict --json
```

**Pass:** Exit code 1 with explicit `missing_oracle` rows for absent surfaces,
OR exit code 0 only when `vv oracle-coverage --profile hard-target` also passes
(zero missing on oracle side).

**Artifact:** Compare JSON + oracle-coverage JSON.

**Status:** ❌ **OPEN** — compare currently passes with partial oracle (27/52
surfaces on smoke fixture).

---

## Phase 1 — Model Is The Same Object

### Reward 3: Parameter and steady-state parity

**Commands:**

```powershell
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile parameters --tolerance-profile strict --json
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile steady-state --tolerance-profile strict --json
```

**Pass:** Both return `"passed": true`.

**Artifact:** `outputs/release/compare.log` or JSON stdout.

**Status:** ✅ **PASS** (release fixture, 2026-06-23)

---

### Reward 4: Matrix parity (equilibrium + transition + measurement)

**Command:**

```powershell
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile matrix --tolerance-profile strict --json
```

**Pass:** `"passed": true` for all 16 matrix arrays; max abs diff ≤ `1e-10`.

**Artifact:** Compare JSON with labeled mismatch coordinates (empty on pass).

**Status:** ✅ **PASS** (release fixture)

---

### Reward 5: Observable metadata parity (names, order, transforms, sources)

**Command:**

```powershell
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile model-metadata --tolerance-profile strict --json
```

**Pass:** `"passed": true` on all seven metadata arrays:

- `metadata/observable_names`
- `metadata/observable_sources`
- `metadata/observable_forward_transforms`
- `metadata/observable_reverse_transforms`
- `metadata/pseudo_observable_names`
- `metadata/pseudo_observable_forward_transforms`
- `metadata/pseudo_observable_reverse_transforms`

**Work required:** Update `tools/oracle_julia/export_model1002.jl` to emit real
reporting-transform and FRED source strings (not `"identity"` / empty
placeholders). Re-export release oracle, re-run candidate export, re-compare.

**Artifact:** Updated oracle HDF5 + compare JSON.

**Status:** ❌ **OPEN** — names/order pass; transform/source strings fail because
Julia oracle exports placeholders (documented in `docs/release_fixture.md`).

---

## Phase 2 — Filtering And Estimation Match

### Reward 6: Kalman filter parity at full history length

**Prerequisite:** `vv export-suite` must write `kalman.npz` when `--data` is
provided (today only `posterior.npz` is written; Kalman histories are computed
but not exported).

**Commands:**

```powershell
# After export-suite exports kalman:
uv run nydsge vv export-suite --output-dir outputs/release/candidate --data outputs/release/observables_release.csv --shock-samples outputs/release/oracle/m1002_ss10_release.h5 --allow-empty-data-columns --horizon 40 --json

uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile kalman --tolerance-profile strict --json
```

**Pass:** `"passed": true` on all seven Kalman arrays at shape `(235, …)`:

- `kalman/log_likelihood`
- `kalman/total_log_likelihood`
- `kalman/predicted_states`, `kalman/filtered_states`
- `kalman/predicted_covariances`, `kalman/filtered_covariances`
- `kalman/final_filtered_state`

**Artifact:** `outputs/release/candidate/kalman.npz` + compare JSON.

**Status:** ❌ **OPEN** — Julia oracle has Kalman arrays; Python candidate does
not export them yet. Posterior scalars already pass (Reward 7).

---

### Reward 7: Current-parameter posterior parity

**Command:**

```powershell
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile posterior --tolerance-profile strict --json
```

**Pass:** `"passed": true` on all six posterior arrays.

**Artifact:** Compare JSON.

**Status:** ✅ **PASS** (release fixture; worst `log_likelihood` abs diff `7.2e-10`)

---

### Reward 8: Sampler deterministic replay smoke

**Commands:**

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out tests/fixtures/oracle/m1002_ss10_sampler_smoke.h5 --include-sampler true

uv run nydsge vv sampler-posterior-replay --oracle tests/fixtures/oracle/m1002_ss10_sampler_smoke.h5 --data tests/fixtures/oracle/sampler_observables.csv --tolerance-profile strict --json
```

**Pass:** Exit 0; proposal/previous vectors replay to matching log-posterior
components within strict tolerance.

**Artifact:** Replay JSON report.

**Status:** ✅ **PASS** (smoke fixture; checked-in tests cover this)

---

### Reward 9: Production-scale sampler parity

**Goal:** Match Julia over a longer MH chain, not just a 2-draw smoke replay.

**Commands (target — adjust draw count if artifacts are too large):**

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out outputs/release/oracle/m1002_ss10_sampler.h5 --include-sampler true --data-out outputs/release/observables_release.csv

uv run nydsge estimate --data outputs/release/observables_release.csv --mode-input <julia-mode-or-saved> --mh-draws 1000 --mh-burnin 100 --sampler-output outputs/release/candidate/sampler.npz

uv run nydsge vv sampler-compare --oracle outputs/release/oracle/m1002_ss10_sampler.h5 --candidate outputs/release/candidate/sampler.npz --tolerance-profile strict --json
```

**Pass:**

- `vv sampler-compare` passes on metadata (acceptance, proposal covariance health)
- `vv sampler-posterior-replay` passes on exported proposal traces
- Per-parameter effective sample sizes are finite and documented

**Artifact:** Sampler NPZ + compare JSON + diagnostics JSON.

**Status:** ❌ **OPEN** — smoke only; production chain not validated.

---

### Reward 10: Mode optimization and Hessian parity

**Command pattern:**

```powershell
uv run nydsge estimate --data outputs/release/observables_release.csv --optimize --parameters <subset> --maxiter 50 --hessian --mode-output outputs/release/candidate/mode.npz
```

**Pass:**

- Mode archive schema validates
- Optimized parameters match Julia mode within strict tolerance **or** documented
  equivalent convergence (same log-posterior at optimum within `1e-8`)
- Hessian shape and diagonal signs match Julia where exported

**Artifact:** `mode.npz`, Hessian archive, comparison notes.

**Status:** ❌ **OPEN** — optimizer runs; Julia mode/Hessian parity not release-tested.

---

## Phase 3 — Forecasts And Histories Match

### Reward 11: Mode forecast parity (40-quarter horizon)

**Command:**

```powershell
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile forecast-mode --tolerance-profile strict --json
```

**Pass:** `"passed": true`; `forecast_mode/observables` shape `(40, 19)`.

**Status:** ✅ **PASS** (release fixture)

---

### Reward 12: Mode history (`histobs`) parity

**Command:**

```powershell
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile forecast-mode-history --tolerance-profile strict --json
```

**Pass:** `"passed": true`; `forecast_mode/history_observables` shape `(235, 19)`.

**Status:** ✅ **PASS** (release fixture)

---

### Reward 13: Full forecast sample parity (1000 draws × 40 quarters)

**Command:**

```powershell
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile forecast-full --tolerance-profile strict --json
```

**Pass:** `"passed": true`; arrays at production shape:

- `forecast_full/state_samples` → `(1000, 40, 84)`
- `forecast_full/observable_samples` → `(1000, 40, 19)`

**Note:** Release recipe uses **shared deterministic zero-shock draws** between
oracle and candidate. This validates the full-forecast **machinery** at production
dimensions.

**Status:** ✅ **PASS** (release fixture, zero-shock shared samples)

---

### Reward 14: Full history sample parity

**Command:**

```powershell
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile forecast-full-history --tolerance-profile strict --json
```

**Pass:** `"passed": true`; `forecast_full/history_observable_samples` →
`(1000, 235, 19)`.

**Status:** ✅ **PASS** (release fixture)

---

### Reward 15: Means/bands parity

**Command:**

```powershell
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile hard-target --tolerance-profile strict --json
```

**Pass:** All `meansbands_*` arrays in the hard-target profile pass strict
tolerance (mode and full, forecast and history, mean/lower/upper).

**Status:** ✅ **PASS** (numerical means/bands on release fixture)

---

### Reward 16: Stochastic full-forecast parity (optional stretch)

**Goal:** Same forecast distribution when draws are **not** pre-shared.

**Commands:**

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out outputs/release/oracle/m1002_ss10_stochastic.h5 --include-full-forecast true --full-draws 1000 --seed 123 --horizon 40 --data-out outputs/release/observables_release.csv

uv run nydsge vv export-suite --output-dir outputs/release/candidate_stochastic --data outputs/release/observables_release.csv --full-draws 1000 --seed 123 --horizon 40 --allow-empty-data-columns --json

uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate_stochastic --profile forecast-full --tolerance-profile strict --json
```

**Pass:** Draw-by-draw arrays match with identical seed, OR means/quantile bands
match within `forecast` tolerance profile (`1e-8`) with documented seed alignment.

**Status:** ❌ **OPEN** — not validated; release uses shared zero-shock samples.

---

## Phase 4 — Data Pipeline: Same Inputs, Same Outputs

### Reward 17: Transformed observable CSV matches Julia

**Commands:**

```powershell
julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out outputs/release/oracle/m1002_ss10_release.h5 --data-out outputs/release/observables_release.csv --horizon 40

uv run nydsge data build --input outputs/release/observables_release.csv --output outputs/release/python_replay.csv --model m1002 --subspec ss10
```

**Pass:** Python reload of Julia-exported CSV reproduces the same in-sample matrix
used by Kalman/forecast (validated via Reward 7/11–14 passing on that CSV).

**Status:** ✅ **PASS** (indirectly — release replay uses Julia CSV and all
downstream surfaces pass)

---

### Reward 18: Raw sources → observables → model outputs (no Julia CSV shortcut)

**Goal:** Build observables from raw source files on disk, then match Julia
end-to-end.

**Commands:**

```powershell
uv run nydsge data prepare-sources --source-root <raw-bundle> --vintage 181115 --start-date 1959-Q3 --end-date 2018-Q3 --json
uv run nydsge data build-sources --source-root <raw-bundle> --output outputs/e2e/observables_181115.csv --start-date 1959-Q3 --end-date 2018-Q3

julia +1.8 --project=tools/oracle_julia tools/oracle_julia/export_model1002.jl --out outputs/e2e/oracle.h5 --include-kalman true --include-posterior true --include-forecast true --include-full-forecast true --full-draws 1000 --data-out outputs/e2e/observables_julia.csv --horizon 40

uv run nydsge vv export-suite --output-dir outputs/e2e/candidate --data outputs/e2e/observables_181115.csv --shock-samples outputs/e2e/oracle.h5 --allow-empty-data-columns --horizon 40 --json

uv run nydsge vv compare --oracle-dir outputs/e2e/oracle --candidate-dir outputs/e2e/candidate --profile hard-target --tolerance-profile strict --json
```

**Pass:** Python-built CSV produces candidate arrays matching Julia oracle built
from its own CSV, **or** the two CSVs match column-wise within `1e-10` and
Rewards 6–15 pass using Python-built CSV.

**Artifact:** Documented raw bundle path + e2e compare log.

**Status:** ❌ **OPEN** — not demonstrated at release scale.

---

## Phase 5 — User-Facing Equivalence

### Reward 19: CLI forecast path matches fixture export

**Commands:**

```powershell
uv run nydsge forecast --data outputs/release/observables_release.csv --horizon 40 --include-history --json
uv run nydsge meansbands --input-type mode --draws 1000 --seed 123 --horizon 40 --data outputs/release/observables_release.csv --json
```

**Pass:** Outputs are finite, labeled, and consistent with `vv export-suite`
arrays for the same settings (spot-check via `test_cli` or manual NPZ diff).

**Status:** ✅ **PASS** (covered by tests + release fixture consistency)

---

### Reward 20: Report numeric outputs are well-defined

**Commands:**

```powershell
uv run nydsge report irf --horizon 40 --no-plots --output-dir outputs/report/irf --json
uv run nydsge report historical-decomposition --data outputs/release/observables_release.csv --no-plots --output-dir outputs/report/hd --json
```

**Pass:**

- Exit 0; NPZ arrays exported with axis labels
- Historical decomposition reconciles: contributions + baseline = smoothed path
  within documented tolerance (test: `test_historical_decomposition_reconciles_to_smoothed_path`)

**Artifact:** `outputs/report/**` NPZ + manifest JSON.

**Status:** ✅ **PASS** (unit tests + CLI smoke)

---

## Phase 6 — The Single Combined Gate

### Reward 21: Full hard-target gate (release)

**Commands:**

```powershell
# 1. Coverage pre-check (oracle must be complete)
uv run nydsge vv oracle-coverage --oracle-dir outputs/release/oracle --profile hard-target --json

# 2. Candidate export (must include kalman once Reward 6 is done)
uv run nydsge vv export-suite --output-dir outputs/release/candidate --data outputs/release/observables_release.csv --shock-samples outputs/release/oracle/m1002_ss10_release.h5 --allow-empty-data-columns --horizon 40 --json

# 3. Strict compare
uv run nydsge vv compare --oracle-dir outputs/release/oracle --candidate-dir outputs/release/candidate --profile hard-target --tolerance-profile strict --json
```

**Pass:**

- `oracle-coverage`: `"passed": true`, `"missing": []`
- `compare`: `"passed": true` on **every** required hard-target array (52 surfaces
  today, including metadata strings once Reward 5 closes)
- No `missing_oracle`, `missing_candidate`, or `shape_mismatch` rows

**Artifact:** `outputs/release/compare.log` committed to evidence folder or
documented regeneration in `docs/release_fixture.md`.

**Status:** ⚠️ **PARTIAL** — numerical surfaces pass individually; combined gate
fails on metadata strings (Reward 5) and Kalman is outside hard-target profile
but required by plan §3 (Reward 6).

---

### Reward 22: Julia-free reproduction

**Goal:** A machine with only Python can reproduce the answer after one-time
fixture generation.

**Commands (on a clean machine without Julia):**

```powershell
uv run nydsge solve --json
uv run nydsge forecast --data <cached-observables.csv> --horizon 40 --include-history --json
uv run nydsge estimate --data <cached-observables.csv> --json
uv run pytest -ra
```

**Pass:** All commands succeed; spot-check NPZ arrays against checked-in reference
hashes or a small curated smoke fixture set.

**Work required:** Curate a **small** checked-in smoke fixture (or LFS bundle)
so CI does not require Julia. Large release fixtures stay gitignored under
`outputs/`.

**Status:** ❌ **OPEN** — oracle binaries are gitignored; fresh clone needs Julia
to regenerate.

---

## Definition Of Done

**The parity goal is accomplished** when Rewards **0–1**, **3–4**, **5–7**,
**11–15**, **17**, **19–21** are all ✅ **PASS** on the release fixture, and
Rewards **8–10**, **16**, **18**, **22** are either ✅ or explicitly deferred
with written rationale in this file.

Minimum bar for your stated goal (“run either repo, same answer”):

| Must pass | Reward |
|---|---|
| Same model object | 3, 4, 5 |
| Same filtered data likelihood | 6, 7 |
| Same forecasts & histories | 11, 12, 13, 14, 15 |
| Same data in | 17 or 18 |
| No silent proof gaps | 2, 21 |
| No Julia at runtime | 1, 22 |

Sampler production parity (9–10) is required for **estimation equivalence**, not
for **forecasting at fixed parameters** (which is the Fed’s published forecast
use case).

---

## Suggested Work Order

1. **Reward 6** — export Kalman in `export-suite` (~small code change, high value)
2. **Reward 5** — fix Julia metadata exporter strings
3. **Reward 2** — harden `vv compare` missing-surface detection
4. **Reward 21** — re-run combined gate; update `docs/release_fixture.md`
5. **Reward 18** — raw-source end-to-end
6. **Rewards 9–10** — sampler/mode if estimation parity is in scope
7. **Reward 22** — curated smoke fixtures for Julia-free CI

---

## Status Dashboard

| Reward | Name | Status |
|--------|------|--------|
| 0 | Repo hygiene | ✅ |
| 1 | Runtime purity | ✅ |
| 2 | No silent compare skips | ❌ |
| 3 | Parameters + steady state | ✅ |
| 4 | Matrices | ✅ |
| 5 | Metadata strings | ❌ |
| 6 | Kalman histories | ❌ |
| 7 | Posterior | ✅ |
| 8 | Sampler smoke replay | ✅ |
| 9 | Sampler production | ❌ |
| 10 | Mode/Hessian | ❌ |
| 11 | Mode forecast | ✅ |
| 12 | Mode history | ✅ |
| 13 | Full forecast samples | ✅ |
| 14 | Full history samples | ✅ |
| 15 | Means/bands | ✅ |
| 16 | Stochastic forecast | ❌ |
| 17 | Julia CSV replay | ✅ |
| 18 | Raw sources e2e | ❌ |
| 19 | CLI forecast path | ✅ |
| 20 | Report numeric outputs | ✅ |
| 21 | Combined hard-target | ⚠️ partial |
| 22 | Julia-free CI | ❌ |

**Score: 13 / 22 closed; 1 partial; 8 open**

Update the dashboard when a reward closes.