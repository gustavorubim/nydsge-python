# Contributing to nydsge

Thanks for your interest in `nydsge`, a native Python port of the New York Fed
DSGE workflow ([FRBNY-DSGE/DSGE.jl](https://github.com/FRBNY-DSGE/DSGE.jl)). The
project's goal is **numerical parity with the upstream Julia implementation, with
no Julia runtime dependency**. Contributions are welcome as long as they preserve
that contract.

[`PORTING_PLAN.md`](PORTING_PLAN.md) is the authoritative spec for scope,
validation gates, and tolerance policy. Please read it before proposing changes
to model surfaces.

## Non-negotiable constraints

- **Runtime purity.** `src/nydsge/` must be pure Python. It must not call Julia,
  require WSL, shell out to migration tools, or read oracle fixtures to compute
  production results. Julia is allowed only under `tools/oracle_julia/` to
  generate parity evidence. `uv run nydsge vv runtime-purity` enforces this.
- **NumPy/SciPy float64 CPU is the reference path.** Torch and JAX backends are
  optional accelerators validated against the CPU reference; they never replace
  it as the source of truth.
- **Parity is proven, not asserted.** Every numerical surface needs a candidate
  export and a labeled oracle comparison. Do not loosen a tolerance profile to
  hide an unexplained mismatch, and do not let a comparison "pass" because an
  oracle surface is missing.

## Development setup

```powershell
uv sync
```

Run the full gate before opening a pull request. All four must pass:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check .
uv run pytest
```

## Coding standards

- Match the style, naming, and structure of the surrounding code. Where it makes
  parity easier to audit, preserve upstream DSGE.jl naming, matrix dimensions,
  labels, and data orientation; document any orientation differences explicitly.
- Keep functions typed and prefer small, testable units.
- Generated artifacts (figures, `.npz`/`.h5` arrays, run folders) belong under
  `outputs/` and are gitignored; never place production code there. Promote
  anything reusable into `scripts/` or `src/nydsge/`.

## Changing or adding a model surface

1. Translate the behavior from the upstream DSGE.jl source area referenced in
   `docs/port_matrix.csv`.
2. Add a Python candidate export and compare it against a Julia oracle fixture
   under the appropriate profile and the `strict` tolerance:

   ```powershell
   uv run nydsge vv export-suite --output-dir tests/fixtures/candidate --data path\to\observables.csv --horizon 40 --json
   uv run nydsge vv compare --oracle-dir tests/fixtures/oracle --candidate-dir tests/fixtures/candidate --profile <surface> --tolerance-profile strict --json
   ```

3. Add unit/CLI tests under `tests/` and update `docs/port_matrix.csv` status.
4. For the combined gate, confirm
   `uv run nydsge vv compare --profile hard-target --tolerance-profile strict`
   still passes.

## Tolerance policy

| Profile | Tolerance | Use |
|---|---|---|
| `strict` / `cpu-oracle` | `atol = rtol = 1e-10` | CPU oracle, matrices, hard target |
| `forecast` | `atol = rtol = 1e-8` | forecast / means-bands accumulation |
| `accelerator` | `atol = rtol = 1e-5` | Torch / JAX vs NumPy CPU |

Every comparison must name its tolerance profile and report shape, max absolute
difference, max relative difference, and labels on mismatch.

## Commits and pull requests

- Use clear, conventional commit subjects (`feat:`, `fix:`, `chore:`,
  `docs:`, `test:`, `build:`) and keep commits focused.
- Describe what changed and how it was validated (paste the relevant gate
  output). Note any new or regenerated fixtures and the recipe used.
- Keep generated binaries out of source control unless intentionally curated per
  the artifact policy in `PORTING_PLAN.md`.

## Licensing of contributions

`nydsge` is distributed under the BSD 3-Clause License (see [`LICENSE`](LICENSE)),
matching upstream DSGE.jl. By contributing, you agree that your contributions are
licensed under the same terms.
