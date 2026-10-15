# Julia-free daily-driver workflow

Use the NumPy/SciPy float64 CPU path as the reference. Julia is not required for
installation, testing, estimation or analysis. Optional oracle regeneration is a
separate developer task. This workflow does not change operational desk scripts
or select a production forecast specification for the user.

## Clean native environment

From a fresh checkout of the reviewed commit, with `uv` installed:

```powershell
git rev-parse HEAD
uv sync --locked --python 3.12 --extra plot
uv run --no-sync nydsge doctor --json
uv run --no-sync nydsge vv runtime-purity
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
uv run --no-sync ty check .
uv run --no-sync python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"
uv run --no-sync pytest --basetemp outputs/clean_pytest_tmp --junitxml outputs/clean_tests.xml
```

Use native Windows, Linux or macOS, not WSL. CPU tests need no credentials or
network data retrieval after dependency installation. Set `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` before Python starts to avoid
oversubscription; the analytics CLI does this for its own bounded child process.
Record Python, uv, NumPy/SciPy, OS and BLAS versions with the run. A clean clone
must use the same commit and lockfile; `--locked` must fail if they disagree.
Hosted CI runs the same gate on native Windows/Linux. Optional accelerator skips
are not accelerator certification.

## Offline, dated economic analysis

Follow the [analytics input contract](../general_analytics_tools/README.md) and
copy its example configuration into ignored `inputs/`. Supply real official
quarterly levels and an immutable source manifest, including a genuine quarter
before `presample_start` for transformations. Use actual source/retrieval URLs,
UTC times, vintage, revision policy, seasonal adjustment, frequency aggregation
and units. Missing values remain missing. A retrieval timestamp is not a release
publication date or a historical point-in-time vintage.

```powershell
uv run --no-sync python -m general_analytics_tools run --config inputs/my_analysis.json --output-dir outputs/my_unique_run --timeout-seconds 600
uv run --no-sync python -m general_analytics_tools plot --run-dir outputs/my_unique_run
```

The unique output directory must not already exist. These commands consume local
inputs, perform bounded CPU computation and archive hashes; they do not fetch a
new release or run estimation. Inspect status and validation before interpreting
figures. A failed/timeout run is incomplete. A diagnostic HD is not a validated
structural attribution; retain its residual and status.

Choose `history_end`, `conditioning_end`, and each conditioning channel explicitly.
A quarterly flow average needs all released constituent observations before being
declared complete. If only hours are complete, assimilate only `obs_hours`.
Expected-rate rows use explicit formation/publication dates and horizons 1â€“6;
horizon one is the next quarter. Default `point_in_time` history excludes later
publications. `analysis_vintage` is retrospective and must be labelled accordingly.
Annual-percent rate inputs divide by four into native quarterly percentage-point
units; annual fractions multiply by 25. No log conversion is applied to rates.

GDP/core-price growth plots annualize quarterly log differences by four; GDP here
is the model's per-capita measurement, not automatically headline aggregate GDP.
The nominal-rate plot multiplies its quarterly model rate by four. Figure units,
source vintage, origin, caption and conditioning must remain visible.

Held-rate paths are sequential surprises. Forward guidance is announced at its
specified quarter and can affect activity before its target quarter. A 25bp
quarterly rate constraint is not automatically a 25bp intraday policy surprise.
Record whether anticipated innovations have zero variance under the baseline
prior; an off-support hypothetical has no defensible standard-deviation label.

## Parameters and uncertainty

Keep public-default, optimized-mode and externally supplied parameter-draw runs
separate. Record parameter coordinate names/order, regime mapping, source, file
hash and estimation diagnostics. M7 is not a converged posterior. A finite mode,
a successful filter or a passing source-parity test does not make it one.

For native parameter-draw forecasts use the documented function signature and
explicit draw inputs; check each draw's validity and keep the original model
unchanged. Preserve the forecast's uncertainty metadata when calculating bands.
Do not describe arbitrary bounded test draws as posterior uncertainty. Fixed-mode
future-shock bands omit parameter uncertainty; parameter-draw bands omit any
uncertainty not represented in their inputs. There is no automatic revision or
model-misspecification band.

## Native ss104 estimation interface

The same dated native-observation frame used above can be passed to the generic
Python interface. The forecast-origin/sample settings must match its dates;
there is no inference of a data vintage from a filename.

```python
from nydsge.estimate import estimate

# model and data are the explicitly configured ss104 model and dated frame.
fixed = estimate(model, data)
candidate = estimate(
    model, data, optimize=True, parameter_names=["alpha"],
    optimizer_method="BFGS", maxiter=2, compute_hessian=True,
)
print(candidate.optimization.success)  # inspect; a bounded demo may not converge
```

The input model remains unchanged. This is a native local candidate search,
not proof of a validated posterior or of global optimization. Inspect finite
objective values, gradients, bounds and `optimization.success` before adopting
anything. The optional finite-difference Hessian is raw curvature of the negative
log posterior in optimization coordinates, **not** a posterior covariance; do not
silently invert an indefinite or poorly conditioned matrix.

The CLI preserves ss104 CSV quarter dates:

```powershell
uv run --no-sync nydsge estimate --subspec ss104 --data inputs/observations.csv
uv run --no-sync nydsge estimate --subspec ss104 --data inputs/observations.csv --optimize --parameters alpha --maxiter 2 --hessian
```

Its generic ss104 default forecast origin is `2026-Q3`; use the Python API with
explicit settings for another origin rather than silently reinterpreting dates.
The CLI example's optimizer method is its default. Generic ss104 Metropolis–
Hastings remains explicitly unsupported; an interface for supplied draws does
not certify their producing sampler or convergence.

## Native ss104 bands from explicit inputs

This API example assumes `inputs/observations.csv` already contains validated
native transformed observations, ending before `date_forecast_start`. It does
not download data or infer a vintage. Keep its source manifest alongside it.
The seed below controls future innovations, not parameter estimation.

```python
import pandas as pd
from nydsge.models import Model1002
from nydsge.forecast import compute_meansbands

model = Model1002("ss104", settings={"date_forecast_start": "2026-Q3"})
data = pd.read_csv("inputs/observations.csv")
data = data.set_index(pd.PeriodIndex(data.pop("date"), freq="Q"))
bands = compute_meansbands(
    model, input_type="full", cond_type="none", output_vars=["forecastobs"],
    data=data, horizon=8, draws=32, seed=20261015, draw_future_shocks=True,
)
print(bands.uncertainty_sources)  # retain this with the exported bands
```

This is a **fixed-public-default, future-innovation** example; 32 draws are a
bounded demonstration, not stable tail-coverage evidence. To include externally
supplied parameter variation, pass `parameter_draws` (rows are draws, columns
are model-space values), `parameter_names` (the exact ordered coordinate labels),
and a nonempty `parameter_draw_provenance`. Record the producing estimate,
convergence diagnostics and input hash. Keep `draw_future_shocks=False` for
parameter variation alone, or explicitly enable both sources. Invalid draws must
fail rather than be silently dropped. Do not pass an ss10 sampler as an ss104
posterior. Use explicit conditional histories and `cond_type` when needed;
a future path is not a conditioning observation unless actually supplied.

## Acceptance before using a result

Retain configuration, source snapshots/manifests, transforms, code commit,
lockfile/runtime, parameters, dates, seeds and output hashes. Verify observation
availability, target realization, timing, array finiteness and reconstruction in
plotted units. Do not publish an HD as validated unless its strict `1e-8` gate
passes; keep initial/deterministic/remainder terms inspectable. Test a frozen
rerun before updating a baseline, and preserve the previous archive.

For a macro read, use one consolidated identification-caveat section: demand,
markup and monetary residuals are conditional model interpretations, not causal
identification by labels. State decision-relevant risks separately. Representative
agent linearization, ragged-edge coverage, input revisions and missing credit or
distribution channels can matter. None of these outputs is a Fed staff forecast.

## Outstanding numerical investigation

The HD investigation identifies sensitivity in near-singular innovation matrices
as a source of accumulated reconstruction error. This diagnosis is not a blanket
fix or cross-machine parity certification. Keep failing archives diagnostic;
retain original plotted-unit and oracle gates without relaxation.

The `future_model_innovations` source draws from dated model QQ. This includes
measurement-state innovations where QQ assigns them positive variance, alongside
economic shocks. It does not draw additional observation noise from EE or the
conditional terminal-state covariance. Parameter rows re-estimate conditional
state means at each supplied vector; they do not sample latent-state uncertainty.

## Portable committed-snapshot smoke

A clean checkout can reproduce a bounded native smoke without any prior output
archive or Julia installation:

```powershell
uv run python scripts/smoke_daily_driver.py --output-dir outputs/my_unique_smoke --draws 32 --horizon 8
```

This uses only the committed October public snapshots, Q2-frozen transforms and
one semi-conditioned Q3 quarter. The manifest records exact source hashes, the
conservative October 12 survey availability, native quarterly units, seed and
uncertainty sources. It tests zero-shock equality, deterministic seeded reruns,
explicit two-point parameter sensitivity and fixed-point posterior evaluation.
The sensitivity grid is not a posterior. It does not retrieve new data, claim
publication-time accuracy or run estimation. Output directories must be new.
