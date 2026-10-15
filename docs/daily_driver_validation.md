# Julia-free daily-driver improvements: validation, 2026-10-15

Base: main `f668db4aae6c8521b5cb549efefc5d0854eaee76`.
Scope: native Python interfaces, numerical diagnostics, CPU CI and precise
reproducibility guidance. No operational desk baseline, credentials, private
inputs or estimated parameter files were changed. No live estimation or Julia
runtime was used. This is a staged improvement, not a claim that every audited
gap has been closed.

## Implemented and exercised

- Native ss104 full ensembles and means/bands accept explicitly named model-space
  parameter rows, with parameter-regime coordinates and mandatory provenance.
  Each row solves and filters independently; invalid rows fail the whole call.
  Fixed-parameter ensembles solve/filter once. Both paths use native regime
  forecasts, with future innovations starting after conditional quarters.
- Dated QQ simulation supports zero/singular covariance and seeded reruns.
  `future_model_innovations` includes economic and measurement-state innovations
  carried by QQ; it excludes extra EE observation noise and initial/terminal-state
  draws. Supplied parameters are not certified as a posterior. Metadata survives
  means/bands and reverse transforms.
- Generic `estimate()` and CLI preserve dated ss104 history, support fixed-point
  posterior evaluation and selected-coordinate native optimization/Hessian work,
  retain optimizer failure status and isolate the caller model. Reusing an
  explicitly unsuccessful mode is rejected. ss104 MH remains explicitly unsupported.
  A numerical Hessian is not automatically a validated posterior covariance.
- Native multi-regime attribution now verifies effective covariance/initialization
  consistency and exports state/observation remainders, conditioning diagnostics
  and worst quarters. Independent joint-Gaussian boundary tests cover singular
  QQ, changing regimes, missing observations, intercepts and custom initial priors.
- Python-only hosted CPU workflow runs locked dependencies, complete tests and
  committed fixtures on Windows/Linux and Python 3.12/3.14. Permissions are read
  only; actions are pinned; no secrets, publishing, deployment or Julia execution.
  A bounded public-snapshot smoke exercises normal computation without oracle input.
- The feature matrix replaces blanket parity claims with original per-surface
  thresholds, source pins, metadata-validation limits and open numerical exceptions.

## Executed local evidence

- **634 passed, 4 skipped**, full suite in **93.01 s**. Skips are optional backends.
- Full formatting, Ruff, ty and runtime-purity checks passed; purity checks 37
  runtime Python files. Final source formatting covers 104 Python files.
- The unchanged default and three perturbed ss104 Julia-oracle fixtures pass
  their existing gates. No tolerances, fixtures or economic calibration changed.
- New distribution tests pass zero shocks, conditional-window/nonzero path
  equivalence, singular QQ, seeds, invalid bounds/metadata and model isolation.
- Estimation tests include a genuine dated native likelihood replay and bounded
  synthetic optimization/status checks; they do not fit a new economic posterior.
- Actual committed public-input smoke: eight future quarters, 32 QQ draws,
  zero-shock and seeded rerun equality array-exact, explicit two-point alpha
  sensitivity (not posterior draws), finite fixed-default posterior evaluation.
  Source CSV hashes and conservative survey availability are recorded in manifests.
- The full toolkit frozen-input pipeline completed and rendered six inspected
  figure types. A new three-panel native future-innovation bands figure was
  visually inspected; small-draw tails are labelled illustrative.

The smoke uses October 6 public snapshots, Q2-frozen population transforms and
one semi-conditioned Q3 quarter; conservative survey availability is October 12.
It is retrospective, not a real-time July/September information set or new release
capture. The GDP panel is per capita annualized log growth, core PCE annualized
log inflation and effective FFR annual percent. No empirical accuracy claim follows.

## Open numerical and scientific gates

Full real-input HD remains **diagnostic** under the unchanged `1e-8` plotted-unit
gate: core PCE about `2.70e-8`, GDP `2.86e-6`, FFR `3.04e-6`. The covariance condition
number reaches `5.95e6`; state reconstruction discrepancy reaches `1.56e-5`.
Joseph updates and arithmetic reordering did not establish a valid fix. Forcing
unsupported discrepancies into shocks would change the inference and was rejected.
See [the full diagnosis](hd_numerics.md). A separately validated stable
filter/smoother investigation remains necessary before certifying this sample's HD.

Fresh comparison with the existing independent box oracle retains conditional
observable error `1.53848e-7 > 1e-7` and smoother errors above `5e-6`. Original
thresholds remain unchanged. The new stochastic interfaces are native-invariant
tested, not newly independently oracle-certified Julia distribution workflows.
M7 remains unconverged; historical input gaps and verified publication dates,
fully independent raw-data measurement validation and real-time out-of-sample
forecast evaluation remain separate open work.

## Rerun

```powershell
uv sync --locked --extra plot
uv run ruff format --check .
uv run ruff check .
uv run ty check .
uv run nydsge vv runtime-purity
uv run pytest --basetemp outputs/my_unique_tests
uv run python scripts/smoke_daily_driver.py --output-dir outputs/my_unique_smoke --draws 32 --horizon 8
```

Always choose new output directories. The portable script requires only a clean
checkout and committed public inputs, records the exact code SHA/status and runtime
versions, and neither retrieves fresh data nor uses oracle results for computation.
Hosted check results must be read from the PR for its exact head; the workflow
definition alone does not establish a passing hosted run.

Daily-driver verdict: materially better for reproducible native conditional
forecasts and explicitly sourced uncertainty experiments. Strict real-sample HD,
cross-machine numerical robustness and posterior/empirical forecast readiness are
not certified by this PR. The PR is for review and must not be auto-merged.
