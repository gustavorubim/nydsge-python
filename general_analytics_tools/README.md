# General analytics tools

Reusable **offline, fixed-default ss104** baseline forecasts, isolated monetary
surprises, and diagnostic historical decomposition, with the same six-panel
plotting style used in the ad hoc Econer analysis. No dates or machine paths are
baked into the computation. No data download, estimation, sampling, GPU, Julia
runtime or external publication is performed.

## Run and plot separately

From the repository root, install the locked CPU/plot environment and make a
local config based on [example_config.json](example_config.json):

```powershell
uv sync --locked --extra plot
uv run python -m general_analytics_tools run --config inputs/my_analysis.json --output-dir outputs/my_analysis --timeout-seconds 600
uv run python -m general_analytics_tools plot --run-dir outputs/my_analysis
```

The output directory **must not exist**. Omit `--output-dir` on `run` for a unique
UTC timestamp/UUID under `outputs/general_analytics_tools/`. Rendering does not
solve or filter a model. It verifies archive hashes, then creates a new
`figures/` directory; use `plot --output-dir outputs/my_other_figures` to restyle
an archive without overwriting earlier figures. Keep inputs, archives and large
generated files in ignored `inputs/` or `outputs/`, not in this source folder.

The CLI runs its own compute subprocess with a hard timeout (default 600 seconds,
maximum 1800). Only that subprocess is killed on timeout; it starts no child
workers. CPU math uses one OMP/OpenBLAS/MKL thread. History is capped at 500
quarters, future forecasts at 40 quarters, conditioning at four quarters and
scenarios at eight. It never controls other jobs. A timeout/failure leaves a
labelled incomplete archive that the plot command rejects.

## Inputs and measurement

Paths in a config resolve relative to **that config file**, not a user's home.
`inputs.levels_csv` is a quarterly, unique, ordered contiguous level-data CSV
with a `date` column and native source mnemonics such as `GDP`, `GDPDEF`,
`CNP16OV`, `AWHNONAG`, `CE16OV`, `PCEPILFE`, `COMPNFB`, `PCE`, `FPI`, `DFF`,
`GDI`, `GS10` and supported spread inputs. Unavailable measurements stay NaN.
Use existing `nydsge` data functions to create these inputs; **do not fabricate
missing values or add CPI as a substitute column**. This toolkit calls
`current_public_observables` and native `prepare_ss104_observations`, retaining
ss104's documented historical ZLB masks and model gap initialization.

`inputs.source_manifest` is a JSON object, for example this **schema illustration,
not a real data vintage**:

```json
{
  "vintage": "REPLACE_WITH_ACTUAL_VINTAGE",
  "retrieved_at_utc": "REPLACE_WITH_ACTUAL_UTC_RETRIEVAL",
  "source_urls": ["https://fred.stlouisfed.org/series/PCEPILFE"],
  "measurement_notes": "Record SA/NSA, units, source frequency, quarterly aggregation, revisions and complete input coverage here."
}
```

Use genuine official source URLs and actual retrieval/as-of times. Retrieval
after a brief's cutoff is **not** a cutoff-vintage capture. Preserve immutable
source snapshots and their raw manifests alongside the input levels. The toolkit
records hashes, but cannot prove a producer's vintage/aggregation declaration.

Only explicitly listed `conditioning_observables` enter quarters after
`history_end`. Every channel must also be declared in
`complete_conditioning_observables` and finite in each conditional quarter.
The producer must verify all required monthly/daily observations are released
and the quarter is complete; three-month completeness cannot be inferred from
an already averaged quarterly level. Use no channels and equal history/conditioning
end dates for a purely historical-origin forecast. Native ss104 expected-rate
rows remain missing unless an explicit `expectations_csv` and metadata are supplied.

For the original hours-only extension, list **only `obs_hours`**. Its native
measurement uses separate quarterly means of SA production/nonsupervisory
weekly hours and SA household employment, divided by the HP1600 trend of NSA
civilian population:

`100 log[3 mean(AWHNONAG) mean(CE16OV) / (100 HP1600(mean(CNP16OV)))]`.

Do not substitute payroll jobs, unemployment, all-worker weekly hours or hourly
earnings. Real wage growth uses `COMPNFB / GDPDEF`; CPI-deflated earnings are not
the same measurement. CPI is neither core PCE nor the GDP deflator. New CPI news
may motivate policy scenarios, but does not update these measurements.

Population HP filtering can revise transformed historical measurements when a
new quarter is appended. The toolkit separately runs the original history,
the extended-population/all-missing conditional quarter, and the observed update.
`endpoint_and_observation_news.csv` separates those sequential contributions;
neither contribution is an identified structural shock. It does not overwrite
the original baseline. Historical revisions in other input levels must be
documented by the input producer.

## Dated expected-rate inputs and reproducible sample boundaries

Supply **one genuine quarterly input before `presample_start`**. It participates
in differences and the population HP window, then is removed from transformed
observations before filtering, smoothing and likelihood scoring. A missing lead
quarter is an error. This corrects the old boundary convention and can change
forecasts; do not compare runs with different transform windows as survey effects.
Optional `inputs.public_sources_csv` uses the native public-source merge, filling
missing FRED columns without replacing finite observations.

To assimilate expectations, add `inputs.expectations_csv` and `rate_expectations`:

```json
{
  "kind": "survey",
  "source_url": "https://www.newyorkfed.org/markets/market-intelligence/survey-of-market-expectations",
  "vintage": "REPLACE_WITH_ACTUAL_SNAPSHOT",
  "as_of": "REPLACE_WITH_TIMEZONE_AWARE_UTC_TIMESTAMP",
  "units": "annual_percent",
  "rate_concept": "REPLACE_WITH_ACTUAL_POINT_RATE_CONCEPT",
  "mapping_notes": "State why this concept proxies native expected effective FFR",
  "historical_vintage_policy": "point_in_time"
}
```

CSV columns are `survey_date,available_date,target_quarter,rate`. OIS inputs use
`kind: "ois"` and `observation_date` instead of `survey_date`. Units are explicitly
`annual_percent` (divide by four), `annual_fraction` (multiply by 25), or
`quarterly_model_rate` (unchanged). No percent/log-rate conversion is applied.
The native noisy measurement rows are `obs_exp_nominalrate1` through `6`;
horizon one means the **next quarter**, never the same quarter. Only the latest
eligible formation date within each observation quarter is used. Missing targets
stay missing; no interpolation, extrapolation or later-quarter backfill occurs.
These point measurements do not require a complete monthly flow quarter.

By default, historical publication must precede both the analysis as-of and each
historical quarter end. A conditional nowcast quarter may use publications up to
the analysis as-of. Explicit `historical_vintage_policy: "analysis_vintage"`
instead permits retrospective publication within historical rows and must be
labelled as such. A conservative retrieval date is not a verified publication
date. The committed survey CSV has conservative October 12 availability, so its
July formation does not prove July information availability. Preserve the source
workbook hash and state if the workbook has not been independently retrieved.

Survey target rates and OIS quotes proxy the native expected effective quarterly
rate only through the supplied mapping assumption. This adds no term-premium,
risk-neutral or target-to-effective spread measurement equation. OIS support is
an explicit input contract, not evidence that a particular market feed is valid.
`survey_ablation: true` exports same-model, same-data, same-window on/off forecasts
in `expectations_ablation.csv`; it also works as a zero-effect check when no
expectations are supplied. `population_endpoint: "freeze_history"` preserves the
original history transforms and appends only conditional rows; default `recompute`
retains endpoint revision attribution. Both choices are recorded in the manifest.

## Verified policy paths

Optional `policy_paths` contains at most eight distinct labelled specifications:

```json
[
  {
    "label": "Hold 3.75 percent for four quarters",
    "kind": "held_rate",
    "information": "sequential_surprises",
    "start_quarter": "2027Q1",
    "rates_annual_percent": [3.75, 3.75, 3.75, 3.75]
  },
  {
    "label": "Announce next-quarter 25bp deviation",
    "kind": "forward_guidance",
    "information": "announced",
    "announcement_quarter": "2027Q1",
    "start_quarter": "2027Q2",
    "deviations_bp": [25]
  }
]
```

Choose exactly one target vector: absolute annual percent rates or basis-point
deviations from the same baseline. Null entries leave quarters unconstrained.
Held-rate paths solve a fresh unanticipated active monetary innovation each
target quarter; agents do not know later innovations in advance. Guidance solves
native `rm_ait_shl1..6` innovations **at announcement**, which may affect output
and inflation immediately while targeting rates one to six quarters later.
Rank-deficient or unreachable targets fail. Actual rate targets are verified to
`1e-8` annual percentage points, and innovations, constraints, information and
errors are archived. Zero prior variance on anticipated shocks is recorded:
such paths are off-prior-support hypotheticals, with no standard-deviation or
posterior-probability claim. An effective quarterly rate path does not identify
an intraday FOMC surprise. Operational desk scripts and configurations remain
separate; toolkit support does not certify parity with an unseen adapter.

## Full and cyclical historical decomposition

Price and wage markups are separate groups. Core PCE, GDP per capita and Fed
funds have full decomposition figures; FFR uses annual rate percentage points.
The cyclical dashboard subtracts initial-state and deterministic contributions
from fitted and observed totals, retaining every shock group and numerical
remainder. It is a model-centred accounting view, not a newly estimated output gap.
`decomposition_cyclical.csv` contains the cyclical components;
`decomposition_totals.csv` retains full, initial, deterministic, cyclical totals,
measurement gaps and remainders. Full arrays and original levels remain available.
The full-sample gate remains **1e-8 in plotted units**, including for cyclical
figures. A diagnostic result never becomes a pass by dropping a remainder or
subtracting a large baseline; status applies only to the specific run.

### Guard rolling yield history

If using a rolling `BAMLC8A0C15PYEY` download, optionally add both
`inputs.preserved_yield_csv` and `inputs.current_yield_csv`: two-column daily
date/value snapshots. The toolkit verifies exact overlapping values, preserves
missing historical dates, exports the union/restored dates, and rebuilds quarterly
means via the native loader. Conflicting overlap is rejected for explicit
revision review. The preserved snapshot must cover complete historical quarters;
unioning incomplete snapshots cannot invent missing observations. Without these
inputs, no history repair is claimed. Never interpret a truncated quarter's
changed mean as economic news.

## Parameters, forecasts and scenarios

Version 1 supports `parameters: "public_defaults"` only. These are the published
ss104 code defaults, **not** a NY Fed production posterior or staff forecast.
Mode/posterior files are deliberately rejected. The experimental M7 mode remains
unconverged and must not be renamed a validated/default configuration. Numerical
solver/parity validation is different from empirical calibration or forecast
validation. A future estimated-point extension needs explicit independent
convergence, provenance and measurement validation; this toolkit does not supply it.

The first forecast quarter follows `conditioning_end`. Partial-information
quarters are labelled conditional **nowcasts/latent states**; only supplied
measurements are actual. The baseline uses zero future innovations. GDP and
investment are **per capita**. Growth/inflation plots multiply quarterly log
changes by four: annualized **log** percent, not exact compounded SAAR or YoY.
Fed funds is four times the quarterly rate, in annual percentage points. Hours
is a `100 log` level; differences are log points. Never mix these units.

`policy_impacts_bp` defaults to `[25, 50, -25]`. At the first future quarter,
one primitive monetary innovation is normalized by its actual response:

`shock = (impact_bp / 100) / [4 Z_rate R_shock]`.

The active unanticipated channel is selected from `rm_ait_sh`, then `rm_sh`,
requiring nonzero contemporaneous rate loading and positive shock variance.
Optionally set `policy_shock` to one of these names. An inactive/unsupported
selection is rejected. No anticipated rate innovation is silently substituted.
The actual impact is verified to `1e-10` annual pp. Later innovations are zero;
**subsequent rates follow endogenous dynamics, not a held rate path**. A quarterly
model surprise does not automatically equal a 25 bp intraday FOMC surprise.

## Historical decomposition and interpretation

The native public HD API supports single-regime systems only. The toolkit's
separate diagnostic attribution uses native regime solutions, Kalman filter and
disturbance smoother; it recovers conditional disturbance means as `Q R' r`.
It exports every shock contribution plus the posterior initial-state term,
deterministic `CCC`/`DD` terms, numerical reconstruction remainder and the
observed-minus-fitted measurement gap. None is forced to zero or hidden.

Core PCE/GDP/Fed funds reconstructions are checked over the **whole sample** against an
immutable threshold of **1e-8 plotted units**. A failure does not invalidate
an independently verified forecast, but its HD chart is prominently
**DIAGNOSTIC**, including a residual strip and the actual maximum error. The
known earlier GDP diagnostic missed this gate; inspect the current run rather
than copying its outcome. Passing reconstruction alone is not independent
Julia validation or proof of empirical shock identification.

Each plot names its vintage, units and forecast origin. HD legends retain
initial/deterministic/remainder terms; model policy residuals are distinct from
systematic policy responses. Price/wage markup and demand labels are conditional
structural interpretations. PCE measurement innovations are not structural
supply shocks; government-spending shocks do not identify tax/transfer policy.

There are **no uncertainty bands**: fixed-point paths do not represent parameter,
smoothed-state or predictive uncertainty. Do not add bands unless a separately
supported uncertainty calculation supplies them. State decision-relevant risks:
default-parameter sensitivity, incomplete information, revisions, HP endpoints,
policy credibility assumptions and omitted energy/distribution channels.

## Output contract and agent checklist

- `manifest.json`: code SHA/status and source-file hashes (including uncommitted
  toolkit code), versions/runtime import path, exact config/input hashes,
  vintage, scenario normalization and validation results.
- `parameters.json`: exact model-space parameter values by regime.
- Frozen levels, transformed observations, forecasts, endpoint/news attribution,
  grouped HD CSV and raw decomposition arrays.
- `figures/`: baseline dashboard; policy-surprise dashboard; core PCE and GDP
  and Fed funds full HD charts; cyclical dashboard; `figures.json` hashes and one-line captions.

Before delivery: verify the intended revision/import path; audit actual source
coverage and revisions; run in a new directory; inspect `status.json` and every
HD gate; visually inspect all figures; disclose latent measurements, uncertainty
omissions and empirical limitations. Save requested figures through the user's
supported artifact channel. Existing dated outputs are reference evidence,
never fresh model runs. No private data, credentials, large outputs or scratch
configurations belong in commits. See [AGENTS.md](AGENTS.md).
