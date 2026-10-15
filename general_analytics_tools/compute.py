"""Bounded, offline ss104 orchestration; use native solvers, filter and forecasts."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

from general_analytics_tools.decomposition import disturbance_attribution, export_decomposition
from general_analytics_tools.expectations import rate_expectations
from general_analytics_tools.inputs import load_config, quarter, read_levels, sha256
from general_analytics_tools.policy_paths import policy_paths
from nydsge.forecast import forecast_linear_system
from nydsge.kalman import kalman_log_likelihood, model_process_covariances, smooth_kalman_result
from nydsge.models import Model1002
from nydsge.regimes import RegimeSwitchingSystem, regime_indices_from_dates
from nydsge.runtime import RuntimeConfig
from nydsge.scenarios import current_public_observables
from nydsge.solve import compute_system, validate_system
from nydsge.ss104_workflow import forecast_ss104, prepare_ss104_observations

VARIABLES = (
    ("obs_gdp", "Real GDP per capita growth", 4, "Annualized log growth (%)"),
    ("obs_corepce", "Core PCE inflation", 4, "Annualized log inflation (%)"),
    ("obs_gdpdeflator", "GDP deflator inflation", 4, "Annualized log inflation (%)"),
    ("obs_nominalrate", "Effective Fed funds rate", 4, "Annual rate (%)"),
    ("obs_hours", "Hours per capita", 1, "100 log hours per capita"),
    ("obs_investment", "Real investment per capita growth", 4, "Annualized log growth (%)"),
)
IDENTIFICATION = (
    "Illustrative linear representative-agent ss104 with public defaults, not a Fed staff forecast "
    "or production posterior. Shock attribution is conditional on model restrictions, not causal "
    "identification. Policy residual differs from systematic policy response. "
    "CPI is not assimilated as core PCE or GDP-deflator inflation. "
    "Measurement innovations are not structural supply shocks; "
    "government spending does not identify taxes/transfers. Missing observations imply latent "
    "conditional estimates. State, parameter and predictive uncertainty are not represented."
)


def git_text(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def policy_surprises(
    system: RegimeSwitchingSystem,
    state: np.ndarray,
    regimes: np.ndarray,
    names: list[str],
    shocks: list[str],
    impacts_bp: list[float],
    selected: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Normalize on the actual impact of a single unanticipated primitive innovation."""
    horizon = len(regimes)
    baseline = forecast_linear_system(system, state, horizon=horizon, regime_indices=regimes)
    active = system.period_systems(regimes, horizon)[0]
    rate = names.index("obs_nominalrate")
    candidates = [selected] if selected else ["rm_ait_sh", "rm_sh"]
    selected_index = None
    impact = None
    for name in candidates:
        if name not in shocks or name not in ("rm_sh", "rm_ait_sh"):
            continue
        j = shocks.index(name)
        loading = float(4 * active.measurement.ZZ[rate] @ active.transition.RRR[:, j])
        if np.isfinite(loading) and abs(loading) > 1e-12 and active.measurement.QQ[j, j] > 0:
            selected_index, impact = j, loading
            break
    if selected_index is None or impact is None:
        raise ValueError("No active supported monetary surprise; check regime/shock selection.")
    arrays, records = {}, []
    for bp in impacts_bp:
        label = f"{bp:+g} bp surprise"
        innovations = np.zeros((horizon, len(shocks)))
        innovations[0, selected_index] = (bp / 100) / impact
        output = forecast_linear_system(
            system, state, horizon=horizon, regime_indices=regimes, shocks=innovations
        )
        np.testing.assert_allclose(
            4 * (output.observables[0, rate] - baseline.observables[0, rate]),
            bp / 100,
            atol=1e-10,
            rtol=0,
        )
        if not np.isfinite(output.observables).all():
            raise ValueError("Nonfinite policy scenario.")
        arrays[label] = output.observables
        records.append(
            {
                "label": label,
                "impact_bp": bp,
                "shock": shocks[selected_index],
                "model_units": float(innovations[0, selected_index]),
                "shock_standard_deviations": float(
                    innovations[0, selected_index]
                    / np.sqrt(active.measurement.QQ[selected_index, selected_index])
                ),
                "annual_rate_impact_per_model_unit": impact,
                "later_innovations": "zero; subsequent rates endogenous, not held",
            }
        )
    return baseline.observables, arrays, records


def dated_observations(model: Model1002, levels: pd.DataFrame) -> pd.DataFrame:
    observations = current_public_observables(model, levels)
    observations.index = pd.PeriodIndex(observations.pop("date").map(quarter), freq="Q")
    return observations.loc[
        observations.index >= quarter(str(model.get_setting("date_presample_start")))
    ]


def compute(config_path: Path, output: Path) -> dict[str, Any]:
    runtime = Path(str(sys.modules["nydsge"].__file__)).resolve()
    expected = Path(__file__).resolve().parents[1] / "src/nydsge"
    if not runtime.is_relative_to(expected.resolve()):
        raise ValueError(
            "Runtime source differs from toolkit checkout; run uv sync in this checkout."
        )
    config = load_config(config_path)
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "status.json"
    status_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    try:
        result = _compute(config, config_path, output)
        status_path.write_text(
            json.dumps({"status": "completed", "HD": result["HD"]}), encoding="utf-8"
        )
        return result
    except Exception as error:
        status_path.write_text(
            json.dumps({"status": "failed", "error": str(error)}), encoding="utf-8"
        )
        raise


def _compute(config: dict[str, Any], config_path: Path, output: Path) -> dict[str, Any]:
    levels, source = read_levels(config, output)
    sample = config["sample"]
    end, cond_end = quarter(sample["history_end"]), quarter(sample["conditioning_end"])
    start_forecast = end + 1
    model = Model1002(
        "ss104",
        runtime=RuntimeConfig(backend="numpy", device="cpu", dtype="float64"),
        settings={
            "data_vintage": str(source["vintage"]),
            "date_presample_start": str(quarter(sample["presample_start"])),
            "date_mainsample_start": str(quarter(sample["mainsample_start"])),
            "date_forecast_start": str(start_forecast),
        },
    )
    names, shocks = list(model.observables), list(model.indexes.exogenous_shocks)
    channels = config.get("conditioning_observables", [])
    if set(channels) - set(names):
        raise ValueError("Unknown conditioning observable; CPI is not a substitute measurement.")
    model.set_setting("cond_full_names", channels)
    historical_levels = levels.loc[levels.date.map(quarter) <= end]
    historical = dated_observations(model, historical_levels)
    revised = dated_observations(model, levels)
    if "expectations_csv" in config["inputs"]:
        expectation_rows = rate_expectations(
            pd.read_csv(config["inputs"]["expectations_csv"]),
            pd.PeriodIndex(revised.index, freq="Q"),
            config["rate_expectations"],
            end,
        )
        expectation_rows.to_csv(output / "expected_rate_observations.csv", index_label="date")
        revised.loc[:, expectation_rows.columns] = expectation_rows
        historical.loc[:, expectation_rows.columns] = expectation_rows.reindex(historical.index)
        channels = list(dict.fromkeys([*channels, *expectation_rows.columns]))
        model.set_setting("cond_full_names", channels)
    if config.get("population_endpoint") == "freeze_history":
        revised.loc[historical.index, :] = historical
    endpoint = revised.copy()
    endpoint.loc[endpoint.index > end, :] = np.nan
    observed = endpoint.copy()
    for channel in channels:
        selected = revised.loc[revised.index > end, channel]
        if (
            channel in config.get("conditioning_observables", [])
            and not np.isfinite(selected.to_numpy()).all()
        ):
            raise ValueError(f"Incomplete conditioning measurement: {channel}.")
        observed.loc[observed.index > end, channel] = selected
    for label, frame in (
        ("historical", historical),
        ("endpoint", endpoint),
        ("observed", observed),
    ):
        frame.to_csv(output / f"{label}_observations.csv", index_label="date")
    print("Solving fixed-default ss104 and filtering dated observations", flush=True)
    system = compute_system(model)
    if not isinstance(system, RegimeSwitchingSystem):
        raise TypeError("Expected the native ss104 regime system.")
    for active in system.systems:
        validate_system(active)
    conditional_count = int(cond_end.ordinal - end.ordinal)
    horizon = config["forecast_quarters"] + conditional_count
    original = forecast_ss104(model, system=system, data=historical, horizon=horizon)
    kind = "full" if conditional_count else "none"
    population = forecast_ss104(
        model, system=system, data=endpoint, horizon=horizon, cond_type=kind
    )
    forecast = forecast_ss104(model, system=system, data=observed, horizon=horizon, cond_type=kind)
    if config.get("survey_ablation"):
        without = observed.copy()
        expected_names = [f"obs_exp_nominalrate{i}" for i in range(1, 7)]
        without.loc[:, expected_names] = np.nan
        off = forecast_ss104(model, system=system, data=without, horizon=horizon, cond_type=kind)
        ablation = []
        for t, date in enumerate(
            pd.period_range(cond_end + 1, periods=config["forecast_quarters"], freq="Q")
        ):
            for name, _, factor, unit in VARIABLES:
                i = names.index(name)
                ablation.append(
                    {
                        "quarter": str(date),
                        "variable": name,
                        "with_expectations": float(factor * forecast.observables[t, i]),
                        "without_expectations": float(factor * off.observables[t, i]),
                        "difference": float(
                            factor * (forecast.observables[t, i] - off.observables[t, i])
                        ),
                        "units": unit,
                    }
                )
        pd.DataFrame(ablation).to_csv(output / "expectations_ablation.csv", index=False)
    prepared = prepare_ss104_observations(model, observed)
    dates = pd.PeriodIndex(prepared.index, freq="Q")
    regimes = regime_indices_from_dates(model.get_setting("regime_dates"), list(dates))
    process = model_process_covariances(model, system, len(dates), start_date=dates[0])
    presample = int((dates < quarter(sample["mainsample_start"])).sum())
    filtered = kalman_log_likelihood(
        system,
        prepared.to_numpy(),
        regime_indices=regimes,
        process_covariances=process,
        log_likelihood_start=presample,
    )
    smoothed = smooth_kalman_result(system, filtered, data=prepared.to_numpy())
    states = smoothed.smoothed_states
    if states is None or not np.isfinite(states).all() or not np.isfinite(filtered.log_likelihood):
        raise ValueError("Invalid native filter/smoother result.")
    native_states = forecast.history_states
    if native_states is None or forecast.log_likelihood is None:
        raise ValueError("Native forecast did not return history and likelihood.")
    if conditional_count:
        if forecast.conditional_states is None:
            raise ValueError("Native forecast did not return conditional states.")
        native_states = np.vstack((native_states, forecast.conditional_states))
    np.testing.assert_allclose(states, native_states, atol=1e-8, rtol=1e-10)
    qq = np.asarray([s.measurement.QQ.copy() for s in system.period_systems(regimes, len(dates))])
    anticipated = [
        i for i, name in enumerate(shocks) if name in {f"rm_shl{h}" for h in range(1, 7)}
    ]
    zlb = quarter(str(model.get_setting("date_zlb_start")))
    for t in np.flatnonzero(dates < zlb):
        qq[t, anticipated, :] = 0
        qq[t, :, anticipated] = 0
    hd = disturbance_attribution(
        system,
        filtered,
        states,
        prepared.to_numpy(),
        regimes,
        shock_covariances=qq,
        process_covariances=process,
    )
    decomposition, validation = export_decomposition(hd, dates, names, shocks)
    decomposition.to_csv(output / "decomposition.csv", index=False)
    cyclical = decomposition.loc[
        ~decomposition.component.isin(["Initial state", "Deterministic terms"])
    ].copy()
    cyclical.to_csv(output / "decomposition_cyclical.csv", index=False)
    totals = []
    for name in ("obs_corepce", "obs_gdp", "obs_nominalrate"):
        i = names.index(name)
        for t, date in enumerate(dates):
            totals.append(
                {
                    "quarter": str(date),
                    "variable": name,
                    "fitted": float(4 * hd["fitted"][t, i]),
                    "initial": float(4 * hd["initial"][t, i]),
                    "deterministic": float(4 * hd["deterministic"][t, i]),
                    "cyclical_fitted": float(
                        4 * (hd["fitted"][t, i] - hd["initial"][t, i] - hd["deterministic"][t, i])
                    ),
                    "measurement_gap": None
                    if not np.isfinite(hd["measurement_gap"][t, i])
                    else float(4 * hd["measurement_gap"][t, i]),
                    "cyclical_actual": None
                    if not np.isfinite(prepared.iloc[t, i])
                    else float(
                        4 * (prepared.iloc[t, i] - hd["initial"][t, i] - hd["deterministic"][t, i])
                    ),
                    "units": "annual rate pp" if name == "obs_nominalrate" else "annualized log pp",
                    "remainder": float(4 * hd["remainder"][t, i]),
                }
            )
    pd.DataFrame(totals).to_csv(output / "decomposition_totals.csv", index=False)
    np.savez_compressed(
        output / "decomposition_arrays.npz",
        contributions=hd["contributions"],
        initial=hd["initial"],
        deterministic=hd["deterministic"],
        fitted=hd["fitted"],
        shocks=hd["shocks"],
        remainder=hd["remainder"],
        measurement_gap=hd["measurement_gap"],
        state_remainder=hd["state_remainder"],
        innovation_condition=hd["innovation_condition"],
        smoother_information_error=hd["smoother_information_error"],
    )
    future = pd.period_range(cond_end + 1, periods=config["forecast_quarters"], freq="Q")
    future_regimes = regime_indices_from_dates(model.get_setting("regime_dates"), list(future))
    baseline, scenarios, scenario_records = policy_surprises(
        system,
        states[-1],
        future_regimes,
        names,
        shocks,
        config["policy_impacts_bp"],
        config.get("policy_shock"),
    )
    np.testing.assert_allclose(baseline, forecast.observables, atol=1e-10, rtol=0)
    path_arrays, path_records = policy_paths(
        system, states[-1], future_regimes, future, names, shocks, config.get("policy_paths", [])
    )
    if set(path_arrays) & set(scenarios):
        raise ValueError("Policy paths and surprise labels must be distinct.")
    scenarios.update(path_arrays)
    scenario_records.extend(path_records)
    rows = []
    for label, array in [("Baseline", baseline), *scenarios.items()]:
        for t, date in enumerate(future):
            for name, _, factor, unit in VARIABLES:
                i = names.index(name)
                rows.append(
                    {
                        "quarter": str(date),
                        "variable": name,
                        "scenario": label,
                        "value": float(factor * array[t, i]),
                        "units": unit,
                        "difference": float(factor * (array[t, i] - baseline[t, i])),
                    }
                )
    pd.DataFrame(rows).to_csv(output / "forecasts.csv", index=False)
    history_rows = []
    for t, date in enumerate(dates):
        for name, _, factor, unit in VARIABLES:
            i = names.index(name)
            actual = prepared.iloc[t, i]
            history_rows.append(
                {
                    "quarter": str(date),
                    "variable": name,
                    "actual": None if not np.isfinite(actual) else float(factor * actual),
                    "fitted": float(factor * hd["fitted"][t, i]),
                    "units": unit,
                }
            )
    pd.DataFrame(history_rows).to_csv(output / "history.csv", index=False)
    attribution = []
    for label, candidate in (("population_endpoint", population), ("observed_news", forecast)):
        conditional = candidate.conditional_observables
        array = (
            candidate.observables
            if conditional is None
            else np.vstack((conditional, candidate.observables))
        )
        reference = original.observables
        if label == "observed_news":
            p = population.conditional_observables
            reference = (
                population.observables if p is None else np.vstack((p, population.observables))
            )
        for t, date in enumerate(pd.period_range(start_forecast, periods=horizon, freq="Q")):
            for name, _, factor, _ in VARIABLES:
                i = names.index(name)
                attribution.append(
                    {
                        "quarter": str(date),
                        "variable": name,
                        "component": label,
                        "difference": float(factor * (array[t, i] - reference[t, i])),
                    }
                )
    pd.DataFrame(attribution).to_csv(output / "endpoint_and_observation_news.csv", index=False)
    params = {
        str(r): {name: float(p.value) for name, p in model.regime_snapshot(r).parameters.items()}
        for r in model.get_setting("regime_dates")
    }
    (output / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    config_dump = json.loads(json.dumps(config))
    (output / "config.json").write_text(json.dumps(config_dump, indent=2), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    files = [*Path(__file__).parent.glob("*.py"), *(root / "src/nydsge").rglob("*.py")]
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_sha": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--short"),
        "code_sha256": {str(p.relative_to(root)): sha256(p) for p in files},
        "config_sha256": sha256(config_path),
        "runtime_source": str(sys.modules["nydsge"].__file__),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "runtime": "NumPy/SciPy CPU float64; one thread requested by CLI",
        "model": "ss104",
        "parameters": "public_defaults",
        "source": source,
        "observable_names": names,
        "shock_names": shocks,
        "sample": sample,
        "conditioning_observables": channels,
        "rate_expectations": config.get("rate_expectations"),
        "population_endpoint": config.get("population_endpoint", "recompute"),
        "input_lead_quarter": str(quarter(sample["presample_start"]) - 1),
        "forecast_start": str(future[0]),
        "forecast_end": str(future[-1]),
        "history_plot_start": config.get("plot_history_start", str(dates[max(0, len(dates) - 24)])),
        "policy_scenarios": scenario_records,
        "HD": validation,
        "filter_log_likelihood": float(filtered.log_likelihood),
        "forecast_filter_loglik_difference": float(
            abs(filtered.log_likelihood - forecast.log_likelihood)
        ),
        "validation": [
            "native regime-system validation",
            "native smoother agreement 1e-8",
            "native forecast agreement 1e-10",
            "policy rate impact normalization 1e-10",
        ],
        "bands": "omitted; no parameter/state/predictive uncertainty claimed",
        "identification_caveat": IDENTIFICATION,
        "decision_risks": (
            "Fixed-default sensitivity; missing measurements; revisions and HP endpoints; "
            "energy/distribution channels not explicitly resolved."
        ),
        "output_sha256": {
            p.name: sha256(p) for p in output.iterdir() if p.is_file() and p.name != "status.json"
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result
