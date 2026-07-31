from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nydsge.data import (
    FredFetcher,
    df_to_matrix,
    load_current_fred_series,
    load_current_fred_source,
    parse_data_sources,
    reverse_transform_observables,
    transform_data,
)
from nydsge.estimate import estimate, estimation_mode_from_result, save_estimation_mode
from nydsge.forecast import (
    forecast_linear_system,
    forecast_one,
    observable_irf,
    solve_shocks_for_observable_targets,
)
from nydsge.models import Model1002
from nydsge.solve import compute_system

DEFAULT_UNEMPLOYMENT_TARGETS = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0)
DEFAULT_REFRESH_PARAMETERS = (
    "rho_g",
    "rho_b",
    "rho_mu",
    "rho_ztil",
    "sigma_g",
    "sigma_b",
    "sigma_mu",
    "sigma_ztil",
)
REFERENCE_DSGE_VERSION = "1.3.0"
REFERENCE_DSGE_TREE = "e746a4a5ab9c26d897239e722b0f19d4bb3bd77e"
UPSTREAM_DSGE_URL = f"https://github.com/FRBNY-DSGE/DSGE.jl/tree/v{REFERENCE_DSGE_VERSION}"
NYFED_MARCH_2026_URL = (
    "https://libertystreeteconomics.newyorkfed.org/2026/03/"
    "the-new-york-fed-dsge-model-forecast-march-2026/"
)


@dataclass(frozen=True)
class UnemploymentHoursBridge:
    start_date: str
    end_date: str
    observations: int
    intercept: float
    slope: float
    r_squared: float
    pre_covid_slope: float | None
    post_1990_slope: float | None


@dataclass(frozen=True)
class ScenarioStudyArtifacts:
    output_dir: Path
    report: Path
    metadata: Path
    forecast_csv: Path
    summary_csv: Path
    irf_csv: Path
    observables_csv: Path
    fred_levels_csv: Path
    unemployment_csv: Path
    model_mode: Path | None
    figures: tuple[Path, ...]


def current_public_observables(model: Model1002, fred_levels: pd.DataFrame) -> pd.DataFrame:
    """Build model observables while preserving unavailable non-FRED inputs as missing."""

    levels = fred_levels.copy()
    for observable in model.observable_mappings.values():
        for source_name in observable.source_names:
            mnemonic = source_name.split("__", 1)[0]
            if mnemonic not in levels.columns:
                levels[mnemonic] = np.nan
    return transform_data(model, levels)


def estimate_unemployment_hours_bridge(
    observables: pd.DataFrame,
    unemployment: pd.DataFrame,
    *,
    start_date: str = "1985-Q1",
    end_date: str | None = None,
) -> UnemploymentHoursBridge:
    """Estimate a transparent quarterly change bridge from unemployment to model hours."""

    joined = observables[["date", "obs_hours"]].merge(
        unemployment[["date", "UNRATE"]], on="date", how="inner", validate="one_to_one"
    )
    joined = _date_window(joined, start_date=start_date, end_date=end_date)
    joined["delta_hours"] = joined["obs_hours"].diff()
    joined["delta_unemployment"] = joined["UNRATE"].diff()
    sample = joined.dropna(subset=["delta_hours", "delta_unemployment"]).copy()
    if len(sample) < 20:
        msg = "The unemployment-hours bridge requires at least 20 quarterly observations."
        raise ValueError(msg)
    intercept, slope, r_squared = _ols_change_bridge(sample)
    pre_covid = sample.loc[_period_index(sample["date"]) < _period("2020-Q1")]
    post_1990 = sample.loc[_period_index(sample["date"]) >= _period("1990-Q1")]
    return UnemploymentHoursBridge(
        start_date=str(sample.iloc[0]["date"]),
        end_date=str(sample.iloc[-1]["date"]),
        observations=len(sample),
        intercept=intercept,
        slope=slope,
        r_squared=r_squared,
        pre_covid_slope=(None if len(pre_covid) < 20 else _ols_change_bridge(pre_covid)[1]),
        post_1990_slope=(None if len(post_1990) < 20 else _ols_change_bridge(post_1990)[1]),
    )


def build_unemployment_scenario_path(
    baseline: np.ndarray,
    *,
    target: float,
    ramp_quarters: int = 4,
    hold_quarters: int = 4,
    return_quarters: int = 8,
) -> np.ndarray:
    """Ramp from the current rate to a target, hold, then return to baseline."""

    baseline_path = np.asarray(baseline, dtype=np.float64)
    if baseline_path.ndim != 1 or baseline_path.size == 0:
        msg = "Baseline unemployment must be a non-empty one-dimensional path."
        raise ValueError(msg)
    if min(ramp_quarters, hold_quarters, return_quarters) < 1:
        msg = "Ramp, hold, and return durations must all be positive."
        raise ValueError(msg)
    path = baseline_path.copy()
    ramp_end = min(ramp_quarters, path.size - 1)
    if ramp_end > 0:
        path[: ramp_end + 1] = np.linspace(path[0], target, ramp_end + 1)
    hold_start = ramp_end + 1
    hold_end = min(hold_start + hold_quarters, path.size)
    path[hold_start:hold_end] = target
    return_end = min(hold_end + return_quarters, path.size)
    if hold_end < return_end:
        destination = baseline_path[return_end - 1]
        path[hold_end:return_end] = np.linspace(
            target,
            destination,
            return_end - hold_end + 1,
        )[1:]
    return path


def run_ai_economy_study(
    *,
    output_dir: Path | str,
    start_date: str = "1964-Q1",
    model_end_date: str = "2026-Q1",
    horizon: int = 20,
    unemployment_targets: tuple[float, ...] = DEFAULT_UNEMPLOYMENT_TARGETS,
    bridge_start_date: str = "1985-Q1",
    refresh_model: bool = True,
    refresh_maxiter: int = 60,
    fetcher: FredFetcher | None = None,
    make_plots: bool = True,
) -> ScenarioStudyArtifacts:
    """Run a current-data AI/labor-market DSGE scenario study and save its artifacts."""

    if horizon < 17:
        msg = "The default ramp/hold/return scenario design requires a horizon of at least 17."
        raise ValueError(msg)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    forecast_start = _next_quarter(model_end_date)
    model = Model1002(
        subspec="ss10",
        settings={
            "data_vintage": model_end_date.replace("-Q", "q"),
            "date_forecast_start": forecast_start,
        },
    )

    fred_mnemonics = parse_data_sources(model).get("FRED", [])
    fred_levels = load_current_fred_source(
        fred_mnemonics,
        start_date=start_date,
        end_date=model_end_date,
        fetcher=fetcher,
    )
    unemployment = load_current_fred_series("UNRATE", fetcher=fetcher)
    unemployment = _date_window(unemployment, start_date=start_date)
    observables = current_public_observables(model, fred_levels)

    fred_levels_path = destination / "fred_levels_current.csv"
    observables_path = destination / "observables_current.csv"
    unemployment_path = destination / "unemployment_quarterly.csv"
    fred_levels.to_csv(fred_levels_path, index=False)
    observables.to_csv(observables_path, index=False)
    unemployment.to_csv(unemployment_path, index=False)

    bridge = estimate_unemployment_hours_bridge(
        observables,
        unemployment,
        start_date=bridge_start_date,
        end_date=model_end_date,
    )
    observations = df_to_matrix(model, observables)
    model_refresh, model_mode_path = _refresh_model_shock_block(
        model,
        observations,
        output_dir=destination,
        start_date=str(observables.iloc[0]["date"]),
        enabled=refresh_model,
        maxiter=refresh_maxiter,
    )

    baseline_output = forecast_one(
        model,
        input_type="mode",
        cond_type="none",
        output_vars=["histstates", "forecastobs", "forecastpseudo"],
        check_empty_columns=False,
        horizon=horizon,
        data=observables,
        history_method="smoothed",
    )
    if baseline_output.history_states is None or baseline_output.history_states.shape[0] == 0:
        msg = "Current-data filtering did not produce a usable terminal state."
        raise RuntimeError(msg)
    start_state = baseline_output.history_states[-1]
    baseline_model = np.asarray(baseline_output.observables, dtype=np.float64)
    baseline_report = reverse_transform_observables(model, baseline_model)
    system = compute_system(model)
    dates = _quarter_sequence(forecast_start, horizon)

    baseline_unemployment = _forecast_unemployment_baseline(
        unemployment,
        dates=dates,
        estimation_start=bridge_start_date,
    )
    forecast_rows = _forecast_rows(
        scenario="baseline",
        dates=dates,
        unemployment=baseline_unemployment,
        values=baseline_report,
        baseline=baseline_report,
        variable_names=list(model.observables),
    )
    scenario_summaries: list[dict[str, Any]] = []
    shock_arrays: dict[str, np.ndarray] = {}
    hours_index = list(model.observables).index("obs_hours")
    for target in unemployment_targets:
        scenario_name = f"unemployment_{target:g}pct"
        unemployment_scenario = build_unemployment_scenario_path(
            baseline_unemployment,
            target=float(target),
        )
        targets = np.full_like(baseline_model, np.nan, dtype=np.float64)
        targets[:, hours_index] = baseline_model[:, hours_index] + bridge.slope * (
            unemployment_scenario - baseline_unemployment
        )
        conditioned = solve_shocks_for_observable_targets(system, start_state, targets)
        scenario_report = reverse_transform_observables(model, conditioned.observables)
        forecast_rows.extend(
            _forecast_rows(
                scenario=scenario_name,
                dates=dates,
                unemployment=unemployment_scenario,
                values=scenario_report,
                baseline=baseline_report,
                variable_names=list(model.observables),
            )
        )
        shock_arrays[scenario_name] = conditioned.shocks
        scenario_summaries.append(
            _scenario_summary(
                scenario_name,
                target=float(target),
                values=scenario_report,
                baseline=baseline_report,
                variable_names=list(model.observables),
                condition_error=conditioned.max_abs_error,
                shocks=conditioned.shocks,
                shock_scales=np.sqrt(np.clip(np.diag(system.measurement.QQ), 0.0, None)),
            )
        )

    irf_rows, structural_rows, structural_summaries = _structural_scenarios(
        model,
        system=system,
        start_state=start_state,
        baseline_model=baseline_model,
        baseline_report=baseline_report,
        dates=dates,
    )
    forecast_rows.extend(structural_rows)
    scenario_summaries.extend(structural_summaries)

    forecast_frame = pd.DataFrame(forecast_rows)
    summary_frame = pd.DataFrame(scenario_summaries)
    irf_frame = pd.DataFrame(irf_rows)
    forecast_path = destination / "scenario_forecasts.csv"
    summary_path = destination / "scenario_summary.csv"
    irf_path = destination / "selected_irfs.csv"
    forecast_frame.to_csv(forecast_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    irf_frame.to_csv(irf_path, index=False)
    shock_scenario_names = list(shock_arrays)
    np.savez(
        destination / "conditional_shocks.npz",
        scenario_names=np.asarray(shock_scenario_names, dtype="U"),
        shocks=np.stack([shock_arrays[name] for name in shock_scenario_names], axis=0),
    )

    data_quality = _data_quality_summary(observables, model_end_date=model_end_date)
    metadata_payload = {
        "study": "ai_economy_unemployment_and_demand_scenarios",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "model": {
            "local_spec": model.spec,
            "local_subspec": model.subspec,
            "forecast_start": forecast_start,
            "horizon_quarters": horizon,
            "reference_dsge_version": REFERENCE_DSGE_VERSION,
            "reference_dsge_git_tree": REFERENCE_DSGE_TREE,
            "upstream_url": UPSTREAM_DSGE_URL,
            "production_model_equivalence": False,
            "note": (
                "The local ss10 parity model is state-updated with current data and its "
                "shock block is optionally MAP-refreshed; it is not the private March 2026 "
                "NY Fed production parameterization."
            ),
            "refresh": model_refresh,
        },
        "data": {
            "model_data_start": start_date,
            "model_data_end": model_end_date,
            "unemployment_last_date": str(unemployment.iloc[-1]["date"]),
            "fred_series": fred_mnemonics,
            "quality": data_quality,
        },
        "bridge": asdict(bridge),
        "scenarios": {
            "unemployment_targets_percent": list(unemployment_targets),
            "unemployment_baseline": (
                "Latest available complete-quarter unemployment rate held constant"
            ),
            "labor_path": "4-quarter ramp, 4-quarter hold, 8-quarter return",
            "structural": {
                "ai_investment_boom": "+1 standard deviation mu_sh (MEI)",
                "ai_productivity_acceleration": "+1 standard deviation ztil_sh",
                "demand_contraction": "-1 standard deviation g_sh",
            },
        },
        "sources": [
            {"label": "FRED public series", "url": "https://fred.stlouisfed.org/"},
            {"label": "FRED unemployment rate", "url": "https://fred.stlouisfed.org/series/UNRATE"},
            {"label": "NY Fed DSGE forecast, March 2026", "url": NYFED_MARCH_2026_URL},
            {"label": "DSGE.jl upstream revision", "url": UPSTREAM_DSGE_URL},
        ],
    }
    metadata_path = destination / "study_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    figure_paths = (
        _make_study_plots(destination, forecast_frame, summary_frame, irf_frame)
        if make_plots
        else ()
    )
    report_path = destination / "study_report.md"
    report_path.write_text(
        _study_report_markdown(metadata_payload, summary_frame, figure_paths),
        encoding="utf-8",
    )
    return ScenarioStudyArtifacts(
        output_dir=destination,
        report=report_path,
        metadata=metadata_path,
        forecast_csv=forecast_path,
        summary_csv=summary_path,
        irf_csv=irf_path,
        observables_csv=observables_path,
        fred_levels_csv=fred_levels_path,
        unemployment_csv=unemployment_path,
        model_mode=model_mode_path,
        figures=figure_paths,
    )


def _refresh_model_shock_block(
    model: Model1002,
    observations: np.ndarray,
    *,
    output_dir: Path,
    start_date: str,
    enabled: bool,
    maxiter: int,
) -> tuple[dict[str, Any], Path | None]:
    baseline = estimate(model, observations, start_date=start_date)
    if not enabled:
        return {
            "enabled": False,
            "baseline_log_posterior": baseline.log_posterior,
            "parameters": {},
        }, None
    refreshed = estimate(
        model,
        observations,
        start_date=start_date,
        optimize=True,
        parameter_names=list(DEFAULT_REFRESH_PARAMETERS),
        optimizer_method="Powell",
        maxiter=maxiter,
    )
    if refreshed.optimization is None:
        raise RuntimeError("Targeted model refresh did not return an optimization result.")
    mode = estimation_mode_from_result(refreshed)
    mode_path = save_estimation_mode(mode, output_dir / "updated_shock_mode.npz")
    return {
        "enabled": True,
        "scope": "targeted MAP refresh of demand, MEI, and technology shock persistence/scale",
        "parameters": {
            name: refreshed.parameter_values[name] for name in DEFAULT_REFRESH_PARAMETERS
        },
        "baseline_log_posterior": baseline.log_posterior,
        "updated_log_posterior": refreshed.log_posterior,
        "objective_improvement": refreshed.log_posterior - baseline.log_posterior,
        "optimizer_success": mode.success,
        "optimizer_method": "Powell",
        "optimizer_message": mode.message,
        "iterations": mode.iterations,
        "function_evaluations": mode.function_evaluations,
    }, mode_path


def _structural_scenarios(
    model: Model1002,
    *,
    system: Any,
    start_state: np.ndarray,
    baseline_model: np.ndarray,
    baseline_report: np.ndarray,
    dates: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    definitions = (
        ("ai_investment_boom", "mu_sh", 1.0),
        ("ai_productivity_acceleration", "ztil_sh", 1.0),
        ("demand_contraction", "g_sh", -1.0),
    )
    irf = observable_irf(system, horizon=len(dates), normalization="one_sd")
    shock_names = list(model.indexes.exogenous_shocks)
    variable_names = list(model.observables)
    irf_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scenario, shock_name, sign in definitions:
        shock_index = shock_names.index(shock_name)
        shocks = np.zeros((len(dates), len(shock_names)), dtype=np.float64)
        shocks[0, shock_index] = sign * irf.shock_scales[shock_index]
        outcome = forecast_linear_system(
            system,
            start_state,
            horizon=len(dates),
            shocks=shocks,
        )
        reported = reverse_transform_observables(model, outcome.observables)
        forecast_rows.extend(
            _forecast_rows(
                scenario=scenario,
                dates=dates,
                unemployment=np.full(len(dates), np.nan),
                values=reported,
                baseline=baseline_report,
                variable_names=variable_names,
            )
        )
        summaries.append(
            _scenario_summary(
                scenario,
                target=None,
                values=reported,
                baseline=baseline_report,
                variable_names=variable_names,
                condition_error=None,
                shocks=shocks,
                shock_scales=irf.shock_scales,
            )
        )
        response = sign * irf.observables[:, :, shock_index]
        response_report = reverse_transform_observables(model, baseline_model + response)
        response_report -= baseline_report
        for step, date in enumerate(dates):
            for variable_index, variable in enumerate(variable_names):
                irf_rows.append(
                    {
                        "scenario": scenario,
                        "shock": shock_name,
                        "shock_sign": int(sign),
                        "horizon": step,
                        "date": date,
                        "variable": variable,
                        "response": response_report[step, variable_index],
                    }
                )
    return irf_rows, forecast_rows, summaries


def _forecast_rows(
    *,
    scenario: str,
    dates: list[str],
    unemployment: np.ndarray,
    values: np.ndarray,
    baseline: np.ndarray,
    variable_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, date in enumerate(dates):
        for variable_index, variable in enumerate(variable_names):
            rows.append(
                {
                    "scenario": scenario,
                    "date": date,
                    "horizon": step,
                    "unemployment_rate": unemployment[step],
                    "variable": variable,
                    "value": values[step, variable_index],
                    "baseline_value": baseline[step, variable_index],
                    "deviation": values[step, variable_index] - baseline[step, variable_index],
                }
            )
    return rows


def _scenario_summary(
    scenario: str,
    *,
    target: float | None,
    values: np.ndarray,
    baseline: np.ndarray,
    variable_names: list[str],
    condition_error: float | None,
    shocks: np.ndarray,
    shock_scales: np.ndarray,
) -> dict[str, Any]:
    def minimum_deviation(name: str) -> float:
        index = variable_names.index(name)
        return float(np.min(values[:, index] - baseline[:, index]))

    def maximum_deviation(name: str) -> float:
        index = variable_names.index(name)
        return float(np.max(values[:, index] - baseline[:, index]))

    positive_scales = shock_scales > 1.0e-12
    standardized = np.zeros_like(shocks)
    standardized[:, positive_scales] = shocks[:, positive_scales] / shock_scales[positive_scales]
    shock_norm = np.sqrt(np.sum(standardized**2, axis=1))
    return {
        "scenario": scenario,
        "unemployment_target": target,
        "gdp_growth_min_deviation_pp": minimum_deviation("obs_gdp"),
        "hours_min_deviation_log_points": minimum_deviation("obs_hours"),
        "wage_growth_min_deviation_pp": minimum_deviation("obs_wages"),
        "core_pce_min_deviation_pp": minimum_deviation("obs_corepce"),
        "consumption_growth_min_deviation_pp": minimum_deviation("obs_consumption"),
        "investment_growth_min_deviation_pp": minimum_deviation("obs_investment"),
        "policy_rate_min_deviation_pp": minimum_deviation("obs_nominalrate"),
        "spread_max_deviation_pp": maximum_deviation("obs_spread"),
        "max_quarterly_joint_shock_norm_sd": float(np.max(shock_norm)),
        "conditioning_max_abs_error": condition_error,
    }


def _forecast_unemployment_baseline(
    unemployment: pd.DataFrame,
    *,
    dates: list[str],
    estimation_start: str,
) -> np.ndarray:
    sample = _date_window(unemployment, start_date=estimation_start).dropna(subset=["UNRATE"])
    first_forecast = _period(dates[0])
    last_completed = _period(str(first_forecast - 1))
    known = sample.loc[_period_index(sample["date"]) <= last_completed]
    if known.empty:
        raise ValueError("No unemployment observation is available by the forecast start.")
    known_period = _period(str(known.iloc[-1]["date"]))
    level = float(known.iloc[-1]["UNRATE"])
    if known_period.ordinal < last_completed.ordinal:
        raise ValueError("The unemployment series is not current through the model cutoff.")
    return np.full(len(dates), level, dtype=np.float64)


def _ols_change_bridge(sample: pd.DataFrame) -> tuple[float, float, float]:
    x = sample["delta_unemployment"].to_numpy(dtype=np.float64)
    y = sample["delta_hours"].to_numpy(dtype=np.float64)
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ np.asarray([intercept, slope])
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 0.0 if total_sum == 0.0 else 1.0 - residual_sum / total_sum
    return float(intercept), float(slope), float(r_squared)


def _data_quality_summary(observables: pd.DataFrame, *, model_end_date: str) -> dict[str, Any]:
    variables = [column for column in observables.columns if column != "date"]
    all_missing = [column for column in variables if observables[column].notna().sum() == 0]
    latest_values = {
        column: (
            None
            if observables[column].dropna().empty
            else str(observables.loc[observables[column].last_valid_index(), "date"])
        )
        for column in variables
    }
    return {
        "rows": len(observables),
        "columns": len(variables),
        "first_date": str(observables.iloc[0]["date"]),
        "last_date": str(observables.iloc[-1]["date"]),
        "requested_model_end_date": model_end_date,
        "all_missing_observables": all_missing,
        "all_missing_count": len(all_missing),
        "latest_finite_date_by_observable": latest_values,
    }


def _make_study_plots(
    output_dir: Path,
    forecasts: pd.DataFrame,
    summary: pd.DataFrame,
    irfs: pd.DataFrame,
) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    unemployment = forecasts.loc[
        forecasts["variable"] == "obs_gdp",
        ["scenario", "date", "unemployment_rate"],
    ].drop_duplicates()
    unemployment = unemployment.loc[unemployment["scenario"].str.startswith("unemployment_")]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    unemployment["target"] = unemployment["scenario"].str.extract(r"([0-9.]+)pct").astype(float)
    unemployment = unemployment.sort_values(["target", "date"])
    target_groups = tuple(unemployment.groupby("target", sort=True))
    color_map = LinearSegmentedColormap.from_list(
        "study_blues",
        ("#c9dcf2", "#104b8f"),
    )
    colors = color_map(np.linspace(0.0, 1.0, len(target_groups)))
    for color, (target, group) in zip(colors, target_groups, strict=True):
        ax.plot(group["date"], group["unemployment_rate"], color=color, label=f"{target:g}%")
    ax.set_title(
        "Conditional unemployment paths\nFour-quarter ramp, hold, and return to 2026-Q2 baseline"
    )
    ax.set_ylabel("Percent of labor force")
    ax.set_xlabel("Quarter")
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=60)
    ax.legend(ncol=2, fontsize=8, title="Peak target")
    fig.tight_layout()
    path = figure_dir / "unemployment_paths.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    stress = summary.dropna(subset=["unemployment_target"]).sort_values("unemployment_target")
    metrics = (
        ("gdp_growth_min_deviation_pp", "GDP growth"),
        ("consumption_growth_min_deviation_pp", "Consumption growth"),
        ("investment_growth_min_deviation_pp", "Investment growth"),
        ("core_pce_min_deviation_pp", "Core PCE inflation"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        ax.plot(stress["unemployment_target"], stress[metric], marker="o", color="#2864dc")
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel("Worst deviation from baseline (pp)")
        ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Peak unemployment target (%)")
    fig.suptitle(
        "Worst macro deviations by unemployment target\n"
        "Percentage-point trough relative to the current-data DSGE baseline"
    )
    fig.tight_layout()
    path = figure_dir / "stress_summary.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    selected_variables = (
        "obs_gdp",
        "obs_investment",
        "obs_corepce",
        "obs_nominalrate",
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    scenario_colors = {
        "ai_investment_boom": "#2864dc",
        "ai_productivity_acceleration": "#7a55b3",
        "demand_contraction": "#333333",
    }
    variable_titles = {
        "obs_gdp": "GDP growth",
        "obs_investment": "Investment growth",
        "obs_corepce": "Core PCE inflation",
        "obs_nominalrate": "Policy rate",
    }
    for ax, variable in zip(axes.ravel(), selected_variables, strict=True):
        subset = irfs.loc[irfs["variable"] == variable]
        for scenario, group in subset.groupby("scenario", sort=False):
            ax.plot(
                group["horizon"],
                group["response"],
                color=scenario_colors[scenario],
                label=scenario.replace("_", " "),
            )
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(variable_titles[variable])
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Quarters after shock")
    fig.suptitle(
        "Selected structural impulse responses\n"
        "One-standard-deviation shock; reporting-unit deviation from baseline"
    )
    fig.tight_layout()
    path = figure_dir / "selected_irfs.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return tuple(paths)


def _study_report_markdown(
    metadata: dict[str, Any],
    summary: pd.DataFrame,
    figures: tuple[Path, ...],
) -> str:
    bridge = metadata["bridge"]
    refresh = metadata["model"]["refresh"]
    target_count = len(metadata["scenarios"]["unemployment_targets_percent"])
    stress = summary.dropna(subset=["unemployment_target"]).copy()
    display_columns = [
        "unemployment_target",
        "gdp_growth_min_deviation_pp",
        "consumption_growth_min_deviation_pp",
        "investment_growth_min_deviation_pp",
        "core_pce_min_deviation_pp",
        "max_quarterly_joint_shock_norm_sd",
    ]
    table = _markdown_table(stress[display_columns].round(3))
    figure_links = "\n".join(f"![{path.stem}](figures/{path.name})" for path in figures)
    missing = metadata["data"]["quality"]["all_missing_observables"]
    if refresh.get("enabled", False):
        refresh_text = (
            "The targeted shock-block refresh improved the log posterior by "
            f"{refresh.get('objective_improvement', 0.0):.2f}; optimizer success was "
            f"`{refresh.get('optimizer_success', False)}`."
        )
    else:
        refresh_text = "The targeted shock-block refresh was disabled for this run."
    return f"""# AI, Labor Displacement, and U.S. Macro Scenarios

## Technical summary

This study updates the local Model1002 information set through
{metadata["data"]["model_data_end"]} and evaluates {target_count} unemployment stress paths plus
three structural shocks associated with AI investment, productivity, and aggregate demand.
The unemployment cases are conditional paths, not causal AI estimates. They translate
unemployment gaps into the model's hours observable using a historical quarterly bridge and
then solve for the minimum-norm structural shock sequence consistent with that labor path.

The local model is the validated `ss10` Python parity target, checked against upstream DSGE.jl
    release `{metadata["model"]["reference_dsge_version"]}` (tree
    `{metadata["model"]["reference_dsge_git_tree"][:12]}`). It is not the private March 2026
NY Fed production model. {refresh_text}

## Stress results

{table}

The 10% and 15% unemployment cases are extreme linear-model stress tests. Their joint shock
norms should be read as an extrapolation warning, not as calibrated forecast probabilities.

{figure_links}

## Scope, data, and definitions

- Model data: public FRED series from {metadata["data"]["model_data_start"]} through
  {metadata["data"]["model_data_end"]}.
- Latest unemployment auxiliary quarter: {metadata["data"]["unemployment_last_date"]}.
- Unemployment baseline: latest complete-quarter rate held constant over the horizon.
- Unemployment path: four-quarter ramp, four-quarter hold, eight-quarter return.
- Bridge: Δ log hours per capita = intercept + slope × Δ unemployment; slope
  `{bridge["slope"]:.3f}`, R² `{bridge["r_squared"]:.3f}`, n = `{bridge["observations"]}`.
- Entirely unavailable current public observables are preserved as missing rather than filled
  with non-equivalent proxies: {", ".join(missing)}.

## Methodology

The state is filtered and smoothed using all available current observables. The demand
contraction is a negative one-standard-deviation `g_sh`; the AI capital-deepening case is a
positive one-standard-deviation marginal-efficiency-of-investment `mu_sh`; and the AI
productivity case is a positive one-standard-deviation neutral technology `ztil_sh`.
Structural IRFs use the model's estimated shock covariance. Scenario forecast values are in
the reporting units declared by Model1002.

## Limitations and robustness

The model is linear, representative-agent, and does not contain unemployment, worker
heterogeneity, sectoral reallocation, adoption lags, or an identified AI shock. The
unemployment-hours bridge is descriptive. Post-2020 observations materially influence it;
the saved metadata includes pre-COVID and post-1990 sensitivity slopes. Missing survey,
Fernald TFP, and expected-rate series reduce the current-state information set. Parameter
uncertainty is not propagated through the deterministic stress paths.

## Recommended next steps

Treat 5-7% as conditional planning cases and 8-15% as tail stresses. A publication-quality
causal AI study should add sector/occupation exposure data, identify an adoption or investment
instrument, and estimate heterogeneous labor-market transitions outside this representative-
agent DSGE.

## Further questions

The next useful extensions are probabilistic conditional bands, a sectoral labor block,
alternative unemployment-hours bridges, and a model comparison against a BVAR/local-
projection benchmark.
"""


def _date_window(
    frame: pd.DataFrame,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    periods = _period_index(frame["date"])
    mask = np.ones(len(frame), dtype=bool)
    if start_date is not None:
        mask &= periods >= _period(start_date)
    if end_date is not None:
        mask &= periods <= _period(end_date)
    return frame.loc[mask].sort_values("date").reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _period(value: str) -> pd.Period:
    period = pd.Period(str(value).replace("-Q", "Q"), freq="Q")
    if not isinstance(period, pd.Period):
        raise ValueError(f"Invalid quarterly date: {value}")
    return period


def _period_index(values: pd.Series[Any]) -> pd.PeriodIndex:
    return pd.PeriodIndex([_period(str(value)) for value in values], freq="Q")


def _next_quarter(value: str) -> str:
    return _quarter_label(_period(value) + 1)


def _quarter_sequence(start: str, horizon: int) -> list[str]:
    first = _period(start)
    return [_quarter_label(first + offset) for offset in range(horizon)]


def _quarter_label(value: pd.Period) -> str:
    return f"{value.year}-Q{value.quarter}"
