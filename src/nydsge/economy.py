"""Quarterly U.S. macro projection and scenario package.

The package deliberately separates:

* DSGE structural scenarios and historical shock decompositions;
* statistical CPI basket contribution accounting; and
* unemployment paths imposed through an external hours/unemployment bridge.

That separation prevents accounting categories, identified model shocks, and
external conditioning assumptions from being presented as interchangeable.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nydsge.data import (
    FredFetcher,
    df_to_matrix,
    load_current_fred_series,
    parse_data_sources,
    reverse_transform_observables,
    reverse_transform_pseudo_observables,
)
from nydsge.estimate import estimate, estimation_mode_from_result, save_estimation_mode
from nydsge.forecast import (
    forecast_linear_system,
    forecast_linear_system_samples,
    forecast_one,
    historical_decomposition,
    solve_shocks_for_observable_targets,
)
from nydsge.models import Model1002
from nydsge.scenarios import (
    DEFAULT_REFRESH_PARAMETERS,
    build_unemployment_scenario_path,
    current_public_observables,
    estimate_unemployment_hours_bridge,
)
from nydsge.solve import System, compute_system

FIGURE_DPI = 170
REFERENCE_DSGE_VERSION = "1.3.0"
REFERENCE_DSGE_TREE = "e746a4a5ab9c26d897239e722b0f19d4bb3bd77e"

SHOCK_GROUPS: dict[str, tuple[str, ...]] = {
    "Government spending": ("g_sh",),
    "Private demand": ("b_sh",),
    "Productivity and supply": ("ztil_sh", "zp_sh"),
    "Price and wage markups": ("lambda_f_sh", "lambda_w_sh"),
    "Monetary policy": (
        "rm_sh",
        "rm_shl1",
        "rm_shl2",
        "rm_shl3",
        "rm_shl4",
        "rm_shl5",
        "rm_shl6",
    ),
    "Inflation target": ("pi_star_sh",),
    "Investment and finance": (
        "mu_sh",
        "sigma_omega_sh",
        "mu_e_sh",
        "gamma_sh",
    ),
    "Measurement innovations": (
        "lr_sh",
        "tfp_sh",
        "gdpdef_sh",
        "corepce_sh",
        "gdp_sh",
        "gdi_sh",
    ),
}

GROUP_COLORS = {
    "Initial conditions and trend": "#C8CDD3",
    "Government spending": "#E7B64A",
    "Private demand": "#D9822B",
    "Productivity and supply": "#6E9E45",
    "Price and wage markups": "#A65A52",
    "Monetary policy": "#34699A",
    "Inflation target": "#8FB8D8",
    "Investment and finance": "#A56CC1",
    "Measurement innovations": "#7A8088",
    "Observed-model residual": "#3F454C",
}

CPI_DRIVER_COLUMNS = {
    "Food": "food_effect",
    "Energy": "energy_effect",
    "Core goods": "core_goods_effect",
    "Shelter": "shelter_effect",
    "Other core services": "other_core_services_effect",
    "BLS rounding residual": "rounding_residual",
}

CPI_COLORS = {
    "Food": "#E7B64A",
    "Energy": "#D9822B",
    "Core goods": "#4C78A8",
    "Shelter": "#B279A2",
    "Other core services": "#59A14F",
    "BLS rounding residual": "#B9BDC5",
}


@dataclass(frozen=True)
class ShockComponent:
    shock: str
    size_sd: float
    start: int = 0
    duration: int = 1
    decay: float = 1.0


@dataclass(frozen=True)
class StructuralScenario:
    name: str
    label: str
    components: tuple[ShockComponent, ...]


@dataclass(frozen=True)
class PolicyScenario:
    name: str
    label: str
    rate_deviation_pp: tuple[float, ...]


@dataclass(frozen=True)
class QuarterlyEconomyConfig:
    start_date: str
    model_end_date: str
    horizon: int
    stochastic_draws: int
    seed: int
    historical_tail_quarters: int
    refresh_model: bool
    refresh_maxiter: int
    fred_levels_path: Path | None
    cpi_summary_path: Path | None
    cpi_detail_path: Path | None
    cpi_goods_path: Path | None
    unemployment_targets: tuple[float, ...]
    unemployment_bridge_start: str
    policy_scenarios: tuple[PolicyScenario, ...]
    structural_scenarios: tuple[StructuralScenario, ...]
    source_path: Path
    source_sha256: str


@dataclass(frozen=True)
class QuarterlyEconomyArtifacts:
    output_dir: Path
    report: Path
    metadata: Path
    baseline_forecast: Path
    scenario_forecast: Path
    scenario_summary: Path
    historical_decomposition: Path
    cpi_decomposition: Path | None
    model_mode: Path | None
    figures: tuple[Path, ...]


def load_quarterly_economy_config(path: Path | str) -> QuarterlyEconomyConfig:
    """Load and strictly validate a quarterly package JSON configuration."""

    source_path = Path(path).resolve()
    raw_bytes = source_path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("Quarterly economy configuration must be a JSON object.")

    base = source_path.parent

    def optional_path(name: str) -> Path | None:
        value = raw.get(name)
        if value in {None, ""}:
            return None
        candidate = Path(str(value))
        return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    policy_scenarios = tuple(
        PolicyScenario(
            name=str(item["name"]),
            label=str(item.get("label", item["name"])),
            rate_deviation_pp=tuple(float(value) for value in item["rate_deviation_pp"]),
        )
        for item in _mapping_list(raw, "policy_scenarios")
    )
    structural_scenarios = tuple(
        StructuralScenario(
            name=str(item["name"]),
            label=str(item.get("label", item["name"])),
            components=tuple(
                ShockComponent(
                    shock=str(component["shock"]),
                    size_sd=float(component["size_sd"]),
                    start=int(component.get("start", 0)),
                    duration=int(component.get("duration", 1)),
                    decay=float(component.get("decay", 1.0)),
                )
                for component in _mapping_list(item, "components")
            ),
        )
        for item in _mapping_list(raw, "structural_scenarios")
    )
    config = QuarterlyEconomyConfig(
        start_date=_quarter_label(_period(str(raw.get("start_date", "1964-Q1")))),
        model_end_date=_quarter_label(_period(str(raw["model_end_date"]))),
        horizon=int(raw.get("horizon", 20)),
        stochastic_draws=int(raw.get("stochastic_draws", 500)),
        seed=int(raw.get("seed", 0)),
        historical_tail_quarters=int(raw.get("historical_tail_quarters", 24)),
        refresh_model=bool(raw.get("refresh_model", False)),
        refresh_maxiter=int(raw.get("refresh_maxiter", 60)),
        fred_levels_path=optional_path("fred_levels_path"),
        cpi_summary_path=optional_path("cpi_summary_path"),
        cpi_detail_path=optional_path("cpi_detail_path"),
        cpi_goods_path=optional_path("cpi_goods_path"),
        unemployment_targets=tuple(float(value) for value in raw.get("unemployment_targets", [])),
        unemployment_bridge_start=_quarter_label(
            _period(str(raw.get("unemployment_bridge_start", "1985-Q1")))
        ),
        policy_scenarios=policy_scenarios,
        structural_scenarios=structural_scenarios,
        source_path=source_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
    _validate_config(config)
    return config


def run_quarterly_economy_package(
    *,
    config_path: Path | str,
    output_dir: Path | str,
    fetcher: FredFetcher | None = None,
    make_plots: bool = True,
) -> QuarterlyEconomyArtifacts:
    """Run the configured data update, forecast, scenarios, and decompositions."""

    config = load_quarterly_economy_config(config_path)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    forecast_start = _quarter_label(_period(config.model_end_date) + 1)
    model = Model1002(
        subspec="ss10",
        settings={
            "data_vintage": config.model_end_date.replace("-Q", "q"),
            "date_forecast_start": forecast_start,
        },
    )

    fred_names = parse_data_sources(model).get("FRED", [])
    fred_levels = _load_fred_levels(config, fred_names, fetcher=fetcher)
    observables = current_public_observables(model, fred_levels)
    data_quality = assess_quarterly_data_quality(
        observables,
        start_date=config.start_date,
        model_end_date=config.model_end_date,
    )
    fred_path = destination / "fred_levels.csv"
    observables_path = destination / "observables.csv"
    fred_levels.to_csv(fred_path, index=False)
    observables.to_csv(observables_path, index=False)

    observations = df_to_matrix(model, observables)
    refresh_metadata, model_mode = _refresh_model(
        model,
        observations,
        output_dir=destination,
        start_date=config.start_date,
        enabled=config.refresh_model,
        maxiter=config.refresh_maxiter,
    )
    system = compute_system(model)
    baseline = forecast_one(
        model,
        input_type="mode",
        cond_type="none",
        output_vars=[
            "histstates",
            "histobs",
            "histpseudo",
            "forecastobs",
            "forecastpseudo",
        ],
        check_empty_columns=False,
        horizon=config.horizon,
        data=observables,
        history_method="smoothed",
    )
    if baseline.history_states is None or baseline.history_states.shape[0] == 0:
        raise RuntimeError("Current-data filtering did not produce a terminal model state.")
    if baseline.pseudo_observables is None:
        raise RuntimeError("Baseline forecast did not produce pseudo-observables.")
    start_state = baseline.history_states[-1]
    dates = _quarter_sequence(forecast_start, config.horizon)
    baseline_obs_model = np.asarray(baseline.observables, dtype=np.float64)
    baseline_pseudo_model = np.asarray(baseline.pseudo_observables, dtype=np.float64)
    baseline_obs = reverse_transform_observables(model, baseline_obs_model)
    baseline_pseudo = reverse_transform_pseudo_observables(model, baseline_pseudo_model)

    bands = _stochastic_baseline_bands(
        model,
        system,
        start_state,
        horizon=config.horizon,
        draws=config.stochastic_draws,
        seed=config.seed,
    )
    baseline_frame = _baseline_forecast_frame(
        model,
        dates=dates,
        observables=baseline_obs,
        pseudo_observables=baseline_pseudo,
        bands=bands,
    )
    baseline_path = destination / "baseline_forecast_all_variables.csv"
    baseline_frame.to_csv(baseline_path, index=False)
    _write_baseline_wide(destination, dates, model, baseline_obs, baseline_pseudo)
    _write_history(destination, model, observables, baseline)

    unemployment, unemployment_baseline, bridge = _unemployment_inputs(
        config,
        observables,
        dates=dates,
        fetcher=fetcher,
    )
    unemployment.to_csv(destination / "unemployment.csv", index=False)

    scenario_frame, scenario_summary, scenario_shocks = _run_scenarios(
        config,
        model,
        system,
        start_state,
        dates=dates,
        baseline_obs_model=baseline_obs_model,
        baseline_pseudo_model=baseline_pseudo_model,
        baseline_obs=baseline_obs,
        baseline_pseudo=baseline_pseudo,
        unemployment_baseline=unemployment_baseline,
        bridge_slope=bridge.slope,
    )
    scenario_path = destination / "scenario_forecast_all_variables.csv"
    summary_path = destination / "scenario_summary.csv"
    scenario_frame.to_csv(scenario_path, index=False)
    scenario_summary.to_csv(summary_path, index=False)
    _save_scenario_shocks(destination / "scenario_shocks.npz", scenario_shocks)

    historical_frame, historical_quality = _historical_group_decomposition(
        model,
        system,
        observables,
        tail_quarters=config.historical_tail_quarters,
    )
    historical_path = destination / "historical_decomposition_grouped.csv"
    historical_frame.to_csv(historical_path, index=False)

    cpi_frame: pd.DataFrame | None = None
    cpi_detail: pd.DataFrame | None = None
    cpi_goods: pd.DataFrame | None = None
    cpi_quality: dict[str, Any] | None = None
    cpi_path: Path | None = None
    if config.cpi_summary_path is not None:
        cpi_frame, cpi_detail, cpi_goods, cpi_quality = build_cpi_accounting(
            config.cpi_summary_path,
            detail_path=config.cpi_detail_path,
            goods_path=config.cpi_goods_path,
            expected_latest_quarter=config.model_end_date,
        )
        cpi_path = destination / "cpi_decomposition.csv"
        cpi_frame.to_csv(cpi_path, index=False)
        if cpi_detail is not None:
            cpi_detail.to_csv(destination / "cpi_detail_decomposition.csv", index=False)
        if cpi_goods is not None:
            cpi_goods.to_csv(destination / "cpi_goods_latest.csv", index=False)

    figures = (
        _make_figures(
            destination,
            model=model,
            baseline=baseline_frame,
            scenarios=scenario_frame,
            historical=historical_frame,
            cpi=cpi_frame,
            cpi_detail=cpi_detail,
        )
        if make_plots
        else ()
    )
    metadata_payload = {
        "package": "quarterly_economy",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "configuration": {
            **_config_manifest(config),
            "path": str(config.source_path),
            "sha256": config.source_sha256,
        },
        "model": {
            "spec": model.spec,
            "subspec": model.subspec,
            "forecast_start": forecast_start,
            "reference_implementation": {
                "package": "DSGE.jl",
                "version": REFERENCE_DSGE_VERSION,
                "git_tree_sha1": REFERENCE_DSGE_TREE,
                "scope": "Model1002 ss10 parity target",
            },
            "production_model_equivalence": False,
            "refresh": refresh_metadata,
        },
        "data": {
            "fred_series": fred_names,
            "fred_levels_path": str(fred_path),
            "observables_path": str(observables_path),
            "quality": data_quality,
        },
        "baseline": {
            "kind": "conditional mean under zero future structural shocks",
            "stochastic_bands": {
                "draws": config.stochastic_draws,
                "seed": config.seed,
                "lower_quantile": 0.05,
                "upper_quantile": 0.95,
                "parameter_uncertainty_included": False,
            },
        },
        "scenarios": {
            "policy_conditioning": (
                "Exact obs_nominalrate deviation path solved using rm_sh innovations only"
            ),
            "structural_units": "multiples of model shock standard deviations",
            "stress_flag_threshold_sd": 4.0,
            "extreme_flag_threshold_sd": 8.0,
            "unemployment": {
                "targets_percent": list(config.unemployment_targets),
                "bridge": asdict(bridge),
                "interpretation": (
                    "External descriptive hours-unemployment bridge; unemployment is not "
                    "a Model1002 observable."
                ),
            },
        },
        "historical_decomposition": historical_quality,
        "cpi_accounting": cpi_quality,
        "artifacts": {
            "baseline_forecast": str(baseline_path),
            "scenario_forecast": str(scenario_path),
            "scenario_summary": str(summary_path),
            "historical_decomposition": str(historical_path),
            "cpi_decomposition": None if cpi_path is None else str(cpi_path),
            "figures": [str(path) for path in figures],
        },
    }
    metadata_path = destination / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    report_path = destination / "quarterly_economic_report.md"
    report_path.write_text(
        _report_markdown(
            metadata_payload,
            baseline_frame,
            scenario_summary,
            historical_frame,
            cpi_frame,
            figures,
        ),
        encoding="utf-8",
    )
    return QuarterlyEconomyArtifacts(
        output_dir=destination,
        report=report_path,
        metadata=metadata_path,
        baseline_forecast=baseline_path,
        scenario_forecast=scenario_path,
        scenario_summary=summary_path,
        historical_decomposition=historical_path,
        cpi_decomposition=cpi_path,
        model_mode=model_mode,
        figures=figures,
    )


def assess_quarterly_data_quality(
    observables: pd.DataFrame,
    *,
    start_date: str,
    model_end_date: str,
) -> dict[str, Any]:
    """Fail closed on date-grid defects and report observable availability."""

    if "date" not in observables:
        raise ValueError("Observable data must include a date column.")
    periods = _period_index(observables["date"])
    if periods.has_duplicates:
        raise ValueError("Observable data contain duplicate quarterly dates.")
    expected = pd.period_range(_period(start_date), _period(model_end_date), freq="Q")
    if not periods.equals(expected):
        missing = [str(period) for period in expected.difference(periods)]
        extra = [str(period) for period in periods.difference(expected)]
        raise ValueError(
            "Observable quarterly grid does not match the configured sample; "
            f"missing={missing}, extra={extra}."
        )
    variables = [column for column in observables if column != "date"]
    latest_finite: dict[str, str | None] = {}
    missing_at_end: list[str] = []
    all_missing: list[str] = []
    for variable in variables:
        finite = observables[variable].notna()
        if not finite.any():
            all_missing.append(variable)
            latest_finite[variable] = None
        else:
            latest_finite[variable] = str(observables.loc[finite, "date"].iloc[-1])
        if pd.isna(observables.iloc[-1][variable]):
            missing_at_end.append(variable)
    observed_cells = int(observables[variables].notna().to_numpy().sum())
    total_cells = int(observables[variables].size)
    return {
        "status": "partial" if missing_at_end else "complete",
        "rows": len(observables),
        "variables": len(variables),
        "first_date": str(observables.iloc[0]["date"]),
        "last_date": str(observables.iloc[-1]["date"]),
        "observed_cell_share": observed_cells / total_cells,
        "all_missing_observables": all_missing,
        "missing_at_model_end": missing_at_end,
        "latest_finite_date_by_observable": latest_finite,
    }


def build_structural_shock_path(
    scenario: StructuralScenario,
    *,
    shock_names: Sequence[str],
    shock_scales: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Compose a scenario from timed, decaying structural shock components."""

    shocks = np.zeros((horizon, len(shock_names)), dtype=np.float64)
    name_to_index = {name: index for index, name in enumerate(shock_names)}
    for component in scenario.components:
        if component.shock not in name_to_index:
            raise ValueError(
                f"Scenario '{scenario.name}' references unknown shock '{component.shock}'."
            )
        shock_index = name_to_index[component.shock]
        for offset in range(component.duration):
            step = component.start + offset
            if step >= horizon:
                break
            shocks[step, shock_index] += (
                component.size_sd * component.decay**offset * float(shock_scales[shock_index])
            )
    return shocks


def build_cpi_accounting(
    summary_path: Path | str,
    *,
    detail_path: Path | str | None = None,
    goods_path: Path | str | None = None,
    expected_latest_quarter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any]]:
    """Build reconciled CPI contribution accounting from BLS Table 7 snapshots."""

    summary = pd.read_csv(summary_path)
    required = {
        "quarter",
        "source_url",
        "headline_cpi_yoy",
        "food_effect",
        "energy_effect",
        "core_goods_effect",
        "services_ex_energy_effect",
        "shelter_effect",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"CPI summary is missing required columns: {missing}.")
    _validate_quarter_frame(summary, "CPI summary")
    numeric_columns = sorted(required.difference({"quarter", "source_url"}))
    if summary[numeric_columns].isna().any().any():
        raise ValueError("CPI summary contains missing required numeric values.")
    wide = summary.copy()
    wide["other_core_services_effect"] = wide["services_ex_energy_effect"] - wide["shelter_effect"]
    published_sum = wide[
        [
            "food_effect",
            "energy_effect",
            "core_goods_effect",
            "services_ex_energy_effect",
        ]
    ].sum(axis=1)
    wide["rounding_residual"] = wide["headline_cpi_yoy"] - published_sum
    wide["reconciliation_error"] = (
        wide[list(CPI_DRIVER_COLUMNS.values())].sum(axis=1) - wide["headline_cpi_yoy"]
    )
    max_error = float(wide["reconciliation_error"].abs().max())
    max_rounding = float(wide["rounding_residual"].abs().max())
    if max_error > 1.0e-10:
        raise ValueError(f"CPI accounting fails reconciliation: {max_error:.3e}.")
    if max_rounding > 0.10:
        raise ValueError(f"CPI rounding residual is unexpectedly large: {max_rounding:.3f}.")
    latest = str(wide.iloc[-1]["quarter"])
    if expected_latest_quarter is not None and _period(latest) != _period(expected_latest_quarter):
        raise ValueError(
            "CPI snapshot is stale relative to the configured model end: "
            f"{latest} != {expected_latest_quarter}."
        )

    detail: pd.DataFrame | None = None
    detail_quality: dict[str, float] = {}
    if detail_path is not None:
        detail = pd.read_csv(detail_path)
        _validate_quarter_frame(detail, "CPI detail")
        if list(detail["quarter"].astype(str)) != list(wide["quarter"].astype(str)):
            raise ValueError("CPI detail quarters do not match the summary snapshot.")
        detail = _build_cpi_detail(detail, wide)
        detail_quality = {
            "max_abs_food_reconciliation_error_pp": float(
                detail["food_reconciliation_error"].abs().max()
            ),
            "max_abs_shelter_reconciliation_error_pp": float(
                detail["shelter_reconciliation_error"].abs().max()
            ),
            "max_abs_other_services_reconciliation_error_pp": float(
                detail["other_core_services_reconciliation_error"].abs().max()
            ),
        }
        if max(detail_quality.values(), default=0.0) > 1.0e-10:
            raise ValueError("Detailed CPI component accounting fails reconciliation.")

    goods: pd.DataFrame | None = None
    goods_control_error: float | None = None
    if goods_path is not None:
        goods = pd.read_csv(goods_path)
        goods_required = {
            "category",
            "group",
            "relative_importance",
            "yoy_pct",
            "contribution_pp",
            "source_url",
        }
        missing_goods = sorted(goods_required.difference(goods.columns))
        if missing_goods:
            raise ValueError(f"CPI goods detail is missing columns: {missing_goods}.")
        goods_numeric = list(goods_required.difference({"category", "group", "source_url"}))
        if goods[goods_numeric].isna().any().any():
            raise ValueError("CPI goods detail contains missing numeric values.")
        control = float(
            goods.loc[goods["group"].isin(["Food goods", "Energy goods"]), "contribution_pp"].sum()
            + wide.iloc[-1]["core_goods_effect"]
        )
        goods_control_error = float(goods["contribution_pp"].sum() - control)
        if abs(goods_control_error) > 0.01:
            raise ValueError(
                f"CPI goods detail does not match its control total: {goods_control_error:.3f}."
            )

    quality: dict[str, Any] = {
        "status": "reconciled",
        "method": "BLS CPI-U Table 7 effect-on-All-items accounting",
        "interpretation": "Statistical basket contributions, not causal policy shocks",
        "first_quarter": str(wide.iloc[0]["quarter"]),
        "latest_quarter": latest,
        "observations": len(wide),
        "max_abs_reconciliation_error_pp": max_error,
        "max_abs_rounding_residual_pp": max_rounding,
        "goods_control_error_pp": goods_control_error,
        **detail_quality,
    }
    return wide, detail, goods, quality


def _mapping_list(mapping: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    value = mapping.get(name, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Configuration field '{name}' must be a list of objects.")
    return value


def _validate_config(config: QuarterlyEconomyConfig) -> None:
    if config.horizon < 4:
        raise ValueError("Quarterly economy horizon must be at least four quarters.")
    if config.stochastic_draws < 0:
        raise ValueError("stochastic_draws must be nonnegative.")
    if config.historical_tail_quarters < 4:
        raise ValueError("historical_tail_quarters must be at least four.")
    if config.refresh_maxiter <= 0:
        raise ValueError("refresh_maxiter must be positive.")
    names = [scenario.name for scenario in (*config.policy_scenarios, *config.structural_scenarios)]
    if len(names) != len(set(names)) or "baseline" in names:
        raise ValueError("Scenario names must be unique and cannot be 'baseline'.")
    for scenario in config.policy_scenarios:
        if not scenario.rate_deviation_pp:
            raise ValueError(f"Policy scenario '{scenario.name}' has an empty rate path.")
        if len(scenario.rate_deviation_pp) > config.horizon:
            raise ValueError(f"Policy scenario '{scenario.name}' exceeds the forecast horizon.")
        if not np.isfinite(scenario.rate_deviation_pp).all():
            raise ValueError(f"Policy scenario '{scenario.name}' contains a nonfinite rate.")
    for scenario in config.structural_scenarios:
        if not scenario.components:
            raise ValueError(f"Structural scenario '{scenario.name}' has no components.")
        for component in scenario.components:
            if component.start < 0 or component.duration < 1:
                raise ValueError(
                    f"Scenario '{scenario.name}' component timing must be nonnegative."
                )
            if not np.isfinite([component.size_sd, component.decay]).all():
                raise ValueError(f"Scenario '{scenario.name}' contains a nonfinite component.")
    for path in (
        config.fred_levels_path,
        config.cpi_summary_path,
        config.cpi_detail_path,
        config.cpi_goods_path,
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)


def _load_fred_levels(
    config: QuarterlyEconomyConfig,
    fred_names: Sequence[str],
    *,
    fetcher: FredFetcher | None,
) -> pd.DataFrame:
    if config.fred_levels_path is None:
        grid = pd.DataFrame(
            {
                "date": [
                    _quarter_label(period)
                    for period in pd.period_range(
                        _period(config.start_date),
                        _period(config.model_end_date),
                        freq="Q",
                    )
                ]
            }
        )
        for mnemonic in fred_names:
            series = load_current_fred_series(mnemonic, fetcher=fetcher)
            if series.empty:
                grid[mnemonic] = np.nan
                continue
            _validate_ragged_source(mnemonic, series)
            periods = _period_index(series["date"])
            selected = series.loc[
                (periods >= _period(config.start_date))
                & (periods <= _period(config.model_end_date)),
                ["date", mnemonic],
            ]
            grid = grid.merge(selected, on="date", how="left", validate="one_to_one")
        return grid
    frame = pd.read_csv(config.fred_levels_path)
    if "date" not in frame:
        raise ValueError("Configured FRED levels snapshot must include a date column.")
    periods = _period_index(frame["date"])
    mask = (periods >= _period(config.start_date)) & (periods <= _period(config.model_end_date))
    return frame.loc[mask].sort_values("date").reset_index(drop=True)


def _validate_ragged_source(mnemonic: str, series: pd.DataFrame) -> None:
    if "date" not in series or mnemonic not in series:
        raise ValueError(f"FRED:{mnemonic} response is missing required columns.")
    periods = _period_index(series["date"])
    if periods.has_duplicates:
        raise ValueError(f"FRED:{mnemonic} response contains duplicate quarters.")
    finite = series[mnemonic].notna().to_numpy()
    if not finite.any():
        return
    finite_indexes = np.flatnonzero(finite)
    first = int(finite_indexes[0])
    last = int(finite_indexes[-1])
    if not finite[first : last + 1].all():
        window = series.loc[first:last]
        missing = list(window.loc[window[mnemonic].isna(), "date"].astype(str))
        raise ValueError(f"FRED:{mnemonic} contains interior missing quarterly values: {missing}.")


def _refresh_model(
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
            "scope": "fixed published ss10 parameterization",
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
        "scope": (
            "Targeted MAP refresh of persistence and scale for government-spending, "
            "private-demand, MEI, and neutral-technology shocks"
        ),
        "certification": (
            "Python local estimation; equations and fixed-parameter model are Julia-parity "
            "validated, but this optimizer result is not a Julia-estimation oracle."
        ),
        "parameters": {
            name: refreshed.parameter_values[name] for name in DEFAULT_REFRESH_PARAMETERS
        },
        "baseline_log_posterior": baseline.log_posterior,
        "updated_log_posterior": refreshed.log_posterior,
        "objective_improvement": refreshed.log_posterior - baseline.log_posterior,
        "optimizer_success": mode.success,
        "optimizer_message": mode.message,
        "iterations": mode.iterations,
        "function_evaluations": mode.function_evaluations,
    }, mode_path


def _stochastic_baseline_bands(
    model: Model1002,
    system: System,
    start_state: np.ndarray,
    *,
    horizon: int,
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if draws == 0:
        empty_obs = np.full((horizon, len(model.observables)), np.nan)
        empty_pseudo = np.full((horizon, len(model.pseudo_observable_mappings)), np.nan)
        return {
            "obs_lower": empty_obs,
            "obs_upper": empty_obs.copy(),
            "pseudo_lower": empty_pseudo,
            "pseudo_upper": empty_pseudo.copy(),
        }
    sampled = forecast_linear_system_samples(
        system,
        start_state,
        horizon=horizon,
        draws=draws,
        seed=seed,
        include_pseudo=True,
    )
    if sampled.observable_samples is None or sampled.pseudo_observable_samples is None:
        raise RuntimeError("Stochastic baseline did not produce forecast samples.")
    obs_shape = sampled.observable_samples.shape
    pseudo_shape = sampled.pseudo_observable_samples.shape
    obs_report = reverse_transform_observables(
        model, sampled.observable_samples.reshape(-1, obs_shape[-1])
    ).reshape(obs_shape)
    pseudo_report = reverse_transform_pseudo_observables(
        model, sampled.pseudo_observable_samples.reshape(-1, pseudo_shape[-1])
    ).reshape(pseudo_shape)
    return {
        "obs_lower": np.quantile(obs_report, 0.05, axis=0),
        "obs_upper": np.quantile(obs_report, 0.95, axis=0),
        "pseudo_lower": np.quantile(pseudo_report, 0.05, axis=0),
        "pseudo_upper": np.quantile(pseudo_report, 0.95, axis=0),
    }


def _baseline_forecast_frame(
    model: Model1002,
    *,
    dates: Sequence[str],
    observables: np.ndarray,
    pseudo_observables: np.ndarray,
    bands: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for kind, names, values, lower, upper in (
        (
            "observable",
            list(model.observables),
            observables,
            bands["obs_lower"],
            bands["obs_upper"],
        ),
        (
            "pseudo_observable",
            list(model.pseudo_observable_mappings),
            pseudo_observables,
            bands["pseudo_lower"],
            bands["pseudo_upper"],
        ),
    ):
        for step, date in enumerate(dates):
            for index, name in enumerate(names):
                rows.append(
                    {
                        "date": date,
                        "horizon": step,
                        "variable_kind": kind,
                        "variable": name,
                        "baseline": float(values[step, index]),
                        "p05": float(lower[step, index]),
                        "p95": float(upper[step, index]),
                    }
                )
    return pd.DataFrame(rows)


def _write_baseline_wide(
    destination: Path,
    dates: Sequence[str],
    model: Model1002,
    observables: np.ndarray,
    pseudo_observables: np.ndarray,
) -> None:
    pd.DataFrame(observables, columns=list(model.observables)).assign(date=list(dates))[
        ["date", *model.observables]
    ].to_csv(destination / "baseline_observables_wide.csv", index=False)
    pseudo_names = list(model.pseudo_observable_mappings)
    pd.DataFrame(pseudo_observables, columns=pseudo_names).assign(date=list(dates))[
        ["date", *pseudo_names]
    ].to_csv(destination / "baseline_pseudo_observables_wide.csv", index=False)


def _write_history(
    destination: Path,
    model: Model1002,
    observables: pd.DataFrame,
    baseline: Any,
) -> None:
    if baseline.history_observables is None:
        raise RuntimeError("Baseline run did not return observable history.")
    history_obs = reverse_transform_observables(model, baseline.history_observables)
    dates = list(observables["date"].astype(str))[-history_obs.shape[0] :]
    pd.DataFrame(history_obs, columns=list(model.observables)).assign(date=dates)[
        ["date", *model.observables]
    ].to_csv(destination / "history_observables_reporting_units.csv", index=False)
    if baseline.history_pseudo_observables is not None:
        history_pseudo = reverse_transform_pseudo_observables(
            model, baseline.history_pseudo_observables
        )
        names = list(model.pseudo_observable_mappings)
        pd.DataFrame(history_pseudo, columns=names).assign(date=dates)[["date", *names]].to_csv(
            destination / "history_pseudo_observables_reporting_units.csv", index=False
        )


def _unemployment_inputs(
    config: QuarterlyEconomyConfig,
    observables: pd.DataFrame,
    *,
    dates: Sequence[str],
    fetcher: FredFetcher | None,
) -> tuple[pd.DataFrame, np.ndarray, Any]:
    unemployment = load_current_fred_series("UNRATE", fetcher=fetcher)
    periods = _period_index(unemployment["date"])
    unemployment = unemployment.loc[
        (periods >= _period(config.start_date)) & (periods <= _period(config.model_end_date))
    ].reset_index(drop=True)
    if unemployment.empty:
        raise ValueError("No unemployment data are available through the model cutoff.")
    bridge = estimate_unemployment_hours_bridge(
        observables,
        unemployment,
        start_date=config.unemployment_bridge_start,
        end_date=config.model_end_date,
    )
    end_rows = unemployment.loc[
        _period_index(unemployment["date"]) <= _period(config.model_end_date)
    ].dropna(subset=["UNRATE"])
    if end_rows.empty:
        raise ValueError("No finite unemployment value is available by the model cutoff.")
    level = float(end_rows.iloc[-1]["UNRATE"])
    baseline = np.full(len(dates), level, dtype=np.float64)
    return unemployment, baseline, bridge


def _run_scenarios(
    config: QuarterlyEconomyConfig,
    model: Model1002,
    system: System,
    start_state: np.ndarray,
    *,
    dates: Sequence[str],
    baseline_obs_model: np.ndarray,
    baseline_pseudo_model: np.ndarray,
    baseline_obs: np.ndarray,
    baseline_pseudo: np.ndarray,
    unemployment_baseline: np.ndarray,
    bridge_slope: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rows = _scenario_rows(
        "baseline",
        "Baseline",
        "baseline",
        dates,
        model,
        baseline_obs,
        baseline_pseudo,
        baseline_obs,
        baseline_pseudo,
        unemployment=unemployment_baseline,
    )
    summaries: list[dict[str, Any]] = []
    shock_paths: dict[str, np.ndarray] = {}
    shock_names = list(model.indexes.exogenous_shocks)
    shock_scales = np.sqrt(np.clip(np.diag(system.measurement.QQ), 0.0, None))
    rate_index = list(model.observables).index("obs_nominalrate")
    rm_index = shock_names.index("rm_sh")

    for definition in config.policy_scenarios:
        targets = np.full_like(baseline_obs_model, np.nan)
        requested = np.asarray(definition.rate_deviation_pp, dtype=np.float64)
        targets[: requested.size, rate_index] = (
            baseline_obs_model[: requested.size, rate_index] + requested / 4.0
        )
        conditioned = solve_shocks_for_observable_targets(
            system,
            start_state,
            targets,
            allowed_shock_indices=[rm_index],
        )
        pseudo_model = _pseudo_from_states(system, conditioned.states)
        obs = reverse_transform_observables(model, conditioned.observables)
        pseudo = reverse_transform_pseudo_observables(model, pseudo_model)
        rows.extend(
            _scenario_rows(
                definition.name,
                definition.label,
                "policy_path",
                dates,
                model,
                obs,
                pseudo,
                baseline_obs,
                baseline_pseudo,
            )
        )
        shock_paths[definition.name] = conditioned.shocks
        summaries.append(
            _scenario_summary(
                definition.name,
                definition.label,
                "policy_path",
                model,
                obs,
                pseudo,
                baseline_obs,
                baseline_pseudo,
                conditioned.shocks,
                shock_scales,
                conditioning_error=conditioned.max_abs_error,
            )
        )

    for definition in config.structural_scenarios:
        shocks = build_structural_shock_path(
            definition,
            shock_names=shock_names,
            shock_scales=shock_scales,
            horizon=config.horizon,
        )
        forecast = forecast_linear_system(
            system,
            start_state,
            horizon=config.horizon,
            shocks=shocks,
            include_pseudo=True,
        )
        if forecast.pseudo_observables is None:
            raise RuntimeError("Structural scenario did not return pseudo-observables.")
        obs = reverse_transform_observables(model, forecast.observables)
        pseudo = reverse_transform_pseudo_observables(model, forecast.pseudo_observables)
        rows.extend(
            _scenario_rows(
                definition.name,
                definition.label,
                "structural",
                dates,
                model,
                obs,
                pseudo,
                baseline_obs,
                baseline_pseudo,
            )
        )
        shock_paths[definition.name] = shocks
        summaries.append(
            _scenario_summary(
                definition.name,
                definition.label,
                "structural",
                model,
                obs,
                pseudo,
                baseline_obs,
                baseline_pseudo,
                shocks,
                shock_scales,
            )
        )

    hours_index = list(model.observables).index("obs_hours")
    for target in config.unemployment_targets:
        name = f"unemployment_{target:g}pct"
        label = f"Unemployment reaches {target:g}%"
        unemployment = build_unemployment_scenario_path(
            unemployment_baseline,
            target=target,
        )
        targets = np.full_like(baseline_obs_model, np.nan)
        targets[:, hours_index] = baseline_obs_model[:, hours_index] + bridge_slope * (
            unemployment - unemployment_baseline
        )
        conditioned = solve_shocks_for_observable_targets(system, start_state, targets)
        pseudo_model = _pseudo_from_states(system, conditioned.states)
        obs = reverse_transform_observables(model, conditioned.observables)
        pseudo = reverse_transform_pseudo_observables(model, pseudo_model)
        rows.extend(
            _scenario_rows(
                name,
                label,
                "external_unemployment_condition",
                dates,
                model,
                obs,
                pseudo,
                baseline_obs,
                baseline_pseudo,
                unemployment=unemployment,
            )
        )
        shock_paths[name] = conditioned.shocks
        summary = _scenario_summary(
            name,
            label,
            "external_unemployment_condition",
            model,
            obs,
            pseudo,
            baseline_obs,
            baseline_pseudo,
            conditioned.shocks,
            shock_scales,
            conditioning_error=conditioned.max_abs_error,
        )
        summary["unemployment_target_percent"] = target
        summaries.append(summary)
    return pd.DataFrame(rows), pd.DataFrame(summaries), shock_paths


def _scenario_rows(
    name: str,
    label: str,
    scenario_type: str,
    dates: Sequence[str],
    model: Model1002,
    observables: np.ndarray,
    pseudo_observables: np.ndarray,
    baseline_observables: np.ndarray,
    baseline_pseudo_observables: np.ndarray,
    *,
    unemployment: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, names, values, baseline in (
        ("observable", list(model.observables), observables, baseline_observables),
        (
            "pseudo_observable",
            list(model.pseudo_observable_mappings),
            pseudo_observables,
            baseline_pseudo_observables,
        ),
    ):
        for step, date in enumerate(dates):
            for index, variable in enumerate(names):
                rows.append(
                    {
                        "scenario": name,
                        "scenario_label": label,
                        "scenario_type": scenario_type,
                        "date": date,
                        "horizon": step,
                        "variable_kind": kind,
                        "variable": variable,
                        "value": float(values[step, index]),
                        "baseline": float(baseline[step, index]),
                        "deviation": float(values[step, index] - baseline[step, index]),
                        "unemployment_rate": (
                            np.nan if unemployment is None else float(unemployment[step])
                        ),
                    }
                )
    return rows


def _scenario_summary(
    name: str,
    label: str,
    scenario_type: str,
    model: Model1002,
    observables: np.ndarray,
    pseudo_observables: np.ndarray,
    baseline_observables: np.ndarray,
    baseline_pseudo_observables: np.ndarray,
    shocks: np.ndarray,
    shock_scales: np.ndarray,
    *,
    conditioning_error: float | None = None,
) -> dict[str, Any]:
    obs_names = list(model.observables)
    pseudo_names = list(model.pseudo_observable_mappings)
    obs_deviation = observables - baseline_observables
    pseudo_deviation = pseudo_observables - baseline_pseudo_observables

    def obs_min(variable: str) -> float:
        return float(np.min(obs_deviation[:, obs_names.index(variable)]))

    def obs_max(variable: str) -> float:
        return float(np.max(obs_deviation[:, obs_names.index(variable)]))

    standardized = np.zeros_like(shocks)
    positive = shock_scales > 1.0e-12
    standardized[:, positive] = shocks[:, positive] / shock_scales[positive]
    max_component = float(np.max(np.abs(standardized)))
    max_joint = float(np.max(np.linalg.norm(standardized, axis=1)))
    if max_component > 8.0 or max_joint > 8.0:
        domain_flag = "extreme_extrapolation"
    elif max_component > 4.0 or max_joint > 4.0:
        domain_flag = "stress_extrapolation"
    else:
        domain_flag = "within_4sd"
    productivity_index = pseudo_names.index("LaborProductivityGrowth")
    output_gap_index = pseudo_names.index("OutputGap")
    return {
        "scenario": name,
        "label": label,
        "scenario_type": scenario_type,
        "gdp_growth_min_deviation_pp": obs_min("obs_gdp"),
        "gdp_growth_max_deviation_pp": obs_max("obs_gdp"),
        "core_pce_min_deviation_pp": obs_min("obs_corepce"),
        "core_pce_max_deviation_pp": obs_max("obs_corepce"),
        "policy_rate_min_deviation_pp": obs_min("obs_nominalrate"),
        "policy_rate_max_deviation_pp": obs_max("obs_nominalrate"),
        "hours_min_deviation_log_points": obs_min("obs_hours"),
        "investment_min_deviation_pp": obs_min("obs_investment"),
        "consumption_min_deviation_pp": obs_min("obs_consumption"),
        "productivity_growth_min_deviation_pp": float(
            np.min(pseudo_deviation[:, productivity_index])
        ),
        "productivity_growth_max_deviation_pp": float(
            np.max(pseudo_deviation[:, productivity_index])
        ),
        "output_gap_min_deviation": float(np.min(pseudo_deviation[:, output_gap_index])),
        "max_component_shock_sd": max_component,
        "max_quarterly_joint_shock_norm_sd": max_joint,
        "linear_domain_flag": domain_flag,
        "conditioning_max_abs_error": conditioning_error,
    }


def _pseudo_from_states(system: System, states: np.ndarray) -> np.ndarray:
    if system.pseudo_measurement is None:
        raise RuntimeError("Solved system does not include pseudo-measurement matrices.")
    return states @ np.asarray(system.pseudo_measurement.ZZ_pseudo).T + np.asarray(
        system.pseudo_measurement.DD_pseudo
    )


def _save_scenario_shocks(path: Path, scenario_shocks: Mapping[str, np.ndarray]) -> None:
    names = list(scenario_shocks)
    arrays = np.stack([scenario_shocks[name] for name in names], axis=0)
    np.savez(
        path,
        scenario_names=np.asarray(names, dtype="U"),
        shocks=arrays,
    )


def _historical_group_decomposition(
    model: Model1002,
    system: System,
    data: pd.DataFrame,
    *,
    tail_quarters: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    decomposition = historical_decomposition(
        model,
        data=data,
        check_empty_columns=False,
    )
    shock_names = list(model.indexes.exogenous_shocks)
    assigned = {shock for shocks in SHOCK_GROUPS.values() for shock in shocks}
    if assigned != set(shock_names):
        raise ValueError(
            "Historical shock taxonomy must assign every shock exactly once; "
            f"missing={sorted(set(shock_names) - assigned)}, "
            f"extra={sorted(assigned - set(shock_names))}."
        )
    dates = list(data["date"].astype(str))[-decomposition.observed.shape[0] :]
    obs_rows, obs_error = _group_decomposition_rows(
        dates,
        variable_kind="observable",
        variable_names=list(model.observables),
        shock_names=shock_names,
        contributions=decomposition.observable_contributions,
        baseline=decomposition.observable_baseline,
        smoothed=decomposition.smoothed_observables,
        observed=decomposition.observed,
        transform=lambda values: reverse_transform_observables(model, values),
        tail_quarters=tail_quarters,
    )
    if system.pseudo_measurement is None:
        raise RuntimeError("Historical productivity decomposition requires pseudo measurements.")
    pseudo_zz = np.asarray(system.pseudo_measurement.ZZ_pseudo)
    pseudo_dd = np.asarray(system.pseudo_measurement.DD_pseudo)
    pseudo_contributions = np.einsum(
        "pn,tnj->tpj",
        pseudo_zz,
        decomposition.state_contributions,
    )
    pseudo_baseline = decomposition.state_baseline @ pseudo_zz.T + pseudo_dd
    pseudo_smoothed = decomposition.smoothed_states @ pseudo_zz.T + pseudo_dd
    pseudo_rows, pseudo_error = _group_decomposition_rows(
        dates,
        variable_kind="pseudo_observable",
        variable_names=list(model.pseudo_observable_mappings),
        shock_names=shock_names,
        contributions=pseudo_contributions,
        baseline=pseudo_baseline,
        smoothed=pseudo_smoothed,
        observed=pseudo_smoothed,
        transform=lambda values: reverse_transform_pseudo_observables(model, values),
        tail_quarters=tail_quarters,
    )
    if decomposition.reconciliation_max_abs_error > 1.0e-5:
        raise ValueError(
            "Historical decomposition fails model-unit reconciliation: "
            f"{decomposition.reconciliation_max_abs_error:.3e}."
        )
    if max(obs_error, pseudo_error) > 1.0e-5:
        raise ValueError(
            "Historical decomposition fails report-unit reconciliation: "
            f"{max(obs_error, pseudo_error):.3e}."
        )
    rows = pd.DataFrame([*obs_rows, *pseudo_rows])
    quality = {
        "method": "RTS-smoothed structural shock contributions",
        "shock_groups": {name: list(shocks) for name, shocks in SHOCK_GROUPS.items()},
        "model_unit_reconciliation_max_abs_error": (decomposition.reconciliation_max_abs_error),
        "report_unit_allocation": (
            "Exact for linear transforms; proportional total-change allocation for "
            "nonlinear annualized log-growth transforms"
        ),
        "observable_report_unit_reconciliation_max_abs_error": obs_error,
        "pseudo_report_unit_reconciliation_max_abs_error": pseudo_error,
        "fiscal_definition": (
            "Government-spending shock g_sh only; no identified tax or transfer shock"
        ),
        "corepce_sh_definition": (
            "AR measurement innovation for observed core PCE, not a structural supply shock"
        ),
    }
    return rows, quality


def _group_decomposition_rows(
    dates: Sequence[str],
    *,
    variable_kind: str,
    variable_names: Sequence[str],
    shock_names: Sequence[str],
    contributions: np.ndarray,
    baseline: np.ndarray,
    smoothed: np.ndarray,
    observed: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
    tail_quarters: int,
) -> tuple[list[dict[str, Any]], float]:
    smoothed_report = transform(smoothed)
    observed_report = transform(observed)
    epsilon = 1.0e-7
    derivative = transform(np.full_like(smoothed, epsilon)) / epsilon
    scale = np.divide(
        smoothed_report,
        smoothed,
        out=derivative,
        where=np.abs(smoothed) > 1.0e-10,
    )
    grouped: dict[str, np.ndarray] = {}
    for group, shocks in SHOCK_GROUPS.items():
        indexes = [shock_names.index(shock) for shock in shocks]
        grouped[group] = contributions[:, :, indexes].sum(axis=2) * scale
    grouped["Initial conditions and trend"] = baseline * scale
    residual = observed_report - smoothed_report
    grouped["Observed-model residual"] = residual
    reconstructed = np.sum(np.stack(list(grouped.values()), axis=0), axis=0)
    finite = np.isfinite(observed_report)
    max_error = float(np.max(np.abs(reconstructed[finite] - observed_report[finite])))
    start = max(0, len(dates) - tail_quarters)
    rows: list[dict[str, Any]] = []
    for period in range(start, len(dates)):
        for variable_index, variable in enumerate(variable_names):
            for group, values in grouped.items():
                rows.append(
                    {
                        "date": dates[period],
                        "variable_kind": variable_kind,
                        "variable": variable,
                        "driver": group,
                        "contribution": float(values[period, variable_index]),
                        "smoothed": float(smoothed_report[period, variable_index]),
                        "observed": float(observed_report[period, variable_index]),
                    }
                )
    return rows, max_error


def _build_cpi_detail(detail: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "quarter",
        "food_effect",
        "food_at_home_effect",
        "cereals_bakery_effect",
        "meats_poultry_fish_eggs_effect",
        "dairy_effect",
        "fruits_vegetables_effect",
        "nonalcoholic_beverages_effect",
        "other_food_at_home_effect",
        "food_away_from_home_effect",
        "services_ex_energy_effect",
        "shelter_effect",
        "rent_primary_effect",
        "lodging_away_effect",
        "oer_residences_effect",
        "tenants_insurance_effect",
        "water_sewer_trash_effect",
        "medical_services_effect",
        "transportation_services_effect",
        "recreation_services_effect",
        "education_communication_services_effect",
        "other_personal_services_effect",
    }
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise ValueError(f"CPI detail is missing required columns: {missing}.")
    result = detail.copy()
    for column in ("food_effect", "services_ex_energy_effect", "shelter_effect"):
        if float((result[column] - summary[column]).abs().max()) > 1.0e-10:
            raise ValueError(f"CPI detail does not match summary column '{column}'.")
    food_parts = [
        "cereals_bakery_effect",
        "meats_poultry_fish_eggs_effect",
        "dairy_effect",
        "fruits_vegetables_effect",
        "nonalcoholic_beverages_effect",
        "other_food_at_home_effect",
        "food_away_from_home_effect",
    ]
    shelter_parts = [
        "rent_primary_effect",
        "lodging_away_effect",
        "oer_residences_effect",
        "tenants_insurance_effect",
    ]
    service_parts = [
        "water_sewer_trash_effect",
        "medical_services_effect",
        "transportation_services_effect",
        "recreation_services_effect",
        "education_communication_services_effect",
        "other_personal_services_effect",
    ]
    result["food_detail_residual"] = result["food_effect"] - result[food_parts].sum(axis=1)
    result["shelter_detail_residual"] = result["shelter_effect"] - result[shelter_parts].sum(axis=1)
    result["other_core_services_effect"] = (
        result["services_ex_energy_effect"] - result["shelter_effect"]
    )
    result["other_core_services_residual_effect"] = result["other_core_services_effect"] - result[
        service_parts
    ].sum(axis=1)
    result["food_reconciliation_error"] = (
        result[food_parts].sum(axis=1) + result["food_detail_residual"] - result["food_effect"]
    )
    result["shelter_reconciliation_error"] = (
        result[shelter_parts].sum(axis=1)
        + result["shelter_detail_residual"]
        - result["shelter_effect"]
    )
    result["other_core_services_reconciliation_error"] = (
        result[service_parts].sum(axis=1)
        + result["other_core_services_residual_effect"]
        - result["other_core_services_effect"]
    )
    return result


def _validate_quarter_frame(frame: pd.DataFrame, label: str) -> None:
    if "quarter" not in frame:
        raise ValueError(f"{label} must include a quarter column.")
    periods = _period_index(frame["quarter"])
    if periods.has_duplicates:
        raise ValueError(f"{label} contains duplicate quarters.")
    if not periods.is_monotonic_increasing:
        raise ValueError(f"{label} quarters must be sorted.")


def _make_figures(
    destination: Path,
    *,
    model: Model1002,
    baseline: pd.DataFrame,
    scenarios: pd.DataFrame,
    historical: pd.DataFrame,
    cpi: pd.DataFrame | None,
    cpi_detail: pd.DataFrame | None,
) -> tuple[Path, ...]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = destination / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.append(
        _plot_baseline_panel(
            plt,
            baseline.loc[baseline["variable_kind"] == "observable"],
            list(model.observables),
            figure_dir / "baseline_all_observables.png",
            title="Baseline forecast — all Model1002 observables",
            labels={
                name: model.observable_mappings[name].description for name in model.observables
            },
        )
    )
    paths.append(
        _plot_baseline_panel(
            plt,
            baseline.loc[baseline["variable_kind"] == "pseudo_observable"],
            list(model.pseudo_observable_mappings),
            figure_dir / "baseline_all_pseudo_observables.png",
            title="Baseline forecast — all model-implied macro variables",
            labels={
                name: mapping.description
                for name, mapping in model.pseudo_observable_mappings.items()
            },
        )
    )
    paths.append(_plot_policy_pce(plt, scenarios, figure_dir / "fed_funds_core_pce.png"))
    paths.append(_plot_scenario_panel(plt, scenarios, figure_dir / "scenario_rundown.png"))
    for variable, title, filename in (
        ("obs_corepce", "DSGE historical decomposition — core PCE", "pce_decomposition.png"),
        ("obs_gdp", "DSGE historical decomposition — GDP growth", "gdp_decomposition.png"),
        (
            "LaborProductivityGrowth",
            "DSGE historical decomposition — labor-productivity growth",
            "productivity_decomposition.png",
        ),
    ):
        paths.append(
            _plot_grouped_history(
                plt,
                historical.loc[historical["variable"] == variable],
                figure_dir / filename,
                title=title,
            )
        )
    paths.append(
        _plot_policy_history(
            plt,
            historical,
            figure_dir / "fiscal_vs_monetary_history.png",
        )
    )
    if cpi is not None:
        paths.append(_plot_cpi(plt, cpi, figure_dir / "cpi_component_decomposition.png"))
    if cpi_detail is not None:
        paths.append(_plot_cpi_detail(plt, cpi_detail, figure_dir / "cpi_detail.png"))
    return tuple(paths)


def _plot_baseline_panel(
    plt: Any,
    frame: pd.DataFrame,
    variables: Sequence[str],
    path: Path,
    *,
    title: str,
    labels: Mapping[str, str],
) -> Path:
    columns = 3
    rows = (len(variables) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(17, 3.15 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for index, ax in enumerate(axes_array):
        if index >= len(variables):
            ax.axis("off")
            continue
        variable = variables[index]
        subset = frame.loc[frame["variable"] == variable].sort_values("horizon")
        if subset.empty:
            raise ValueError(f"Baseline panel is missing variable '{variable}'.")
        x = subset["horizon"].to_numpy(dtype=int)
        ax.fill_between(
            x,
            subset["p05"],
            subset["p95"],
            color="#9EC1E6",
            alpha=0.38,
            linewidth=0.0,
        )
        ax.plot(x, subset["baseline"], color="#184E88", linewidth=1.9)
        label = textwrap.fill(labels.get(variable, variable), width=34)
        ax.set_title(f"{label}\n[{variable}]", fontsize=8.8, color="#1F2937")
        tick_positions = x[::4]
        if x[-1] not in tick_positions:
            tick_positions = np.append(tick_positions, x[-1])
        date_by_horizon = dict(
            zip(
                subset["horizon"].to_numpy(dtype=int),
                subset["date"].astype(str),
                strict=True,
            )
        )
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [date_by_horizon[int(position)] for position in tick_positions],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_xlim(float(x[0]), float(x[-1]))
        if float(subset["p05"].min()) <= 0.0 <= float(subset["p95"].max()):
            ax.axhline(0.0, color="#6B7280", linewidth=0.7, alpha=0.65)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9CA3AF")
    fig.suptitle(
        title,
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#111827",
    )
    fig.text(
        0.04,
        0.976,
        (
            "Blue line: zero-future-shock conditional mean. Shading: 5th–95th "
            "percentile future-shock band. Panels use independent y-scales."
        ),
        ha="left",
        va="top",
        fontsize=9,
        color="#4B5563",
    )
    fig.text(
        0.5,
        0.005,
        "Forecast quarter • values shown in each variable's model reporting units",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )
    fig.tight_layout(rect=[0.02, 0.025, 1, 0.955])
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_policy_pce(plt: Any, scenarios: pd.DataFrame, path: Path) -> Path:
    subset = scenarios.loc[
        (scenarios["scenario_type"].isin(["baseline", "policy_path"]))
        & (scenarios["variable"] == "obs_corepce")
    ]
    fig, ax = plt.subplots(figsize=(13, 6.8))
    for scenario, group in subset.groupby("scenario", sort=False):
        label = str(group.iloc[0]["scenario_label"])
        ax.plot(
            group["horizon"],
            group["value"],
            linewidth=2.5 if scenario == "baseline" else 1.7,
            label=label,
        )
    margin = max(0.04, float(subset["value"].max() - subset["value"].min()) * 0.08)
    ax.set_ylim(float(subset["value"].min()) - margin, float(subset["value"].max()) + margin)
    ax.set_title(
        "Core PCE under exact fed-funds-rate path ablations",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Annualized quarterly inflation (%)")
    ax.set_xlabel("Forecast quarter")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_scenario_panel(plt: Any, scenarios: pd.DataFrame, path: Path) -> Path:
    selected = (
        "obs_gdp",
        "obs_corepce",
        "obs_nominalrate",
        "obs_hours",
        "OutputGap",
        "LaborProductivityGrowth",
    )
    non_unemployment = scenarios.loc[
        ~scenarios["scenario_type"].isin(["external_unemployment_condition"])
    ]
    definitions = (
        non_unemployment.loc[non_unemployment["scenario"] != "baseline"][
            ["scenario", "scenario_label", "scenario_type"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    color_map = plt.get_cmap("tab20")
    colors = {
        str(row["scenario"]): color_map(index / max(1, len(definitions) - 1))
        for index, row in definitions.iterrows()
    }
    handles: list[Any] = []
    labels: list[str] = []
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    for axis_index, (ax, variable) in enumerate(zip(axes.ravel(), selected, strict=True)):
        subset = non_unemployment.loc[non_unemployment["variable"] == variable]
        for scenario, group in subset.groupby("scenario", sort=False):
            if scenario == "baseline":
                continue
            line = ax.plot(
                group["horizon"],
                group["deviation"],
                linewidth=1.35,
                alpha=0.9,
                color=colors[str(scenario)],
                linestyle=("--" if str(group.iloc[0]["scenario_type"]) == "policy_path" else "-"),
            )[0]
            if axis_index == 0:
                handles.append(line)
                labels.append(str(group.iloc[0]["scenario_label"]))
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_title(variable)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Scenario deviations from baseline — structural and policy ablations",
        fontsize=16,
        fontweight="bold",
    )
    fig.supxlabel("Forecast quarter", y=0.145)
    fig.supylabel("Deviation in reporting units")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=7.0,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.18, 1, 0.97])
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_grouped_history(
    plt: Any,
    frame: pd.DataFrame,
    path: Path,
    *,
    title: str,
) -> Path:
    if frame.empty:
        raise ValueError(f"No historical decomposition rows are available for {title}.")
    wide = frame.pivot(index="date", columns="driver", values="contribution")
    dates = list(wide.index)
    x = np.arange(len(wide))
    fig, ax = plt.subplots(figsize=(15, 6.8))
    _stacked_bars(ax, x, wide)
    observed = frame.drop_duplicates("date").set_index("date").loc[dates, "observed"]
    ax.plot(x, observed, color="#111827", linewidth=2.0, marker="o", markersize=3)
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold")
    ax.set_ylabel("Contribution in reporting units")
    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=45, ha="right")
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_policy_history(plt: Any, historical: pd.DataFrame, path: Path) -> Path:
    selected = historical.loc[
        (historical["variable"].isin(["obs_corepce", "obs_gdp"]))
        & (historical["driver"].isin(["Government spending", "Monetary policy"]))
    ]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, variable in zip(axes, ("obs_corepce", "obs_gdp"), strict=True):
        subset = selected.loc[selected["variable"] == variable]
        dates = list(dict.fromkeys(subset["date"]))
        x = np.arange(len(dates))
        wide = subset.pivot(index="date", columns="driver", values="contribution").loc[dates]
        _stacked_bars(ax, x, wide)
        ax.set_title(variable)
        ax.axhline(0.0, color="#444", linewidth=0.8)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(np.arange(len(dates)))
    axes[-1].set_xticklabels(dates, rotation=45, ha="right")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(
        "Government-spending and monetary-policy shock contributions",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _stacked_bars(ax: Any, x: np.ndarray, wide: pd.DataFrame) -> None:
    positive = np.zeros(len(wide))
    negative = np.zeros(len(wide))
    for driver in wide.columns:
        values = wide[driver].to_numpy(dtype=float)
        bottoms = np.where(values >= 0.0, positive, negative)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            width=0.8,
            label=driver,
            color=GROUP_COLORS.get(str(driver), "#999999"),
            edgecolor="white",
            linewidth=0.25,
        )
        positive += np.where(values >= 0.0, values, 0.0)
        negative += np.where(values < 0.0, values, 0.0)


def _plot_cpi(plt: Any, cpi: pd.DataFrame, path: Path) -> Path:
    x = np.arange(len(cpi))
    fig, ax = plt.subplots(figsize=(15, 7))
    positive = np.zeros(len(cpi))
    negative = np.zeros(len(cpi))
    for label, column in CPI_DRIVER_COLUMNS.items():
        values = cpi[column].to_numpy(dtype=float)
        bottoms = np.where(values >= 0.0, positive, negative)
        ax.bar(
            x,
            values,
            bottom=bottoms,
            label=label,
            color=CPI_COLORS[label],
            width=0.8,
            edgecolor="white",
            linewidth=0.3,
        )
        positive += np.where(values >= 0.0, values, 0.0)
        negative += np.where(values < 0.0, values, 0.0)
    ax.plot(x, cpi["headline_cpi_yoy"], color="#111827", linewidth=2.0, marker="o")
    ax.set_title(
        "Headline CPI contribution accounting",
        loc="left",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Percentage-point contribution to 12-month CPI")
    ax.set_xticks(x)
    ax.set_xticklabels(cpi["quarter"], rotation=45, ha="right")
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _plot_cpi_detail(plt: Any, detail: pd.DataFrame, path: Path) -> Path:
    panels = (
        (
            "Food",
            "food_effect",
            [
                "cereals_bakery_effect",
                "meats_poultry_fish_eggs_effect",
                "dairy_effect",
                "fruits_vegetables_effect",
                "nonalcoholic_beverages_effect",
                "other_food_at_home_effect",
                "food_away_from_home_effect",
                "food_detail_residual",
            ],
        ),
        (
            "Shelter",
            "shelter_effect",
            [
                "rent_primary_effect",
                "oer_residences_effect",
                "lodging_away_effect",
                "tenants_insurance_effect",
                "shelter_detail_residual",
            ],
        ),
        (
            "Other core services",
            "other_core_services_effect",
            [
                "water_sewer_trash_effect",
                "medical_services_effect",
                "transportation_services_effect",
                "recreation_services_effect",
                "education_communication_services_effect",
                "other_personal_services_effect",
                "other_core_services_residual_effect",
            ],
        ),
    )
    x = np.arange(len(detail))
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    for ax, (title, parent, parts) in zip(axes, panels, strict=True):
        positive = np.zeros(len(detail))
        negative = np.zeros(len(detail))
        for column in parts:
            values = detail[column].to_numpy(dtype=float)
            bottoms = np.where(values >= 0.0, positive, negative)
            ax.bar(x, values, bottom=bottoms, width=0.8, label=column)
            positive += np.where(values >= 0.0, values, 0.0)
            negative += np.where(values < 0.0, values, 0.0)
        ax.plot(x, detail[parent], color="#111827", linewidth=1.8)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axhline(0.0, color="#444", linewidth=0.8)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(detail["quarter"], rotation=45, ha="right")
    fig.suptitle(
        "Food, shelter, and other core-services CPI detail",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _report_markdown(
    metadata: Mapping[str, Any],
    baseline: pd.DataFrame,
    summary: pd.DataFrame,
    historical: pd.DataFrame,
    cpi: pd.DataFrame | None,
    figures: Sequence[Path],
) -> str:
    config = metadata["configuration"]
    data_quality = metadata["data"]["quality"]
    selected = baseline.loc[
        (baseline["variable_kind"] == "observable")
        & (baseline["variable"].isin(["obs_gdp", "obs_corepce", "obs_nominalrate"]))
        & (baseline["horizon"] < 4)
    ]
    baseline_table = selected.pivot(
        index="date", columns="variable", values="baseline"
    ).reset_index()
    summary_columns = [
        "scenario",
        "gdp_growth_min_deviation_pp",
        "core_pce_min_deviation_pp",
        "core_pce_max_deviation_pp",
        "policy_rate_max_deviation_pp",
        "max_quarterly_joint_shock_norm_sd",
        "linear_domain_flag",
    ]
    latest_cpi = None if cpi is None else cpi.iloc[-1]
    cpi_text = (
        "No CPI component snapshot was configured."
        if latest_cpi is None
        else (
            f"The latest CPI snapshot is {latest_cpi['quarter']}: headline CPI was "
            f"{latest_cpi['headline_cpi_yoy']:.1f}% year over year. Shelter contributed "
            f"{latest_cpi['shelter_effect']:.3f} percentage points, food "
            f"{latest_cpi['food_effect']:.3f}, energy {latest_cpi['energy_effect']:.3f}, "
            f"core goods {latest_cpi['core_goods_effect']:.3f}, and other core services "
            f"{latest_cpi['other_core_services_effect']:.3f}."
        )
    )
    latest_pce = historical.loc[
        (historical["variable"] == "obs_corepce") & (historical["driver"] == "Monetary policy")
    ].iloc[-1]
    baseline_figures = [path for path in figures if path.stem.startswith("baseline_all_")]
    supporting_figures = [path for path in figures if not path.stem.startswith("baseline_all_")]
    baseline_figures_md = _report_figures_markdown(
        baseline_figures,
        empty_message=(
            "_Baseline forecast panels were not rendered because plot generation "
            "was disabled for this run._"
        ),
    )
    supporting_figures_md = _report_figures_markdown(
        supporting_figures,
        empty_message=(
            "_Supporting scenario and decomposition figures were not rendered because "
            "plot generation was disabled for this run._"
        ),
    )
    observable_count = baseline.loc[baseline["variable_kind"] == "observable", "variable"].nunique()
    pseudo_count = baseline.loc[
        baseline["variable_kind"] == "pseudo_observable", "variable"
    ].nunique()
    return f"""# Quarterly U.S. Economic Projection and Scenario Package

## Executive Summary

The information set ends in **{config["model_end_date"]}** and the forecast starts in
**{metadata["model"]["forecast_start"]}**. The package includes all Model1002 observables and
pseudo-observables, 90% future-shock bands, exact fed-funds-rate path ablations, configurable
structural and compounded scenarios, unemployment stress paths, DSGE historical shock
decompositions, and a separate CPI basket decomposition.

The data panel is **{data_quality["status"]}**: {len(data_quality["missing_at_model_end"])}
observables are unavailable in the final quarter and remain missing rather than being filled
with non-equivalent proxies. The model can filter ragged-edge data, but forecasts should be
read with that reduced final-quarter information set in mind.

{cpi_text}

The latest displayed DSGE monetary-policy contribution to annualized quarterly core PCE is
{latest_pce["contribution"]:.3f} percentage points. This is an unexpected-shock contribution,
not the full effect of the systematic Taylor-rule response.

## Baseline forecast

{_markdown_table(baseline_table.round(3))}

The baseline is the zero-future-shock conditional mean from the smoothed terminal state.
The shaded ranges in the figures are 5th–95th percentile bands from future structural-shock
draws with fixed parameters; they do not include parameter uncertainty.

### All-variable baseline forecast panels

The first panel covers all **{observable_count} observables** and the second covers all
**{pseudo_count} model-implied variables**. Each small multiple uses its own vertical scale
so that the projected path remains visible; compare direction and uncertainty within a
panel, not vertical magnitudes across panels.

{baseline_figures_md}

## Scenario findings

{_markdown_table(summary[summary_columns].round(3))}

Policy paths constrain only the observed nominal rate and are solved with `rm_sh` innovations
only. Structural scenarios are additive shock paths in model standard deviations. Any row
marked `stress_extrapolation` or `extreme_extrapolation` lies outside the package's preferred
local-linear interpretation range.

## Inflation, fiscal, monetary, and productivity decomposition

The DSGE decomposition uses RTS-smoothed structural innovations. “Government spending” is
the model's `g_sh` innovation; Model1002 ss10 does **not** separately identify taxes,
transfers, or a Biden-administration tax shock. “Monetary policy” combines the unexpected
and anticipated policy-rate shocks. The `corepce_sh` contribution is classified as a
measurement innovation because that is how it enters the model—not as a structural supply
shock.

The CPI bars are different: they are BLS effect-on-All-items accounting contributions for
food, energy, core goods, shelter, and other core services. They reconcile to headline CPI
but are not causal fiscal or monetary estimates and should not be added to the DSGE bars.

## Visual evidence

{supporting_figures_md}

## Data quality and reproducibility

- Config SHA-256: `{metadata["configuration"]["sha256"]}`
- Observed-cell share: {data_quality["observed_cell_share"]:.3f}
- Missing at the cutoff: {", ".join(data_quality["missing_at_model_end"])}
- Julia parity reference: DSGE.jl {REFERENCE_DSGE_VERSION}, tree `{REFERENCE_DSGE_TREE[:12]}`
- Parameter refresh: `{metadata["model"]["refresh"]["enabled"]}`
- Historical report-unit reconciliation error:
  {metadata["historical_decomposition"]["observable_report_unit_reconciliation_max_abs_error"]:.3e}

## Limitations

This is the public Model1002 `ss10` parity target, not the private current New York Fed
production model. The targeted MAP refresh is a local Python estimation and is explicitly
not represented as Julia optimizer parity. The model is linear and representative-agent.
Unemployment is outside its measurement system and is imposed through a descriptive
hours/unemployment bridge. Large unemployment and policy paths can require implausibly large
joint shocks; those outputs are stress tests, not probabilities. Historical decompositions
are model-dependent attributions and can change with the data vintage, smoothing sample, and
parameterization.

## Recommended next steps

Use the baseline and scenarios for quarterly planning, but benchmark high-pain disinflation
cases against a BVAR, local projections, and a labor-search model before treating them as
policy estimates. For AI questions, add sector and occupation exposure data or a
heterogeneous-agent/search-and-matching block so unemployment and reallocation are modeled
directly.

## Further questions

The most useful next extensions are parameter-uncertainty bands, alternative monetary-policy
identification, a fiscal instrument split for spending/taxes/transfers, and automated BLS
Table 7 snapshot ingestion with source-vintage archiving.
"""


def _report_figures_markdown(
    figures: Sequence[Path],
    *,
    empty_message: str,
) -> str:
    if not figures:
        return empty_message
    return "\n\n".join(f"![{path.stem.replace('_', ' ')}](figures/{path.name})" for path in figures)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _config_manifest(config: QuarterlyEconomyConfig) -> dict[str, Any]:
    return {
        "start_date": config.start_date,
        "model_end_date": config.model_end_date,
        "horizon": config.horizon,
        "stochastic_draws": config.stochastic_draws,
        "seed": config.seed,
        "historical_tail_quarters": config.historical_tail_quarters,
        "refresh_model": config.refresh_model,
        "refresh_maxiter": config.refresh_maxiter,
        "fred_levels_path": (
            None if config.fred_levels_path is None else str(config.fred_levels_path)
        ),
        "cpi_summary_path": (
            None if config.cpi_summary_path is None else str(config.cpi_summary_path)
        ),
        "cpi_detail_path": (
            None if config.cpi_detail_path is None else str(config.cpi_detail_path)
        ),
        "cpi_goods_path": (None if config.cpi_goods_path is None else str(config.cpi_goods_path)),
        "unemployment_targets": list(config.unemployment_targets),
        "policy_scenarios": [
            {
                "name": scenario.name,
                "label": scenario.label,
                "rate_deviation_pp": list(scenario.rate_deviation_pp),
            }
            for scenario in config.policy_scenarios
        ],
        "structural_scenarios": [
            {
                "name": scenario.name,
                "label": scenario.label,
                "components": [asdict(component) for component in scenario.components],
            }
            for scenario in config.structural_scenarios
        ],
    }


def _period(value: str) -> pd.Period:
    period = pd.Period(str(value).replace("-Q", "Q"), freq="Q")
    if not isinstance(period, pd.Period):
        raise ValueError(f"Invalid quarterly date: {value}")
    return period


def _period_index(values: pd.Series[Any]) -> pd.PeriodIndex:
    return pd.PeriodIndex([_period(str(value)) for value in values], freq="Q")


def _quarter_label(value: pd.Period) -> str:
    return f"{value.year}-Q{value.quarter}"


def _quarter_sequence(start: str, horizon: int) -> list[str]:
    first = _period(start)
    return [_quarter_label(first + offset) for offset in range(horizon)]
