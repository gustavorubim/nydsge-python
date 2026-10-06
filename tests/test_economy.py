from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nydsge.economy import (
    REFERENCE_DSGE_TREE,
    SHOCK_GROUPS,
    ShockComponent,
    StructuralScenario,
    assess_quarterly_data_quality,
    build_cpi_accounting,
    build_structural_shock_path,
    load_quarterly_economy_config,
    run_quarterly_economy_package,
)
from nydsge.models import Model1002


def test_default_quarterly_config_is_valid_and_pinned_to_reference_tree() -> None:
    root = Path(__file__).resolve().parents[1]

    config = load_quarterly_economy_config(root / "configs" / "quarterly_economy.json")

    assert config.model_end_date == "2026-Q2"
    assert config.horizon == 20
    assert config.policy_scenarios
    assert config.structural_scenarios
    assert len(config.source_sha256) == 64
    assert REFERENCE_DSGE_TREE == "e746a4a5ab9c26d897239e722b0f19d4bb3bd77e"


def test_shock_taxonomy_assigns_every_model_shock_exactly_once() -> None:
    model = Model1002()
    assigned = [shock for shocks in SHOCK_GROUPS.values() for shock in shocks]

    assert set(assigned) == set(model.indexes.exogenous_shocks)
    assert len(assigned) == len(set(assigned))
    assert "corepce_sh" in SHOCK_GROUPS["Measurement innovations"]
    assert SHOCK_GROUPS["Government spending"] == ("g_sh",)


def test_structural_scenario_components_compound_with_timing_and_decay() -> None:
    scenario = StructuralScenario(
        name="compound",
        label="Compound",
        components=(
            ShockComponent("a", 2.0, start=0, duration=3, decay=0.5),
            ShockComponent("a", -1.0, start=1, duration=1),
            ShockComponent("b", 1.5, start=2, duration=1),
        ),
    )

    shocks = build_structural_shock_path(
        scenario,
        shock_names=["a", "b"],
        shock_scales=np.array([0.2, 2.0]),
        horizon=4,
    )

    np.testing.assert_allclose(
        shocks,
        np.array(
            [
                [0.4, 0.0],
                [0.0, 0.0],
                [0.1, 3.0],
                [0.0, 0.0],
            ]
        ),
    )


def test_data_quality_rejects_missing_quarter_and_reports_ragged_edge() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2025-Q1", "2025-Q2", "2025-Q3"],
            "obs_a": [1.0, 2.0, np.nan],
            "obs_b": [np.nan, np.nan, np.nan],
        }
    )

    quality = assess_quarterly_data_quality(
        frame,
        start_date="2025-Q1",
        model_end_date="2025-Q3",
    )

    assert quality["status"] == "partial"
    assert quality["all_missing_observables"] == ["obs_b"]
    assert quality["missing_at_model_end"] == ["obs_a", "obs_b"]

    with pytest.raises(ValueError, match="quarterly grid"):
        assess_quarterly_data_quality(
            frame.iloc[[0, 2]],
            start_date="2025-Q1",
            model_end_date="2025-Q3",
        )


def test_committed_cpi_snapshots_reconcile_through_2026q2() -> None:
    root = Path(__file__).resolve().parents[1]

    wide, detail, goods, quality = build_cpi_accounting(
        root / "data" / "cpi" / "bls_table7_quarterly.csv",
        detail_path=root / "data" / "cpi" / "bls_table7_detail_quarterly.csv",
        goods_path=root / "data" / "cpi" / "bls_table7_goods_latest.csv",
        expected_latest_quarter="2026-Q2",
    )

    assert detail is not None
    assert goods is not None
    assert wide.iloc[-1]["headline_cpi_yoy"] == 3.5
    assert quality["status"] == "reconciled"
    assert quality["max_abs_reconciliation_error_pp"] <= 1.0e-10
    assert quality["max_abs_food_reconciliation_error_pp"] <= 1.0e-10


def _frozen_package_inputs(
    tmp_path: Path, *, last_quarter: str = "2026-Q2"
) -> tuple[dict[str, Any], Callable[[str], bytes]]:
    periods = pd.period_range("2014-Q3", last_quarter, freq="Q")
    steps = np.arange(len(periods), dtype=np.float64)
    dates = [f"{period.year}-Q{period.quarter}" for period in periods]
    levels = pd.DataFrame(
        {
            "date": dates,
            "GDP": 18000.0 * np.exp(0.006 * steps),
            "CNP16OV": 245000.0 + 300.0 * steps,
            "GDPDEF": 95.0 * np.exp(0.005 * steps),
            "AWHNONAG": 33.5 + 0.02 * np.sin(steps),
            "CE16OV": 145000.0 + 250.0 * steps + 20.0 * np.sin(steps),
            "COMPNFB": 105.0 * np.exp(0.007 * steps),
            "PCEPILFE": 100.0 * np.exp(0.005 * steps),
            "DFF": 2.0 + 0.5 * np.sin(steps / 5.0),
            "PCE": 12000.0 * np.exp(0.006 * steps),
            "FPI": 2500.0 * np.exp(0.008 * steps),
            "BAA": 5.0 + 0.2 * np.sin(steps / 4.0),
            "BAMLC8A0C15PYEY": 5.2 + 0.2 * np.sin(steps / 4.0),
            "GS10": 3.0 + 0.1 * np.sin(steps / 6.0),
            "GDI": 17900.0 * np.exp(0.006 * steps),
        }
    )
    levels_path = tmp_path / "levels.csv"
    levels.to_csv(levels_path, index=False)
    config = {
        "start_date": dates[0],
        "model_end_date": "2026-Q2",
        "horizon": 4,
        "stochastic_draws": 2,
        "seed": 7,
        "historical_tail_quarters": 4,
        "refresh_model": False,
        "refresh_maxiter": 2,
        "fred_levels_path": str(levels_path),
        "cpi_summary_path": None,
        "cpi_detail_path": None,
        "cpi_goods_path": None,
        "unemployment_targets": [],
        "unemployment_bridge_start": "2015-Q1",
        "policy_scenarios": [
            {
                "name": "tightening_25bp",
                "label": "25 bp tightening",
                "rate_deviation_pp": [0.25],
            }
        ],
        "structural_scenarios": [
            {
                "name": "productivity",
                "label": "Productivity",
                "components": [{"shock": "ztil_sh", "size_sd": 1.0}],
            }
        ],
    }
    monthly_dates = [f"{period.year}-{3 * period.quarter:02d}-01" for period in periods]
    unemployment = 5.0 + 0.15 * np.sin(steps / 3.0)
    payload = "DATE,UNRATE\n" + "\n".join(
        f"{date},{value}" for date, value in zip(monthly_dates, unemployment, strict=True)
    )
    return config, lambda _: payload.encode()


def _run_frozen_package(
    tmp_path: Path, config: dict[str, Any], fetcher: Callable[[str], bytes]
) -> tuple[str, dict[str, Any]]:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifacts = run_quarterly_economy_package(
        config_path=config_path,
        output_dir=tmp_path / "run",
        fetcher=fetcher,
        make_plots=False,
    )
    report = artifacts.report.read_text(encoding="utf-8")
    return report, json.loads(artifacts.metadata.read_text(encoding="utf-8"))


def test_quarterly_package_smoke_with_frozen_levels(tmp_path) -> None:
    config, fetcher = _frozen_package_inputs(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    artifacts = run_quarterly_economy_package(
        config_path=config_path,
        output_dir=tmp_path / "run",
        fetcher=fetcher,
        make_plots=False,
    )

    assert artifacts.report.exists()
    assert artifacts.metadata.exists()
    assert artifacts.baseline_forecast.exists()
    assert artifacts.scenario_summary.exists()
    report = artifacts.report.read_text(encoding="utf-8")
    assert "### All-variable baseline forecast panels" in report
    assert "all **19 observables**" in report
    assert "**21 model-implied variables**" in report
    assert "Baseline forecast panels were not rendered" in report
    # No saved mode, no refresh, no partial quarter: the legacy labels.
    assert "- Parameter source: DSGE.jl ss10 starting values/calibration; not re-estimated" in (
        report
    )
    assert "Parameter refresh" not in report
    assert "- Full information set ends: **2026-Q2**" in report
    assert "- Partial-quarter conditioning: none" in report
    assert "- First unconditioned forecast quarter: **2026-Q3**" in report
    assert "- Scenarios start: **2026-Q3**" in report
    metadata = json.loads(artifacts.metadata.read_text(encoding="utf-8"))
    assert metadata["data"]["quality"]["last_date"] == "2026-Q2"
    assert metadata["model"]["parameter_source"]["kind"] == "starting_values"
    assert metadata["model"]["refresh"]["enabled"] is False
    assert metadata["model"]["information_set"]["partial_quarters"] == []
    assert (
        metadata["historical_decomposition"]["observable_report_unit_reconciliation_max_abs_error"]
        < 1.0e-5
    )


def test_report_labels_saved_mode_with_partial_quarter_conditioning(tmp_path) -> None:
    config, fetcher = _frozen_package_inputs(tmp_path, last_quarter="2026-Q3")
    levels = pd.read_csv(config["fred_levels_path"])
    levels.loc[levels["date"] == "2026-Q3", "DFF"] = 3.667
    levels.to_csv(config["fred_levels_path"], index=False)
    public = pd.DataFrame(
        {
            "date": levels["date"],
            "ASACX10": 2.3,
            "FYCCZA": 4.787,
            "TFPKQ": 1.0,
            "TFPJQ": 0.5,
        }
    )
    public_path = tmp_path / "public.csv"
    public.to_csv(public_path, index=False)
    defaults = Model1002(subspec="ss10").parameters
    mode_path = tmp_path / "saved_mode.json"
    mode_path.write_text(
        json.dumps(
            {
                "status": "converged",
                "sample": {"end": "2026-Q2"},
                "log_posterior": -1039.15,
                "estimated_at": "2026-10-05",
                "estimated_parameters": ["rho_g", "sigma_g"],
                "parameter_values": {
                    "rho_g": float(defaults["rho_g"].value),
                    "sigma_g": float(defaults["sigma_g"].value),
                },
            }
        ),
        encoding="utf-8",
    )
    config.update(
        {
            "refresh_model": False,
            "mode_path": str(mode_path),
            "public_sources_path": str(public_path),
            "n_mon_anticipated_shocks": 0,
            "conditioning_end_date": "2026-Q3",
            "conditioning_observables": [
                "obs_nominalrate",
                "obs_spread",
                "obs_longrate",
                "obs_longinflation",
            ],
        }
    )

    report, metadata = _run_frozen_package(tmp_path, config, fetcher)

    q3_values = metadata["model"]["information_set"]["partial_quarters"][0]["observed_report_units"]
    spread = q3_values["obs_spread"]
    spf10 = q3_values["obs_longinflation"]  # SPF CPI 2.3 less the 0.5 pp wedge, PCE basis
    assert "Parameter refresh" not in report
    assert "Parameter source: re-estimated" not in report
    assert (
        "- Parameter source: loaded saved posterior mode `saved_mode.json` "
        f"(`{mode_path.resolve()}`); saved log posterior -1039.150;"
    ) in report
    assert "mode sample ends 2026-Q2; estimated 2026-10-05; not re-estimated in this run" in report
    assert "- Full information set ends: **2026-Q2**" in report
    assert (
        "- Partial-quarter conditioning: 2026-Q3: FFR 3.667 (actual quarter average), "
        f"spread {spread:.3f}, 10y 4.787, SPF10 {spf10:.3f} (PCE basis). Only these "
        "observables enter the Kalman filter in 2026-Q3"
    ) in " ".join(report.split())
    assert "- First unconditioned forecast quarter: **2026-Q4**" in report
    assert "- Scenarios start: **2026-Q4**" in report
    assert "The information set ends in" not in report
    assert spf10 == pytest.approx(400.0 * (np.exp(0.45 / 100.0) - 1.0))

    model_meta = metadata["model"]
    assert model_meta["refresh"]["enabled"] is False
    assert model_meta["refresh"]["source"] == "saved_mode"
    assert model_meta["parameter_source"]["kind"] == "saved_mode"
    assert model_meta["parameter_source"]["label"] in report
    information = model_meta["information_set"]
    assert information["full_information_end"] == "2026-Q2"
    assert information["first_unconditioned_quarter"] == "2026-Q4"
    assert information["scenario_start"] == "2026-Q4"
    (q3,) = information["partial_quarters"]
    assert q3["quarter"] == "2026-Q3"
    assert set(q3["observed_report_units"]) == set(config["conditioning_observables"])
    assert q3["observed_report_units"]["obs_nominalrate"] == pytest.approx(3.667)
    assert q3["requested_but_missing"] == []


def test_report_labels_parameter_refresh(tmp_path) -> None:
    config, fetcher = _frozen_package_inputs(tmp_path)
    config.update({"refresh_model": True, "refresh_maxiter": 1})

    report, metadata = _run_frozen_package(tmp_path, config, fetcher)

    refresh = metadata["model"]["refresh"]
    assert refresh["enabled"] is True
    assert refresh["source"] == "re_estimated"
    assert metadata["model"]["parameter_source"]["kind"] == "re_estimated"
    before = refresh["baseline_log_posterior"]
    after = refresh["updated_log_posterior"]
    expected = (
        "- Parameter source: re-estimated in this run: partial 8-parameter refresh "
        "(rho_g, rho_b, rho_mu, rho_ztil, sigma_g, sigma_b, sigma_mu, sigma_ztil) via Powell, "
        f"{refresh['iterations']} iterations ({refresh['function_evaluations']} function "
        f"evaluations, optimizer success `{refresh['optimizer_success']}`); log posterior "
        f"{before:.3f} -> {after:.3f} ({after - before:+.3f})"
    )
    assert expected in report
    assert report.count(expected) == 2  # executive summary and reproducibility section
    assert "loaded saved posterior mode" not in report
    assert "- Partial-quarter conditioning: none" in report
    assert "- First unconditioned forecast quarter: **2026-Q3**" in report
