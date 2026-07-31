from __future__ import annotations

import json
from pathlib import Path

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


def test_quarterly_package_smoke_with_frozen_levels(tmp_path) -> None:
    periods = pd.period_range("2014-Q3", "2026-Q2", freq="Q")
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
        "model_end_date": dates[-1],
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
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monthly_dates = [f"{period.year}-{3 * period.quarter:02d}-01" for period in periods]
    unemployment = 5.0 + 0.15 * np.sin(steps / 3.0)
    payload = "DATE,UNRATE\n" + "\n".join(
        f"{date},{value}" for date, value in zip(monthly_dates, unemployment, strict=True)
    )

    artifacts = run_quarterly_economy_package(
        config_path=config_path,
        output_dir=tmp_path / "run",
        fetcher=lambda _: payload.encode(),
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
    metadata = json.loads(artifacts.metadata.read_text(encoding="utf-8"))
    assert metadata["data"]["quality"]["last_date"] == "2026-Q2"
    assert (
        metadata["historical_decomposition"]["observable_report_unit_reconciliation_max_abs_error"]
        < 1.0e-5
    )
