"""Synthetic contract tests, not economic evidence."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from general_analytics_tools.decomposition import shock_groups
from general_analytics_tools.expectations import rate_expectations
from general_analytics_tools.inputs import quarter, read_levels
from general_analytics_tools.policy_paths import policy_paths
from nydsge.forecast import forecast_linear_system
from nydsge.regimes import RegimeSwitchingSystem
from nydsge.solve import Measurement, System, Transition


def metadata(**changes):
    return {
        "kind": "survey",
        "source_url": "https://example.org/synthetic-test",
        "vintage": "SYNTHETIC",
        "as_of": "2026-10-12T00:00:00Z",
        "units": "annual_percent",
        "rate_concept": "synthetic effective rate",
        "mapping_notes": "Synthetic test only.",
        **changes,
    }


def records():
    return pd.DataFrame(
        {
            "survey_date": ["2026-07-15"] * 3,
            "available_date": ["2026-10-12"] * 3,
            "target_quarter": ["2026Q3", "2026Q4", "2027Q1"],
            "rate": [9.0, 4.0, 3.0],
        }
    )


def test_expected_rates_align_units_and_do_not_shift_same_quarter_or_fill_missing():
    dates = pd.period_range("2026Q2", periods=2, freq="Q")
    result = rate_expectations(records(), dates, metadata(), quarter("2026Q2"))
    assert result.loc["2026Q2"].isna().all()
    np.testing.assert_allclose(result.loc["2026Q3"].iloc[:2], [1, 0.75])
    assert result.loc["2026Q3"].iloc[2:].isna().all()
    for units, multiplier in [("annual_fraction", 0.01), ("quarterly_model_rate", 0.25)]:
        frame = records()
        frame["rate"] *= multiplier
        pd.testing.assert_frame_equal(
            result, rate_expectations(frame, dates, metadata(units=units), quarter("2026Q2"))
        )


def test_expected_rates_pit_and_analysis_vintage_are_explicit_and_no_future_leak():
    dates = pd.period_range("2026Q3", periods=1, freq="Q")
    assert rate_expectations(records(), dates, metadata(), dates[0]).isna().all().all()
    assert (
        rate_expectations(
            records(), dates, metadata(historical_vintage_policy="analysis_vintage"), dates[0]
        )
        .notna()
        .any()
        .any()
    )
    assert (
        rate_expectations(records(), dates, metadata(as_of="2026-10-11T00:00:00Z"), dates[0] - 1)
        .isna()
        .all()
        .all()
    )
    ois = records().rename(columns={"survey_date": "observation_date"})
    pd.testing.assert_frame_equal(
        rate_expectations(records(), dates, metadata(), dates[0] - 1),
        rate_expectations(ois, dates, metadata(kind="ois"), dates[0] - 1),
    )
    with pytest.raises(ValueError, match="duplicate"):
        rate_expectations(pd.concat([records(), records()]), dates, metadata(), dates[0])
    with pytest.raises(ValueError, match="units"):
        rate_expectations(records(), dates, metadata(units="basis_points"), dates[0])


def test_extra_input_quarter_is_retained_and_required(tmp_path):
    frame = pd.DataFrame({"date": ["1999Q4", "2000Q1", "2000Q2"], "GDP": [100, 110, 121]})
    frame.to_csv(tmp_path / "levels.csv", index=False)
    (tmp_path / "source.json").write_text(
        json.dumps(
            {
                "vintage": "synthetic",
                "retrieved_at_utc": "2026-10-01T00:00:00Z",
                "source_urls": ["https://example.org/synthetic"],
            }
        )
    )
    config = {
        "inputs": {
            "levels_csv": str(tmp_path / "levels.csv"),
            "source_manifest": str(tmp_path / "source.json"),
        },
        "sample": {"presample_start": "2000Q1", "conditioning_end": "2000Q2"},
    }
    result, _ = read_levels(config, tmp_path)
    assert len(result) == 3 and result.GDP.pct_change().iloc[1] == pytest.approx(0.1)
    frame.iloc[1:].to_csv(tmp_path / "levels.csv", index=False)
    with pytest.raises(ValueError, match="one input quarter"):
        read_levels(config, tmp_path)


def news_system():
    # State 2 is next quarter's policy news, with an immediate information effect on output.
    t = np.array([[0.5, 1.0], [0, 0]])
    r = np.eye(2)
    z = np.array([[1.0, 0.0], [0.0, 1.0]])
    active = System(
        Transition(t, r, np.zeros(2)),
        Measurement(z, np.zeros(2), np.diag([1.0, 0.0]), np.zeros((2, 2))),
    )
    return RegimeSwitchingSystem((active,))


def test_held_and_guidance_targets_zero_shock_and_announcement_information():
    system = news_system()
    dates = pd.period_range("2027Q1", periods=4, freq="Q")
    regimes = np.ones(4, dtype=int)
    names, shocks = ["obs_nominalrate", "output"], ["rm_ait_sh", "rm_ait_shl1"]
    specs = [
        {
            "label": "held",
            "kind": "held_rate",
            "information": "sequential_surprises",
            "start_quarter": "2027Q2",
            "deviations_bp": [25, 25],
        },
        {
            "label": "news",
            "kind": "forward_guidance",
            "information": "announced",
            "announcement_quarter": "2027Q2",
            "start_quarter": "2027Q3",
            "deviations_bp": [25],
        },
        {
            "label": "zero",
            "kind": "held_rate",
            "information": "sequential_surprises",
            "start_quarter": "2027Q1",
            "deviations_bp": [0, 0],
        },
    ]
    result, report = policy_paths(system, np.zeros(2), regimes, dates, names, shocks, specs)
    np.testing.assert_allclose(4 * result["held"][1:3, 0], [0.25, 0.25], atol=1e-14)
    assert 4 * result["news"][2, 0] == pytest.approx(0.25)
    np.testing.assert_array_equal(result["news"][0], [0, 0])
    assert result["news"][1, 1] != 0  # information at announcement, before target
    np.testing.assert_array_equal(
        result["zero"],
        forecast_linear_system(system, np.zeros(2), horizon=4, regime_indices=regimes).observables,
    )
    assert report[1]["anticipated_shock_support"][0]["positive_prior_variance"] is False
    assert all(row["max_target_error_annual_pp"] <= 1e-8 for row in report)
    specs[1]["start_quarter"] = "2027Q2"
    with pytest.raises(ValueError, match="horizons 1 through 6"):
        policy_paths(system, np.zeros(2), regimes, dates, names, shocks, [specs[1]])


def test_price_and_wage_markups_are_separate_and_exhaustive():
    groups = shock_groups(["lambda_f_sh", "lambda_w_sh", "g_sh", "other"])
    assert groups["Price markups"] == [0] and groups["Wage markups"] == [1]
    assert sorted(i for group in groups.values() for i in group) == list(range(4))


def test_native_transform_uses_lead_then_excludes_it_from_filter_grid():
    from general_analytics_tools.compute import dated_observations
    from nydsge.models import Model1002
    from nydsge.scenarios import current_public_observables

    model = Model1002("ss104", settings={"date_presample_start": "2000Q1"})
    levels = pd.DataFrame(
        {
            "date": ["1999-Q4", "2000-Q1", "2000-Q2", "2000-Q3"],
            "GDP": [100.0, 110.0, 121.0, 133.1],
            "GDPDEF": [100.0, 100.0, 100.0, 100.0],
            "CNP16OV": [100.0, 100.0, 100.0, 100.0],
        }
    )
    full = current_public_observables(model, levels)
    result = dated_observations(model, levels)
    assert list(result.index.astype(str)) == ["2000Q1", "2000Q2", "2000Q3"]
    assert result.iloc[0].obs_gdp == pytest.approx(full.iloc[1].obs_gdp)
    assert np.isfinite(result.iloc[0].obs_gdp)
    truncated = current_public_observables(model, levels.iloc[1:])
    assert pd.isna(truncated.iloc[0].obs_gdp)


def test_absolute_hold_and_unreachable_guidance_are_verified():
    system = news_system()
    dates = pd.period_range("2027Q1", periods=3, freq="Q")
    regimes = np.ones(3, dtype=int)
    names, shocks = ["obs_nominalrate", "output"], ["rm_ait_sh", "rm_ait_shl1"]
    hold = {
        "label": "constant",
        "kind": "held_rate",
        "information": "sequential_surprises",
        "start_quarter": "2027Q1",
        "rates_annual_percent": [3.75, None, 3.75],
    }
    result, _ = policy_paths(system, np.array([1.0, 0.0]), regimes, dates, names, shocks, [hold])
    np.testing.assert_allclose(4 * result["constant"][[0, 2], 0], [3.75, 3.75], atol=1e-14)
    active = system.systems[0]
    bad = System(
        Transition(active.transition.TTT, np.array([[1.0, 0.0], [0.0, 0.0]]), np.zeros(2)),
        active.measurement,
    )
    spec = {
        "label": "unreachable",
        "kind": "forward_guidance",
        "information": "announced",
        "announcement_quarter": "2027Q1",
        "start_quarter": "2027Q2",
        "deviations_bp": [25],
    }
    with pytest.raises(ValueError, match="not independently reachable"):
        policy_paths(
            RegimeSwitchingSystem((bad,)), np.zeros(2), regimes, dates, names, shocks, [spec]
        )
