from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nydsge.survey_ffr import expected_ffr_observations, parse_survey_ffr


def test_official_survey_units_last_target_and_vintage_selection():
    rows = pd.DataFrame(
        {
            "survey_release_date": ["2026-07-15"] * 3 + ["2026-09-02"],
            "panel_type": ["Combined"] * 4,
            "question_tag": ["fftr_pathofmodes"] * 4,
            "aggregation": ["pctl50"] * 4,
            "horizon_date": ["2026-10-28", "2026-12-09", "2027-03-31", "2026-12-31"],
            "aggregation_value": [0.0388, 0.0363, 0.0338, 0.0313],
        }
    )
    july = parse_survey_ffr(rows.iloc[:3], available_date="2026-08-19")
    sept = parse_survey_ffr(rows.iloc[3:], available_date="2026-10-07")
    records = pd.concat([july, sept])
    result = expected_ffr_observations(records, ["2021Q3", "2026Q3", "2026Q4"], as_of="2026-09-30")
    assert result.loc["2026Q3", "obs_exp_nominalrate1"] == 0.0363 * 25
    assert result.loc["2026Q3", "obs_exp_nominalrate2"] == 0.0338 * 25
    assert result.loc["2026Q3"].iloc[2:].isna().all()
    assert result.loc["2021Q3"].isna().all()
    assert result.loc["2026Q4"].isna().all()
    assert expected_ffr_observations(records, ["2026Q3"], as_of="2026-08-01").isna().all().all()
    with pytest.raises(ValueError, match="Duplicate"):
        parse_survey_ffr(pd.concat([rows, rows.iloc[:1]]), available_date="2026-10-07")
    with pytest.raises(ValueError, match="no median"):
        parse_survey_ffr(rows, available_date="2026-10-07", panel="Dealer")
    assert np.isfinite(july.rate).all()


def test_real_sme_point_path_fixture():
    rows = pd.read_csv(Path(__file__).parent / "fixtures" / "sme_jul2026_ffr.csv")
    records = parse_survey_ffr(rows, available_date="2026-10-12")
    actual = expected_ffr_observations(records, ["2026Q3"], as_of="2026-10-12")
    np.testing.assert_allclose(actual.iloc[0], [0.9075] * 5 + [0.845], atol=1e-15, rtol=0)
