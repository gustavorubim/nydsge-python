"""Small synthetic regressions for dated transforms and conditional-origin contracts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nydsge.data import (
    FredApiError,
    date_labels_for_sample,
    df_to_matrix,
    load_fred_api_source,
    transform_data,
    validate_quarterly_grid,
)
from nydsge.models import Model1002
from nydsge.regimes import RegimeSwitchingSystem
from nydsge.solve import Measurement, PseudoMeasurement, System, Transition
from nydsge.ss104_workflow import forecast_ss104


@pytest.mark.parametrize(
    "dates, message",
    [
        (["2020Q1", "2020Q3"], "contiguous"),
        (["2020Q1", "2020-Q1"], "duplicate"),
        (["2020Q1", "2020-03-31"], "duplicate"),
        (["2020Q1", "2020Q5"], "Invalid quarterly"),
    ],
)
def test_raw_transforms_and_observable_matrix_reject_bad_quarter_grid(dates, message):
    model = Model1002(settings={"date_forecast_start": "2020Q4", "date_mainsample_start": "2020Q1"})
    with pytest.raises(ValueError, match=message):
        transform_data(model, pd.DataFrame({"date": dates}))
    observations = pd.DataFrame({"date": dates, **{name: [0.0, 1.0] for name in model.observables}})
    with pytest.raises(ValueError, match=message):
        df_to_matrix(model, observations)


def test_grid_uses_calendar_order_preserves_missing_quarter_row_and_partial_windows():
    model = Model1002(settings={"date_forecast_start": "2020Q4", "date_mainsample_start": "2020Q1"})
    frame = pd.DataFrame(
        {
            "date": ["2020Q3", "2020-Q1", "2020-06-30"],
            **{name: [3.0, 1.0, np.nan] for name in model.observables},
        }
    )
    matrix = df_to_matrix(model, frame)
    assert matrix.shape[0] == 3
    assert matrix[0, 0] == 1.0 and matrix[2, 0] == 3.0
    assert np.isnan(matrix[1]).all()
    # A utility transform/matrix can represent a partial research window;
    # the forecast boundary separately enforces its origin/end requirements.
    assert df_to_matrix(model, frame.iloc[[1]]).shape[0] == 1
    with pytest.raises(ValueError, match="end at 2020-Q4"):
        validate_quarterly_grid(frame, required_end="2020Q4")
    with pytest.raises(ValueError, match="start at 2019-Q4"):
        validate_quarterly_grid(frame, required_start="2019Q4")


def test_explicit_date_index_is_validated_instead_of_silently_discarded():
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.PeriodIndex(["2020Q1", "2020Q3"], freq="Q"))
    with pytest.raises(ValueError, match="contiguous"):
        validate_quarterly_grid(frame)
    frame.index = pd.period_range("2020Q1", periods=2, freq="Q")
    valid = validate_quarterly_grid(frame, required_start="2020Q1", required_end="2020Q2")
    assert list(valid["date"].astype(str)) == ["2020Q1", "2020Q2"]
    model = Model1002(settings={"date_forecast_start": "2020Q3", "date_mainsample_start": "2020Q1"})
    assert date_labels_for_sample(model, frame) == ["2020-Q1", "2020-Q2"]


class ScalarModel:
    observables = {"obs_nominalrate": None}

    def get_setting(self, key: str, default: Any = None) -> Any:
        return {
            "date_forecast_start": "2026Q3",
            "date_mainsample_start": "2020Q1",
            "pgap_ygap_init_date": "2020Q1",
            "pgap_value": 0.0,
            "ygap_value": 0.0,
            "regime_dates": {1: "2020Q1"},
            "cond_full_names": ["obs_nominalrate"],
            "cond_semi_names": ["obs_nominalrate"],
        }.get(key, default)


def scalar_system():
    return RegimeSwitchingSystem(
        (
            System(
                Transition(np.array([[0.5]]), np.ones((1, 1)), np.zeros(1)),
                Measurement(np.ones((1, 1)), np.zeros(1), np.ones((1, 1)), np.ones((1, 1))),
                PseudoMeasurement(np.ones((1, 1)), np.zeros(1)),
            ),
        )
    )


@pytest.mark.parametrize("kind", ["full", "semi"])
def test_ss104_conditional_forecasts_require_actual_conditional_rows_and_prior_quarter(kind):
    model, system = ScalarModel(), scalar_system()
    with pytest.raises(ValueError, match="conditioning requires dated observations"):
        forecast_ss104(model, system=system, cond_type=kind, horizon=2)
    history = pd.DataFrame(
        {"obs_nominalrate": [3.0]}, index=pd.period_range("2026Q2", periods=1, freq="Q")
    )
    with pytest.raises(ValueError, match="conditioning requires dated observations"):
        forecast_ss104(model, system=system, data=history, cond_type=kind, horizon=2)
    for start in ["2026Q3", "2026Q4"]:
        late = history.copy()
        late.index = pd.period_range(start, periods=1, freq="Q")
        with pytest.raises(ValueError, match="quarter preceding forecast start"):
            forecast_ss104(model, system=system, data=late, cond_type=kind, horizon=2)
    valid = pd.DataFrame(
        {"obs_nominalrate": [3.0, 3.1]}, index=pd.period_range("2026Q2", periods=2, freq="Q")
    )
    output = forecast_ss104(model, system=system, data=valid, cond_type=kind, horizon=2)
    assert len(output.history_states) == 1
    assert len(output.conditional_states) == 1
    assert len(output.states) == 1  # Q4 follows the correctly dated Q3 condition.


@pytest.mark.parametrize(
    "code,message",
    [
        (400, "Invalid API key"),
        (429, "Rate limit exceeded"),
        (500, "Internal server error"),
        (400, "Invalid realtime range"),
    ],
)
def test_optional_fred_series_does_not_swallow_api_failures(code, message):
    def fetcher(url):
        if "BAMLC8A0C15PYEY" in url:
            return json.dumps({"error_code": code, "error_message": message}).encode()
        return b'{"observations":[{"date":"2020-01-01","value":"1.0"}]}'

    with pytest.raises(FredApiError, match=message) as error:
        load_fred_api_source(
            ["GDP", "BAMLC8A0C15PYEY"],
            api_key="synthetic",
            fetcher=fetcher,
            start_date="2020Q1",
            end_date="2020Q1",
        )
    assert error.value.code == code


@pytest.mark.parametrize("payload", [b"{}", b'{"observations":{}}', b"not-json"])
def test_optional_fred_series_does_not_swallow_malformed_payload(payload):
    with pytest.raises(ValueError):
        load_fred_api_source(
            ["BAMLC8A0C15PYEY"],
            api_key="synthetic",
            fetcher=lambda _: payload,
            start_date="2020Q1",
            end_date="2020Q1",
        )
