from __future__ import annotations

import numpy as np
import pandas as pd

from nydsge.models import Model1002
from nydsge.scenarios import (
    build_unemployment_scenario_path,
    current_public_observables,
    estimate_unemployment_hours_bridge,
)


def test_estimate_unemployment_hours_bridge_recovers_known_change_slope() -> None:
    dates = [
        f"{period.year}-Q{period.quarter}"
        for period in pd.period_range("1985Q1", periods=48, freq="Q")
    ]
    unemployment_changes = np.asarray([0.2, -0.1, 0.0, 0.3] * 12, dtype=np.float64)
    unemployment = 6.0 + np.cumsum(unemployment_changes)
    hours = -45.0 + np.cumsum(0.05 - 1.75 * unemployment_changes)
    observables = pd.DataFrame({"date": dates, "obs_hours": hours})
    unemployment_frame = pd.DataFrame({"date": dates, "UNRATE": unemployment})

    bridge = estimate_unemployment_hours_bridge(
        observables,
        unemployment_frame,
        start_date="1985-Q1",
    )

    assert np.isclose(bridge.intercept, 0.05)
    assert np.isclose(bridge.slope, -1.75)
    assert np.isclose(bridge.r_squared, 1.0)


def test_build_unemployment_scenario_path_hits_holds_and_returns_from_target() -> None:
    baseline = np.full(20, 4.25)

    scenario = build_unemployment_scenario_path(baseline, target=8.0)

    assert scenario[0] == baseline[0]
    assert scenario[4] == 8.0
    np.testing.assert_allclose(scenario[5:8], 8.0)
    assert scenario[16] == baseline[16]
    np.testing.assert_allclose(scenario[17:], baseline[17:])


def test_current_public_observables_keeps_non_fred_inputs_explicitly_missing() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    levels = pd.DataFrame(
        {
            "date": ["2016-Q3", "2016-Q4"],
            "GDP": [100.0, 102.0],
            "CNP16OV": [10.0, 10.1],
            "GDPDEF": [100.0, 101.0],
            "AWHNONAG": [34.0, 34.1],
            "CE16OV": [150.0, 151.0],
            "COMPNFB": [100.0, 101.0],
            "PCEPILFE": [100.0, 100.5],
            "DFF": [0.4, 0.5],
            "PCE": [100.0, 101.0],
            "FPI": [100.0, 103.0],
            "BAA": [4.5, 4.6],
            "BAMLC8A0C15PYEY": [np.nan, np.nan],
            "GS10": [1.5, 1.6],
            "GDI": [100.0, 102.0],
        }
    )

    observables = current_public_observables(model, levels)

    assert observables["obs_gdp"].notna().sum() == 1
    assert observables["obs_longinflation"].isna().all()
    assert observables["obs_longrate"].isna().all()
    assert observables["obs_tfp"].isna().all()
