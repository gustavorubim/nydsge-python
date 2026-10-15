"""Explicit, dated expected-rate measurements; no interpolation or vintage inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nydsge.survey_ffr import expected_ffr_observations


def rate_expectations(
    records: pd.DataFrame, dates: pd.PeriodIndex, metadata: dict[str, Any], history_end: pd.Period
) -> pd.DataFrame:
    """Map survey/OIS point paths to the native six noisy expected-FFR rows.

    OIS is an explicitly declared proxy for the same measurement, not a new
    term-premium or risk-neutral measurement equation.
    """
    required = ("kind", "source_url", "vintage", "as_of", "units", "rate_concept", "mapping_notes")
    if not all(metadata.get(key) for key in required):
        raise ValueError(
            "Expected-rate metadata requires source, vintage, as_of, units and mapping."
        )
    if metadata["kind"] not in ("survey", "ois") or not metadata["source_url"].startswith(
        "https://"
    ):
        raise ValueError("Expected-rate kind must be survey/ois with an HTTPS source.")
    factors = {"annual_percent": 0.25, "annual_fraction": 25.0, "quarterly_model_rate": 1.0}
    if metadata["units"] not in factors:
        raise ValueError("Unsupported expected-rate units.")
    policy = metadata.get("historical_vintage_policy", "point_in_time")
    if policy not in ("point_in_time", "analysis_vintage"):
        raise ValueError("Unsupported historical vintage policy.")
    as_of = pd.Timestamp(metadata["as_of"])
    if (
        not isinstance(as_of, pd.Timestamp)
        or as_of.tzinfo is None
        or as_of.utcoffset() != pd.Timedelta(0)
    ):
        raise ValueError("Expected-rate as_of must be timezone-aware UTC.")
    as_of = as_of.tz_localize(None)
    frame = records.copy()
    if metadata["kind"] == "ois":
        frame = frame.rename(columns={"observation_date": "survey_date"})
    for key in ("survey_date", "available_date"):
        frame[key] = pd.to_datetime(frame[key], errors="raise")
        if frame[key].isna().any():
            raise ValueError("Missing expected-rate dates.")
    if (frame.available_date < frame.survey_date).any():
        raise ValueError("Expected-rate publication precedes formation.")
    frame["target_quarter"] = pd.PeriodIndex(frame.target_quarter, freq="Q")
    frame["rate"] = pd.to_numeric(frame.rate, errors="raise") * factors[metadata["units"]]
    if (
        not np.isfinite(frame.rate).all()
        or frame.duplicated(["survey_date", "target_quarter"]).any()
    ):
        raise ValueError("Nonfinite rates or duplicate expected-rate targets.")
    if (frame.target_quarter < frame.survey_date.dt.to_period("Q")).any():
        raise ValueError("Expected-rate targets precede formation quarter.")
    rows = []
    for date in dates:
        cutoff = as_of
        if policy == "point_in_time" and date <= history_end:
            cutoff = min(cutoff, date.end_time)
        rows.append(expected_ffr_observations(frame, [date], as_of=cutoff.isoformat()).iloc[0])
    return pd.DataFrame(rows, index=dates)
