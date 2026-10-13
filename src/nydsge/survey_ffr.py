"""Public NY Fed SME/SPD median point paths, in Model1002 quarterly units.

Machine-readable files: https://www.newyorkfed.org/markets/market-intelligence/
survey-of-market-expectations . No extrapolation or substitutes for missing quarters.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd


def parse_survey_ffr(
    data: pd.DataFrame, *, available_date: str, panel: str = "Combined"
) -> pd.DataFrame:
    """Parse the official long-table workbook (2023+); require its publication date.

    `survey_release_date` is the question release date, not data publication.
    `aggregation_value` is an annual fraction. Use pctl50 of fftr_pathofmodes;
    take the last dated target in each quarter (including explicit quarter ends).
    """
    columns = {
        "survey_release_date",
        "panel_type",
        "question_tag",
        "aggregation",
        "horizon_date",
        "aggregation_value",
    }
    if not columns <= set(data):
        raise ValueError(f"Survey workbook is missing columns: {sorted(columns - set(data))}")
    rows = data.loc[
        (data.panel_type == panel)
        & (data.question_tag == "fftr_pathofmodes")
        & (data.aggregation == "pctl50")
    ].copy()
    if rows.empty:
        raise ValueError(f"Survey has no median FFR point path for panel {panel!r}.")
    rows["survey_date"] = pd.to_datetime(rows.survey_release_date, errors="raise")
    rows["target_date"] = pd.to_datetime(rows.horizon_date, errors="raise")
    rows["target_quarter"] = rows.target_date.dt.to_period("Q")
    rows["available_date"] = pd.Timestamp(available_date)
    if (rows.available_date < rows.survey_date).any():
        raise ValueError("Survey publication cannot precede question release.")
    rows["rate"] = pd.to_numeric(rows.aggregation_value, errors="raise") * 25.0
    if not np.isfinite(rows.rate).all() or ((rows.rate < -5) | (rows.rate > 25)).any():
        raise ValueError("Invalid survey rate or units.")
    if rows.duplicated(["survey_date", "target_date"]).any():
        raise ValueError("Duplicate median survey target.")
    rows = rows.sort_values("target_date").drop_duplicates(
        ["survey_date", "target_quarter"], keep="last"
    )
    return rows[["survey_date", "available_date", "target_quarter", "rate"]].reset_index(drop=True)


def load_survey_ffr(
    path: str | Path | bytes, *, available_date: str, panel: str = "Combined"
) -> pd.DataFrame:
    source = io.BytesIO(path) if isinstance(path, bytes) else path
    return parse_survey_ffr(pd.read_excel(source), available_date=available_date, panel=panel)


def expected_ffr_observations(
    records: pd.DataFrame, dates, *, as_of: str | None = None
) -> pd.DataFrame:
    """Use the latest published survey formed in each observation quarter.

    Later vintages never populate an earlier quarter. Quarters before 2021Q4
    and absent target horizons remain NaN. `as_of` prevents publication lookahead.
    """
    dates = pd.PeriodIndex(dates, freq="Q")
    result = pd.DataFrame(
        np.nan, index=dates, columns=[f"obs_exp_nominalrate{i}" for i in range(1, 7)]
    )
    records = records.copy()
    records["survey_date"] = pd.to_datetime(records.survey_date)
    records["available_date"] = pd.to_datetime(records.available_date)
    records["target_quarter"] = pd.PeriodIndex(records.target_quarter, freq="Q")
    cutoff = pd.Timestamp(as_of) if as_of else None
    for date in dates:
        if date < pd.Period("2021Q4", freq="Q"):
            continue
        eligible = records.loc[
            (records.survey_date.dt.to_period("Q") == date)
            & (records.available_date <= (cutoff if cutoff is not None else date.end_time))
        ]
        if eligible.empty:
            continue
        latest = eligible.loc[eligible.survey_date == eligible.survey_date.max()]
        if latest.duplicated("target_quarter").any():
            raise ValueError("Conflicting median rates for the same survey target quarter.")
        path = latest.set_index("target_quarter").rate
        for horizon in range(1, 7):
            result.loc[date, f"obs_exp_nominalrate{horizon}"] = path.get(date + horizon, np.nan)
    return result


def merge_expected_ffr(
    observations: pd.DataFrame, records: pd.DataFrame, *, as_of: str | None = None
) -> pd.DataFrame:
    """Attach survey observations to a transformed, quarterly-indexed history frame."""
    result = observations.copy()
    rates = expected_ffr_observations(records, result.index, as_of=as_of)
    for name in rates:
        result[name] = rates[name]
    return result
