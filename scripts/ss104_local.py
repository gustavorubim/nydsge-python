"""Offline ss104 construction on the existing 2026-10-06 public snapshots."""

from __future__ import annotations

import hashlib

import pandas as pd
from c1c2_common import FRED_SNAPSHOT, PUBLIC_SNAPSHOT, ROOT, build_observables

from nydsge.models import Model1002
from nydsge.ss104_workflow import prepare_ss104_observations
from nydsge.survey_ffr import merge_expected_ffr

SURVEY_SNAPSHOT = ROOT / "data/public/ss104_survey_ffr_local.csv"
AS_OF = "2026-10-12"


def build_local_model() -> Model1002:
    return Model1002(
        "ss104",
        settings={
            "data_vintage": "261006",
            "date_presample_start": "1959-Q3",
            "date_mainsample_start": "1960-Q1",
            "date_forecast_start": "2026-Q3",
        },
    )


def build_local_data(model, *, end="2026-Q3"):
    frame = build_observables(model, end=end)
    frame.index = pd.PeriodIndex(frame.pop("date"), freq="Q")
    frame = merge_expected_ffr(frame, pd.read_csv(SURVEY_SNAPSHOT), as_of=AS_OF)
    return prepare_ss104_observations(model, frame)


def input_provenance():
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in (FRED_SNAPSHOT, PUBLIC_SNAPSHOT, SURVEY_SNAPSHOT)
    }
