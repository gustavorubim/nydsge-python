"""Shared data/model construction for the C1/C2 re-estimation scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nydsge.data import df_to_matrix
from nydsge.models import Model1002
from nydsge.public_sources import load_public_sources_csv, merge_public_sources
from nydsge.scenarios import current_public_observables

ROOT = Path(__file__).resolve().parents[1]
FRED_SNAPSHOT = ROOT / "data/public/fred_levels_20261006.csv"
PUBLIC_SNAPSHOT = ROOT / "data/public/nonfred_sources_20261006.csv"
PRESAMPLE_START = "1959-Q3"
MAINSAMPLE_START = "1960-Q1"
MODEL_END = "2026-Q2"
FORECAST_START = "2026-Q3"
N_PRESAMPLE = 2


def qidx(label: str) -> int:
    y, q = label.split("-Q")
    return int(y) * 4 + int(q)


def build_model(n_ant: int = 0, **extra: object) -> Model1002:
    settings = {
        "data_vintage": "261006",
        "date_presample_start": PRESAMPLE_START,
        "date_mainsample_start": MAINSAMPLE_START,
        "date_forecast_start": FORECAST_START,
        "n_mon_anticipated_shocks": n_ant,
    }
    settings.update(extra)
    return Model1002("ss10", settings=settings)


def build_observables(
    model: Model1002,
    *,
    with_public: bool = True,
    end: str = MODEL_END,
    start: str = PRESAMPLE_START,
) -> pd.DataFrame:
    levels = pd.read_csv(FRED_SNAPSHOT)
    if with_public:
        levels = merge_public_sources(levels, load_public_sources_csv(PUBLIC_SNAPSHOT))
    q = levels["date"].map(qidx)
    # one extra leading quarter so first-differenced observables exist at `start`
    levels = levels.loc[(q >= qidx(start) - 1) & (q <= qidx(end))].reset_index(drop=True)
    obs = current_public_observables(model, levels)
    obs = obs.loc[obs["date"].map(qidx) >= qidx(start)].reset_index(drop=True)
    return obs


def observation_matrix(model: Model1002, obs: pd.DataFrame) -> np.ndarray:
    return np.asarray(df_to_matrix(model, obs), dtype=np.float64)
