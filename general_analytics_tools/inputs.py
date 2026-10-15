"""Explicit input contracts and optional preservation of rolling yield history."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nydsge.data import load_current_fred_series


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quarter(value: str) -> pd.Period:
    result = pd.Period(value.replace("-Q", "Q"), freq="Q")
    if not isinstance(result, pd.Period):
        raise ValueError("A finite quarterly date is required.")
    return result


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "model",
        "parameters",
        "inputs",
        "sample",
        "conditioning_observables",
        "complete_conditioning_observables",
        "forecast_quarters",
        "policy_impacts_bp",
        "plot_history_start",
        "policy_shock",
        "rate_expectations",
        "survey_ablation",
        "population_endpoint",
        "policy_paths",
    }
    if set(config) - allowed:
        raise ValueError(f"Unsupported config keys: {sorted(set(config) - allowed)}")
    if config.get("model") != "ss104" or config.get("parameters") != "public_defaults":
        raise ValueError("This toolkit supports ss104 public_defaults only; no M7/mode/posterior.")
    sample = config["sample"]
    start, main, end, cond = (
        quarter(sample[k])
        for k in ("presample_start", "mainsample_start", "history_end", "conditioning_end")
    )
    if (
        not start.ordinal <= main.ordinal <= end.ordinal <= cond.ordinal
        or cond.ordinal > end.ordinal + 4
    ):
        raise ValueError("Require ordered sample dates and at most four conditional quarters.")
    horizon = config["forecast_quarters"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 40:
        raise ValueError("forecast_quarters must be an integer from 1 to 40.")
    impacts = config.get("policy_impacts_bp", [25, 50, -25])
    if not 1 <= len(impacts) <= 8 or len(set(impacts)) != len(impacts):
        raise ValueError("Provide 1–8 distinct policy impacts.")
    if any(isinstance(x, bool) or not np.isfinite(x) or not 0 < abs(x) <= 200 for x in impacts):
        raise ValueError("Policy impacts must be finite, nonzero and within +/-200 bp.")
    config["policy_impacts_bp"] = impacts
    if config.get("population_endpoint", "recompute") not in ("recompute", "freeze_history"):
        raise ValueError("population_endpoint must be recompute or freeze_history.")
    if ("expectations_csv" in config["inputs"]) != ("rate_expectations" in config):
        raise ValueError("Expected-rate inputs and metadata must be supplied together.")
    if not isinstance(config.get("survey_ablation", False), bool):
        raise ValueError("survey_ablation must be boolean.")
    paths = config.get("policy_paths", [])
    if not isinstance(paths, list) or len(paths) + len(impacts) > 8:
        raise ValueError("At most eight combined surprise and policy-path scenarios.")
    channels = config.get("conditioning_observables", [])
    if len(channels) != len(set(channels)):
        raise ValueError("Conditioning channels must be unique.")
    declared = config.get("complete_conditioning_observables", [])
    if set(channels) - set(declared):
        raise ValueError("Every conditioning channel must be declared complete in the config.")
    paths = config["inputs"]
    for key in ("levels_csv", "source_manifest"):
        if key not in paths:
            raise ValueError(f"inputs.{key} is required.")
    for key, value in paths.items():
        resolved = (path.parent / value).resolve()
        if not resolved.is_file():
            raise ValueError(f"Missing input {key}: {resolved}")
        paths[key] = str(resolved)
    if ("preserved_yield_csv" in paths) != ("current_yield_csv" in paths):
        raise ValueError(
            "Yield preservation requires both preserved_yield_csv and current_yield_csv."
        )
    return config


def preserve_yield_history(
    old: pd.DataFrame, new: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve missing dated observations; conflicting overlap requires explicit review."""
    name = "BAMLC8A0C15PYEY"
    frames = []
    for source in (old, new):
        frame = source.copy()
        if frame.shape[1] != 2:
            raise ValueError("Yield files must have exactly date and value columns.")
        frame.columns = ["date", name]
        frame["date"] = pd.to_datetime(frame.date, errors="raise").dt.strftime("%Y-%m-%d")
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if frame.date.duplicated().any():
            raise ValueError("Duplicate yield dates.")
        frames.append(frame)
    a, b = frames
    overlap = a.merge(b, on="date", suffixes=("_old", "_new"))
    if overlap.empty:
        raise ValueError("Yield snapshots need verified overlapping observations.")
    if not np.allclose(overlap.iloc[:, 1], overlap.iloc[:, 2], atol=0, rtol=0, equal_nan=True):
        raise ValueError("Conflicting yield overlap: review genuine revisions before preservation.")
    restored = a.loc[~a.date.isin(b.date)]
    union = pd.concat([a, b]).drop_duplicates("date", keep="last").sort_values("date")
    return union.reset_index(drop=True), restored.reset_index(drop=True)


def read_levels(config: dict[str, Any], output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    inputs = config["inputs"]
    manifest = json.loads(Path(inputs["source_manifest"]).read_text(encoding="utf-8"))
    if not all(manifest.get(k) for k in ("vintage", "retrieved_at_utc", "source_urls")):
        raise ValueError("Source manifest requires vintage, retrieved_at_utc and source_urls.")
    retrieved = datetime.fromisoformat(str(manifest["retrieved_at_utc"]).replace("Z", "+00:00"))
    if retrieved.tzinfo is None or retrieved.utcoffset() != timedelta(0):
        raise ValueError("retrieved_at_utc must be a timezone-aware UTC timestamp.")
    if not isinstance(manifest["source_urls"], list) or not all(
        isinstance(url, str) and url.startswith("https://") for url in manifest["source_urls"]
    ):
        raise ValueError("source_urls must be a nonempty list of actual HTTPS source URLs.")
    levels = pd.read_csv(inputs["levels_csv"])
    if "date" not in levels:
        raise ValueError("Quarterly levels require a date column.")
    if "public_sources_csv" in inputs:
        from nydsge.public_sources import load_public_sources_csv, merge_public_sources

        levels["date"] = levels.date.map(lambda x: str(quarter(x)).replace("Q", "-Q"))
        public = load_public_sources_csv(inputs["public_sources_csv"])
        public["date"] = public.date.map(lambda x: str(quarter(x)).replace("Q", "-Q"))
        levels = merge_public_sources(levels, public)
    dates = pd.PeriodIndex(levels.date.map(quarter), freq="Q")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Quarterly levels must be unique and ordered.")
    sample = config["sample"]
    start, end = quarter(sample["presample_start"]) - 1, quarter(sample["conditioning_end"])
    levels = levels.loc[(dates >= start) & (dates <= end)].copy()
    dates = pd.PeriodIndex(levels.date.map(quarter), freq="Q")
    expected = pd.period_range(start, end, freq="Q")
    if not dates.equals(expected) or len(dates) > 500:
        raise ValueError(
            "Require one input quarter before presample_start and a contiguous grid, "
            "at most 500 quarters; use NaN."
        )
    if "preserved_yield_csv" in inputs:
        union, restored = preserve_yield_history(
            pd.read_csv(inputs["preserved_yield_csv"]), pd.read_csv(inputs["current_yield_csv"])
        )
        union.to_csv(output / "preserved_yield_history.csv", index=False)
        restored.to_csv(output / "preserved_yield_dates.csv", index=False)
        quarterly = load_current_fred_series(
            "BAMLC8A0C15PYEY", fetcher=lambda _: union.to_csv(index=False).encode()
        )
        quarter_dates = pd.PeriodIndex(quarterly.date.map(quarter), freq="Q")
        values = pd.Series(quarterly.BAMLC8A0C15PYEY.to_numpy(), index=quarter_dates)
        replacement = values.reindex(dates).to_numpy()
        if "BAMLC8A0C15PYEY" not in levels:
            levels["BAMLC8A0C15PYEY"] = np.nan
        mask = np.isfinite(replacement)
        levels.loc[mask, "BAMLC8A0C15PYEY"] = replacement[mask]
        manifest["yield_missing_dates_preserved"] = len(restored)
    manifest["input_sha256"] = {key: sha256(Path(value)) for key, value in inputs.items()}
    levels.to_csv(output / "frozen_levels.csv", index=False)
    return levels, manifest
