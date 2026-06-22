from __future__ import annotations

import io
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from nydsge.core import DSGEModel, Observable

FRED_GRAPH_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
DOTENV_FILENAME = ".env"
FredFetcher = Callable[[str], bytes]


class DataNotAvailableError(FileNotFoundError):
    """Raised when no local data fixture or input dataset is available."""


def annual_to_quarter(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / 4.0


def quarter_to_annual(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) * 4.0


def one_quarter_pct_change(values: Any) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    out = np.full_like(series, np.nan, dtype=np.float64)
    out[1:] = 100.0 * np.diff(np.log(series))
    return out


def loggrowth_to_pct_annualized(values: Any) -> np.ndarray:
    return 400.0 * (np.exp(np.asarray(values, dtype=np.float64) / 100.0) - 1.0)


def loggrowth_to_pct_annualized_percapita(values: Any) -> np.ndarray:
    return loggrowth_to_pct_annualized(values)


def identity(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def reverse_transform_array(transform_name: str, values: Any) -> np.ndarray:
    registry: dict[str, Callable[[Any], np.ndarray]] = {
        "identity": identity,
        "quarter_to_annual": quarter_to_annual,
        "loggrowth_to_pct_annualized": loggrowth_to_pct_annualized,
        "loggrowth_to_pct_annualized_percapita": loggrowth_to_pct_annualized_percapita,
    }
    try:
        transform = registry[transform_name]
    except KeyError as err:
        msg = f"Unknown reverse transform: {transform_name}"
        raise KeyError(msg) from err
    return transform(values)


def reverse_transform_observables(model: DSGEModel, values: Any) -> np.ndarray:
    """Convert model-unit observables into the reporting units declared on the model."""

    observable_values = np.asarray(values, dtype=np.float64)
    if observable_values.ndim != 2:
        msg = "Observable forecasts must have shape (periods, observables)."
        raise ValueError(msg)

    expected_count = len(model.observable_mappings)
    if observable_values.shape[1] != expected_count:
        msg = f"Observable forecasts must have {expected_count} columns."
        raise ValueError(msg)

    transformed = np.empty_like(observable_values, dtype=np.float64)
    for column, observable in enumerate(model.observable_mappings.values()):
        transformed[:, column] = reverse_transform_array(
            observable.reverse_transform,
            observable_values[:, column],
        )
    return transformed


def reverse_transform_pseudo_observables(model: DSGEModel, values: Any) -> np.ndarray:
    """Convert model-unit pseudo-observables into their declared reporting units."""

    pseudo_values = np.asarray(values, dtype=np.float64)
    if pseudo_values.ndim != 2:
        msg = "Pseudo-observable forecasts must have shape (periods, pseudo-observables)."
        raise ValueError(msg)

    expected_count = len(model.pseudo_observable_mappings)
    if pseudo_values.shape[1] != expected_count:
        msg = f"Pseudo-observable forecasts must have {expected_count} columns."
        raise ValueError(msg)

    transformed = np.empty_like(pseudo_values, dtype=np.float64)
    for column, pseudo_observable in enumerate(model.pseudo_observable_mappings.values()):
        transformed[:, column] = reverse_transform_array(
            pseudo_observable.reverse_transform,
            pseudo_values[:, column],
        )
    return transformed


def transform_data(
    model: DSGEModel,
    levels: pd.DataFrame,
    *,
    population_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Transform raw level data into model observables.

    This is the executable counterpart to `Model1002`'s observable metadata. It
    supports the core public FRBNY observables and leaves the exact FRED/non-FRED
    data acquisition workflow to the later data-builder port.
    """

    working = prepare_population_data(
        model,
        _sort_by_date(levels.copy()),
        population_forecast=population_forecast,
    )
    transformed = pd.DataFrame(index=working.index)
    if "date" in working.columns:
        transformed["date"] = working["date"]
    for observable in model.observable_mappings.values():
        transformed[observable.name] = _apply_forward_transform(model, observable, working)
    return transformed.reset_index(drop=True)


def build_data_csv(
    model: DSGEModel,
    *,
    input_path: Path | str,
    output_path: Path | str,
    population_forecast_path: Path | str | None = None,
) -> pd.DataFrame:
    levels = pd.read_csv(input_path)
    population_forecast = (
        None if population_forecast_path is None else pd.read_csv(population_forecast_path)
    )
    transformed = transform_data(model, levels, population_forecast=population_forecast)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(destination, index=False)
    return transformed


def build_data_csv_from_sources(
    model: DSGEModel,
    *,
    source_root: Path | str,
    output_path: Path | str,
    vintage: str | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    population_forecast_path: Path | str | None = None,
) -> pd.DataFrame:
    levels = load_data_levels_from_sources(
        model,
        source_root=source_root,
        vintage=vintage,
        start_date=start_date,
        end_date=end_date,
    )
    population_forecast = (
        None if population_forecast_path is None else pd.read_csv(population_forecast_path)
    )
    transformed = transform_data(model, levels, population_forecast=population_forecast)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(destination, index=False)
    return transformed


def download_current_fred_source_csv(
    model: DSGEModel,
    *,
    output_path: Path | str,
    start_date: Any | None = None,
    end_date: Any | None = None,
    aggregation: str = "mean",
    fetcher: FredFetcher | None = None,
) -> pd.DataFrame:
    mnemonics = parse_data_sources(model).get("FRED", [])
    levels = load_current_fred_source(
        mnemonics,
        start_date=start_date,
        end_date=end_date,
        aggregation=aggregation,
        fetcher=fetcher,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    levels.to_csv(destination, index=False)
    return levels


def download_fred_api_source_csv(
    model: DSGEModel,
    *,
    output_path: Path | str,
    api_key: str | None = None,
    realtime_start: str | None = None,
    realtime_end: str | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    aggregation: str = "mean",
    fetcher: FredFetcher | None = None,
) -> pd.DataFrame:
    mnemonics = parse_data_sources(model).get("FRED", [])
    levels = load_fred_api_source(
        mnemonics,
        api_key=api_key,
        realtime_start=realtime_start,
        realtime_end=realtime_end,
        start_date=start_date,
        end_date=end_date,
        aggregation=aggregation,
        fetcher=fetcher,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    levels.to_csv(destination, index=False)
    return levels


def load_fred_api_source(
    mnemonics: list[str],
    *,
    api_key: str | None = None,
    realtime_start: str | None = None,
    realtime_end: str | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
    aggregation: str = "mean",
    fetcher: FredFetcher | None = None,
) -> pd.DataFrame:
    required_dates = _required_quarter_indexes(start_date=start_date, end_date=end_date)
    merged: pd.DataFrame | None = None
    resolved_api_key = _resolve_fred_api_key(api_key)
    for mnemonic in mnemonics:
        series = load_fred_api_series(
            mnemonic,
            api_key=resolved_api_key,
            realtime_start=realtime_start,
            realtime_end=realtime_end,
            observation_start=start_date,
            observation_end=end_date,
            aggregation=aggregation,
            fetcher=fetcher,
        )
        if required_dates is not None:
            _validate_source_date_coverage(f"FRED:{mnemonic}", series["date"], required_dates)
            date_index = _quarter_index(series["date"])
            required_set = set(required_dates.tolist())
            series = series.loc[[date in required_set for date in date_index]].reset_index(
                drop=True
            )
        merged = series if merged is None else pd.merge(merged, series, on="date", how="outer")
    if merged is None:
        return pd.DataFrame({"date": []})
    return _sort_by_date(merged)


def load_fred_api_series(
    mnemonic: str,
    *,
    api_key: str,
    realtime_start: str | None = None,
    realtime_end: str | None = None,
    observation_start: Any | None = None,
    observation_end: Any | None = None,
    aggregation: str = "mean",
    fetcher: FredFetcher | None = None,
) -> pd.DataFrame:
    url = _fred_observations_url(
        mnemonic,
        api_key=api_key,
        realtime_start=realtime_start,
        realtime_end=realtime_end,
        observation_start=observation_start,
        observation_end=observation_end,
    )
    raw = _default_fetcher(url) if fetcher is None else fetcher(url)
    payload = json.loads(raw.decode("utf-8"))
    observations = payload.get("observations", [])
    if not isinstance(observations, list):
        msg = f"FRED API response for {mnemonic} did not include observations."
        raise ValueError(msg)
    values = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [item.get("date") for item in observations],
                errors="coerce",
            ),
            mnemonic: pd.to_numeric(
                [item.get("value") for item in observations],
                errors="coerce",
            ),
        }
    ).dropna(subset=["date"])
    if values.empty:
        msg = f"FRED API response for {mnemonic} did not include usable dated observations."
        raise ValueError(msg)
    values["date"] = [quarter_label(value) for value in values["date"]]
    return _aggregate_quarterly_series(values, mnemonic, aggregation=aggregation)


def load_current_fred_source(
    mnemonics: list[str],
    *,
    start_date: Any | None = None,
    end_date: Any | None = None,
    aggregation: str = "mean",
    fetcher: FredFetcher | None = None,
) -> pd.DataFrame:
    """Download current public FRED graph CSV data and aggregate to quarters."""

    required_dates = _required_quarter_indexes(start_date=start_date, end_date=end_date)
    merged: pd.DataFrame | None = None
    for mnemonic in mnemonics:
        series = load_current_fred_series(
            mnemonic,
            aggregation=aggregation,
            fetcher=fetcher,
        )
        if required_dates is not None:
            _validate_source_date_coverage(f"FRED:{mnemonic}", series["date"], required_dates)
            date_index = _quarter_index(series["date"])
            required_set = set(required_dates.tolist())
            series = series.loc[[date in required_set for date in date_index]].reset_index(
                drop=True
            )
        merged = series if merged is None else pd.merge(merged, series, on="date", how="outer")
    if merged is None:
        return pd.DataFrame({"date": []})
    return _sort_by_date(merged)


def load_current_fred_series(
    mnemonic: str,
    *,
    aggregation: str = "mean",
    fetcher: FredFetcher | None = None,
) -> pd.DataFrame:
    url = _fred_graph_csv_url(mnemonic)
    raw = _default_fetcher(url) if fetcher is None else fetcher(url)
    downloaded = pd.read_csv(io.BytesIO(raw), na_values=["."])
    if downloaded.shape[1] < 2:
        msg = f"FRED CSV for {mnemonic} did not include a value column."
        raise ValueError(msg)
    date_column = downloaded.columns[0]
    value_column = downloaded.columns[1]
    values = pd.DataFrame(
        {
            "date": pd.to_datetime(downloaded[date_column], errors="coerce"),
            mnemonic: pd.to_numeric(downloaded[value_column], errors="coerce"),
        }
    ).dropna(subset=["date"])
    if values.empty:
        msg = f"FRED CSV for {mnemonic} did not include usable dated observations."
        raise ValueError(msg)
    values["date"] = [quarter_label(value) for value in values["date"]]
    return _aggregate_quarterly_series(values, mnemonic, aggregation=aggregation)


def load_data_levels_from_sources(
    model: DSGEModel,
    *,
    source_root: Path | str,
    vintage: str | None = None,
    start_date: Any | None = None,
    end_date: Any | None = None,
) -> pd.DataFrame:
    """Merge local raw source files declared by the model's observable metadata."""

    root = Path(source_root)
    data_sources = parse_data_sources(model)
    selected_vintage = str(
        vintage if vintage is not None else model.get_setting("data_vintage", "")
    )
    required_dates = _required_quarter_indexes(start_date=start_date, end_date=end_date)
    merged: pd.DataFrame | None = None
    for source, mnemonics in data_sources.items():
        path = _find_source_file(root, source=source, vintage=selected_vintage)
        if path is None:
            msg = (
                f"No local source file found for {source}. "
                f"Expected one of: {_candidate_source_paths_text(root, source, selected_vintage)}"
            )
            raise DataNotAvailableError(msg)
        source_df = _select_source_columns(
            pd.read_csv(path),
            source=source,
            mnemonics=mnemonics,
            required_dates=required_dates,
        )
        merged = (
            source_df if merged is None else pd.merge(merged, source_df, on="date", how="outer")
        )
    if merged is None:
        msg = "Model does not declare any data source series."
        raise DataNotAvailableError(msg)
    merged = _sort_by_date(merged)
    if required_dates is not None:
        date_index = _quarter_index(merged["date"])
        required_set = set(required_dates.tolist())
        merged = merged.loc[[date in required_set for date in date_index]].reset_index(drop=True)
    return merged


def parse_data_sources(model: DSGEModel) -> dict[str, list[str]]:
    """Return source -> ordered mnemonic names from observable input metadata."""

    data_sources: dict[str, list[str]] = {}
    for observable in model.observable_mappings.values():
        for source_name in observable.source_names:
            mnemonic, source = _split_source_name(source_name)
            data_sources.setdefault(source, [])
            if mnemonic not in data_sources[source]:
                data_sources[source].append(mnemonic)
    return data_sources


def prepare_population_data(
    model: DSGEModel,
    levels: pd.DataFrame,
    *,
    population_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add DSGE.jl-style population helper series used by per-capita transforms."""

    population = str(model.get_setting("population_mnemonic", "CNP16OV__FRED"))
    column = _resolve_column(levels, population)
    working = levels.copy()
    population_values = working[column].to_numpy(dtype=np.float64)
    working["dlpopulation_recorded"] = one_quarter_pct_change(population_values)
    if _setting_bool(model.get_setting("hpfilter_population", True)):
        hp_lambda = float(model.get_setting("population_hpfilter_lambda", 1600.0))
        hp_input = population_values
        if (
            _setting_bool(model.get_setting("use_population_forecast", True))
            and population_forecast is not None
        ):
            forecast_values = _population_forecast_values(
                model,
                population_forecast,
                recorded_dates=working.get("date"),
            )
            hp_input = np.concatenate([population_values, forecast_values])
        filtered = hpfilter(hp_input, hp_lambda)[: population_values.shape[0]]
        working["filtered_population_recorded"] = filtered
        working["dlfiltered_population_recorded"] = one_quarter_pct_change(filtered)
    return working


def hpfilter(values: Any, smoothing: float = 1600.0) -> np.ndarray:
    """Return the Hodrick-Prescott trend component for a single quarterly series."""

    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 1:
        msg = "HP filter input must be one-dimensional."
        raise ValueError(msg)
    if smoothing < 0.0:
        msg = "HP filter smoothing parameter must be nonnegative."
        raise ValueError(msg)
    trend = np.full_like(series, np.nan, dtype=np.float64)
    finite = np.isfinite(series)
    if not finite.any():
        return trend
    first = int(np.argmax(finite))
    last = int(len(series) - np.argmax(finite[::-1]) - 1)
    segment = series[first : last + 1]
    if not np.isfinite(segment).all():
        return trend
    if segment.size < 3 or smoothing == 0.0:
        trend[first : last + 1] = segment
        return trend
    second_difference = np.zeros((segment.size - 2, segment.size), dtype=np.float64)
    rows = np.arange(segment.size - 2)
    second_difference[rows, rows] = 1.0
    second_difference[rows, rows + 1] = -2.0
    second_difference[rows, rows + 2] = 1.0
    system = np.eye(segment.size, dtype=np.float64) + smoothing * (
        second_difference.T @ second_difference
    )
    trend[first : last + 1] = np.linalg.solve(system, segment)
    return trend


def _apply_forward_transform(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    registry: dict[str, Callable[[DSGEModel, Observable, pd.DataFrame], np.ndarray]] = {
        "identity": _identity_observable,
        "gdp_growth": _gdp_growth,
        "hours_per_capita": _hours_per_capita,
        "real_wage_growth": _real_wage_growth,
        "gdp_deflator_growth": _gdp_deflator_growth,
        "core_pce_growth": _core_pce_growth,
        "nominal_rate": _nominal_rate,
        "consumption_growth": _consumption_growth,
        "investment_growth": _investment_growth,
        "baa_10y_spread": _baa_10y_spread,
        "long_inflation_expectations": _long_inflation_expectations,
        "long_rate": _long_rate,
        "fernald_tfp": _fernald_tfp,
        "gdi_growth": _gdi_growth,
        "anticipated_rate": _identity_observable,
        "expected_ffr_spd": _identity_observable,
        "flexible_ait_gap": _identity_observable,
        "anticipated_gdp_growth": _identity_observable,
    }
    try:
        transform = registry[observable.forward_transform]
    except KeyError as err:
        msg = f"Unknown forward transform for {observable.name}: {observable.forward_transform}"
        raise KeyError(msg) from err
    return transform(model, observable, levels)


def candidate_data_paths(model: DSGEModel) -> list[Path]:
    vintage = model.get_setting("data_vintage", "")
    dataroot = Path(str(model.get_setting("dataroot", "save/input_data")))
    data_id = model.get_setting("data_id", None)
    names = [
        f"data_{vintage}.csv",
        f"data_vint={vintage}.csv",
    ]
    if data_id is not None:
        names.append(f"data_dsid={int(data_id):02d}_vint={vintage}.csv")
    return [dataroot / "data" / name for name in names] + [dataroot / name for name in names]


def load_data(
    model: DSGEModel,
    *,
    path: Path | str | None = None,
    try_disk: bool = True,
    check_empty_columns: bool = True,
    summary_statistics: str = "low",
) -> pd.DataFrame:
    del summary_statistics
    if path is not None:
        df = pd.read_csv(path)
        if check_empty_columns:
            _raise_for_empty_columns(df)
        return df

    if not try_disk:
        msg = "Network/FRED data building is not ported yet; provide a local CSV fixture."
        raise DataNotAvailableError(msg)

    for path in candidate_data_paths(model):
        if path.exists():
            df = pd.read_csv(path)
            if check_empty_columns:
                _raise_for_empty_columns(df)
            return df

    checked = ", ".join(str(path) for path in candidate_data_paths(model))
    msg = f"No local data CSV found for {model.spec} {model.subspec}. Checked: {checked}"
    raise DataNotAvailableError(msg)


def df_to_matrix(
    model: DSGEModel,
    df: pd.DataFrame,
    *,
    cond_type: str = "none",
    in_sample: bool = True,
) -> np.ndarray:
    del cond_type
    working = _sort_by_date(df.copy())

    columns = list(model.observables.keys())
    if any(column not in working.columns for column in columns):
        working = transform_data(model, working)
    working = filter_data_by_sample(model, working, in_sample=in_sample)

    missing = [column for column in columns if column not in working.columns]
    if missing:
        msg = "DataFrame is missing observable columns: " + ", ".join(missing)
        raise KeyError(msg)
    return working.loc[:, columns].to_numpy(dtype=np.float64)


def filter_data_by_sample(
    model: DSGEModel,
    df: pd.DataFrame,
    *,
    in_sample: bool = True,
) -> pd.DataFrame:
    if "date" not in df.columns:
        return df
    forecast_start = model.get_setting("date_forecast_start", None)
    if forecast_start is None:
        return df

    forecast_start_index = _quarter_to_index(forecast_start)
    date_index = _quarter_index(df["date"])
    mask = date_index < forecast_start_index if in_sample else date_index >= forecast_start_index
    if in_sample:
        mainsample_start = model.get_setting("date_mainsample_start", None)
        if mainsample_start is not None:
            mask &= date_index >= _quarter_to_index(mainsample_start)
    return df.loc[mask].reset_index(drop=True)


def date_labels_for_sample(
    model: DSGEModel,
    df: pd.DataFrame,
    *,
    in_sample: bool = True,
) -> list[str]:
    if "date" not in df.columns:
        return []
    filtered = filter_data_by_sample(model, _sort_by_date(df.copy()), in_sample=in_sample)
    if "date" not in filtered.columns:
        return []
    return [quarter_label(value) for value in filtered["date"]]


def quarter_labels_from_start(start: Any, periods: int) -> list[str]:
    if periods < 0:
        msg = "Number of periods must be nonnegative."
        raise ValueError(msg)
    start_index = _quarter_to_index(start)
    return [_quarter_label_from_index(start_index + offset) for offset in range(periods)]


def quarter_label(value: Any) -> str:
    return _quarter_label_from_index(_quarter_to_index(value))


def _sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        return df.sort_values("date").reset_index(drop=True)
    return df


def _raise_for_empty_columns(df: pd.DataFrame) -> None:
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        msg = f"Input data has empty columns: {', '.join(empty_cols)}"
        raise ValueError(msg)


def _split_source_name(source_name: str) -> tuple[str, str]:
    if "__" not in source_name:
        return source_name, "FRED"
    mnemonic, source = source_name.split("__", 1)
    return mnemonic, source.upper()


def _find_source_file(root: Path, *, source: str, vintage: str) -> Path | None:
    for path in _candidate_source_paths(root, source, vintage):
        if path.exists():
            return path
    return None


def _candidate_source_paths(root: Path, source: str, vintage: str) -> list[Path]:
    lower = source.lower()
    upper = source.upper()
    candidates = [root / f"{lower}_{vintage}.csv", root / f"{upper}_{vintage}.csv"]
    if vintage:
        candidates.extend(
            [
                root / f"{lower}_vint={vintage}.csv",
                root / f"{upper}_vint={vintage}.csv",
            ]
        )
    candidates.extend([root / f"{lower}.csv", root / f"{upper}.csv"])
    return candidates


def _candidate_source_paths_text(root: Path, source: str, vintage: str) -> str:
    return ", ".join(str(path) for path in _candidate_source_paths(root, source, vintage))


def _fred_graph_csv_url(mnemonic: str) -> str:
    return f"{FRED_GRAPH_CSV_URL}?{urlencode({'id': mnemonic})}"


def _fred_observations_url(
    mnemonic: str,
    *,
    api_key: str,
    realtime_start: str | None,
    realtime_end: str | None,
    observation_start: Any | None,
    observation_end: Any | None,
) -> str:
    params = {
        "series_id": mnemonic,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
    }
    if realtime_start is not None:
        params["realtime_start"] = _fred_realtime_date(realtime_start)
    if realtime_end is not None:
        params["realtime_end"] = _fred_realtime_date(realtime_end)
    if observation_start is not None:
        params["observation_start"] = _quarter_boundary_date(observation_start, start=True)
    if observation_end is not None:
        params["observation_end"] = _quarter_boundary_date(observation_end, start=False)
    return f"{FRED_OBSERVATIONS_URL}?{urlencode(params)}"


def _resolve_fred_api_key(api_key: str | None) -> str:
    resolved = api_key or os.environ.get("FRED_API_KEY") or _dotenv_value("FRED_API_KEY")
    if not resolved:
        msg = "FRED API access requires --api-key, FRED_API_KEY, or a local .env file."
        raise ValueError(msg)
    return resolved


def _dotenv_value(key: str, *, start: Path | None = None) -> str | None:
    search_start = Path.cwd() if start is None else start
    for directory in (search_start, *search_start.parents):
        path = directory / DOTENV_FILENAME
        if not path.is_file():
            continue
        value = _read_dotenv_value(path, key)
        if value:
            return value
    return None


def _read_dotenv_value(path: Path, key: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    prefix = f"{key}="
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        parsed = _strip_dotenv_comment(value).strip()
        if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
            parsed = parsed[1:-1]
        return parsed
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith(f"export {prefix}"):
            value = _strip_dotenv_comment(
                line.removeprefix("export ").split("=", 1)[1]
            ).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value
    return None


def _strip_dotenv_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
        if char == "#" and quote is None:
            return value[:index]
    return value


def _fred_realtime_date(value: str) -> str:
    text = str(value)
    if len(text) == 6 and text.isdigit():
        return f"20{text[:2]}-{text[2:4]}-{text[4:6]}"
    return str(pd.Timestamp(text).date())


def _quarter_boundary_date(value: Any, *, start: bool) -> str:
    if hasattr(value, "year") and hasattr(value, "month"):
        timestamp = pd.Timestamp(value)
        return str(timestamp.date())
    text = str(value)
    if "-Q" in text:
        year_text, quarter_text = text.split("-Q", 1)
        year = int(year_text)
        quarter = int(quarter_text[0])
        if start:
            month = 3 * (quarter - 1) + 1
            day = 1
        else:
            month = 3 * quarter
            day = 31 if month in {3, 12} else 30
        return f"{year:04d}-{month:02d}-{day:02d}"
    return str(pd.Timestamp(text).date())


def _default_fetcher(url: str) -> bytes:
    with urlopen(url, timeout=30.0) as response:
        return response.read()


def _aggregate_quarterly_series(
    values: pd.DataFrame,
    mnemonic: str,
    *,
    aggregation: str,
) -> pd.DataFrame:
    if aggregation not in {"mean", "last"}:
        msg = "FRED aggregation must be 'mean' or 'last'."
        raise ValueError(msg)
    grouped = values.groupby("date", sort=False)[mnemonic]
    if aggregation == "mean":
        aggregated = grouped.mean()
    else:
        aggregated = grouped.last()
    return aggregated.reset_index()


def _select_source_columns(
    df: pd.DataFrame,
    *,
    source: str,
    mnemonics: list[str],
    required_dates: np.ndarray | None = None,
) -> pd.DataFrame:
    if "date" not in df.columns:
        msg = f"{source} source file is missing required date column."
        raise KeyError(msg)
    if required_dates is not None:
        _validate_source_date_coverage(source, df["date"], required_dates)
    selected = pd.DataFrame({"date": df["date"]})
    missing: list[str] = []
    for mnemonic in mnemonics:
        try:
            column = _resolve_column(df, mnemonic)
        except KeyError:
            missing.append(mnemonic)
            continue
        selected[mnemonic] = df[column]
    if missing:
        msg = f"{source} source file is missing required columns: {', '.join(missing)}"
        raise KeyError(msg)
    return selected


def _required_quarter_indexes(*, start_date: Any | None, end_date: Any | None) -> np.ndarray | None:
    if start_date is None and end_date is None:
        return None
    if start_date is None or end_date is None:
        msg = "Source date validation requires both start_date and end_date."
        raise ValueError(msg)
    start_index = _quarter_to_index(start_date)
    end_index = _quarter_to_index(end_date)
    if end_index < start_index:
        msg = "Source date validation requires end_date at or after start_date."
        raise ValueError(msg)
    return np.arange(start_index, end_index + 1, dtype=np.int64)


def _validate_source_date_coverage(
    source: str,
    dates: pd.Series[Any],
    required_dates: np.ndarray,
) -> None:
    source_dates = _quarter_index(dates)
    unique_dates = set(source_dates.tolist())
    if len(unique_dates) != len(source_dates):
        msg = f"{source} source file has duplicate quarterly dates."
        raise ValueError(msg)
    missing = [date for date in required_dates.tolist() if date not in unique_dates]
    if missing:
        labels = ", ".join(_quarter_label_from_index(date) for date in missing)
        msg = f"{source} source file is missing required quarterly dates: {labels}"
        raise ValueError(msg)


def _identity_observable(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del model
    return _series(levels, observable.source_names[0])


def _gdp_growth(model: DSGEModel, observable: Observable, levels: pd.DataFrame) -> np.ndarray:
    del observable
    real = _real_per_capita(model, levels, "GDP", scale=1000.0)
    return one_quarter_pct_change(real)


def _hours_per_capita(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del observable
    hours = _series(levels, "AWHNONAG") * _series(levels, "CE16OV")
    weekly_hours = _per_capita(model, levels, hours)
    return 100.0 * np.log(3.0 * weekly_hours / 100.0)


def _real_wage_growth(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del model, observable
    return one_quarter_pct_change(_nominal_to_real(levels, _series(levels, "COMPNFB")))


def _gdp_deflator_growth(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del model, observable
    return one_quarter_pct_change(_series(levels, "GDPDEF"))


def _core_pce_growth(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del model, observable
    return one_quarter_pct_change(_series(levels, "PCEPILFE"))


def _nominal_rate(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del model, observable
    return annual_to_quarter(_series(levels, "DFF"))


def _consumption_growth(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del observable
    real = _real_per_capita(model, levels, "PCE", scale=1000.0)
    return one_quarter_pct_change(real)


def _investment_growth(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del observable
    real = _real_per_capita(model, levels, "FPI", scale=10000.0)
    return one_quarter_pct_change(real)


def _baa_10y_spread(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del model, observable
    baa = _series(levels, "BAA")
    if "date" in levels.columns and _has_column(levels, "BAMLC8A0C15PYEY"):
        replacement = _series(levels, "BAMLC8A0C15PYEY")
        after_splice = _quarter_index(levels["date"]) >= _quarter_to_index("2016-Q4")
        baa = np.where(after_splice, replacement, baa)
    return annual_to_quarter(baa - _series(levels, "GS10"))


def _long_inflation_expectations(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del observable
    if model.subspec == "ss102" and _has_column(levels, "PCE10"):
        return annual_to_quarter(_series(levels, "PCE10"))
    return annual_to_quarter(_series(levels, "ASACX10") - 0.5)


def _long_rate(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del model, observable
    return annual_to_quarter(_series(levels, "FYCCZA"))


def _fernald_tfp(
    model: DSGEModel,
    observable: Observable,
    levels: pd.DataFrame,
) -> np.ndarray:
    del model, observable
    tfp = _series(levels, "TFPKQ")
    alpha = _series(levels, "TFPJQ")
    mean = float(np.nanmean(tfp))
    return (tfp - mean) / (4.0 * (1.0 - alpha))


def _gdi_growth(model: DSGEModel, observable: Observable, levels: pd.DataFrame) -> np.ndarray:
    del observable
    real = _real_per_capita(model, levels, "GDI", scale=1000.0)
    return one_quarter_pct_change(real)


def _real_per_capita(
    model: DSGEModel,
    levels: pd.DataFrame,
    column: str,
    *,
    scale: float,
) -> np.ndarray:
    return scale * _nominal_to_real(levels, _per_capita(model, levels, _series(levels, column)))


def _per_capita(model: DSGEModel, levels: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    population = str(model.get_setting("population_mnemonic", "CNP16OV__FRED"))
    if (
        _setting_bool(model.get_setting("hpfilter_population", True))
        and "filtered_population_recorded" in levels.columns
    ):
        population = "filtered_population_recorded"
    return np.asarray(values, dtype=np.float64) / _series(levels, population)


def _population_forecast_values(
    model: DSGEModel,
    population_forecast: pd.DataFrame,
    *,
    recorded_dates: pd.Series[Any] | None,
) -> np.ndarray:
    if population_forecast.empty:
        return np.empty(0, dtype=np.float64)
    forecast = _sort_by_date(population_forecast.copy())
    population = str(model.get_setting("population_mnemonic", "CNP16OV__FRED"))
    try:
        column = _resolve_column(forecast, population)
    except KeyError:
        if "POPULATION" not in forecast.columns:
            raise
        column = "POPULATION"
    if recorded_dates is not None and "date" in forecast.columns and not recorded_dates.empty:
        last_recorded = np.max(_quarter_index(recorded_dates))
        forecast = forecast.loc[_quarter_index(forecast["date"]) >= last_recorded]
    return forecast[column].to_numpy(dtype=np.float64)


def _nominal_to_real(levels: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) / _series(levels, "GDPDEF") * 100.0


def _series(levels: pd.DataFrame, source: str) -> np.ndarray:
    column = _resolve_column(levels, source)
    return levels[column].to_numpy(dtype=np.float64)


def _has_column(levels: pd.DataFrame, source: str) -> bool:
    try:
        _resolve_column(levels, source)
    except KeyError:
        return False
    return True


def _resolve_column(levels: pd.DataFrame, source: str) -> str:
    base = source.split("__", 1)[0]
    for candidate in (source, base):
        if candidate in levels.columns:
            return candidate
    msg = f"DataFrame is missing source column for {source}."
    raise KeyError(msg)


def _quarter_index(values: pd.Series[Any]) -> np.ndarray:
    return values.map(_quarter_to_index).to_numpy(dtype=np.int64)


def _quarter_to_index(value: Any) -> int:
    if hasattr(value, "year") and hasattr(value, "month"):
        month = int(value.month)
        quarter = (month - 1) // 3 + 1
        return int(value.year) * 4 + quarter
    text = str(value)
    if "-Q" in text:
        year, quarter = text.split("-Q", 1)
        return int(year) * 4 + int(quarter[0])
    parsed = pd.Timestamp(text)
    quarter = (int(parsed.month) - 1) // 3 + 1
    return int(parsed.year) * 4 + quarter


def _quarter_label_from_index(value: int) -> str:
    year = (value - 1) // 4
    quarter = value - year * 4
    return f"{year}-Q{quarter}"


def _setting_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)
