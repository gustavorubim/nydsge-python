from __future__ import annotations

import io
from email.message import Message
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest

from nydsge.data import (
    DataNotAvailableError,
    FredApiError,
    annual_to_quarter,
    build_data_csv,
    build_data_csv_from_sources,
    data_source_requirements,
    df_to_matrix,
    download_current_fred_source_csv,
    download_fred_api_source_csv,
    filter_data_by_sample,
    hpfilter,
    load_current_fred_series,
    load_data,
    load_data_levels_from_sources,
    load_fred_api_series,
    load_fred_api_source,
    one_quarter_pct_change,
    parse_data_sources,
    prepare_population_data,
    reverse_transform_array,
    reverse_transform_observables,
    transform_data,
)
from nydsge.models import Model1002


def test_df_to_matrix_uses_model_observable_order() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    df = pd.DataFrame(
        {
            "date": ["2000-Q2", "2000-Q1"],
            **{name: [float(idx + 1), float(idx)] for idx, name in enumerate(model.observables)},
        }
    )
    matrix = df_to_matrix(model, df)
    assert matrix.shape == (2, len(model.observables))
    assert matrix[0, 0] == 0.0
    assert matrix[1, 0] == 1.0


def test_df_to_matrix_uses_rows_before_forecast_start_for_in_sample_data() -> None:
    model = Model1002(
        settings={
            "date_forecast_start": "2018-Q4",
            "n_mon_anticipated_shocks": 0,
        }
    )
    df = pd.DataFrame(
        {
            "date": ["2018-Q3", "2018-Q4", "2019-Q1"],
            **{
                name: [float(idx), 10.0 + idx, 20.0 + idx]
                for idx, name in enumerate(model.observables)
            },
        }
    )

    in_sample = df_to_matrix(model, df)
    out_of_sample = df_to_matrix(model, df, in_sample=False)

    assert in_sample.shape == (1, len(model.observables))
    assert out_of_sample.shape == (2, len(model.observables))
    assert in_sample[0, 0] == 0.0
    assert out_of_sample[0, 0] == 10.0


def test_df_to_matrix_respects_mainsample_start_for_in_sample_data() -> None:
    model = Model1002(
        settings={
            "date_presample_start": "2018-Q1",
            "date_mainsample_start": "2018-Q2",
            "date_forecast_start": "2018-Q4",
            "n_mon_anticipated_shocks": 0,
        }
    )
    df = pd.DataFrame(
        {
            "date": ["2018-Q1", "2018-Q2", "2018-Q3", "2018-Q4"],
            **{
                name: [float(idx), 10.0 + idx, 20.0 + idx, 30.0 + idx]
                for idx, name in enumerate(model.observables)
            },
        }
    )

    in_sample = df_to_matrix(model, df)

    assert in_sample.shape == (2, len(model.observables))
    assert in_sample[:, 0].tolist() == [10.0, 20.0]


def test_df_to_matrix_can_include_presample_rows_for_sampler_likelihood() -> None:
    model = Model1002(
        settings={
            "date_presample_start": "2018-Q1",
            "date_mainsample_start": "2018-Q2",
            "date_forecast_start": "2018-Q4",
            "n_mon_anticipated_shocks": 0,
        }
    )
    df = pd.DataFrame(
        {
            "date": ["2018-Q1", "2018-Q2", "2018-Q3", "2018-Q4"],
            **{
                name: [float(idx), 10.0 + idx, 20.0 + idx, 30.0 + idx]
                for idx, name in enumerate(model.observables)
            },
        }
    )

    in_sample = df_to_matrix(model, df)
    with_presample = df_to_matrix(model, df, include_presample=True)

    assert in_sample[:, 0].tolist() == [10.0, 20.0]
    assert with_presample[:, 0].tolist() == [0.0, 10.0, 20.0]


def test_filter_data_by_sample_accepts_timestamp_dates() -> None:
    model = Model1002(settings={"date_forecast_start": pd.Timestamp("2018-10-01")})
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-07-01", "2018-10-01"]),
            "value": [1.0, 2.0],
        }
    )

    filtered = filter_data_by_sample(model, df)

    assert filtered["value"].tolist() == [1.0]


def test_basic_transforms() -> None:
    assert annual_to_quarter([4.0])[0] == 1.0
    growth = one_quarter_pct_change([1.0, 2.0])
    assert growth.shape == (2,)


def test_load_data_accepts_explicit_csv_path(tmp_path) -> None:
    path = tmp_path / "observables.csv"
    pd.DataFrame({"obs_gdp": [1.0], "obs_hours": [2.0]}).to_csv(path, index=False)

    loaded = load_data(Model1002(), path=path)

    assert loaded.loc[0, "obs_gdp"] == 1.0


def test_load_data_explicit_csv_rejects_empty_columns(tmp_path) -> None:
    path = tmp_path / "observables.csv"
    pd.DataFrame({"obs_gdp": [np.nan]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="empty columns"):
        load_data(Model1002(), path=path)


def test_parse_data_sources_groups_observable_mnemonics_by_source() -> None:
    model = Model1002(
        settings={
            "add_anticipated_obs_gdp": True,
            "n_anticipated_obs_gdp": 1,
            "n_mon_anticipated_shocks": 0,
        }
    )

    sources = parse_data_sources(model)

    assert "FRED" in sources
    assert "ANTGDP" in sources
    assert sources["FRED"][:3] == ["GDP", "CNP16OV", "GDPDEF"]
    assert sources["ANTGDP"] == ["antgdp1"]


def test_data_source_requirements_reports_candidate_paths_and_optional_columns(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    (tmp_path / "fred_181115.csv").write_text("date,GDP\n2016-Q3,1.0\n", encoding="utf-8")

    requirements = data_source_requirements(model, source_root=tmp_path, vintage="181115")
    by_source = {requirement.source: requirement for requirement in requirements}

    assert by_source["FRED"].existing_path == tmp_path / "fred_181115.csv"
    assert "BAMLC8A0C15PYEY" in by_source["FRED"].optional_mnemonics
    assert by_source["DLX"].existing_path is None
    assert tmp_path / "dlx_181115.csv" in by_source["DLX"].candidate_paths


def test_load_data_levels_from_sources_merges_local_source_files(tmp_path) -> None:
    model = Model1002(
        settings={
            "add_anticipated_obs_gdp": True,
            "n_anticipated_obs_gdp": 1,
            "n_mon_anticipated_shocks": 0,
        }
    )
    _raw_levels_fixture().to_csv(tmp_path / "fred_181115.csv", index=False)
    _raw_levels_fixture().to_csv(tmp_path / "dlx_181115.csv", index=False)
    pd.DataFrame(
        {
            "date": ["2016-Q4", "2016-Q3"],
            "antgdp1": [0.4, 0.5],
        }
    ).to_csv(tmp_path / "antgdp_181115.csv", index=False)

    levels = load_data_levels_from_sources(model, source_root=tmp_path)

    assert list(levels["date"]) == ["2016-Q3", "2016-Q4"]
    assert "GDP" in levels.columns
    assert "antgdp1" in levels.columns
    assert np.isclose(levels.loc[0, "antgdp1"], 0.5)


def test_load_data_levels_from_sources_fills_optional_missing_columns(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    raw = _raw_levels_fixture().drop(columns=["BAMLC8A0C15PYEY"])
    raw.to_csv(tmp_path / "fred_181115.csv", index=False)
    raw.to_csv(tmp_path / "dlx_181115.csv", index=False)

    levels = load_data_levels_from_sources(model, source_root=tmp_path)

    assert "BAMLC8A0C15PYEY" in levels.columns
    assert levels["BAMLC8A0C15PYEY"].isna().all()


def test_load_data_levels_from_sources_validates_and_filters_required_dates(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    raw = pd.concat(
        [_raw_levels_fixture().iloc[[1]].copy() for _ in range(4)],
        ignore_index=True,
    )
    raw["date"] = ["2016-Q1", "2016-Q2", "2016-Q3", "2016-Q4"]
    raw.to_csv(tmp_path / "fred_181115.csv", index=False)
    raw.to_csv(tmp_path / "dlx_181115.csv", index=False)

    levels = load_data_levels_from_sources(
        model,
        source_root=tmp_path,
        start_date="2016-Q2",
        end_date="2016-Q3",
    )

    assert list(levels["date"]) == ["2016-Q2", "2016-Q3"]


def test_load_data_levels_from_sources_reports_missing_required_dates(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    raw = _raw_levels_fixture()
    raw.to_csv(tmp_path / "fred_181115.csv", index=False)
    raw.to_csv(tmp_path / "dlx_181115.csv", index=False)

    with pytest.raises(ValueError, match="2016-Q2"):
        load_data_levels_from_sources(
            model,
            source_root=tmp_path,
            start_date="2016-Q2",
            end_date="2016-Q4",
        )


def test_load_data_levels_from_sources_reports_missing_required_values(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    raw = _raw_levels_fixture()
    raw.loc[raw["date"] == "2016-Q3", "GDP"] = np.nan
    raw.to_csv(tmp_path / "fred_181115.csv", index=False)
    raw.to_csv(tmp_path / "dlx_181115.csv", index=False)

    with pytest.raises(ValueError, match="FRED:GDP.*2016-Q3"):
        load_data_levels_from_sources(
            model,
            source_root=tmp_path,
            start_date="2016-Q3",
            end_date="2016-Q4",
        )


def test_load_data_levels_from_sources_requires_complete_date_bounds(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    _raw_levels_fixture().to_csv(tmp_path / "fred_181115.csv", index=False)
    _raw_levels_fixture().to_csv(tmp_path / "dlx_181115.csv", index=False)

    with pytest.raises(ValueError, match="both start_date and end_date"):
        load_data_levels_from_sources(model, source_root=tmp_path, start_date="2016-Q3")


def test_build_data_csv_from_sources_writes_transformed_observables(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    output_path = tmp_path / "built" / "observables.csv"
    _raw_levels_fixture().to_csv(tmp_path / "fred_181115.csv", index=False)
    _raw_levels_fixture().to_csv(tmp_path / "dlx_181115.csv", index=False)

    transformed = build_data_csv_from_sources(
        model,
        source_root=tmp_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert "obs_gdp" in transformed.columns
    assert list(transformed["date"]) == ["2016-Q3", "2016-Q4"]


def test_load_data_levels_from_sources_reports_missing_source_file(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})

    with pytest.raises(DataNotAvailableError, match="FRED"):
        load_data_levels_from_sources(model, source_root=tmp_path)


def test_load_current_fred_series_aggregates_public_graph_csv_to_quarters() -> None:
    def fetcher(url: str) -> bytes:
        assert "id=GDP" in url
        return b"observation_date,GDP\n2016-01-01,10.0\n2016-02-01,20.0\n2016-04-01,40.0\n"

    series = load_current_fred_series("GDP", fetcher=fetcher)

    assert list(series["date"]) == ["2016-Q1", "2016-Q2"]
    assert np.isclose(series.loc[0, "GDP"], 15.0)
    assert np.isclose(series.loc[1, "GDP"], 40.0)


def test_download_current_fred_source_csv_writes_required_fred_series(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    output_path = tmp_path / "fred_181115.csv"

    def fetcher(url: str) -> bytes:
        series_id = url.rsplit("id=", 1)[1]
        return (f"observation_date,{series_id}\n2016-07-01,1.0\n2016-10-01,2.0\n").encode()

    levels = download_current_fred_source_csv(
        model,
        output_path=output_path,
        start_date="2016-Q3",
        end_date="2016-Q4",
        fetcher=fetcher,
    )

    assert output_path.exists()
    assert list(levels["date"]) == ["2016-Q3", "2016-Q4"]
    assert "GDP" in levels.columns
    assert "DFF" in levels.columns


def test_download_current_fred_source_csv_validates_required_quarter_coverage(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})

    def fetcher(url: str) -> bytes:
        series_id = url.rsplit("id=", 1)[1]
        return f"observation_date,{series_id}\n2016-07-01,1.0\n".encode()

    with pytest.raises(ValueError, match="2016-Q4"):
        download_current_fred_source_csv(
            model,
            output_path=tmp_path / "fred.csv",
            start_date="2016-Q3",
            end_date="2016-Q4",
            fetcher=fetcher,
        )


def test_download_current_fred_source_csv_validates_required_quarter_values(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})

    def fetcher(url: str) -> bytes:
        series_id = url.rsplit("id=", 1)[1]
        return (f"observation_date,{series_id}\n2016-07-01,.\n2016-10-01,2.0\n").encode()

    with pytest.raises(ValueError, match="FRED:GDP.*2016-Q3"):
        download_current_fred_source_csv(
            model,
            output_path=tmp_path / "fred.csv",
            start_date="2016-Q3",
            end_date="2016-Q4",
            fetcher=fetcher,
        )


def test_load_fred_api_series_uses_realtime_and_observation_parameters() -> None:
    captured_url = ""

    def fetcher(url: str) -> bytes:
        nonlocal captured_url
        captured_url = url
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "value": "1.0"},'
            b'{"date": "2016-08-01", "value": "3.0"},'
            b'{"date": "2016-10-01", "value": "5.0"}'
            b"]}"
        )

    series = load_fred_api_series(
        "GDP",
        api_key="secret",
        realtime_start="181115",
        realtime_end="2018-11-15",
        observation_start="2016-Q3",
        observation_end="2016-Q4",
        fetcher=fetcher,
    )

    assert "series_id=GDP" in captured_url
    assert "api_key=secret" in captured_url
    assert "realtime_start=2018-11-15" in captured_url
    assert "realtime_end=2018-11-15" in captured_url
    assert "output_type=1" in captured_url
    assert "observation_start=2016-07-01" in captured_url
    assert "observation_end=2016-12-31" in captured_url
    assert list(series["date"]) == ["2016-Q3", "2016-Q4"]
    assert np.isclose(series.loc[0, "GDP"], 2.0)


def test_load_fred_api_series_uses_latest_revision_in_realtime_range() -> None:
    captured_url = ""

    def fetcher(url: str) -> bytes:
        nonlocal captured_url
        captured_url = url
        return (
            b'{"observations": ['
            b'{"realtime_start": "2018-01-01", "realtime_end": "2018-01-31",'
            b'"date": "2016-07-01", "value": "1.0"},'
            b'{"realtime_start": "2018-02-01", "realtime_end": "2018-02-28",'
            b'"date": "2016-07-01", "value": "3.0"},'
            b'{"realtime_start": "2018-02-01", "realtime_end": "2018-02-28",'
            b'"date": "2016-10-01", "value": "5.0"}'
            b"]}"
        )

    series = load_fred_api_series(
        "GDP",
        api_key="secret",
        realtime_start="2018-01-01",
        realtime_end="2018-02-28",
        observation_start="2016-Q3",
        observation_end="2016-Q4",
        fetcher=fetcher,
    )

    assert "output_type=1" in captured_url
    assert list(series["date"]) == ["2016-Q3", "2016-Q4"]
    assert series["GDP"].tolist() == [3.0, 5.0]


def test_load_fred_api_series_uses_vintage_dates_mode() -> None:
    captured_url = ""

    def fetcher(url: str) -> bytes:
        nonlocal captured_url
        captured_url = url
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "GDP_20181115": "1.0", "GDP_20181116": "3.0"},'
            b'{"date": "2016-10-01", "GDP_20181115": "4.0", "GDP_20181116": "5.0"}'
            b"]}"
        )

    series = load_fred_api_series(
        "GDP",
        api_key="secret",
        vintage_dates="181115,181116",
        observation_start="2016-Q3",
        observation_end="2016-Q4",
        fetcher=fetcher,
    )

    assert "vintage_dates=2018-11-15%2C2018-11-16" in captured_url
    assert "output_type=2" in captured_url
    assert "realtime_start=" not in captured_url
    assert series["GDP"].tolist() == [3.0, 5.0]


def test_load_fred_api_series_output_type3_uses_rowwise_latest_vintage_values() -> None:
    captured_url = ""

    def fetcher(url: str) -> bytes:
        nonlocal captured_url
        captured_url = url
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "GDP_20190101": "1.0", "GDP_20190301": "2.0"},'
            b'{"date": "2016-10-01", "GDP_20190101": "3.0"}'
            b"]}"
        )

    series = load_fred_api_series(
        "GDP",
        api_key="secret",
        output_type=3,
        vintage_dates="20190101,20190301",
        observation_start="2016-Q3",
        observation_end="2016-Q4",
        fetcher=fetcher,
    )

    assert "output_type=3" in captured_url
    assert "vintage_dates=2019-01-01%2C2019-03-01" in captured_url
    assert series["GDP"].tolist() == [2.0, 3.0]


def test_load_fred_api_series_output_type2_parses_underscore_prefixed_vintage_keys() -> None:
    captured_url = ""

    def fetcher(url: str) -> bytes:
        nonlocal captured_url
        captured_url = url
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "_1234_20190101": "1.0"},'
            b'{"date": "2016-10-01", "_1234_20190101": "2.0"}'
            b"]}"
        )

    series = load_fred_api_series(
        "1234",
        api_key="secret",
        output_type=2,
        vintage_dates="20190101",
        observation_start="2016-Q3",
        observation_end="2016-Q4",
        fetcher=fetcher,
    )

    assert "output_type=2" in captured_url
    assert series["1234"].tolist() == [1.0, 2.0]


def test_load_fred_api_series_output_type4_uses_scalar_output_type() -> None:
    captured_url = ""

    def fetcher(url: str) -> bytes:
        nonlocal captured_url
        captured_url = url
        return (
            b'{"observations": ['
            b'{"realtime_start": "2018-01-01", "realtime_end": "2018-01-31",'
            b'"date": "2016-07-01", "value": "1.0"},'
            b'{"realtime_start": "2018-01-01", "realtime_end": "2018-01-31",'
            b'"date": "2016-10-01", "value": "2.0"}'
            b"]}"
        )

    series = load_fred_api_series(
        "GDP",
        api_key="secret",
        output_type=4,
        realtime_start="20180101",
        realtime_end="20181231",
        observation_start="2016-Q3",
        observation_end="2016-Q4",
        fetcher=fetcher,
    )

    assert "output_type=4" in captured_url
    assert list(series["date"]) == ["2016-Q3", "2016-Q4"]
    assert series["GDP"].tolist() == [1.0, 2.0]


def test_load_fred_api_series_rejects_mixed_vintage_and_realtime_modes() -> None:
    with pytest.raises(ValueError, match="vintage_dates"):
        load_fred_api_series(
            "GDP",
            api_key="secret",
            realtime_start="181115",
            vintage_dates="181116",
        )


def test_load_fred_api_series_rejects_invalid_output_type() -> None:
    with pytest.raises(ValueError, match="output_type"):
        load_fred_api_series("GDP", api_key="secret", output_type=9)


def test_load_fred_api_series_rejects_duplicate_dates_without_realtime_metadata() -> None:
    def fetcher(url: str) -> bytes:
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "value": "1.0"},'
            b'{"date": "2016-07-01", "value": "3.0"}'
            b"]}"
        )

    with pytest.raises(ValueError, match="duplicate observation dates"):
        load_fred_api_series(
            "GDP",
            api_key="secret",
            realtime_start="2018-01-01",
            realtime_end="2018-02-28",
            fetcher=fetcher,
        )


def test_download_fred_api_source_csv_uses_env_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    output_path = tmp_path / "fred_api.csv"

    def fetcher(url: str) -> bytes:
        assert "api_key=from-env" in url
        series_id = url.split("series_id=", 1)[1].split("&", 1)[0]
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "value": "1.0"},'
            b'{"date": "2016-10-01", "value": "2.0"}'
            b"]}"
        ).replace(b"series_id", series_id.encode())

    levels = download_fred_api_source_csv(
        model,
        output_path=output_path,
        start_date="2016-Q3",
        end_date="2016-Q4",
        fetcher=fetcher,
    )

    assert output_path.exists()
    assert list(levels["date"]) == ["2016-Q3", "2016-Q4"]
    assert "GDP" in levels.columns


def test_load_fred_api_source_reports_missing_required_quarter_values(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "from-env")

    def fetcher(url: str) -> bytes:
        assert "series_id=GDP" in url
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "value": "."},'
            b'{"date": "2016-10-01", "value": "2.0"}'
            b"]}"
        )

    with pytest.raises(ValueError, match="FRED:GDP.*2016-Q3"):
        load_fred_api_source(
            ["GDP"],
            start_date="2016-Q3",
            end_date="2016-Q4",
            fetcher=fetcher,
        )


def test_load_fred_api_source_fills_optional_alfred_missing_series(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "from-env")

    def fetcher(url: str) -> bytes:
        if "series_id=BAMLC8A0C15PYEY" in url:
            raise HTTPError(
                url=url,
                code=400,
                msg="Bad Request",
                hdrs=Message(),
                fp=io.BytesIO(
                    b'{"error_message":"Bad Request. The series does not exist in ALFRED."}'
                ),
            )
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "value": "1.0"},'
            b'{"date": "2016-10-01", "value": "2.0"}'
            b"]}"
        )

    levels = load_fred_api_source(
        ["GDP", "BAMLC8A0C15PYEY"],
        realtime_start="181115",
        realtime_end="181115",
        start_date="2016-Q3",
        end_date="2016-Q4",
        fetcher=fetcher,
    )

    assert list(levels["date"]) == ["2016-Q3", "2016-Q4"]
    assert levels["GDP"].tolist() == [1.0, 2.0]
    assert levels["BAMLC8A0C15PYEY"].isna().all()


def test_load_fred_api_source_fills_optional_alfred_missing_json_error(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "from-env")

    def fetcher(url: str) -> bytes:
        if "series_id=BAMLC8A0C15PYEY" in url:
            return (
                b'{"error_code":400,'
                b'"error_message":"Bad Request. The series does not exist in ALFRED."}'
            )
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "value": "1.0"},'
            b'{"date": "2016-10-01", "value": "2.0"}'
            b"]}"
        )

    levels = load_fred_api_source(
        ["GDP", "BAMLC8A0C15PYEY"],
        realtime_start="181115",
        realtime_end="181115",
        start_date="2016-Q3",
        end_date="2016-Q4",
        fetcher=fetcher,
    )

    assert list(levels["date"]) == ["2016-Q3", "2016-Q4"]
    assert levels["GDP"].tolist() == [1.0, 2.0]
    assert levels["BAMLC8A0C15PYEY"].isna().all()


def test_load_fred_api_source_fills_optional_alfred_empty_payload(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "from-env")

    def fetcher(url: str) -> bytes:
        if "series_id=BAMLC8A0C15PYEY" in url:
            return b'{"observations": []}'
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "value": "1.0"},'
            b'{"date": "2016-10-01", "value": "2.0"}'
            b"]}"
        )

    levels = load_fred_api_source(
        ["GDP", "BAMLC8A0C15PYEY"],
        start_date="2016-Q3",
        end_date="2016-Q4",
        fetcher=fetcher,
    )

    assert list(levels["date"]) == ["2016-Q3", "2016-Q4"]
    assert levels["GDP"].tolist() == [1.0, 2.0]
    assert levels["BAMLC8A0C15PYEY"].isna().all()


def test_load_fred_api_source_fills_optional_alfred_missing_without_required_dates(
    monkeypatch,
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "from-env")

    def fetcher(url: str) -> bytes:
        if "series_id=BAMLC8A0C15PYEY" in url:
            raise HTTPError(
                url=url,
                code=400,
                msg="Bad Request",
                hdrs=Message(),
                fp=io.BytesIO(
                    b'{"error_message":"Bad Request. The series does not exist in ALFRED."}'
                ),
            )
        return (
            b'{"observations": '
            b'[{"date": "2016-07-01", "value": "1.0"},'
            b'{"date": "2016-10-01", "value": "2.0"}]'
            b"}"
        )

    levels = load_fred_api_source(
        ["GDP", "BAMLC8A0C15PYEY"],
        realtime_start="181115",
        realtime_end="181115",
        fetcher=fetcher,
    )

    assert list(levels["date"]) == ["2016-Q3", "2016-Q4"]
    assert levels["GDP"].tolist() == [1.0, 2.0]
    assert levels["BAMLC8A0C15PYEY"].isna().all()


def test_load_fred_api_series_raises_json_error_for_required_series() -> None:
    def fetcher(url: str) -> bytes:
        assert "api_key=secret" in url
        return b'{"error_code":400,"error_message":"Bad Request. Invalid realtime range."}'

    with pytest.raises(FredApiError, match="Invalid realtime range") as exc_info:
        load_fred_api_series("GDP", api_key="secret", fetcher=fetcher)

    assert exc_info.value.code == 400


def test_download_fred_api_source_csv_uses_dotenv_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "# local secrets\nFRED_API_KEY='from-dotenv' # keep private\n",
        encoding="utf-8",
    )
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    output_path = tmp_path / "fred_api.csv"

    def fetcher(url: str) -> bytes:
        assert "api_key=from-dotenv" in url
        return (
            b'{"observations": ['
            b'{"date": "2016-07-01", "value": "1.0"},'
            b'{"date": "2016-10-01", "value": "2.0"}'
            b"]}"
        )

    levels = download_fred_api_source_csv(
        model,
        output_path=output_path,
        start_date="2016-Q3",
        end_date="2016-Q4",
        fetcher=fetcher,
    )

    assert output_path.exists()
    assert list(levels["date"]) == ["2016-Q3", "2016-Q4"]


def test_download_fred_api_source_csv_prefers_env_over_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("FRED_API_KEY=from-dotenv\n", encoding="utf-8")
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})

    def fetcher(url: str) -> bytes:
        assert "api_key=from-env" in url
        assert "api_key=from-dotenv" not in url
        return b'{"observations": [{"date": "2016-07-01", "value": "1.0"}]}'

    download_fred_api_source_csv(
        model,
        output_path=tmp_path / "fred_api.csv",
        start_date="2016-Q3",
        end_date="2016-Q3",
        fetcher=fetcher,
    )


def test_download_fred_api_source_csv_requires_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})

    with pytest.raises(ValueError, match="FRED API access"):
        download_fred_api_source_csv(model, output_path=tmp_path / "fred_api.csv")


def test_transform_data_builds_model1002_base_observables_from_levels() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    levels = _raw_levels_fixture()

    transformed = transform_data(model, levels)

    assert list(transformed["date"]) == ["2016-Q3", "2016-Q4"]
    assert list(model.observables) == [column for column in transformed.columns if column != "date"]
    assert np.isnan(transformed.loc[0, "obs_gdp"])
    assert np.isclose(transformed.loc[1, "obs_gdp"], 100.0 * np.log(2.0))
    assert np.isclose(transformed.loc[0, "obs_nominalrate"], 0.5)
    assert np.isclose(transformed.loc[1, "obs_spread"], (8.0 - 1.5) / 4.0)
    assert np.isclose(transformed.loc[1, "obs_longinflation"], (2.5 - 0.5) / 4.0)


def test_transform_data_builds_anticipated_gdp_observables_from_levels() -> None:
    model = Model1002(
        settings={
            "add_anticipated_obs_gdp": True,
            "n_anticipated_obs_gdp": 1,
            "n_mon_anticipated_shocks": 0,
        }
    )
    levels = _raw_levels_fixture().assign(antgdp1=[0.4, 0.5])

    transformed = transform_data(model, levels)

    assert "obs_gdp1" in transformed.columns
    assert np.isclose(transformed.loc[0, "obs_gdp1"], 0.5)
    assert np.isclose(transformed.loc[1, "obs_gdp1"], 0.4)


def test_prepare_population_data_adds_hp_filtered_population_helpers() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    levels = _raw_levels_fixture_with_population_kink()

    prepared = prepare_population_data(model, levels)

    assert "filtered_population_recorded" in prepared.columns
    assert "dlfiltered_population_recorded" in prepared.columns
    assert "dlpopulation_recorded" in prepared.columns
    assert not np.allclose(
        prepared["filtered_population_recorded"],
        prepared["CNP16OV"],
    )


def test_prepare_population_data_uses_population_forecast_for_hp_filter() -> None:
    model = Model1002(
        settings={
            "n_mon_anticipated_shocks": 0,
            "population_hpfilter_lambda": 10.0,
        }
    )
    levels = _raw_levels_fixture_with_population_kink()
    population_forecast = pd.DataFrame(
        {
            "date": ["2016-Q4", "2017-Q1", "2017-Q2"],
            "POPULATION": [10.0, 80.0, 90.0],
        }
    )

    no_forecast = prepare_population_data(model, levels)
    with_forecast = prepare_population_data(
        model,
        levels,
        population_forecast=population_forecast,
    )

    assert not np.allclose(
        no_forecast["filtered_population_recorded"],
        with_forecast["filtered_population_recorded"],
    )


def test_transform_data_uses_hp_filtered_population_for_per_capita_observables() -> None:
    filtered_model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    unfiltered_model = Model1002(
        settings={
            "hpfilter_population": False,
            "n_mon_anticipated_shocks": 0,
        }
    )
    levels = _raw_levels_fixture_with_population_kink()

    filtered = transform_data(filtered_model, levels)
    unfiltered = transform_data(unfiltered_model, levels)

    assert not np.allclose(
        filtered["obs_gdp"].dropna(),
        unfiltered["obs_gdp"].dropna(),
    )
    assert np.allclose(
        unfiltered["obs_gdp"].dropna(),
        one_quarter_pct_change(np.array([10.0, 10.0 / 3.0, 10.0, 10.0]))[1:],
        equal_nan=False,
    )


def test_hpfilter_handles_edge_and_internal_missing_values() -> None:
    edge_missing = hpfilter([np.nan, 1.0, 2.0, 3.0, np.nan])
    assert np.isnan(edge_missing[0])
    assert np.isnan(edge_missing[-1])
    np.testing.assert_allclose(edge_missing[1:-1], [1.0, 2.0, 3.0])

    internal_missing = hpfilter([1.0, np.nan, 3.0])
    assert np.isnan(internal_missing).all()


def test_build_data_csv_writes_transformed_observable_file(tmp_path) -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "built" / "observables.csv"
    _raw_levels_fixture().to_csv(input_path, index=False)

    transformed = build_data_csv(model, input_path=input_path, output_path=output_path)
    loaded = pd.read_csv(output_path)

    assert output_path.exists()
    assert list(loaded.columns) == list(transformed.columns)
    assert list(loaded["date"]) == ["2016-Q3", "2016-Q4"]
    assert list(model.observables) == [column for column in loaded.columns if column != "date"]


def test_build_data_csv_accepts_population_forecast_file(tmp_path) -> None:
    model = Model1002(
        settings={
            "n_mon_anticipated_shocks": 0,
            "population_hpfilter_lambda": 10.0,
        }
    )
    input_path = tmp_path / "raw.csv"
    forecast_path = tmp_path / "population_forecast.csv"
    output_path = tmp_path / "built" / "observables.csv"
    _raw_levels_fixture_with_population_kink().to_csv(input_path, index=False)
    pd.DataFrame(
        {
            "date": ["2016-Q4", "2017-Q1", "2017-Q2"],
            "POPULATION": [10.0, 80.0, 90.0],
        }
    ).to_csv(forecast_path, index=False)

    transformed = build_data_csv(
        model,
        input_path=input_path,
        output_path=output_path,
        population_forecast_path=forecast_path,
    )

    assert output_path.exists()
    assert transformed.shape[0] == 4
    assert list(transformed["date"]) == ["2016-Q1", "2016-Q2", "2016-Q3", "2016-Q4"]


def test_df_to_matrix_transforms_raw_levels_when_observable_columns_are_absent() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    matrix = df_to_matrix(model, _raw_levels_fixture())
    assert matrix.shape == (2, len(model.observables))
    assert np.isclose(matrix[1, list(model.observables).index("obs_nominalrate")], 0.75)


def test_df_to_matrix_filters_raw_levels_after_transforming_observables() -> None:
    model = Model1002(
        settings={
            "date_forecast_start": "2016-Q4",
            "n_mon_anticipated_shocks": 0,
        }
    )

    matrix = df_to_matrix(model, _raw_levels_fixture())

    assert matrix.shape == (1, len(model.observables))
    assert np.isnan(matrix[0, list(model.observables).index("obs_gdp")])
    assert np.isclose(matrix[0, list(model.observables).index("obs_nominalrate")], 0.5)


def test_transform_data_reports_missing_source_column() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    levels = _raw_levels_fixture().drop(columns=["GDPDEF"])
    with pytest.raises(KeyError, match="GDPDEF"):
        transform_data(model, levels)


def test_reverse_transform_array_dispatches_reporting_units() -> None:
    np.testing.assert_allclose(reverse_transform_array("quarter_to_annual", [0.5]), [2.0])
    np.testing.assert_allclose(
        reverse_transform_array("loggrowth_to_pct_annualized", [1.0]),
        [400.0 * (np.exp(0.01) - 1.0)],
    )

    with pytest.raises(KeyError, match="Unknown reverse transform"):
        reverse_transform_array("missing", [1.0])


def test_reverse_transform_observables_uses_model_metadata() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    values = np.zeros((1, len(model.observables)))
    nominal_rate_column = list(model.observables).index("obs_nominalrate")
    deflator_column = list(model.observables).index("obs_gdpdeflator")
    hours_column = list(model.observables).index("obs_hours")
    values[0, nominal_rate_column] = 0.5
    values[0, deflator_column] = 1.0
    values[0, hours_column] = 3.0

    transformed = reverse_transform_observables(model, values)

    assert transformed.shape == values.shape
    assert np.isclose(transformed[0, nominal_rate_column], 2.0)
    assert np.isclose(transformed[0, deflator_column], 400.0 * (np.exp(0.01) - 1.0))
    assert np.isclose(transformed[0, hours_column], 3.0)


def test_reverse_transform_observables_validates_shape() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})

    with pytest.raises(ValueError, match="shape"):
        reverse_transform_observables(model, np.zeros(len(model.observables)))

    with pytest.raises(ValueError, match="columns"):
        reverse_transform_observables(model, np.zeros((1, len(model.observables) - 1)))


def _raw_levels_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2016-Q4", "2016-Q3"],
            "GDP": [220.0, 100.0],
            "GDPDEF": [110.0, 100.0],
            "CNP16OV": [10.0, 10.0],
            "AWHNONAG": [5.0, 4.0],
            "CE16OV": [10.0, 10.0],
            "COMPNFB": [121.0, 100.0],
            "PCEPILFE": [105.0, 100.0],
            "DFF": [3.0, 2.0],
            "PCE": [110.0, 50.0],
            "FPI": [33.0, 15.0],
            "BAA": [20.0, 5.0],
            "BAMLC8A0C15PYEY": [8.0, 7.0],
            "GS10": [1.5, 1.0],
            "ASACX10": [2.5, 2.0],
            "FYCCZA": [4.0, 3.0],
            "TFPKQ": [2.0, 1.0],
            "TFPJQ": [0.25, 0.25],
            "GDI": [330.0, 150.0],
        }
    )


def _raw_levels_fixture_with_population_kink() -> pd.DataFrame:
    base = pd.concat(
        [_raw_levels_fixture().iloc[[1]].copy() for _ in range(4)],
        ignore_index=True,
    )
    base["date"] = ["2016-Q1", "2016-Q2", "2016-Q3", "2016-Q4"]
    base["GDP"] = [100.0, 100.0, 100.0, 100.0]
    base["GDPDEF"] = [100.0, 100.0, 100.0, 100.0]
    base["CNP16OV"] = [10.0, 30.0, 10.0, 10.0]
    return base
