from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import nydsge.cli as cli
from nydsge.cli import app
from nydsge.estimate import (
    EstimationModeResult,
    MetropolisHastingsResult,
    evaluate_log_posterior_for_parameter_values,
    parameter_estimation_vector,
    save_estimation_mode,
    save_sampler_result,
)
from nydsge.models import Model1002
from nydsge.runtime import RuntimeStatus
from nydsge.vv import load_sampler_fixture_result


def test_doctor_json() -> None:
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert "numpy" in result.stdout
    assert "cpu" in result.stdout


def test_models_json() -> None:
    result = CliRunner().invoke(app, ["models", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["models"] == [
        {
            "aliases": ["Model1002", "model1002"],
            "default_subspec": "ss10",
            "description": "New York Fed DSGE Model1002 representative-agent model.",
            "name": "m1002",
        }
    ]


def test_doctor_json_can_resolve_requested_runtime() -> None:
    result = CliRunner().invoke(
        app,
        ["doctor", "--backend", "numpy", "--device", "cpu", "--dtype", "float64", "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["requested_runtime"]["available"] is True
    assert payload["requested_runtime"]["resolved"]["backend"] == "numpy"
    assert payload["requested_runtime"]["resolved"]["device"] == "cpu"


def test_doctor_json_fails_for_unsupported_requested_runtime() -> None:
    result = CliRunner().invoke(
        app,
        ["doctor", "--backend", "numpy", "--device", "cuda", "--json"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["requested_runtime"]["available"] is False
    assert "NumPy backend only supports" in payload["requested_runtime"]["reason"]


def test_doctor_json_fails_for_wsl_platform(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "runtime_report",
        lambda: [
            RuntimeStatus(
                "platform",
                "native",
                False,
                "WSL is not supported; use native Windows, macOS, or Linux",
            ),
            RuntimeStatus("numpy", "cpu", True, "NumPy CPU reference is available"),
        ],
    )

    result = CliRunner().invoke(app, ["doctor", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload[0]["backend"] == "platform"
    assert payload[0]["available"] is False


def test_bench_json_reports_run_and_skips() -> None:
    result = CliRunner().invoke(
        app,
        ["bench", "--horizon", "1", "--repeats", "1", "--include-pseudo", "--json"],
    )

    assert result.exit_code == 0
    assert '"kernel": "forecast"' in result.stdout
    assert '"backend": "numpy"' in result.stdout
    assert '"elapsed_seconds":' in result.stdout
    assert '"pseudo_observables_shape": [' in result.stdout
    assert '"skipped": true' in result.stdout


def test_bench_json_can_run_kalman_kernel() -> None:
    result = CliRunner().invoke(
        app,
        ["bench", "--kernel", "kalman", "--periods", "1", "--repeats", "1", "--json"],
    )

    assert result.exit_code == 0
    assert '"kernel": "kalman"' in result.stdout
    assert '"states_shape": [' in result.stdout


def test_bench_json_can_run_batched_kalman_kernel() -> None:
    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--kernel",
            "kalman-batch",
            "--periods",
            "1",
            "--batches",
            "2",
            "--repeats",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"kernel": "kalman-batch"' in result.stdout
    assert '"batches": 2' in result.stdout
    assert '"states_shape": [' in result.stdout


def test_bench_json_can_run_hard_target_kernel() -> None:
    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--kernel",
            "hard-target",
            "--horizon",
            "1",
            "--periods",
            "1",
            "--draws",
            "1",
            "--repeats",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"kernel": "hard-target"' in result.stdout
    assert '"draws": 1' in result.stdout
    assert '"artifacts": 8' in result.stdout


def test_bench_json_can_compare_oracle_baseline(tmp_path) -> None:
    baseline_path = tmp_path / "oracle_benchmark.json"
    baseline_path.write_text(
        json.dumps(
            {
                "name": "julia-oracle",
                "results": [
                    {
                        "kernel": "forecast",
                        "horizon": 1,
                        "dtype": "float64",
                        "elapsed_seconds": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--horizon",
            "1",
            "--repeats",
            "1",
            "--baseline",
            str(baseline_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    numpy = next(item for item in payload if item["backend"] == "numpy" and item["device"] == "cpu")
    assert numpy["baseline_name"] == "julia-oracle"
    assert numpy["baseline_elapsed_seconds"] == 10.0
    assert numpy["speedup_vs_baseline"] > 0.0


def test_bench_can_write_reference_report(tmp_path) -> None:
    baseline_path = tmp_path / "oracle_benchmark.json"
    output_path = tmp_path / "reports" / "local_benchmark.json"
    baseline_path.write_text(
        json.dumps(
            {
                "name": "julia-oracle",
                "results": [
                    {
                        "kernel": "forecast",
                        "horizon": 1,
                        "dtype": "float64",
                        "elapsed_seconds": 10.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--horizon",
            "1",
            "--repeats",
            "1",
            "--baseline",
            str(baseline_path),
            "--output",
            str(output_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    payload = json.loads(result.stdout)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert report["schema_version"] == 1
    assert report["command"]["kernel"] == "forecast"
    assert report["command"]["baseline_path"] == str(baseline_path)
    assert report["platform"]["system"]
    numpy = next(
        item for item in report["results"] if item["backend"] == "numpy" and item["device"] == "cpu"
    )
    assert numpy["baseline_name"] == "julia-oracle"
    assert numpy["speedup_vs_baseline"] > 0.0


def test_bench_json_all_includes_batched_kalman_kernel() -> None:
    result = CliRunner().invoke(
        app,
        [
            "bench",
            "--kernel",
            "all",
            "--horizon",
            "1",
            "--periods",
            "1",
            "--batches",
            "2",
            "--draws",
            "1",
            "--repeats",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    kernels = {item["kernel"] for item in payload}
    assert {"forecast", "kalman", "kalman-batch", "hard-target"}.issubset(kernels)


def test_bench_rejects_unknown_kernel() -> None:
    result = CliRunner().invoke(app, ["bench", "--kernel", "bad"])

    assert result.exit_code == 2
    assert "Benchmark kernel" in result.stdout


def test_data_build_command_writes_observable_csv(tmp_path) -> None:
    input_path = _write_raw_levels_csv(tmp_path)
    output_path = tmp_path / "built" / "observables.csv"

    result = CliRunner().invoke(
        app,
        [
            "data",
            "build",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"rows": 2' in result.stdout
    assert '"observables": 19' in result.stdout
    assert '"hpfilter_population": true' in result.stdout
    loaded = pd.read_csv(output_path)
    assert "obs_gdp" in loaded.columns
    assert list(loaded["date"]) == ["2016-Q3", "2016-Q4"]


def test_data_build_command_can_disable_hp_filtered_population(tmp_path) -> None:
    input_path = _write_raw_levels_csv(tmp_path)
    output_path = tmp_path / "built" / "observables.csv"

    result = CliRunner().invoke(
        app,
        [
            "data",
            "build",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--no-hpfilter-population",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"hpfilter_population": false' in result.stdout
    assert output_path.exists()


def test_data_build_command_accepts_population_forecast(tmp_path) -> None:
    input_path = _write_raw_levels_csv(tmp_path)
    forecast_path = tmp_path / "population_forecast.csv"
    output_path = tmp_path / "built" / "observables.csv"
    pd.DataFrame(
        {
            "date": ["2016-Q4", "2017-Q1"],
            "POPULATION": [10.0, 12.0],
        }
    ).to_csv(forecast_path, index=False)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "build",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--population-forecast",
            str(forecast_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"population_forecast":' in result.stdout
    assert "population_forecast.csv" in result.stdout
    assert output_path.exists()


def test_data_build_sources_command_writes_observable_csv(tmp_path) -> None:
    raw_path = _write_raw_levels_csv(tmp_path)
    raw = pd.read_csv(raw_path)
    source_root = tmp_path / "sources"
    source_root.mkdir()
    raw.drop(columns=[f"ant{index}" for index in range(1, 7)]).to_csv(
        source_root / "fred_181115.csv",
        index=False,
    )
    raw.drop(columns=[f"ant{index}" for index in range(1, 7)]).to_csv(
        source_root / "dlx_181115.csv",
        index=False,
    )
    raw[["date", *[f"ant{index}" for index in range(1, 7)]]].to_csv(
        source_root / "ois_181115.csv",
        index=False,
    )
    output_path = tmp_path / "built" / "observables.csv"

    result = CliRunner().invoke(
        app,
        [
            "data",
            "build-sources",
            "--source-root",
            str(source_root),
            "--output",
            str(output_path),
            "--start-date",
            "2016-Q3",
            "--end-date",
            "2016-Q4",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"source_root":' in result.stdout
    assert '"start_date": "2016-Q3"' in result.stdout
    assert '"end_date": "2016-Q4"' in result.stdout
    assert '"observables": 19' in result.stdout
    loaded = pd.read_csv(output_path)
    assert "obs_nominalrate6" in loaded.columns
    assert list(loaded["date"]) == ["2016-Q3", "2016-Q4"]


def test_data_sources_command_reports_required_source_files(tmp_path) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    (source_root / "fred_181115.csv").write_text("date,GDP\n2016-Q3,1.0\n", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "data",
            "sources",
            "--source-root",
            str(source_root),
            "--vintage",
            "181115",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    by_source = {source["source"]: source for source in payload["sources"]}
    assert by_source["FRED"]["available"] is True
    assert by_source["FRED"]["fetchable"] is True
    assert "BAMLC8A0C15PYEY" in by_source["FRED"]["optional_mnemonics"]
    assert by_source["DLX"]["available"] is False
    assert "dlx_181115.csv" in " ".join(by_source["DLX"]["candidate_paths"])


def test_data_prepare_sources_command_fetches_canonical_fred_and_reports_gaps(
    tmp_path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "sources"

    def fake_download(*args, **kwargs) -> pd.DataFrame:
        assert kwargs["output_path"] == source_root / "fred_181115.csv"
        assert kwargs["realtime_start"] == "181115"
        assert kwargs["realtime_end"] == "181115"
        source_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "date": ["2016-Q3", "2016-Q4"],
                "GDP": [1.0, 2.0],
            }
        ).to_csv(kwargs["output_path"], index=False)
        return pd.DataFrame(
            {
                "date": ["2016-Q3", "2016-Q4"],
                "GDP": [1.0, 2.0],
            }
        )

    monkeypatch.setattr("nydsge.cli.download_fred_api_source_csv", fake_download)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "prepare-sources",
            "--source-root",
            str(source_root),
            "--vintage",
            "181115",
            "--realtime-start",
            "181115",
            "--realtime-end",
            "181115",
            "--start-date",
            "2016-Q3",
            "--end-date",
            "2016-Q4",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["fred_action"] == "downloaded"
    assert payload["fred_output"].endswith("fred_181115.csv")
    assert payload["ready_for_build"] is False
    assert {source["source"] for source in payload["missing_sources"]} == {"DLX", "OIS"}
    assert (source_root / "fred_181115.csv").exists()


def test_data_prepare_sources_command_defaults_realtime_window_to_vintage(
    tmp_path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "sources"

    def fake_download(*args, **kwargs) -> pd.DataFrame:
        assert kwargs["output_path"] == source_root / "fred_181115.csv"
        assert kwargs["realtime_start"] == "181115"
        assert kwargs["realtime_end"] == "181115"
        source_root.mkdir(parents=True, exist_ok=True)
        data = pd.DataFrame(
            {
                "date": ["2016-Q3"],
                "GDP": [1.0],
            }
        )
        data.to_csv(kwargs["output_path"], index=False)
        return data

    monkeypatch.setattr("nydsge.cli.download_fred_api_source_csv", fake_download)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "prepare-sources",
            "--source-root",
            str(source_root),
            "--vintage",
            "181115",
            "--start-date",
            "2016-Q3",
            "--end-date",
            "2016-Q3",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["vintage"] == "181115"
    assert payload["realtime_start"] == "181115"
    assert payload["realtime_end"] == "181115"
    assert payload["fred_action"] == "downloaded"


def test_data_prepare_sources_command_rejects_mixed_vintage_and_realtime_modes(
    tmp_path,
) -> None:
    result = CliRunner().invoke(
        app,
        [
            "data",
            "prepare-sources",
            "--source-root",
            str(tmp_path / "sources"),
            "--vintage-dates",
            "181115,181116",
            "--realtime-start",
            "181115",
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "cannot be combined" in result.stdout


def test_data_prepare_sources_command_reports_ready_when_all_sources_exist(
    tmp_path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "sources"
    source_root.mkdir()
    for name in ("fred_181115.csv", "dlx_181115.csv", "ois_181115.csv"):
        (source_root / name).write_text("date,value\n2016-Q3,1.0\n", encoding="utf-8")

    def fail_download(*args, **kwargs) -> pd.DataFrame:
        raise AssertionError("existing FRED source should not be downloaded")

    monkeypatch.setattr("nydsge.cli.download_fred_api_source_csv", fail_download)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "prepare-sources",
            "--source-root",
            str(source_root),
            "--vintage",
            "181115",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["fred_action"] == "existing"
    assert payload["ready_for_build"] is True
    assert payload["missing_sources"] == []


def test_data_fetch_fred_command_writes_source_csv(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "fred_181115.csv"

    def fake_download(*args, **kwargs) -> pd.DataFrame:
        pd.DataFrame(
            {
                "date": ["2016-Q3", "2016-Q4"],
                "GDP": [1.0, 2.0],
                "DFF": [0.5, 0.75],
            }
        ).to_csv(kwargs["output_path"], index=False)
        return pd.DataFrame(
            {
                "date": ["2016-Q3", "2016-Q4"],
                "GDP": [1.0, 2.0],
                "DFF": [0.5, 0.75],
            }
        )

    monkeypatch.setattr("nydsge.cli.download_current_fred_source_csv", fake_download)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "fetch-fred",
            "--output",
            str(output_path),
            "--start-date",
            "2016-Q3",
            "--end-date",
            "2016-Q4",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"fred_series": 2' in result.stdout
    assert '"first_date": "2016-Q3"' in result.stdout
    assert output_path.exists()


def test_data_fetch_fred_api_command_writes_source_csv(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "fred_api_181115.csv"

    def fake_download(*args, **kwargs) -> pd.DataFrame:
        assert kwargs["api_key"] == "secret"
        assert kwargs["realtime_start"] == "181115"
        assert kwargs["realtime_end"] == "181115"
        pd.DataFrame(
            {
                "date": ["2016-Q3", "2016-Q4"],
                "GDP": [1.0, 2.0],
                "DFF": [0.5, 0.75],
            }
        ).to_csv(kwargs["output_path"], index=False)
        return pd.DataFrame(
            {
                "date": ["2016-Q3", "2016-Q4"],
                "GDP": [1.0, 2.0],
                "DFF": [0.5, 0.75],
            }
        )

    monkeypatch.setattr("nydsge.cli.download_fred_api_source_csv", fake_download)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "fetch-fred-api",
            "--output",
            str(output_path),
            "--api-key",
            "secret",
            "--realtime-start",
            "181115",
            "--realtime-end",
            "181115",
            "--start-date",
            "2016-Q3",
            "--end-date",
            "2016-Q4",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"fred_series": 2' in result.stdout
    assert '"realtime_start": "181115"' in result.stdout
    assert output_path.exists()


def test_data_fetch_fred_api_command_accepts_vintage_dates_mode(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "fred_api_vintages.csv"

    def fake_download(*args, **kwargs) -> pd.DataFrame:
        assert kwargs["output_path"] == output_path
        assert kwargs["realtime_start"] is None
        assert kwargs["realtime_end"] is None
        assert kwargs["vintage_dates"] == "181115,181116"
        assert kwargs["output_type"] == 2
        data = pd.DataFrame({"date": ["2016-Q3"], "GDP": [1.0]})
        data.to_csv(kwargs["output_path"], index=False)
        return data

    monkeypatch.setattr("nydsge.cli.download_fred_api_source_csv", fake_download)

    result = CliRunner().invoke(
        app,
        [
            "data",
            "fetch-fred-api",
            "--output",
            str(output_path),
            "--vintage-dates",
            "181115,181116",
            "--output-type",
            "2",
            "--start-date",
            "2016-Q3",
            "--end-date",
            "2016-Q3",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["vintage_dates"] == "181115,181116"
    assert payload["output_type"] == 2
    assert payload["rows"] == 1


def test_estimate_command_reports_posterior_metrics(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(app, ["estimate", "--data", str(data_path), "--json"])

    assert result.exit_code == 0
    assert '"log_likelihood":' in result.stdout
    assert '"log_prior":' in result.stdout
    assert '"n_parameters": 95' in result.stdout


def test_estimate_command_can_optimize_selected_parameter(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=1)

    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "--data",
            str(data_path),
            "--optimize",
            "--parameters",
            "alpha",
            "--maxiter",
            "1",
            "--hessian",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"parameter_names": [' in result.stdout
    assert '"alpha"' in result.stdout
    assert '"hessian_shape": [' in result.stdout


def test_estimate_command_can_write_mode_output(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=1)
    mode_path = tmp_path / "mode" / "mode.npz"

    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "--data",
            str(data_path),
            "--optimize",
            "--parameters",
            "alpha",
            "--maxiter",
            "1",
            "--hessian",
            "--mode-output",
            str(mode_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"mode_output":' in result.stdout
    assert mode_path.exists()
    with np.load(mode_path) as archive:
        assert archive["parameter_names"].tolist() == ["alpha"]
        assert archive["estimation_values"].shape == (1,)
        assert archive["hessian"].shape == (1, 1)


def test_estimate_command_can_load_mode_input_for_sampler(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=1)
    mode_path = _write_mode_archive(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "--data",
            str(data_path),
            "--mode-input",
            str(mode_path),
            "--mh-draws",
            "2",
            "--seed",
            "7",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"mode_input":' in result.stdout
    assert '"sampler": {' in result.stdout
    assert '"proposal_covariance_shape": [' in result.stdout


def test_estimate_command_can_seed_sampler_from_hessian(tmp_path) -> None:
    data_path = _write_dated_observable_csv(tmp_path, dates=["2018-Q3"])

    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "--data",
            str(data_path),
            "--optimize",
            "--parameters",
            "alpha",
            "--maxiter",
            "1",
            "--hessian",
            "--mh-draws",
            "2",
            "--proposal-scale",
            "0.0001",
            "--seed",
            "7",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"hessian_shape": [' in result.stdout
    assert '"sampler": {' in result.stdout
    assert '"proposal_covariance_shape": [' in result.stdout


def test_estimate_command_can_run_metropolis_hastings(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=1)

    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "--data",
            str(data_path),
            "--parameters",
            "alpha",
            "--mh-draws",
            "2",
            "--proposal-scale",
            "0.0001",
            "--seed",
            "7",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"sampler": {' in result.stdout
    assert '"parameter_names": [' in result.stdout
    assert '"alpha"' in result.stdout
    assert '"estimation_draws_shape": [' in result.stdout
    assert '"acceptance_rate":' in result.stdout


def test_estimate_command_can_run_blocked_metropolis_hastings(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=1)

    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "--data",
            str(data_path),
            "--parameters",
            "alpha",
            "--mh-draws",
            "4",
            "--mh-blocks",
            "2",
            "--mh-param-blocks",
            "1",
            "--mh-thin",
            "2",
            "--proposal-scale",
            "0.0001",
            "--seed",
            "7",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sampler"] is not None
    assert payload["sampler"]["mh_blocks"] == 2
    assert payload["sampler"]["mh_param_blocks"] == 1
    assert payload["sampler"]["mh_thin"] == 2
    assert payload["sampler"]["estimation_draws_shape"] == [4, 1]


def test_estimate_command_can_load_proposal_covariance_csv(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=1)
    covariance_path = tmp_path / "proposal.csv"
    np.savetxt(covariance_path, np.array([[1.0e-8]]), delimiter=",")

    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "--data",
            str(data_path),
            "--parameters",
            "alpha",
            "--mh-draws",
            "2",
            "--proposal-covariance",
            str(covariance_path),
            "--seed",
            "7",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"sampler": {' in result.stdout
    assert '"parameter_draws_shape": [' in result.stdout


def test_estimate_command_can_write_sampler_output(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=1)
    sampler_path = tmp_path / "chains" / "sampler.npz"

    result = CliRunner().invoke(
        app,
        [
            "estimate",
            "--data",
            str(data_path),
            "--parameters",
            "alpha",
            "--mh-draws",
            "2",
            "--proposal-scale",
            "0.0001",
            "--seed",
            "7",
            "--sampler-output",
            str(sampler_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"sampler_output":' in result.stdout
    assert sampler_path.exists()
    with np.load(sampler_path) as archive:
        assert archive["parameter_draws"].shape == (2, 1)
        assert archive["parameter_names"].tolist() == ["alpha"]


def test_vv_sampler_diagnostics_reports_archive_metrics(tmp_path) -> None:
    sampler_path = _write_sampler_archive(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "sampler-diagnostics",
            "--sampler",
            str(sampler_path),
            "--windows",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sampler"] == str(sampler_path)
    assert payload["parameter_names"] == ["alpha"]
    assert payload["draws"] == 2
    assert payload["acceptance_windows"] == [1.0, 1.0]
    assert payload["proposal_covariance_shape"] == [1, 1]
    assert payload["proposal_covariance_positive_semidefinite"] is True
    assert payload["parameters"][0]["name"] == "alpha"
    assert payload["parameters"][0]["effective_sample_size"] > 0.0
    assert payload["parameters"][0]["monte_carlo_standard_error"] >= 0.0
    assert payload["parameters"][0]["split_rhat"] is None


def test_vv_sampler_fixture_summary_reports_julia_hdf5_metadata(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "m1002_sampler.h5"
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = np.array([[0.1, 1.1, 2.1], [0.2, 1.2, 2.2]])
        handle["sampler/proposal_covariance"] = np.diag([0.5, 2.0])
        handle["sampler/input_proposal_covariance"] = 1.0e-8 * np.eye(2)
        handle["sampler/accepted"] = np.array([1, 0, 1], dtype=np.int8)
        handle["sampler/log_posterior"] = np.array([-3.0, -2.5, -2.0])
        handle["sampler/proposal_parameters"] = np.array([[0.15, 1.15, 2.15], [0.25, 1.25, 2.25]])
        handle["sampler/previous_parameters"] = np.array([[0.05, 1.05, 2.05], [0.15, 1.15, 2.15]])
        handle["sampler/proposal_log_posterior"] = np.array([-2.8, -2.6, -1.9])
        handle["sampler/previous_log_posterior"] = np.array([-3.1, -2.7, -2.1])
        handle["sampler/uniform_draw"] = np.array([0.2, 0.6, 0.4])
        handle["sampler/log_acceptance"] = np.array([0.3, -0.1, 0.2])
        handle.attrs["sampler_parameter_names"] = "alpha,beta"
        handle.attrs["sampler_draws"] = 3
        handle.attrs["sampler_proposal_scale"] = "1.0e-8"
        handle.attrs["sampler_covariance_source"] = "saved_draw_covariance"
        handle.attrs["sampler_trace_available"] = "true"
        handle.attrs["sampler_proposal_trace_available"] = "true"
        handle.attrs["sampler_acceptance_rate"] = "0.6666666666666666"
        handle.attrs["sampler_block_acceptance_rates"] = "0.6666666666666666"
        handle.attrs["sampler_input_proposal_covariance_available"] = "true"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "sampler-fixture-summary",
            "--sampler",
            str(oracle),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["fixture_path"] == str(path)
    assert payload["parameter_names"] == ["alpha", "beta"]
    assert payload["mhparams_shape"] == [2, 3]
    assert payload["parameter_axis"] == 0
    assert payload["draw_axis"] == 1
    assert payload["draws"] == 3
    assert payload["covariance_shape"] == [2, 2]
    assert payload["covariance_source"] == "saved_draw_covariance"
    assert payload["covariance_positive_semidefinite"] is True
    assert payload["input_proposal_covariance_available"] is True
    assert payload["trace_available"] is True
    assert payload["accepted_shape"] == [3]
    assert payload["log_posterior_shape"] == [3]
    assert payload["accepted_draws"] == 2
    assert payload["realized_acceptance_rate"] == 2 / 3
    assert payload["log_posterior_minimum"] == -3.0
    assert payload["log_posterior_maximum"] == -2.0
    assert payload["proposal_trace_available"] is True
    assert payload["proposal_parameters_shape"] == [2, 3]
    assert payload["previous_parameters_shape"] == [2, 3]
    assert payload["proposal_log_posterior_shape"] == [3]
    assert payload["uniform_draw_shape"] == [3]
    assert payload["log_acceptance_shape"] == [3]
    assert payload["proposal_log_posterior_minimum"] == -2.8
    assert payload["proposal_log_posterior_maximum"] == -1.9
    assert payload["log_acceptance_minimum"] == -0.1
    assert payload["log_acceptance_maximum"] == 0.3
    assert payload["metadata"]["proposal_scale"] == 1.0e-8
    assert payload["metadata"]["trace_available"] is True
    assert payload["metadata"]["proposal_trace_available"] is True
    assert payload["metadata"]["acceptance_rate"] == 2 / 3
    assert payload["metadata"]["block_acceptance_rates"] == 2 / 3
    assert payload["unavailable_diagnostics"] == []
    assert payload["unavailable_proposal_diagnostics"] == []


def test_vv_sampler_compare_reports_matching_trace_archive(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle_path = tmp_path / "oracle_sampler.h5"
    with h5py.File(oracle_path, "w") as handle:
        handle["sampler/mhparams"] = np.array([[0.1, 1.1, 2.1], [0.2, 1.2, 2.2]])
        handle["sampler/proposal_covariance"] = np.diag([0.5, 2.0])
        handle["sampler/accepted"] = np.array([1, 0, 1], dtype=np.int8)
        handle["sampler/log_posterior"] = np.array([-3.0, -2.5, -2.0])
        handle.attrs["sampler_parameter_names"] = "alpha,beta"
        handle.attrs["sampler_draws"] = 3
        handle.attrs["sampler_blocks"] = 1
        handle.attrs["sampler_param_blocks"] = 1
        handle.attrs["sampler_thin"] = 1
        handle.attrs["sampler_burnin"] = 0
        handle.attrs["sampler_acceptance_rate"] = "0.6666666666666666"
        handle.attrs["sampler_seed"] = 123
    candidate_path = save_sampler_result(
        load_sampler_fixture_result(oracle_path),
        tmp_path / "candidate_sampler.npz",
    )

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "sampler-compare",
            "--oracle-sampler",
            str(oracle_path),
            "--candidate-sampler",
            str(candidate_path),
            "--windows",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["oracle_sampler"] == str(oracle_path)
    assert payload["candidate_sampler"] == str(candidate_path)
    assert payload["passed"] is True
    assert {item["name"] for item in payload["comparisons"]} == {
        "parameter_draws",
        "log_posterior",
        "accepted",
        "proposal_covariance",
        "diagnostics/core",
        "diagnostics/acceptance_windows",
        "diagnostics/parameters",
    }


def test_vv_sampler_proposal_trace_check_replays_julia_hdf5_trace(tmp_path) -> None:
    sampler_path = _write_sampler_proposal_trace_hdf5(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "sampler-proposal-trace-check",
            "--sampler",
            str(sampler_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sampler_path"] == str(sampler_path)
    assert payload["passed"] is True
    assert payload["tolerance_profile"]["name"] == "strict"
    assert {item["name"] for item in payload["comparisons"]} == {
        "proposal_trace/log_acceptance",
        "proposal_trace/accepted",
        "proposal_trace/retained_log_posterior",
        "proposal_trace/retained_parameters",
    }


def test_vv_sampler_posterior_replay_compares_model_value_trace(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=1)
    model = Model1002()
    observations = np.zeros((1, len(model.observables)), dtype=np.float64)
    sampler_path = _write_sampler_posterior_replay_hdf5(tmp_path, model, observations)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "sampler-posterior-replay",
            "--sampler",
            str(sampler_path),
            "--data",
            str(data_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sampler_path"] == str(sampler_path)
    assert payload["data"] == str(data_path)
    assert payload["passed"] is True
    assert payload["draws"] == 2
    assert payload["tolerance_profile"]["name"] == "strict"
    assert {item["name"] for item in payload["comparisons"]} == {
        "proposal_trace/proposal_log_posterior",
        "proposal_trace/previous_log_posterior",
        "proposal_trace/log_acceptance_from_replay",
        "proposal_trace/proposal_log_likelihood",
        "proposal_trace/previous_log_likelihood",
        "proposal_trace/proposal_log_prior",
        "proposal_trace/previous_log_prior",
    }


def test_presample_period_count_normalizes_timestamp_settings() -> None:
    model = Model1002()
    model.set_setting("date_presample_start", "2017-Q4")
    model.set_setting("date_mainsample_start", pd.Timestamp("2018-04-01"))
    model.set_setting("date_forecast_start", "2018-Q4")
    data = pd.DataFrame(
        {
            "date": ["2017-Q4", "2018-Q1", "2018-Q2"],
            next(iter(model.observables)): [1.0, 2.0, 3.0],
        }
    )

    assert cli._presample_period_count(model, data) == 2


def test_forecast_command_reports_model1002_forecast_shape() -> None:
    result = CliRunner().invoke(app, ["forecast", "--horizon", "3", "--json"])

    assert result.exit_code == 0
    assert '"states_shape": [' in result.stdout
    assert "84" in result.stdout
    assert "19" in result.stdout


def test_forecast_command_can_report_transformed_observables() -> None:
    result = CliRunner().invoke(app, ["forecast", "--horizon", "2", "--transformed", "--json"])

    assert result.exit_code == 0
    assert '"transformed": true' in result.stdout
    assert '"observables_shape": [' in result.stdout


def test_forecast_command_can_use_zlb_rate_path() -> None:
    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--horizon",
            "4",
            "--zlb-rates",
            "0.25,-0.10,0.75",
            "--zlb-floor",
            "0.0",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    n_shocks = len(Model1002().indexes.exogenous_shocks)
    assert payload["cond_type"] == "full"
    assert payload["zlb_rates"] == [0.25, -0.1, 0.75]
    assert payload["conditional_shocks_shape"] == [3, n_shocks]
    assert payload["conditional_states_shape"] == [3, 84]
    assert payload["states_shape"] == [1, 84]


def test_forecast_command_can_run_full_sample_forecast() -> None:
    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--input-type",
            "full",
            "--draws",
            "2",
            "--seed",
            "5",
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"draws": 2' in result.stdout
    assert '"observable_samples_shape": [' in result.stdout


def test_forecast_command_can_use_shock_sample_archive(tmp_path) -> None:
    shock_samples_path = _write_shock_samples_archive(tmp_path, horizon=2)

    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--input-type",
            "full",
            "--shock-samples",
            str(shock_samples_path),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    n_shocks = len(Model1002().indexes.exogenous_shocks)
    assert payload["draws"] == 0
    assert payload["shock_samples_shape"] == [2, 2, n_shocks]
    assert payload["observable_samples_shape"] == [2, 2, 19]


def test_forecast_command_can_use_sampler_draw_archive(tmp_path) -> None:
    sampler_path = _write_sampler_archive(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--input-type",
            "full",
            "--sampler-draws",
            str(sampler_path),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"sampler_draws":' in result.stdout
    assert '"observable_samples_shape": [' in result.stdout


def test_forecast_command_can_use_conditional_sampler_history(tmp_path) -> None:
    sampler_path = _write_sampler_archive(tmp_path)
    data_path = _write_dated_observable_csv(
        tmp_path,
        dates=["2018-Q3", "2018-Q4", "2019-Q1"],
    )

    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--input-type",
            "full",
            "--cond-type",
            "semi",
            "--sampler-draws",
            str(sampler_path),
            "--data",
            str(data_path),
            "--include-history",
            "--horizon",
            "4",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"states_shape": [\n    2,\n    84\n  ]' in result.stdout
    assert '"history_states_shape": [\n    1,\n    84\n  ]' in result.stdout
    assert '"history_state_samples_shape": [\n    2,\n    1,\n    84\n  ]' in result.stdout


def test_forecast_command_can_include_filter_backed_history(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--horizon",
            "1",
            "--data",
            str(data_path),
            "--include-history",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"history_observables_shape": [' in result.stdout
    assert '"log_likelihood":' in result.stdout


def test_forecast_command_can_include_smoothed_history(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--horizon",
            "1",
            "--data",
            str(data_path),
            "--include-history",
            "--history-method",
            "smoothed",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"history_method": "smoothed"' in result.stdout
    assert '"history_states_shape": [' in result.stdout


def test_forecast_command_can_include_pseudo_observables(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "forecast",
            "--horizon",
            "1",
            "--data",
            str(data_path),
            "--include-history",
            "--include-pseudo",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"pseudo_observables_shape": [' in result.stdout
    assert '"history_pseudo_observables_shape": [' in result.stdout


def test_forecast_command_rejects_unported_mode() -> None:
    result = CliRunner().invoke(app, ["forecast", "--input-type", "full"])

    assert result.exit_code == 2
    assert "input_type" in result.stdout


def test_forecast_command_rejects_unsupported_runtime_cleanly() -> None:
    result = CliRunner().invoke(app, ["forecast", "--backend", "numpy", "--device", "cuda"])

    assert result.exit_code == 2
    assert "NumPy backend only supports" in result.stdout


def test_meansbands_command_reports_model1002_band_shape() -> None:
    result = CliRunner().invoke(app, ["meansbands", "--horizon", "3", "--json"])

    assert result.exit_code == 0
    assert '"mean_shape": [' in result.stdout
    assert "19" in result.stdout


def test_meansbands_command_rejects_transformed_states() -> None:
    result = CliRunner().invoke(
        app,
        ["meansbands", "--source", "states", "--transformed", "--json"],
    )

    assert result.exit_code == 2
    assert "--transformed is only valid" in result.stdout


def test_meansbands_command_can_use_histobs_source(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        ["meansbands", "--source", "histobs", "--data", str(data_path), "--json"],
    )

    assert result.exit_code == 0
    assert '"source": "histobs"' in result.stdout
    assert '"mean_shape": [' in result.stdout


def test_meansbands_command_can_run_full_sample_bands() -> None:
    result = CliRunner().invoke(
        app,
        [
            "meansbands",
            "--input-type",
            "full",
            "--draws",
            "3",
            "--seed",
            "5",
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"draws": 3' in result.stdout
    assert '"mean_shape": [' in result.stdout


def test_meansbands_command_can_use_sampler_draw_archive(tmp_path) -> None:
    sampler_path = _write_sampler_archive(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "meansbands",
            "--input-type",
            "full",
            "--sampler-draws",
            str(sampler_path),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"sampler_draws":' in result.stdout
    assert '"mean_shape": [' in result.stdout


def test_meansbands_command_can_use_smoothed_histobs_source(tmp_path) -> None:
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "meansbands",
            "--source",
            "histobs",
            "--data",
            str(data_path),
            "--history-method",
            "smoothed",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"history_method": "smoothed"' in result.stdout
    assert '"mean_shape": [' in result.stdout


def test_meansbands_command_can_use_pseudo_source() -> None:
    result = CliRunner().invoke(
        app,
        ["meansbands", "--source", "forecastpseudo", "--horizon", "2", "--json"],
    )

    assert result.exit_code == 0
    assert '"source": "forecastpseudo"' in result.stdout
    assert '"mean_shape": [' in result.stdout


def test_meansbands_command_rejects_unported_mode() -> None:
    result = CliRunner().invoke(app, ["meansbands", "--input-type", "full"])

    assert result.exit_code == 2
    assert "input_type" in result.stdout


def test_solve_command_reports_model1002_system_shape() -> None:
    result = CliRunner().invoke(app, ["solve"])

    assert result.exit_code == 0
    assert "TTT" in result.stdout
    assert "(84, 84)" in result.stdout


def test_vv_compare_missing_dirs_reports_file_error(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(tmp_path / "missing-oracle"),
            "--candidate-dir",
            str(tmp_path / "missing-candidate"),
        ],
    )
    assert result.exit_code != 0
    assert "Fixture directory does not exist" in result.stdout


def test_vv_backend_parity_json_reports_passes_and_skips() -> None:
    result = CliRunner().invoke(
        app,
        [
            "vv",
            "backend-parity",
            "--kernel",
            "forecast",
            "--horizon",
            "1",
            "--tolerance-profile",
            "accelerator",
            "--include-pseudo",
            "--json",
        ],
    )

    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["tolerance_profile"]["name"] == "accelerator"
    assert any(
        item["kernel"] == "forecast" and item["backend"] == "numpy" for item in payload["results"]
    )
    assert all(item["backend"] != "platform" for item in payload["results"])
    assert any(item["passed"] for item in payload["results"])
    assert any(item["skipped"] for item in payload["results"])


def test_vv_runtime_purity_json_reports_clean_runtime() -> None:
    result = CliRunner().invoke(app, ["vv", "runtime-purity", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["checked_files"] > 0
    assert payload["findings"] == []


def test_vv_export_hard_target_inputs_writes_deterministic_bundle(tmp_path) -> None:
    output_dir = tmp_path / "hard_target_smoke"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-hard-target-inputs",
            "--output-dir",
            str(output_dir),
            "--periods",
            "2",
            "--horizon",
            "3",
            "--draws",
            "4",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dates"] == ["2018-Q2", "2018-Q3"]
    assert payload["history_periods"] == 2
    assert payload["horizon"] == 3
    assert payload["draws"] == 4
    observables = pd.read_csv(output_dir / "observables.csv")
    assert list(observables["date"]) == ["2018-Q2", "2018-Q3"]
    assert list(observables.columns) == ["date", *list(Model1002().observables)]
    assert float(observables.drop(columns=["date"]).to_numpy().max()) == 0.0
    with np.load(output_dir / "zero_shocks.npz") as archive:
        shock_samples = archive["shock_samples"]
    assert shock_samples.shape == (4, 3, len(Model1002().indexes.exogenous_shocks))
    assert float(shock_samples.max()) == 0.0
    manifest = json.loads(
        (output_dir / "hard_target_smoke_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == payload
    assert "--include-posterior" in payload["julia_oracle_command"]
    assert "--shock-samples" in payload["python_candidate_command"]
    assert "hard-target" in payload["compare_command"]


def test_vv_export_hard_target_inputs_uses_canonical_model_name_for_oracle_file(tmp_path) -> None:
    output_dir = tmp_path / "hard_target_smoke_alias"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-hard-target-inputs",
            "--model",
            "Model1002",
            "--output-dir",
            str(output_dir),
            "--periods",
            "2",
            "--horizon",
            "2",
            "--draws",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    oracle_args = payload["julia_oracle_command"]
    assert any(arg.endswith("m1002_ss10_hardtarget.h5") for arg in map(str, oracle_args))


def test_vv_raw_data_smoke_builds_observables_and_candidate_suite(tmp_path) -> None:
    raw_path = _write_raw_levels_csv(tmp_path)
    raw = pd.read_csv(raw_path)
    source_root = tmp_path / "sources"
    source_root.mkdir()
    raw.drop(columns=[f"ant{index}" for index in range(1, 7)]).to_csv(
        source_root / "fred_181115.csv",
        index=False,
    )
    raw.drop(columns=[f"ant{index}" for index in range(1, 7)]).to_csv(
        source_root / "dlx_181115.csv",
        index=False,
    )
    raw[["date", *[f"ant{index}" for index in range(1, 7)]]].to_csv(
        source_root / "ois_181115.csv",
        index=False,
    )
    population_forecast_path = tmp_path / "population_forecast.csv"
    pd.DataFrame(
        {
            "date": ["2017-Q1", "2017-Q2"],
            "CNP16OV__FRED": [260.0, 260.5],
        }
    ).to_csv(population_forecast_path, index=False)
    output_dir = tmp_path / "raw_data_smoke"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "raw-data-smoke",
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_dir),
            "--start-date",
            "2016-Q3",
            "--end-date",
            "2016-Q4",
            "--population-forecast",
            str(population_forecast_path),
            "--no-hpfilter-population",
            "--population-hpfilter-lambda",
            "10",
            "--horizon",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["rows"] == 2
    assert payload["columns"] == 20
    assert payload["population_forecast"] == str(population_forecast_path)
    assert payload["hpfilter_population"] is False
    assert payload["population_hpfilter_lambda"] == 10.0
    assert payload["comparison"]["status"] == "skipped"
    assert (output_dir / "observables.csv").exists()
    exported_kinds = {artifact["kind"] for artifact in payload["exported"]}
    assert {"forecast_mode", "meansbands_mode_histobs", "posterior"}.issubset(exported_kinds)


def _write_observable_csv(tmp_path: Path, *, periods: int) -> Path:
    model = Model1002()
    path = tmp_path / "observables.csv"
    pd.DataFrame({name: [0.0] * periods for name in model.observables}).to_csv(
        path,
        index=False,
    )
    return path


def _write_dated_observable_csv(tmp_path: Path, *, dates: list[str]) -> Path:
    model = Model1002()
    path = tmp_path / "dated_observables.csv"
    pd.DataFrame(
        {
            "date": dates,
            **{name: [0.0] * len(dates) for name in model.observables},
        }
    ).to_csv(path, index=False)
    return path


def _write_sampler_archive(tmp_path: Path) -> Path:
    model = Model1002()
    alpha = model.parameters["alpha"].value
    sampler = MetropolisHastingsResult(
        parameter_names=("alpha",),
        estimation_draws=np.zeros((2, 1), dtype=np.float64),
        parameter_draws=np.array([[alpha], [alpha + 1.0e-3]], dtype=np.float64),
        log_posterior=np.zeros(2, dtype=np.float64),
        accepted=np.array([True, True]),
        acceptance_rate=1.0,
        proposal_covariance=np.eye(1, dtype=np.float64),
        seed=11,
        burnin=0,
    )
    return save_sampler_result(sampler, tmp_path / "sampler.npz")


def _write_sampler_proposal_trace_hdf5(tmp_path: Path) -> Path:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sampler_proposal_trace.h5"
    proposal_log_posterior = np.array([-3.0, -3.2, -2.0], dtype=np.float64)
    previous_log_posterior = np.array([-3.4, -2.5, -2.2], dtype=np.float64)
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = np.array(
            [[0.1, 1.1, 2.1], [0.2, 1.2, 2.2]],
            dtype=np.float64,
        )
        handle["sampler/proposal_covariance"] = np.diag([0.5, 2.0])
        handle["sampler/accepted"] = np.array([1, 0, 1], dtype=np.int8)
        handle["sampler/log_posterior"] = np.array([-3.0, -2.5, -2.0])
        handle["sampler/proposal_parameters"] = np.array(
            [[0.1, 1.3, 2.1], [0.2, 1.4, 2.2]],
            dtype=np.float64,
        )
        handle["sampler/previous_parameters"] = np.array(
            [[0.0, 1.1, 2.0], [0.1, 1.2, 2.1]],
            dtype=np.float64,
        )
        handle["sampler/proposal_log_posterior"] = proposal_log_posterior
        handle["sampler/previous_log_posterior"] = previous_log_posterior
        handle["sampler/uniform_draw"] = np.array([0.2, 0.8, 0.9], dtype=np.float64)
        handle["sampler/log_acceptance"] = proposal_log_posterior - previous_log_posterior
        handle.attrs["sampler_parameter_names"] = "alpha,beta"
        handle.attrs["sampler_draws"] = 3
        handle.attrs["sampler_blocks"] = 1
        handle.attrs["sampler_param_blocks"] = 1
        handle.attrs["sampler_thin"] = 1
        handle.attrs["sampler_burnin"] = 0
        handle.attrs["sampler_trace_available"] = "true"
        handle.attrs["sampler_proposal_trace_available"] = "true"
    return path


def _write_sampler_posterior_replay_hdf5(
    tmp_path: Path,
    model: Model1002,
    observations: np.ndarray,
) -> Path:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sampler_posterior_replay.h5"
    parameter_names = tuple(model.parameters)
    current_values = np.asarray(
        [parameter.value for parameter in model.parameters.values()],
        dtype=np.float64,
    )
    proposal_draws = np.vstack([current_values, current_values])
    previous_draws = np.vstack([current_values, current_values])
    fixed_index = next(
        index for index, parameter in enumerate(model.parameters.values()) if parameter.fixed
    )
    proposal_draws[0, fixed_index] += 1.0e-3
    proposal_components = np.asarray(
        [
            evaluate_log_posterior_for_parameter_values(
                model,
                observations,
                parameter_names,
                draw,
                update_fixed_parameters=False,
            )[:3]
            for draw in proposal_draws
        ],
        dtype=np.float64,
    )
    previous_components = np.asarray(
        [
            evaluate_log_posterior_for_parameter_values(
                model,
                observations,
                parameter_names,
                draw,
                update_fixed_parameters=False,
            )[:3]
            for draw in previous_draws
        ],
        dtype=np.float64,
    )
    proposal_log_posterior = proposal_components[:, 0]
    proposal_log_likelihood = proposal_components[:, 1]
    proposal_log_prior = proposal_components[:, 2]
    previous_log_posterior = previous_components[:, 0]
    previous_log_likelihood = previous_components[:, 1]
    previous_log_prior = previous_components[:, 2]
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = proposal_draws.T
        handle["sampler/proposal_covariance"] = np.eye(len(parameter_names), dtype=np.float64)
        handle["sampler/accepted"] = np.array([1, 1], dtype=np.int8)
        handle["sampler/log_posterior"] = proposal_log_posterior
        handle["sampler/proposal_parameters"] = proposal_draws.T
        handle["sampler/previous_parameters"] = previous_draws.T
        handle["sampler/proposal_log_posterior"] = proposal_log_posterior
        handle["sampler/previous_log_posterior"] = previous_log_posterior
        handle["sampler/proposal_log_likelihood"] = proposal_log_likelihood
        handle["sampler/previous_log_likelihood"] = previous_log_likelihood
        handle["sampler/proposal_log_prior"] = proposal_log_prior
        handle["sampler/previous_log_prior"] = previous_log_prior
        handle["sampler/uniform_draw"] = np.array([0.2, 0.3], dtype=np.float64)
        handle["sampler/log_acceptance"] = proposal_log_posterior - previous_log_posterior
        handle.attrs["sampler_parameter_names"] = ",".join(parameter_names)
        handle.attrs["sampler_draws"] = 2
        handle.attrs["sampler_blocks"] = 1
        handle.attrs["sampler_param_blocks"] = 1
        handle.attrs["sampler_thin"] = 1
        handle.attrs["sampler_burnin"] = 0
        handle.attrs["sampler_trace_available"] = "true"
        handle.attrs["sampler_proposal_trace_available"] = "true"
    return path


def _write_shock_samples_archive(tmp_path: Path, *, horizon: int) -> Path:
    n_shocks = len(Model1002().indexes.exogenous_shocks)
    path = tmp_path / "shock_samples.npz"
    shock_samples = np.zeros((2, horizon, n_shocks), dtype=np.float64)
    shock_samples[1, 0, 0] = 0.25
    np.savez(path, shock_samples=shock_samples)
    return path


def _write_mode_archive(tmp_path: Path) -> Path:
    model = Model1002()
    mode = EstimationModeResult(
        parameter_names=("alpha",),
        estimation_values=parameter_estimation_vector(model, ("alpha",)),
        objective_value=1.0,
        success=True,
        message="fixture",
        iterations=1,
        function_evaluations=2,
        hessian=np.array([[1.0e8]], dtype=np.float64),
    )
    return save_estimation_mode(mode, tmp_path / "mode.npz")


def _write_raw_levels_csv(tmp_path: Path) -> Path:
    path = tmp_path / "raw_levels.csv"
    pd.DataFrame(
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
            "ant1": [0.1, 0.0],
            "ant2": [0.2, 0.1],
            "ant3": [0.3, 0.2],
            "ant4": [0.4, 0.3],
            "ant5": [0.5, 0.4],
            "ant6": [0.6, 0.5],
        }
    ).to_csv(path, index=False)
    return path
