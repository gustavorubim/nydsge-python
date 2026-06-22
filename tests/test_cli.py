from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

import nydsge.cli as cli
from nydsge.cli import app
from nydsge.estimate import (
    EstimationModeResult,
    MetropolisHastingsResult,
    parameter_estimation_vector,
    save_estimation_mode,
    save_sampler_result,
)
from nydsge.models import Model1002
from nydsge.runtime import RuntimeStatus


def test_doctor_json() -> None:
    result = CliRunner().invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert "numpy" in result.stdout
    assert "cpu" in result.stdout


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
