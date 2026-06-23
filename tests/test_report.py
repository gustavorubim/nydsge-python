from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nydsge.cli import app
from nydsge.data import quarter_labels_from_start
from nydsge.forecast import (
    forecast_linear_system,
    historical_decomposition,
    observable_irf,
)
from nydsge.models import Model1002
from nydsge.report import (
    ReportError,
    _check_axis,
    generate_forecast_report,
    generate_historical_decomposition_report,
    generate_irf_report,
)
from nydsge.solve import compute_system

runner = CliRunner()


def _model() -> Model1002:
    return Model1002(settings={"date_forecast_start": "2018-Q4"})


def _simulated_frame(model: Model1002, periods: int, *, seed: int = 0) -> pd.DataFrame:
    """Simulate the solved system to obtain non-trivial in-sample observables."""
    system = compute_system(model)
    rng = np.random.default_rng(seed)
    n_shocks = system.transition.RRR.shape[1]
    n_states = system.transition.TTT.shape[0]
    shocks = rng.standard_normal((periods, n_shocks)) * 0.05
    simulated = forecast_linear_system(
        system,
        np.zeros(n_states),
        horizon=periods,
        shocks=shocks,
    )
    # End the in-sample window at 2018-Q3 so every row precedes date_forecast_start.
    dates = quarter_labels_from_start("2008-Q4", periods)
    frame = {"date": dates}
    for index, name in enumerate(model.observables):
        frame[name] = simulated.observables[:, index]
    return pd.DataFrame(frame)


def test_observable_irf_matches_forecast_linear_system() -> None:
    model = _model()
    system = compute_system(model)
    irf = observable_irf(system, horizon=10, normalization="unit")
    n_shocks = system.transition.RRR.shape[1]
    assert irf.observables.shape == (10, len(model.observables), n_shocks)
    for shock_index in (0, 5, n_shocks - 1):
        shocks = np.zeros((10, n_shocks))
        shocks[0, shock_index] = 1.0
        baseline = forecast_linear_system(
            system,
            np.zeros(system.transition.TTT.shape[0]),
            horizon=10,
            shocks=shocks,
        )
        np.testing.assert_allclose(irf.states[:, :, shock_index], baseline.states, atol=1e-12)


def test_observable_irf_one_sd_normalization_scales_by_sqrt_qq() -> None:
    model = _model()
    system = compute_system(model)
    irf = observable_irf(system, horizon=4, normalization="one_sd")
    expected = np.sqrt(np.clip(np.diag(system.measurement.QQ), 0.0, None))
    np.testing.assert_allclose(irf.shock_scales, expected)
    np.testing.assert_allclose(
        irf.states[0], system.transition.RRR * expected[np.newaxis, :], atol=1e-12
    )


def test_observable_irf_rejects_unknown_normalization() -> None:
    system = compute_system(_model())
    with pytest.raises(ValueError, match="normalization"):
        observable_irf(system, horizon=4, normalization="bogus")


def test_historical_decomposition_reconciles_to_smoothed_path() -> None:
    model = _model()
    data = _simulated_frame(model, 40, seed=7)
    decomposition = historical_decomposition(model, data=data)
    assert decomposition.observable_contributions.shape[0] == 40
    assert decomposition.observable_contributions.shape[1] == len(model.observables)
    # Contributions plus baseline reconstruct the smoothed observable path.
    reconstructed = (
        decomposition.observable_contributions.sum(axis=2) + decomposition.observable_baseline
    )
    np.testing.assert_allclose(reconstructed, decomposition.smoothed_observables, atol=1e-5)
    assert decomposition.reconciliation_max_abs_error < 1e-5


def test_check_axis_raises_on_inconsistent_dimensions() -> None:
    with pytest.raises(ReportError, match="inconsistent"):
        _check_axis("observables", ["a", "b"], 3)


def test_generate_irf_report_numeric_arrays_and_labels(tmp_path) -> None:
    model = _model()
    artifacts = generate_irf_report(
        model,
        output_dir=tmp_path / "irf",
        horizon=8,
        normalization="one_sd",
        make_plots=False,
    )
    assert artifacts.arrays
    array_path = artifacts.arrays[0]
    assert array_path.exists()
    loaded = np.load(array_path)
    labels = artifacts.manifest["labels"]["impulse_responses/observables"]
    assert len(labels["axis0"]) == loaded["observables"].shape[0]
    assert len(labels["axis1"]) == loaded["observables"].shape[1]
    assert len(labels["axis2"]) == loaded["observables"].shape[2]
    assert artifacts.summary.exists()
    assert artifacts.figures == []


def test_generate_historical_decomposition_report_numeric(tmp_path) -> None:
    model = _model()
    data = _simulated_frame(model, 36, seed=3)
    artifacts = generate_historical_decomposition_report(
        model,
        output_dir=tmp_path / "hd",
        data=data,
        make_plots=False,
    )
    array_path = artifacts.arrays[0]
    loaded = np.load(array_path)
    assert loaded["observable_contributions"].shape[0] == 36
    assert artifacts.manifest["method"] == "rts_smoothed_shock_contributions"
    assert artifacts.manifest["reconciliation_max_abs_error"] < 1e-5
    labels = artifacts.manifest["labels"]["historical_decomposition/observable_contributions"]
    assert len(labels["axis1"]) == loaded["observable_contributions"].shape[1]


def test_generate_forecast_report_with_figures(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    model = _model()
    data = _simulated_frame(model, 32, seed=11)
    artifacts = generate_forecast_report(
        model,
        output_dir=tmp_path / "report",
        horizon=8,
        data=data,
        history_method="smoothed",
        make_plots=True,
    )
    # One macro panel + per-observable IRF (+combined) + per-observable decomposition.
    assert len(artifacts.figures) == 1 + (len(model.observables) + 1) + len(model.observables)
    for figure in artifacts.figures:
        assert figure.exists()
        assert figure.stat().st_size > 0
    macro = tmp_path / "report" / "figures" / "macro_forecasts_all_observables.png"
    assert macro.exists()


def test_report_irf_cli_numeric(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "report",
            "irf",
            "--output-dir",
            str(tmp_path / "cli_irf"),
            "--horizon",
            "6",
            "--no-plots",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cli_irf" / "impulse_responses.npz").exists()


def test_report_historical_decomposition_cli_numeric(tmp_path) -> None:
    model = _model()
    data = _simulated_frame(model, 30, seed=5)
    data_path = tmp_path / "observables.csv"
    data.to_csv(data_path, index=False)
    result = runner.invoke(
        app,
        [
            "report",
            "historical-decomposition",
            "--output-dir",
            str(tmp_path / "cli_hd"),
            "--data",
            str(data_path),
            "--no-plots",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "cli_hd" / "historical_decomposition.npz").exists()
