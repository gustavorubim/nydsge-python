"""Small synthetic correctness tests; these fixtures are not economic evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from general_analytics_tools.compute import VARIABLES, policy_surprises
from general_analytics_tools.decomposition import disturbance_attribution, export_decomposition
from general_analytics_tools.inputs import load_config, preserve_yield_history, sha256
from general_analytics_tools.plotting import render
from nydsge.kalman import kalman_log_likelihood, smooth_kalman_result
from nydsge.regimes import RegimeSwitchingSystem
from nydsge.solve import Measurement, System, Transition


def scalar_system(a, q, *, r=None, z=None):
    r = np.ones((1, 1)) if r is None else r
    z = np.ones((1, 1)) if z is None else z
    return System(
        Transition(np.array([[a]]), r, np.zeros(1)),
        Measurement(z, np.zeros(1), np.atleast_2d(q), np.array([[0.3]])),
    )


def test_disturbance_attribution_matches_direct_gaussian_conditioning():
    a = np.array([0.8, 0.5, 0.9])
    q = np.array([0.2, 0.4, 0.1])
    y = np.array([0.1, -0.2, 0.3])
    system = RegimeSwitchingSystem(tuple(scalar_system(x, v) for x, v in zip(a, q, strict=True)))
    filtered = kalman_log_likelihood(system, y[:, None], regime_indices=[1, 2, 3])
    smooth = smooth_kalman_result(system, filtered, data=y[:, None])
    assert smooth.smoothed_states is not None
    result = disturbance_attribution(
        system,
        filtered,
        smooth.smoothed_states,
        y[:, None],
        np.array([1, 2, 3]),
        shock_covariances=q[:, None, None],
        process_covariances=None,
    )
    # Independent joint-Gaussian conditional expectation of initial state and innovations.
    mapping = np.zeros((3, 4))
    mapping[0, :2] = [a[0], 1]
    for t in range(1, 3):
        mapping[t] = a[t] * mapping[t - 1]
        mapping[t, t + 1] = 1
    prior = np.diag([q[0] / (1 - a[0] ** 2), *q])
    expected = prior @ mapping.T @ np.linalg.solve(mapping @ prior @ mapping.T + 0.3 * np.eye(3), y)
    np.testing.assert_allclose(result["shocks"][:, 0], expected[1:], atol=1e-14, rtol=0)
    np.testing.assert_allclose(result["fitted"][:, 0], mapping @ expected, atol=1e-14, rtol=0)
    assert np.max(np.abs(result["remainder"])) < 1e-14


def test_policy_normalizes_signed_loading_and_uses_only_impact_innovation():
    active = scalar_system(0.5, np.diag([0.0, 0.04]), r=np.array([[0.0, -1.0]]))
    system = RegimeSwitchingSystem((active,))
    baseline, scenarios, records = policy_surprises(
        system,
        np.zeros(1),
        np.ones(3, dtype=int),
        ["obs_nominalrate"],
        ["rm_sh", "rm_ait_sh"],
        [25, 50, -25],
    )
    np.testing.assert_allclose(
        4 * (scenarios["+25 bp surprise"][:, 0] - baseline[:, 0]), [0.25, 0.125, 0.0625], atol=1e-14
    )
    np.testing.assert_allclose(
        scenarios["+50 bp surprise"] - baseline, 2 * (scenarios["+25 bp surprise"] - baseline)
    )
    np.testing.assert_allclose(
        scenarios["-25 bp surprise"] - baseline, -(scenarios["+25 bp surprise"] - baseline)
    )
    assert all(record["shock"] == "rm_ait_sh" for record in records)
    assert records[0]["model_units"] < 0  # Negative primitive loading, positive actual FFR impact.
    with pytest.raises(ValueError, match="No active supported"):
        policy_surprises(
            system,
            np.zeros(1),
            np.ones(3, dtype=int),
            ["obs_nominalrate"],
            ["rm_sh", "rm_ait_sh"],
            [25],
            "rm_sh",
        )


def test_rolling_history_is_preserved_without_inventing_or_overwriting_revisions():
    old = pd.DataFrame({"date": ["2023-10-02", "2023-10-03"], "yield": [5.0, 6.0]})
    new = pd.DataFrame({"date": ["2023-10-03", "2023-10-04"], "yield": [6.0, 7.0]})
    combined, restored = preserve_yield_history(old, new)
    assert list(combined.iloc[:, 1]) == [5.0, 6.0, 7.0]
    assert list(restored.date) == ["2023-10-02"]
    new.iloc[0, 1] = 6.1
    with pytest.raises(ValueError, match="Conflicting yield overlap"):
        preserve_yield_history(old, new)


def test_hd_gate_retains_remainder_and_diagnostic_status():
    arrays = {
        "contributions": np.zeros((1, 2, 1)),
        "initial": np.zeros((1, 2)),
        "deterministic": np.ones((1, 2)),
        "remainder": np.array([[1e-8, 2e-9]]),
    }
    table, gates = export_decomposition(
        arrays, pd.period_range("2000Q1", periods=1, freq="Q"), ["obs_gdp", "obs_corepce"], ["g_sh"]
    )
    assert gates["obs_gdp"]["status"] == "diagnostic"
    assert gates["obs_corepce"]["status"] == "reconstruction_pass"
    assert {"Initial state", "Deterministic terms", "Numerical remainder"} <= set(table.component)
    assert (
        table.loc[
            (table.variable == "obs_gdp") & (table.component == "Numerical remainder"),
            "contribution",
        ].iloc[0]
        == 4e-8
    )


@pytest.mark.parametrize(
    "change", ["M7", "undeclared", "horizon", "dates", "duplicate", "mode_path"]
)
def test_config_rejects_unsupported_or_incomplete_requests(tmp_path, change):
    (tmp_path / "levels.csv").write_text("date,GDP\n2000Q1,1\n")
    (tmp_path / "source.json").write_text("{}")
    config = {
        "model": "ss104",
        "parameters": "public_defaults",
        "sample": {
            "presample_start": "2000Q1",
            "mainsample_start": "2000Q1",
            "history_end": "2000Q2",
            "conditioning_end": "2000Q3",
        },
        "inputs": {"levels_csv": "levels.csv", "source_manifest": "source.json"},
        "conditioning_observables": ["obs_hours"],
        "complete_conditioning_observables": ["obs_hours"],
        "forecast_quarters": 8,
    }
    if change == "M7":
        config["parameters"] = "M7"
    elif change == "undeclared":
        config["complete_conditioning_observables"] = []
    elif change == "horizon":
        config["forecast_quarters"] = 41
    elif change == "dates":
        config["sample"]["conditioning_end"] = "2002Q1"
    elif change == "duplicate":
        config["policy_impacts_bp"] = [25, 25]
    else:
        config["mode_path"] = "unconverged.json"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def synthetic_archive(path: Path):
    """Small explicitly synthetic archive tests rendering without solving."""
    path.mkdir()
    dates = ["2020Q4", "2021Q1"]
    names = [row[0] for row in VARIABLES]
    history = [
        {
            "quarter": d,
            "variable": n,
            "actual": 1.0 if t == 0 else np.nan,
            "fitted": 1.0,
            "units": u,
        }
        for t, d in enumerate(dates)
        for n, _, _, u in VARIABLES
    ]
    forecasts = [
        {"quarter": d, "variable": n, "scenario": s, "value": 1.0, "difference": 0.0, "units": u}
        for d in ["2021Q2", "2021Q3"]
        for n, _, _, u in VARIABLES
        for s in ["Baseline", "+25 bp surprise"]
    ]
    pd.DataFrame(history).to_csv(path / "history.csv", index=False)
    pd.DataFrame(forecasts).to_csv(path / "forecasts.csv", index=False)
    arrays = {
        "contributions": np.zeros((2, 6, 1)),
        "initial": np.zeros((2, 6)),
        "deterministic": np.ones((2, 6)) / 4,
        "remainder": np.zeros((2, 6)),
        "measurement_gap": np.zeros((2, 6)),
    }
    table, gates = export_decomposition(arrays, pd.PeriodIndex(dates, freq="Q"), names, ["g_sh"])
    gates["obs_gdp"]["status"] = "diagnostic"
    table.to_csv(path / "decomposition.csv", index=False)
    np.savez(
        path / "decomposition_arrays.npz",
        contributions=arrays["contributions"],
        initial=arrays["initial"],
        deterministic=arrays["deterministic"],
        remainder=arrays["remainder"],
        measurement_gap=arrays["measurement_gap"],
    )
    (path / "status.json").write_text(json.dumps({"status": "completed"}))
    meta = {
        "history_plot_start": dates[0],
        "observable_names": names,
        "HD": gates,
        "sample": {"history_end": "2020Q4", "conditioning_end": "2021Q1"},
        "forecast_start": "2021Q2",
        "forecast_end": "2021Q3",
        "source": {"vintage": "SYNTHETIC TEST ONLY", "retrieved_at_utc": "test fixture"},
        "policy_scenarios": [{"label": "+25 bp surprise"}],
        "output_sha256": {p.name: sha256(p) for p in path.iterdir() if p.name != "status.json"},
    }
    (path / "manifest.json").write_text(json.dumps(meta))


def test_renderer_is_compute_free_and_preserves_diagnostic_label(tmp_path, monkeypatch):
    import matplotlib.axes

    import general_analytics_tools.compute as compute

    archive = tmp_path / "archive"
    synthetic_archive(archive)

    def forbidden(*args, **kwargs):
        raise AssertionError("Rendering must not compute a system")

    monkeypatch.setattr(compute, "compute_system", forbidden)
    titles = []
    original = matplotlib.axes.Axes.set_title

    def title(self, label, *args, **kwargs):
        titles.append(label)
        return original(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", title)
    plots = render(archive)
    assert len(plots) == 4 and all(p.stat().st_size > 1000 for p in plots)
    assert any("DIAGNOSTIC" in label for label in titles)
    assert (archive / "figures/figures.json").is_file()


def test_renderer_rejects_failed_or_tampered_archive(tmp_path):
    archive = tmp_path / "archive"
    synthetic_archive(archive)
    (archive / "history.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        render(archive)
    (archive / "status.json").write_text(json.dumps({"status": "timed_out"}))
    with pytest.raises(ValueError, match="completed"):
        render(archive)


def test_cli_timeout_marks_only_own_archive_incomplete(tmp_path, monkeypatch):
    import subprocess
    import sys

    from general_analytics_tools.__main__ import main

    output = tmp_path / "run"

    def timed_out(command, **kwargs):
        assert command[:4] == [sys.executable, "-m", "general_analytics_tools", "_compute"]
        output.mkdir()
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timed_out)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analytics",
            "run",
            "--config",
            "unused.json",
            "--output-dir",
            str(output),
            "--timeout-seconds",
            "1",
        ],
    )
    with pytest.raises(SystemExit, match="Own compute subprocess timed out"):
        main()
    assert json.loads((output / "status.json").read_text())["status"] == "timed_out"


def test_cli_preserves_existing_output_directory(tmp_path, monkeypatch):
    import sys

    from general_analytics_tools.__main__ import main

    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "preserve.txt"
    sentinel.write_text("existing evidence")
    monkeypatch.setattr(
        sys, "argv", ["analytics", "run", "--config", "unused.json", "--output-dir", str(output)]
    )
    with pytest.raises(SystemExit):
        main()
    assert sentinel.read_text() == "existing evidence"
