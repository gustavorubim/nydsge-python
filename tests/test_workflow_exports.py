"""Regression checks for public CLI and durable uncertainty/date contracts."""

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nydsge.cli import _forecast_manifest, _load_cli_data, _meansbands_manifest, app
from nydsge.forecast import ForecastOutput, MeansBands
from nydsge.models import Model1002
from nydsge.vv import load_fixture_arrays, save_forecast_fixture, save_meansbands_fixture


@pytest.mark.parametrize("subspec", ["ss10", "ss104"])
@pytest.mark.parametrize("cond_type,offset", [("none", 0), ("full", 1), ("semi", 1)])
def test_export_future_band_dates_include_conditional_offset(subspec, cond_type, offset):
    model = Model1002(subspec)
    model.set_setting("date_forecast_start", "2026Q3")
    values = np.zeros((3 - offset, len(model.observables)))
    bands = MeansBands(values, values, values)
    manifest = _meansbands_manifest(
        model,
        bands,
        input_type="mode",
        cond_type=cond_type,
        horizon=3,
        source="observables",
        history_method="filtered",
        data_path=None,
        transformed=False,
    )
    dates = json.loads(json.dumps(manifest))["labels"]["meansbands/mean"]["axis0"]
    assert dates == (["2026-Q3", "2026-Q4", "2027-Q1"] if offset == 0 else ["2026-Q4", "2027-Q1"])


def test_uncertainty_survives_archives_and_manifests(tmp_path):
    model = Model1002("ss104")
    sources = ("parameter_draws", "future_model_innovations")
    provenance = "Two-point sensitivity; not posterior draws"
    values = np.zeros((2, len(model.observables)))
    forecast = ForecastOutput(
        np.zeros((2, 110)),
        values,
        uncertainty_sources=sources,
        parameter_draw_provenance=provenance,
    )
    bands = MeansBands(values, values, values, sources, provenance)
    paths = [save_forecast_fixture(forecast, tmp_path), save_meansbands_fixture(bands, tmp_path)]
    for path in paths:
        metadata = json.loads(path.with_suffix(path.suffix + ".metadata.json").read_text())
        assert metadata["uncertainty_sources"] == list(sources)
        assert metadata["parameter_draw_provenance"] == provenance
        assert metadata["archive_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # Numeric comparison/loading remains pickle-free and has no metadata keys.
    assert set(load_fixture_arrays(tmp_path)) == {
        "forecast/states",
        "forecast/observables",
        "meansbands/mean",
        "meansbands/lower",
        "meansbands/upper",
    }
    kwargs: dict[str, Any] = dict(
        input_type="full",
        cond_type="none",
        horizon=2,
        history_method="filtered",
        data_path=None,
        transformed=False,
    )
    fm = _forecast_manifest(model, forecast, **kwargs)
    bm = _meansbands_manifest(model, bands, source="observables", **kwargs)
    for manifest in (fm, bm):
        assert manifest["uncertainty_sources"] == list(sources)
        assert manifest["parameter_draw_provenance"] == provenance
    # Rewriting deterministic output must replace, not retain, stale metadata.
    save_meansbands_fixture(MeansBands(values, values, values), tmp_path)
    assert (
        json.loads(paths[1].with_suffix(".npz.metadata.json").read_text())[
            "parameter_draw_provenance"
        ]
        is None
    )


def test_ss104_cli_missing_expected_rates_can_estimate_and_export(tmp_path):
    model = Model1002("ss104")
    frame = pd.DataFrame(
        {
            "date": ["2026Q2"],
            **{
                name: [np.nan if name.startswith("obs_exp_nominalrate") else 0.0]
                for name in model.observables
            },
        }
    )
    path = tmp_path / "observables.csv"
    frame.to_csv(path, index=False)
    loaded = _load_cli_data(model, path)
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.filter(like="obs_exp_nominalrate").isna().all().all()
    for kind in ("export-forecast", "export-meansbands"):
        result = CliRunner().invoke(
            app,
            [
                "vv",
                kind,
                "--subspec",
                "ss104",
                "--data",
                str(path),
                "--forecast-start",
                "2026-Q3",
                "--horizon",
                "2",
                "--output-dir",
                str(tmp_path / kind),
                "--json",
            ],
        )
        assert result.exit_code == 0, result.output
        manifest = json.loads((tmp_path / kind / "manifest.json").read_text())
        assert manifest["subspec"] == "ss104"
    result = CliRunner().invoke(
        app, ["estimate", "--subspec", "ss104", "--data", str(path), "--json"]
    )
    assert result.exit_code == 0, result.output


def test_ss10_cli_keeps_default_empty_column_guard(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"date": ["2018Q3"], "obs_gdp": [np.nan]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="empty columns"):
        _load_cli_data(Model1002("ss10"), path)
