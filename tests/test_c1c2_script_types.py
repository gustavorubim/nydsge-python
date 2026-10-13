from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nydsge.forecast import ForecastOutput
from nydsge.solve import System, build_system

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1c2_compare as comparison  # noqa: E402
import c1c2_nyfed_compare as nyfed  # noqa: E402
import estimate_c1c2_mode as mode  # noqa: E402
import mcmc_c1c2 as mcmc  # noqa: E402
import mcmc_c1c2_forecast as forecast  # noqa: E402
from c1c2_common import build_model  # noqa: E402


def test_q4q4_preserves_compounding() -> None:
    values = [1.25, -0.5, 3.75, 2.5]
    quarters = {f"2026-Q{i}": v for i, v in enumerate(values, 1)}
    expected = float(100 * (np.prod([(1 + v / 100) ** 0.25 for v in values]) - 1))
    assert nyfed.q4q4(quarters, 2026) == expected


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -np.inf])
def test_q4q4_missing_or_nonfinite_quarters(invalid: float | None) -> None:
    quarters = {f"2026-Q{i}": 2.0 for i in range(1, 5)}
    assert np.isnan(nyfed.q4q4({}, 2026))
    assert np.isnan(nyfed.q4q4({**quarters, "2026-Q2": invalid}, 2026))


def test_worker_requires_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mode, "_STATE", None)
    with pytest.raises(RuntimeError, match="not been initialized"):
        mode._neg_log_post(np.zeros(1))


@pytest.mark.parametrize("fails", [False, True])
def test_worker_preserves_parameters(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    model = build_model(0)
    original = dict(model.parameters)
    name = next(iter(original))
    data = np.zeros((1, 1))
    monkeypatch.setattr(mode, "_setup", lambda *_: (model, data, (name,), ["2026-Q1"]))
    monkeypatch.setattr(mode, "_STATE", None)
    mode._init_worker(0, False)

    def set_parameter(*_: Any) -> None:
        model.parameters[name] = replace(original[name], value=original[name].value + 1)

    def log_posterior(*_: Any, **__: Any) -> tuple[float, float, float, None]:
        if fails:
            raise ValueError("synthetic posterior failure")
        return 12.0, 10.0, 2.0, None

    monkeypatch.setattr(mode, "_set_parameter_estimation_vector", set_parameter)
    monkeypatch.setattr(mode, "_evaluate_log_posterior", log_posterior)
    assert mode._neg_log_post(np.zeros(1)) == (float("inf") if fails else -12.0)
    assert model.parameters == original


def test_mcmc_target_preserves_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = build_model(0)
    original = dict(model.parameters)
    name = next(iter(original))

    def log_posterior(*_: Any, **kwargs: Any) -> float:
        assert kwargs["jacobian"] is True
        model.parameters[name] = replace(original[name], value=original[name].value + 1)
        return 7.0

    monkeypatch.setattr(mcmc, "_log_posterior_for_estimation_values", log_posterior)
    assert mcmc.log_target(model, np.zeros((1, 1)), (name,), np.zeros(1)) == 7.0
    assert model.parameters == original


def _system() -> System:
    return build_system(
        TTT=np.eye(1),
        RRR=np.eye(1),
        CCC=np.zeros(1),
        ZZ=np.ones((3, 1)),
        DD=np.zeros(3),
        QQ=np.eye(1),
        EE=np.zeros((3, 3)),
    )


def test_metrics_require_pseudo_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comparison, "compute_system", lambda _: _system())
    with pytest.raises(ValueError, match="pseudo-measurement"):
        comparison.metrics(build_model(0), pd.DataFrame())


def test_metrics_reject_regime_switching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comparison, "compute_system", lambda _: object())
    with pytest.raises(ValueError, match="single-regime"):
        comparison.metrics(build_model(0), pd.DataFrame())


def test_forecast_requires_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    model = build_model(0)
    monkeypatch.setattr(
        forecast, "_S", {"model": model, "obs": pd.DataFrame(), "names": list(forecast.VARS)}
    )
    monkeypatch.setattr(forecast, "_set", lambda _: None)
    monkeypatch.setattr(forecast, "compute_system", lambda _: _system())
    output = ForecastOutput(
        states=np.zeros((9, 1)), observables=np.zeros((9, 3)), history_states=np.zeros((1, 1))
    )
    monkeypatch.setattr(forecast, "forecast_one", lambda *_, **__: output)
    monkeypatch.setattr(forecast, "forecast_linear_system_samples", lambda *_, **__: output)
    monkeypatch.setattr(forecast, "reverse_transform_observables", lambda _, values: values)
    with pytest.raises(RuntimeError, match="observable samples"):
        forecast.one_draw({"values": {}, "n_shock": 2, "seed": 1})


def test_forecast_summary_preserves_scalar_and_vector_bands() -> None:
    results = []
    for shift in (0.0, 1.0):
        results.append(
            {
                "mean": {"corepce": [1.0 + shift, 2.0 + shift]},
                "sims": {"corepce": [[1.0 + shift, 2.0 + shift], [3.0 + shift, 4.0 + shift]]},
                "decomp": {
                    "corepce": {
                        "pi_star_sh": 1.0,
                        "lambda_f_sh": 2.0,
                        "lambda_w_sh": 3.0,
                        "rm_sh": 4.0,
                        "rm_shl1-6": 0.0,
                    },
                    "outputgap": {"pi_star_sh": 2.0 + shift},
                    "corepce_initial": 1.0,
                    "smoothed_corepce_2026Q2": 11.0,
                    "long_run_inflation_2026Q2": 2.0,
                    "output_gap_2026Q2": 3.0,
                },
            }
        )
    result = forecast.summarize(results)
    assert result["corepce"]["parameter_only (zero future shocks)"]["median"] == [1.5, 2.5]
    decomp = result["decomposition_2026Q2"]
    assert decomp["n_draws"] == 2
    assert decomp["outputgap:pi_star_sh"]["median"] == 2.5
    assert decomp["share_of_shock_driven_deviation"]["monetary"]["mean"] == 0.4
    assert json.loads(json.dumps(result)) == result
