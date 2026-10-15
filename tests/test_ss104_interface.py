"""Native ss104 interface tests; synthetic inputs are not forecast evidence."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nydsge.core import NotPortedError
from nydsge.estimate import EstimationModeResult, estimate
from nydsge.models import Model1002
from nydsge.ss104_estimation import evaluate_ss104_posterior, ss104_parameter_catalog


def test_native_estimate_fixed_point_presample_and_caller_unchanged():
    model = Model1002(
        "ss104", settings={"date_presample_start": "2019Q4", "date_mainsample_start": "2020Q1"}
    )
    frame = pd.DataFrame(
        np.nan, index=pd.period_range("2019Q4", "2026Q2", freq="Q"), columns=list(model.observables)
    )
    frame["obs_corepce"] = 0.6
    frame["obs_nominalrate"] = 0.9
    before = {k: p.value for k, p in ss104_parameter_catalog(model).items()}
    expected = evaluate_ss104_posterior(model, frame)
    result = estimate(model, frame)
    assert result.log_posterior == pytest.approx(expected[0], abs=1e-10)
    assert result.log_likelihood == pytest.approx(expected[1], abs=1e-10)
    assert result.log_prior == pytest.approx(expected[2], abs=1e-10)
    assert result.optimization is None and result.sampler is None
    assert before == {k: p.value for k, p in ss104_parameter_catalog(model).items()}
    with pytest.raises(ValueError, match="quarterly dates"):
        estimate(model, frame, start_date="2019Q4")
    with pytest.raises(NotPortedError, match="MH sampling"):
        estimate(model, frame, mh_draws=2)
    with pytest.raises(ValueError, match="before forecast start"):
        estimate(model, frame.reindex(pd.period_range("2019Q4", "2026Q3", freq="Q")))


def test_native_mode_quadratic_objective_and_hessian_without_real_estimation(monkeypatch):
    import nydsge.ss104_interface as interface

    model = Model1002("ss104")
    baseline = model.parameters["alpha"].value

    def quadratic(model, data):
        # Synthetic known density in the model coordinate; all other parameters fixed.
        log_likelihood = -20 * (model.parameters["alpha"].value - 0.25) ** 2
        return log_likelihood - 2, log_likelihood, -2.0, None

    monkeypatch.setattr(interface, "evaluate_ss104_posterior", quadratic)
    result = estimate(
        model,
        None,
        optimize=True,
        parameter_names=["alpha"],
        optimizer_method="BFGS",
        maxiter=40,
        compute_hessian=True,
    )
    assert result.optimization is not None and result.optimization.success
    assert result.parameter_values["alpha"] == pytest.approx(0.25, abs=1e-5)
    assert result.hessian is not None and result.hessian.shape == (1, 1)
    assert result.hessian[0, 0] > 0
    assert model.parameters["alpha"].value == baseline
    mode = EstimationModeResult(**result.optimization.__dict__, hessian=result.hessian)
    replay = estimate(model, None, parameter_names=["alpha"], mode=mode)
    assert replay.log_posterior == pytest.approx(result.log_posterior)
    with pytest.raises(ValueError, match="Unconverged"):
        estimate(model, None, parameter_names=["alpha"], mode=replace(mode, success=False))
    with pytest.raises(ValueError, match="coordinates"):
        estimate(model, None, parameter_names=["h"], mode=mode)
    assert model.parameters["alpha"].value == baseline


def test_failed_native_search_status_is_not_converted_to_convergence(monkeypatch):
    from scipy.optimize import OptimizeResult

    import nydsge.ss104_interface as interface

    model = Model1002("ss104")
    original = model.parameters["alpha"].value
    monkeypatch.setattr(
        interface, "evaluate_ss104_posterior", lambda model, data: (-3.0, -1.0, -2.0, None)
    )
    monkeypatch.setattr(
        interface,
        "minimize",
        lambda objective, start, **kwargs: OptimizeResult(
            x=start, fun=3.0, success=False, message="synthetic iteration limit", nit=1, nfev=1
        ),
    )
    result = estimate(model, None, optimize=True, parameter_names=["alpha"], maxiter=1)
    assert result.optimization is not None and not result.optimization.success
    assert "iteration limit" in result.optimization.message
    with pytest.raises(ValueError, match="unique free"):
        estimate(model, None, optimize=True, parameter_names=["alpha", "alpha"])
    assert model.parameters["alpha"].value == original
