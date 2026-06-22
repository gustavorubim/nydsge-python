from __future__ import annotations

import numpy as np
import pytest

from nydsge.estimate import (
    EstimationModeResult,
    MetropolisHastingsResult,
    estimate,
    estimation_mode_from_result,
    estimation_parameter_names,
    finite_difference_hessian,
    load_estimation_mode,
    load_sampler_result,
    parameter_estimation_vector,
    proposal_covariance_from_hessian,
    sampler_diagnostics,
    save_estimation_mode,
    save_sampler_result,
    validate_estimation_mode,
    validate_sampler_result,
)
from nydsge.models import Model1002
from nydsge.parameters import update_parameter_value


def test_estimate_model1002_evaluates_current_parameter_posterior() -> None:
    model = Model1002()
    data = np.zeros((2, len(model.observables)))

    result = estimate(model, data)

    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.log_prior)
    assert result.log_prior != 0.0
    assert np.isclose(result.log_posterior, result.log_likelihood + result.log_prior)
    assert len(result.parameter_values) == len(model.parameters)
    assert result.kalman.filtered_states.shape == (2, 84)


def test_estimate_can_run_metropolis_hastings_sampler() -> None:
    model = Model1002()
    data = np.zeros((1, len(model.observables)))

    result = estimate(
        model,
        data,
        parameter_names=["alpha"],
        mh_draws=4,
        mh_burnin=1,
        proposal_covariance=np.eye(1) * 1.0e-8,
        seed=123,
    )

    assert result.sampler is not None
    assert result.sampler.parameter_names == ("alpha",)
    assert result.sampler.estimation_draws.shape == (4, 1)
    assert result.sampler.parameter_draws.shape == (4, 1)
    assert result.sampler.log_posterior.shape == (4,)
    assert result.sampler.accepted.shape == (4,)
    assert 0.0 <= result.sampler.acceptance_rate <= 1.0
    assert np.isclose(result.log_posterior, result.sampler.log_posterior[-1])
    assert np.isfinite(result.parameter_values["alpha"])


def test_sampler_result_can_round_trip_npz_archive(tmp_path) -> None:
    model = Model1002()
    data = np.zeros((1, len(model.observables)))
    result = estimate(
        model,
        data,
        parameter_names=["alpha"],
        mh_draws=3,
        proposal_covariance=np.eye(1) * 1.0e-8,
        seed=99,
    )
    assert result.sampler is not None

    path = save_sampler_result(result.sampler, tmp_path / "sampler.npz")
    loaded = load_sampler_result(path)

    assert loaded.parameter_names == result.sampler.parameter_names
    np.testing.assert_allclose(loaded.estimation_draws, result.sampler.estimation_draws)
    np.testing.assert_allclose(loaded.parameter_draws, result.sampler.parameter_draws)
    np.testing.assert_allclose(loaded.log_posterior, result.sampler.log_posterior)
    np.testing.assert_array_equal(loaded.accepted, result.sampler.accepted)
    np.testing.assert_allclose(loaded.proposal_covariance, result.sampler.proposal_covariance)
    assert loaded.acceptance_rate == result.sampler.acceptance_rate
    assert loaded.seed == 99
    assert loaded.burnin == 0


def test_sampler_diagnostics_reports_chain_health_metrics() -> None:
    sampler = MetropolisHastingsResult(
        parameter_names=("alpha", "rho"),
        estimation_draws=np.array(
            [
                [0.1, 0.2],
                [0.2, 0.1],
                [0.3, 0.4],
                [0.4, 0.3],
            ],
            dtype=np.float64,
        ),
        parameter_draws=np.array(
            [
                [1.1, 2.2],
                [1.2, 2.1],
                [1.3, 2.4],
                [1.4, 2.3],
            ],
            dtype=np.float64,
        ),
        log_posterior=np.array([-4.0, -3.5, -3.0, -2.5], dtype=np.float64),
        accepted=np.array([True, False, True, True]),
        acceptance_rate=0.75,
        proposal_covariance=np.array([[4.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        seed=11,
        burnin=2,
    )

    diagnostics = sampler_diagnostics(sampler, windows=2)

    assert diagnostics.parameter_names == ("alpha", "rho")
    assert diagnostics.draws == 4
    assert diagnostics.burnin == 2
    assert diagnostics.seed == 11
    assert diagnostics.accepted_draws == 3
    assert diagnostics.acceptance_rate == 0.75
    assert diagnostics.realized_acceptance_rate == 0.75
    assert diagnostics.acceptance_windows == (0.5, 1.0)
    assert diagnostics.proposal_covariance_shape == (2, 2)
    assert diagnostics.proposal_covariance_min_eigenvalue == 1.0
    assert diagnostics.proposal_covariance_max_eigenvalue == 4.0
    assert diagnostics.proposal_covariance_condition_number == 4.0
    assert diagnostics.proposal_covariance_positive_semidefinite is True
    assert diagnostics.log_posterior_mean == -3.25
    assert diagnostics.parameters[0].name == "alpha"
    assert diagnostics.parameters[0].effective_sample_size <= 4.0
    assert diagnostics.parameters[0].monte_carlo_standard_error > 0.0
    assert diagnostics.parameters[0].split_rhat is not None
    assert diagnostics.parameters[0].split_rhat > 1.0
    payload = diagnostics.to_dict()
    assert payload["parameter_names"] == ["alpha", "rho"]
    assert payload["parameters"][0]["name"] == "alpha"
    assert payload["parameters"][0]["monte_carlo_standard_error"] > 0.0
    assert payload["parameters"][0]["split_rhat"] == diagnostics.parameters[0].split_rhat


def test_sampler_diagnostics_validates_windows() -> None:
    sampler = MetropolisHastingsResult(
        parameter_names=("alpha",),
        estimation_draws=np.ones((1, 1), dtype=np.float64),
        parameter_draws=np.ones((1, 1), dtype=np.float64),
        log_posterior=np.zeros(1, dtype=np.float64),
        accepted=np.ones(1, dtype=bool),
        acceptance_rate=1.0,
        proposal_covariance=np.ones((1, 1), dtype=np.float64),
        seed=None,
        burnin=0,
    )

    with pytest.raises(ValueError, match="windows"):
        sampler_diagnostics(sampler, windows=0)


def test_estimation_mode_can_round_trip_npz_archive(tmp_path) -> None:
    model = Model1002()
    mode = EstimationModeResult(
        parameter_names=("alpha",),
        estimation_values=parameter_estimation_vector(model, ("alpha",)),
        objective_value=12.5,
        success=True,
        message="ok",
        iterations=3,
        function_evaluations=9,
        hessian=np.array([[4.0]], dtype=np.float64),
    )

    path = save_estimation_mode(mode, tmp_path / "mode.npz")
    loaded = load_estimation_mode(path)

    assert loaded.parameter_names == mode.parameter_names
    np.testing.assert_allclose(loaded.estimation_values, mode.estimation_values)
    assert loaded.objective_value == mode.objective_value
    assert loaded.success is True
    assert loaded.message == "ok"
    assert loaded.iterations == 3
    assert loaded.function_evaluations == 9
    assert loaded.hessian is not None
    assert mode.hessian is not None
    np.testing.assert_allclose(loaded.hessian, mode.hessian)


def test_estimation_mode_validation_rejects_bad_shapes() -> None:
    mode = EstimationModeResult(
        parameter_names=("alpha",),
        estimation_values=np.ones(2),
        objective_value=0.0,
        success=True,
        message="bad",
        iterations=None,
        function_evaluations=None,
    )

    with pytest.raises(ValueError, match="shape"):
        validate_estimation_mode(mode)


def test_parameter_update_preserves_metadata() -> None:
    model = Model1002()
    original = model.parameters["alpha"]

    updated = update_parameter_value(original, 0.25)

    assert updated.description == original.description
    assert updated.category == original.category
    assert updated.regime == original.regime


def test_estimate_preserves_parameter_metadata_after_sampler() -> None:
    model = Model1002()
    data = np.zeros((1, len(model.observables)))

    estimate(
        model,
        data,
        parameter_names=["alpha"],
        mh_draws=1,
        proposal_covariance=np.eye(1) * 1.0e-8,
        seed=123,
    )

    assert model.parameters["alpha"].category == "structural"
    assert model.parameters["alpha"].regime == "baseline"


def test_sampler_result_validation_rejects_inconsistent_shapes() -> None:
    sampler = MetropolisHastingsResult(
        parameter_names=("alpha",),
        estimation_draws=np.ones((2, 1)),
        parameter_draws=np.ones((2, 2)),
        log_posterior=np.ones(2),
        accepted=np.ones(2, dtype=bool),
        acceptance_rate=0.5,
        proposal_covariance=np.eye(1),
        seed=1,
        burnin=0,
    )

    with pytest.raises(ValueError, match="parameter_draws"):
        validate_sampler_result(sampler)


def test_load_sampler_result_validates_archive_schema(tmp_path) -> None:
    path = tmp_path / "bad_sampler.npz"
    np.savez(
        path,
        parameter_names=np.asarray(["alpha"], dtype=str),
        estimation_draws=np.ones((2, 1)),
        parameter_draws=np.ones((2, 2)),
        log_posterior=np.ones(2),
        accepted=np.ones(2, dtype=bool),
        acceptance_rate=np.asarray([0.5]),
        proposal_covariance=np.eye(1),
        seed=np.asarray([1]),
        burnin=np.asarray([0]),
    )

    with pytest.raises(ValueError, match="parameter_draws"):
        load_sampler_result(path)


def test_estimate_validates_metropolis_hastings_inputs() -> None:
    model = Model1002()
    data = np.zeros((1, len(model.observables)))

    with pytest.raises(ValueError, match="mh_draws"):
        estimate(model, data, proposal_covariance=np.eye(1))

    with pytest.raises(ValueError, match="Proposal covariance"):
        estimate(
            model,
            data,
            parameter_names=["alpha"],
            mh_draws=1,
            proposal_covariance=np.eye(2),
        )


def test_estimate_can_optimize_selected_parameter_and_compute_hessian() -> None:
    model = Model1002()
    data = np.zeros((1, len(model.observables)))

    result = estimate(
        model,
        data,
        start_date="2018-Q3",
        optimize=True,
        parameter_names=["alpha"],
        maxiter=1,
        compute_hessian=True,
    )

    assert result.optimization is not None
    assert result.optimization.parameter_names == ("alpha",)
    assert result.optimization.function_evaluations is not None
    assert result.hessian is not None
    assert result.hessian.shape == (1, 1)
    assert np.isfinite(result.log_posterior)
    assert np.isfinite(result.parameter_values["alpha"])

    mode = estimation_mode_from_result(result)
    assert mode.parameter_names == ("alpha",)
    assert mode.hessian is not None
    assert mode.hessian.shape == (1, 1)


def test_estimate_rejects_hessian_without_optimization() -> None:
    model = Model1002()
    data = np.zeros((1, len(model.observables)))

    with pytest.raises(ValueError, match="optimize=True"):
        estimate(model, data, compute_hessian=True)


def test_estimation_parameter_names_default_uses_prior_backed_free_parameters() -> None:
    model = Model1002()

    names = estimation_parameter_names(model)

    assert "alpha" in names
    assert "delta" not in names
    assert "sigma_ziid" not in names


def test_estimation_parameter_names_rejects_fixed_parameters() -> None:
    model = Model1002()

    with pytest.raises(ValueError, match="fixed"):
        estimation_parameter_names(model, parameter_names=["delta"])


def test_finite_difference_hessian_matches_quadratic() -> None:
    center = np.array([1.0, -2.0])

    def objective(values: np.ndarray) -> float:
        return float(values[0] ** 2 + 3.0 * values[0] * values[1] + 2.0 * values[1] ** 2)

    hessian = finite_difference_hessian(objective, center)

    np.testing.assert_allclose(hessian, np.array([[2.0, 3.0], [3.0, 4.0]]), atol=1.0e-6)


def test_proposal_covariance_from_hessian_inverts_curvature() -> None:
    hessian = np.array([[4.0, 0.5], [0.5, 2.0]])

    covariance = proposal_covariance_from_hessian(hessian, jitter=0.0)

    np.testing.assert_allclose(covariance, np.linalg.inv(hessian))
    np.linalg.cholesky(covariance)


def test_proposal_covariance_from_hessian_rejects_invalid_curvature() -> None:
    with pytest.raises(ValueError, match="square"):
        proposal_covariance_from_hessian(np.ones((1, 2)))

    with pytest.raises(ValueError, match="positive definite"):
        proposal_covariance_from_hessian(np.array([[1.0, 0.0], [0.0, -1.0]]), jitter=0.0)


def test_estimate_can_seed_sampler_with_optimized_hessian_proposal() -> None:
    model = Model1002()
    data = np.zeros((1, len(model.observables)))

    result = estimate(
        model,
        data,
        start_date="2018-Q3",
        optimize=True,
        parameter_names=["alpha"],
        maxiter=1,
        compute_hessian=True,
        mh_draws=2,
        proposal_scale=1.0e-4,
        seed=3,
    )

    assert result.hessian is not None
    assert result.sampler is not None
    assert result.sampler.proposal_covariance.shape == (1, 1)
    assert np.isfinite(result.sampler.proposal_covariance[0, 0])
    assert result.sampler.proposal_covariance[0, 0] > 0.0


def test_estimate_can_load_mode_and_seed_sampler_from_hessian() -> None:
    model = Model1002()
    data = np.zeros((1, len(model.observables)))
    mode = EstimationModeResult(
        parameter_names=("alpha",),
        estimation_values=parameter_estimation_vector(model, ("alpha",)),
        objective_value=1.0,
        success=True,
        message="loaded",
        iterations=1,
        function_evaluations=2,
        hessian=np.array([[1.0e8]], dtype=np.float64),
    )

    result = estimate(
        model,
        data,
        mode=mode,
        mh_draws=2,
        proposal_scale=1.0,
        seed=5,
    )

    assert result.optimization is not None
    assert result.optimization.message == "loaded"
    assert result.sampler is not None
    np.testing.assert_allclose(result.sampler.proposal_covariance, np.array([[1.0e-8]]))


def test_estimate_validates_data_shape() -> None:
    model = Model1002()

    with pytest.raises(ValueError, match="2D"):
        estimate(model, np.zeros(len(model.observables)))
