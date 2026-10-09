"""Estimation-space Jacobian used by the Metropolis-Hastings target (docs/mcmc_c1c2.md)."""

from __future__ import annotations

import numpy as np
import pytest

from nydsge.estimate import (
    _log_posterior_for_estimation_values,
    estimation_parameter_names,
    log_abs_jacobian,
    metropolis_hastings,
    parameter_estimation_vector,
)
from nydsge.models import Model1002
from nydsge.parameters import log_abs_jacobian_to_model_space, transform_to_model_space


@pytest.mark.parametrize(
    ("transform", "bounds", "x"),
    [
        ("identity", None, 0.7),
        ("exponential", None, -0.3),
        ("sqrt", None, 1.4),
        ("sqrt", (0.0, 1.0), 0.8),
        ("sqrt", (1e-5, 0.99999), -2.5),
        ("sqrt", (-0.5, 2.0), 3.0),
    ],
)
def test_log_abs_jacobian_matches_numerical_derivative(transform, bounds, x) -> None:
    h = 1e-6
    derivative = (
        transform_to_model_space(x + h, transform, bounds=bounds)
        - transform_to_model_space(x - h, transform, bounds=bounds)
    ) / (2 * h)
    expected = np.log(abs(derivative))
    assert log_abs_jacobian_to_model_space(x, transform, bounds=bounds) == pytest.approx(
        expected, rel=1e-6, abs=1e-6
    )


def test_bounded_jacobian_is_stable_far_in_the_tails() -> None:
    value = log_abs_jacobian_to_model_space(800.0, "sqrt", bounds=(0.0, 1.0))
    assert np.isfinite(value)
    assert value == pytest.approx(-800.0)


def _small_problem():
    model = Model1002("ss10", settings={"n_mon_anticipated_shocks": 0})
    data = np.zeros((4, len(model.observables)))
    data[:, 0] = [0.3, -0.2, 0.1, 0.4]
    names = estimation_parameter_names(model)
    return model, data, names


def test_target_with_jacobian_adds_log_abs_jacobian() -> None:
    model, data, names = _small_problem()
    original = dict(model.parameters)
    x = parameter_estimation_vector(model, names)
    plain = _log_posterior_for_estimation_values(model, data, original, names, x)
    with_jacobian = _log_posterior_for_estimation_values(
        model, data, original, names, x, jacobian=True
    )
    assert np.isfinite(plain)
    assert with_jacobian - plain == pytest.approx(log_abs_jacobian(original, names, x))


def test_log_likelihood_start_is_forwarded_to_the_target() -> None:
    model, data, names = _small_problem()
    original = dict(model.parameters)
    x = parameter_estimation_vector(model, names)
    full = _log_posterior_for_estimation_values(model, data, original, names, x)
    trimmed = _log_posterior_for_estimation_values(
        model, data, original, names, x, log_likelihood_start=2
    )
    assert np.isfinite(trimmed)
    assert trimmed != pytest.approx(full)


def test_sampler_records_model_space_posterior_and_continues_from_last_state() -> None:
    model, data, names = _small_problem()
    original = dict(model.parameters)
    sub = names[:2]
    x0 = parameter_estimation_vector(model, sub)
    first = metropolis_hastings(
        model,
        data,
        parameter_names=sub,
        draws=3,
        proposal_covariance=np.eye(2) * 1e-4,
        seed=1,
        log_likelihood_start=2,
    )
    expected = _log_posterior_for_estimation_values(
        model,
        data,
        dict(model.parameters),
        sub,
        first.estimation_draws[-1],
        log_likelihood_start=2,
    )
    # recorded values are the model-space log posterior (no Jacobian term)
    assert first.log_posterior[-1] == pytest.approx(expected)
    # the model is left at the last state, so a second call continues the chain
    np.testing.assert_allclose(parameter_estimation_vector(model, sub), first.estimation_draws[-1])
    assert not np.allclose(x0, first.estimation_draws[-1]) or first.acceptance_rate == 0.0
    model.parameters.update(original)
