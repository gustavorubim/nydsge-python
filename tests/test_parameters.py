from __future__ import annotations

from math import lgamma

import numpy as np

from nydsge.core import Parameter
from nydsge.parameters import (
    Prior,
    model_log_prior,
    parameter_log_prior,
    prior_from_metadata,
    transform_to_estimation_space,
    transform_to_model_space,
    update_parameter_value,
)


def test_exponential_transform_roundtrip() -> None:
    model_value = 2.5
    estimation_value = transform_to_estimation_space(model_value, "exponential")
    assert np.isclose(transform_to_model_space(estimation_value, "exponential"), model_value)


def test_bounded_square_root_transform_roundtrip() -> None:
    model_value = 0.7
    estimation_value = transform_to_estimation_space(model_value, "sqrt", bounds=(0.0, 1.0))
    assert np.isclose(
        transform_to_model_space(estimation_value, "sqrt", bounds=(0.0, 1.0)), model_value
    )


def test_update_parameter_value_preserves_metadata() -> None:
    parameter = Parameter(
        name="rho",
        value=0.5,
        fixed=False,
        value_bounds=(0.0, 1.0),
        transform="sqrt",
        scaling="percent",
        description="Persistence",
    )
    updated = update_parameter_value(parameter, 0.0)
    assert updated.name == "rho"
    assert updated.description == "Persistence"
    assert np.isclose(updated.value, 0.5)
    assert updated.scaling == "percent"


def test_parameter_scaled_values() -> None:
    assert np.isclose(Parameter("beta", 0.1402, scaling="discount_rate").scaled_value, 0.998599964)
    assert np.isclose(Parameter("pi_star", 0.5, scaling="gross_rate").scaled_value, 1.005)
    assert np.isclose(
        Parameter("Fomega", 0.03, scaling="fomega").scaled_value,
        1.0 - (1.0 - 0.03) ** 0.25,
    )
    assert np.isclose(
        Parameter("spr", 1.7444, scaling="quarterly_spread").scaled_value,
        (1.0 + 1.7444 / 100.0) ** 0.25,
    )
    assert np.isclose(Parameter("gamma", 0.3673, scaling="percent").scaled_value, 0.003673)


def test_prior_logpdfs() -> None:
    normal = Prior("normal", mean=0.0, std=1.0)
    uniform = Prior("uniform", lower=-1.0, upper=1.0)
    beta = Prior("beta", shape=2.0, scale=2.0)
    assert np.isclose(normal.logpdf(0.0), -0.5 * np.log(2.0 * np.pi))
    assert np.isclose(uniform.logpdf(0.0), -np.log(2.0))
    assert np.isfinite(beta.logpdf(0.5))


def test_prior_logpdfs_support_modelconstructors_parameterization() -> None:
    gamma_alt = Prior("gamma_alt", mean=2.0, std=1.0)
    gamma_equivalent = Prior("gamma", shape=4.0, scale=0.5)
    beta_alt = Prior("beta_alt", mean=0.5, std=0.1)
    beta_equivalent = Prior("beta", shape=12.0, scale=12.0)
    root_inverse_gamma = Prior("root_inverse_gamma", nu=2.0, tau=0.1)

    x = 0.2
    expected_root_inverse_gamma = (
        np.log(2.0)
        - lgamma(1.0)
        + np.log(2.0 * 0.1**2 / 2.0)
        - 1.5 * np.log(x**2)
        - 2.0 * 0.1**2 / (2.0 * x**2)
    )

    assert np.isclose(gamma_alt.logpdf(2.0), gamma_equivalent.logpdf(2.0))
    assert np.isclose(beta_alt.logpdf(0.5), beta_equivalent.logpdf(0.5))
    assert np.isclose(root_inverse_gamma.logpdf(x), expected_root_inverse_gamma)


def test_parameter_log_prior_skips_fixed_and_missing_priors() -> None:
    free = Parameter("rho", 0.0, prior=Prior("normal", mean=0.0, std=1.0))
    fixed = Parameter("alpha", 0.5, fixed=True, prior=Prior("normal", mean=0.0, std=1.0))
    missing = Parameter("beta", 0.5)

    assert np.isclose(parameter_log_prior(free), -0.5 * np.log(2.0 * np.pi))
    assert parameter_log_prior(fixed) == 0.0
    assert parameter_log_prior(missing) == 0.0


def test_model_log_prior_sums_parameter_priors() -> None:
    parameters = {
        "a": Parameter("a", 0.0, prior=Prior("normal", mean=0.0, std=1.0)),
        "b": Parameter("b", 0.0, prior=Prior("normal", mean=0.0, std=1.0)),
    }

    assert np.isclose(model_log_prior(parameters), -np.log(2.0 * np.pi))


def test_prior_from_metadata_accepts_dicts() -> None:
    prior = prior_from_metadata({"name": "uniform", "lower": 0.0, "upper": 1.0})

    assert np.isclose(prior.logpdf(0.5), 0.0)
