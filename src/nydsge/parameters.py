from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Any, Literal, cast

import numpy as np

from nydsge.core import Parameter

TransformName = Literal["identity", "untransformed", "exponential", "sqrt", "square_root"]
PriorName = Literal[
    "normal",
    "uniform",
    "gamma",
    "gamma_alt",
    "beta",
    "beta_alt",
    "root_inverse_gamma",
]


@dataclass(frozen=True)
class Prior:
    name: PriorName
    mean: float | None = None
    std: float | None = None
    lower: float | None = None
    upper: float | None = None
    shape: float | None = None
    scale: float | None = None
    nu: float | None = None
    tau: float | None = None

    def logpdf(self, value: float) -> float:
        return prior_logpdf(self, value)


def transform_to_model_space(
    value: float,
    transform: TransformName,
    *,
    bounds: tuple[float, float] | None = None,
) -> float:
    if transform in {"identity", "untransformed"}:
        return float(value)
    if transform == "exponential":
        return float(np.exp(value))
    if transform in {"sqrt", "square_root"}:
        if bounds is None:
            return float(value * value)
        lower, upper = bounds
        logistic = 1.0 / (1.0 + np.exp(-value))
        return float(lower + (upper - lower) * logistic)
    msg = f"Unsupported transform: {transform}"
    raise ValueError(msg)


def transform_to_estimation_space(
    value: float,
    transform: TransformName,
    *,
    bounds: tuple[float, float] | None = None,
) -> float:
    if transform in {"identity", "untransformed"}:
        return float(value)
    if transform == "exponential":
        if value <= 0.0:
            msg = "Exponential transform requires positive model-space values."
            raise ValueError(msg)
        return float(np.log(value))
    if transform in {"sqrt", "square_root"}:
        if bounds is None:
            if value < 0.0:
                msg = "Square-root transform requires non-negative model-space values."
                raise ValueError(msg)
            return float(np.sqrt(value))
        lower, upper = bounds
        if not lower < value < upper:
            msg = f"Bounded square-root transform requires {lower} < value < {upper}."
            raise ValueError(msg)
        scaled = (value - lower) / (upper - lower)
        return float(np.log(scaled / (1.0 - scaled)))
    msg = f"Unsupported transform: {transform}"
    raise ValueError(msg)


def update_parameter_value(parameter: Parameter, estimation_space_value: float) -> Parameter:
    model_value = transform_to_model_space(
        estimation_space_value,
        _parameter_transform(parameter),
        bounds=parameter.value_bounds,
    )
    return Parameter(
        name=parameter.name,
        value=model_value,
        fixed=parameter.fixed,
        value_bounds=parameter.value_bounds,
        transform=parameter.transform,
        scaling=parameter.scaling,
        prior=parameter.prior,
        description=parameter.description,
        tex_label=parameter.tex_label,
        category=parameter.category,
        regime=parameter.regime,
    )


def parameter_log_prior(parameter: Parameter) -> float:
    if parameter.fixed or parameter.prior is None:
        return 0.0
    prior = prior_from_metadata(parameter.prior)
    return prior.logpdf(parameter.value)


def model_log_prior(parameters: dict[str, Parameter]) -> float:
    return float(sum(parameter_log_prior(parameter) for parameter in parameters.values()))


def prior_from_metadata(metadata: Any) -> Prior:
    if isinstance(metadata, Prior):
        return metadata
    if isinstance(metadata, dict):
        return Prior(**metadata)
    msg = f"Unsupported prior metadata: {metadata!r}"
    raise ValueError(msg)


def prior_logpdf(prior: Prior, value: float) -> float:
    if prior.name == "normal":
        mean = _require(prior.mean, "normal mean")
        std = _require_positive(prior.std, "normal std")
        z = (value - mean) / std
        return float(-0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(std) + z * z))
    if prior.name == "uniform":
        lower = _require(prior.lower, "uniform lower")
        upper = _require(prior.upper, "uniform upper")
        if lower > value or value > upper:
            return float("-inf")
        return float(-np.log(upper - lower))
    if prior.name == "gamma":
        shape = _require_positive(prior.shape, "gamma shape")
        scale = _require_positive(prior.scale, "gamma scale")
        if value <= 0.0:
            return float("-inf")
        return float(
            (shape - 1.0) * np.log(value) - value / scale - lgamma(shape) - shape * np.log(scale)
        )
    if prior.name == "gamma_alt":
        mean = _require_positive(prior.mean, "gamma_alt mean")
        std = _require_positive(prior.std, "gamma_alt std")
        scale = std * std / mean
        shape = mean / scale
        return prior_logpdf(Prior("gamma", shape=shape, scale=scale), value)
    if prior.name == "beta":
        alpha = _require_positive(prior.shape, "beta alpha")
        beta = _require_positive(prior.scale, "beta beta")
        if value <= 0.0 or value >= 1.0:
            return float("-inf")
        log_norm = lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta)
        return float((alpha - 1.0) * np.log(value) + (beta - 1.0) * np.log1p(-value) - log_norm)
    if prior.name == "beta_alt":
        mean = _require(prior.mean, "beta_alt mean")
        std = _require_positive(prior.std, "beta_alt std")
        alpha = (1.0 - mean) * mean * mean / (std * std) - mean
        beta = alpha * (1.0 / mean - 1.0)
        return prior_logpdf(Prior("beta", shape=alpha, scale=beta), value)
    if prior.name == "root_inverse_gamma":
        nu = _require_positive(
            prior.nu if prior.nu is not None else prior.shape,
            "root-inverse-gamma nu",
        )
        tau = _require_positive(
            prior.tau if prior.tau is not None else prior.scale,
            "root-inverse-gamma tau",
        )
        if value <= 0.0:
            return float("-inf")
        return float(
            np.log(2.0)
            - lgamma(nu / 2.0)
            + (nu / 2.0) * np.log(nu * tau * tau / 2.0)
            - ((nu + 1.0) / 2.0) * np.log(value * value)
            - nu * tau * tau / (2.0 * value * value)
        )
    msg = f"Unsupported prior: {prior.name}"
    raise ValueError(msg)


def _parameter_transform(parameter: Parameter) -> TransformName:
    match parameter.transform:
        case "identity" | "untransformed" | "exponential" | "sqrt" | "square_root":
            return cast(TransformName, parameter.transform)
        case _:
            msg = f"Unsupported parameter transform: {parameter.transform}"
            raise ValueError(msg)


def _require(value: float | None, label: str) -> float:
    if value is None:
        msg = f"Missing {label}."
        raise ValueError(msg)
    return value


def _require_positive(value: float | None, label: str) -> float:
    checked = _require(value, label)
    if checked <= 0.0:
        msg = f"{label} must be positive."
        raise ValueError(msg)
    return checked
