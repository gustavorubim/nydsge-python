from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from nydsge.backends import get_backend
from nydsge.core import DSGEModel, NotPortedError, Parameter
from nydsge.kalman import KalmanResult, kalman_log_likelihood, model_process_covariances
from nydsge.parameters import (
    TransformName,
    model_log_prior,
    transform_to_estimation_space,
    update_parameter_value,
)
from nydsge.solve import compute_system


@dataclass(frozen=True)
class OptimizationResult:
    parameter_names: tuple[str, ...]
    estimation_values: np.ndarray
    objective_value: float
    success: bool
    message: str
    iterations: int | None
    function_evaluations: int | None


@dataclass(frozen=True)
class MetropolisHastingsResult:
    parameter_names: tuple[str, ...]
    estimation_draws: np.ndarray
    parameter_draws: np.ndarray
    log_posterior: np.ndarray
    accepted: np.ndarray
    acceptance_rate: float
    proposal_covariance: np.ndarray
    seed: int | None
    burnin: int
    n_blocks: int = 1
    n_param_blocks: int = 1
    mhthin: int = 1
    burnin_blocks: int = 0
    proposal_scale: float = 1.0
    adaptive_accept: bool = False
    target_accept: float = 0.25
    alpha: float = 1.0
    c: float = 0.5


@dataclass(frozen=True)
class SamplerParameterDiagnostics:
    name: str
    mean: float
    std: float
    minimum: float
    maximum: float
    effective_sample_size: float
    integrated_autocorrelation_time: float
    monte_carlo_standard_error: float
    split_rhat: float | None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "effective_sample_size": self.effective_sample_size,
            "integrated_autocorrelation_time": self.integrated_autocorrelation_time,
            "monte_carlo_standard_error": self.monte_carlo_standard_error,
            "split_rhat": self.split_rhat,
        }


@dataclass(frozen=True)
class SamplerDiagnostics:
    parameter_names: tuple[str, ...]
    draws: int
    burnin: int
    seed: int | None
    accepted_draws: int
    acceptance_rate: float
    realized_acceptance_rate: float
    acceptance_windows: tuple[float, ...]
    proposal_covariance_shape: tuple[int, int]
    proposal_covariance_min_eigenvalue: float
    proposal_covariance_max_eigenvalue: float
    proposal_covariance_condition_number: float
    proposal_covariance_positive_semidefinite: bool
    log_posterior_mean: float
    log_posterior_minimum: float
    log_posterior_maximum: float
    parameters: tuple[SamplerParameterDiagnostics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_names": list(self.parameter_names),
            "draws": self.draws,
            "burnin": self.burnin,
            "seed": self.seed,
            "accepted_draws": self.accepted_draws,
            "acceptance_rate": self.acceptance_rate,
            "realized_acceptance_rate": self.realized_acceptance_rate,
            "acceptance_windows": list(self.acceptance_windows),
            "proposal_covariance_shape": list(self.proposal_covariance_shape),
            "proposal_covariance_min_eigenvalue": self.proposal_covariance_min_eigenvalue,
            "proposal_covariance_max_eigenvalue": self.proposal_covariance_max_eigenvalue,
            "proposal_covariance_condition_number": (self.proposal_covariance_condition_number),
            "proposal_covariance_positive_semidefinite": (
                self.proposal_covariance_positive_semidefinite
            ),
            "log_posterior_mean": self.log_posterior_mean,
            "log_posterior_minimum": self.log_posterior_minimum,
            "log_posterior_maximum": self.log_posterior_maximum,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


@dataclass(frozen=True)
class EstimationModeResult:
    parameter_names: tuple[str, ...]
    estimation_values: np.ndarray
    objective_value: float
    success: bool
    message: str
    iterations: int | None
    function_evaluations: int | None
    hessian: np.ndarray | None = None


@dataclass(frozen=True)
class EstimateResult:
    log_posterior: float
    log_likelihood: float
    log_prior: float
    parameter_values: dict[str, float]
    kalman: KalmanResult
    optimization: OptimizationResult | None = None
    hessian: np.ndarray | None = None
    sampler: MetropolisHastingsResult | None = None


def estimate(
    model: DSGEModel,
    data: np.ndarray,
    *,
    verbose: str = "low",
    start_date: Any | None = None,
    proposal_covariance: np.ndarray | None = None,
    optimize: bool = False,
    parameter_names: list[str] | tuple[str, ...] | None = None,
    optimizer_method: str = "Nelder-Mead",
    maxiter: int = 100,
    compute_hessian: bool = False,
    hessian_step: float = 1.0e-4,
    mh_draws: int = 0,
    mh_burnin: int = 0,
    mh_blocks: int = 1,
    mh_param_blocks: int = 1,
    mh_thin: int = 1,
    proposal_scale: float = 1.0,
    mh_burn_blocks: int = 0,
    mh_adaptive_accept: bool = False,
    mh_target_accept: float = 0.25,
    mh_alpha: float = 1.0,
    mh_c: float = 0.5,
    seed: int | None = None,
    mode: EstimationModeResult | None = None,
) -> EstimateResult:
    del verbose
    if maxiter <= 0:
        msg = "maxiter must be positive."
        raise ValueError(msg)
    if mh_draws < 0:
        msg = "mh_draws must be nonnegative."
        raise ValueError(msg)
    if mh_burnin < 0:
        msg = "mh_burnin must be nonnegative."
        raise ValueError(msg)
    if proposal_covariance is not None and mh_draws == 0:
        msg = "mh_draws must be positive when proposal_covariance is provided."
        raise ValueError(msg)
    if mh_burnin and mh_draws == 0:
        msg = "mh_draws must be positive when mh_burnin is provided."
        raise ValueError(msg)
    if mh_blocks < 1:
        msg = "mh_blocks must be positive."
        raise ValueError(msg)
    if mh_param_blocks < 1:
        msg = "mh_param_blocks must be positive."
        raise ValueError(msg)
    if mh_param_blocks > 0 and mh_param_blocks > len(parameter_names or model.parameters):
        msg = "mh_param_blocks cannot exceed number of estimated parameters."
        raise ValueError(msg)
    if mh_thin < 1:
        msg = "mh_thin must be positive."
        raise ValueError(msg)
    if mh_burn_blocks < 0:
        msg = "mh_burn_blocks must be nonnegative."
        raise ValueError(msg)
    if mh_adaptive_accept and not 0.0 <= mh_target_accept <= 1.0:
        msg = "mh_target_accept must be between 0 and 1."
        raise ValueError(msg)
    if mh_adaptive_accept and not 0.0 <= mh_alpha <= 1.0:
        msg = "mh_alpha must be in [0, 1]."
        raise ValueError(msg)
    if mh_adaptive_accept and mh_c <= 0.0:
        msg = "mh_c must be positive."
        raise ValueError(msg)
    if proposal_scale <= 0.0:
        msg = "proposal_scale must be positive."
        raise ValueError(msg)

    if model.spec == "m1002" and model.subspec == "ss10":
        observations = np.asarray(data, dtype=np.float64)
        if observations.ndim != 2:
            msg = "Estimation data must be a 2D array shaped as periods x observables."
            raise ValueError(msg)
        selected_names: tuple[str, ...] = ()
        optimization = None
        hessian = None
        sampler = None
        original_parameters = dict(model.parameters)
        if mode is not None:
            validate_estimation_mode(mode)
            if optimize:
                msg = "mode cannot be combined with optimize=True."
                raise ValueError(msg)
            if compute_hessian:
                msg = "compute_hessian is only valid when optimize=True."
                raise ValueError(msg)
            if parameter_names is not None and tuple(parameter_names) != mode.parameter_names:
                msg = "parameter_names must match the loaded estimation mode."
                raise ValueError(msg)
            selected_names = mode.parameter_names
            _set_parameter_estimation_vector(
                model,
                original_parameters,
                selected_names,
                mode.estimation_values,
            )
            optimization = _optimization_result_from_mode(mode)
            hessian = None if mode.hessian is None else mode.hessian.copy()
        elif optimize:
            selected_names = estimation_parameter_names(model, parameter_names=parameter_names)
            start = parameter_estimation_vector(model, selected_names)
            objective = _negative_log_posterior_objective(
                model,
                observations,
                selected_names,
                start_date=start_date,
            )
            result = minimize(
                objective,
                start,
                method=optimizer_method,
                options={"maxiter": maxiter},
            )
            best = np.asarray(result.x, dtype=np.float64)
            _set_parameter_estimation_vector(model, original_parameters, selected_names, best)
            optimization = _optimization_result(selected_names, best, result)
            if compute_hessian:
                hessian = finite_difference_hessian(objective, best, step=hessian_step)
                _set_parameter_estimation_vector(model, original_parameters, selected_names, best)
        elif compute_hessian:
            msg = "compute_hessian is only valid when optimize=True."
            raise ValueError(msg)

        if mh_draws > 0:
            if not selected_names:
                selected_names = estimation_parameter_names(model, parameter_names=parameter_names)
            sampler_proposal_covariance = proposal_covariance
            if sampler_proposal_covariance is None and hessian is not None:
                sampler_proposal_covariance = proposal_covariance_from_hessian(hessian)
            sampler = metropolis_hastings(
                model,
                observations,
                parameter_names=selected_names,
                draws=mh_draws,
                burnin=mh_burnin,
                n_blocks=mh_blocks,
                n_param_blocks=mh_param_blocks,
                mhthin=mh_thin,
                proposal_covariance=sampler_proposal_covariance,
                proposal_scale=proposal_scale,
                burnin_blocks=mh_burn_blocks,
                adaptive_accept=mh_adaptive_accept,
                target_accept=mh_target_accept,
                alpha=mh_alpha,
                c=mh_c,
                seed=seed,
                start_date=start_date,
            )
        elif parameter_names is not None and not optimize:
            msg = "parameter_names are only valid when optimize=True or mh_draws > 0."
            raise ValueError(msg)

        log_posterior, log_likelihood, log_prior, kalman = _evaluate_log_posterior(
            model,
            observations,
            start_date=start_date,
        )
        return EstimateResult(
            log_posterior=float(log_posterior),
            log_likelihood=float(log_likelihood),
            log_prior=float(log_prior),
            parameter_values={
                name: parameter.value for name, parameter in model.parameters.items()
            },
            kalman=kalman,
            optimization=optimization,
            hessian=hessian,
            sampler=sampler,
        )

    msg = (
        f"Estimation for {model.spec} {model.subspec} is not ported yet. "
        "Next required translations: model-specific state-space setup, optimizer/Hessian, "
        "and Metropolis-Hastings."
    )
    raise NotPortedError(msg)


def estimation_parameter_names(
    model: DSGEModel,
    *,
    parameter_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if parameter_names is None:
        names = [
            name
            for name, parameter in model.parameters.items()
            if not parameter.fixed and parameter.prior is not None
        ]
    else:
        names = list(parameter_names)
    if not names:
        msg = "No estimation parameters were selected."
        raise ValueError(msg)
    missing = [name for name in names if name not in model.parameters]
    if missing:
        msg = "Unknown estimation parameter(s): " + ", ".join(missing)
        raise KeyError(msg)
    fixed = [name for name in names if model.parameters[name].fixed]
    if fixed:
        msg = "Cannot optimize fixed parameter(s): " + ", ".join(fixed)
        raise ValueError(msg)
    return tuple(names)


def parameter_estimation_vector(
    model: DSGEModel,
    parameter_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [
            transform_to_estimation_space(
                model.parameters[name].value,
                _parameter_transform(model.parameters[name]),
                bounds=model.parameters[name].value_bounds,
            )
            for name in parameter_names
        ],
        dtype=np.float64,
    )


def finite_difference_hessian(
    objective: Any,
    center: np.ndarray,
    *,
    step: float = 1.0e-4,
) -> np.ndarray:
    if step <= 0.0:
        msg = "Hessian finite-difference step must be positive."
        raise ValueError(msg)
    x0 = np.asarray(center, dtype=np.float64)
    n_params = x0.size
    hessian = np.zeros((n_params, n_params), dtype=np.float64)
    f0 = float(objective(x0))
    steps = step * np.maximum(1.0, np.abs(x0))
    for i in range(n_params):
        ei = np.zeros(n_params, dtype=np.float64)
        ei[i] = steps[i]
        f_plus = float(objective(x0 + ei))
        f_minus = float(objective(x0 - ei))
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)
        for j in range(i + 1, n_params):
            ej = np.zeros(n_params, dtype=np.float64)
            ej[j] = steps[j]
            f_pp = float(objective(x0 + ei + ej))
            f_pm = float(objective(x0 + ei - ej))
            f_mp = float(objective(x0 - ei + ej))
            f_mm = float(objective(x0 - ei - ej))
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def estimation_mode_from_result(result: EstimateResult) -> EstimationModeResult:
    if result.optimization is None:
        msg = "EstimateResult does not contain an optimization result."
        raise ValueError(msg)
    return EstimationModeResult(
        parameter_names=result.optimization.parameter_names,
        estimation_values=result.optimization.estimation_values.copy(),
        objective_value=result.optimization.objective_value,
        success=result.optimization.success,
        message=result.optimization.message,
        iterations=result.optimization.iterations,
        function_evaluations=result.optimization.function_evaluations,
        hessian=None if result.hessian is None else result.hessian.copy(),
    )


def save_estimation_mode(
    mode: EstimationModeResult,
    path: Path | str,
) -> Path:
    validate_estimation_mode(mode)
    destination = Path(path)
    if destination.suffix != ".npz":
        msg = "Estimation mode results must be written as a .npz archive."
        raise ValueError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        parameter_names=np.asarray(mode.parameter_names, dtype=str),
        estimation_values=mode.estimation_values,
        objective_value=np.asarray([mode.objective_value], dtype=np.float64),
        success=np.asarray([mode.success], dtype=bool),
        message=np.asarray([mode.message], dtype=str),
        iterations=np.asarray([-1 if mode.iterations is None else mode.iterations], dtype=np.int64),
        function_evaluations=np.asarray(
            [-1 if mode.function_evaluations is None else mode.function_evaluations],
            dtype=np.int64,
        ),
        hessian_present=np.asarray([mode.hessian is not None], dtype=bool),
        hessian=(
            np.empty((0, 0), dtype=np.float64)
            if mode.hessian is None
            else np.asarray(mode.hessian, dtype=np.float64)
        ),
    )
    return destination


def load_estimation_mode(path: Path | str) -> EstimationModeResult:
    source = Path(path)
    with np.load(source) as archive:
        required = {
            "parameter_names",
            "estimation_values",
            "objective_value",
            "success",
            "message",
            "iterations",
            "function_evaluations",
            "hessian_present",
            "hessian",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            msg = "Estimation mode archive is missing array(s): " + ", ".join(missing)
            raise KeyError(msg)
        has_hessian = bool(np.ravel(archive["hessian_present"])[0])
        iterations = int(np.ravel(archive["iterations"])[0])
        function_evaluations = int(np.ravel(archive["function_evaluations"])[0])
        mode = EstimationModeResult(
            parameter_names=tuple(str(name) for name in archive["parameter_names"].tolist()),
            estimation_values=np.asarray(archive["estimation_values"], dtype=np.float64),
            objective_value=float(np.ravel(archive["objective_value"])[0]),
            success=bool(np.ravel(archive["success"])[0]),
            message=str(np.ravel(archive["message"])[0]),
            iterations=None if iterations < 0 else iterations,
            function_evaluations=None if function_evaluations < 0 else function_evaluations,
            hessian=(np.asarray(archive["hessian"], dtype=np.float64) if has_hessian else None),
        )
    validate_estimation_mode(mode)
    return mode


def validate_estimation_mode(mode: EstimationModeResult) -> None:
    n_parameters = len(mode.parameter_names)
    if n_parameters == 0:
        msg = "Estimation mode archive must contain at least one parameter name."
        raise ValueError(msg)
    if mode.estimation_values.shape != (n_parameters,):
        msg = f"Estimation mode values must have shape {(n_parameters,)}."
        raise ValueError(msg)
    if not np.all(np.isfinite(mode.estimation_values)):
        msg = "Estimation mode values must contain only finite values."
        raise ValueError(msg)
    if not np.isfinite(mode.objective_value):
        msg = "Estimation mode objective value must be finite."
        raise ValueError(msg)
    if mode.iterations is not None and mode.iterations < 0:
        msg = "Estimation mode iterations must be nonnegative."
        raise ValueError(msg)
    if mode.function_evaluations is not None and mode.function_evaluations < 0:
        msg = "Estimation mode function evaluations must be nonnegative."
        raise ValueError(msg)
    if mode.hessian is not None:
        if mode.hessian.shape != (n_parameters, n_parameters):
            msg = f"Estimation mode Hessian must have shape {(n_parameters, n_parameters)}."
            raise ValueError(msg)
        if not np.all(np.isfinite(mode.hessian)):
            msg = "Estimation mode Hessian must contain only finite values."
            raise ValueError(msg)


def proposal_covariance_from_hessian(
    hessian: np.ndarray,
    *,
    jitter: float = 1.0e-10,
) -> np.ndarray:
    """Build an MH proposal covariance from a negative-log-posterior Hessian."""
    matrix = np.asarray(hessian, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        msg = "Hessian must be a square 2D array."
        raise ValueError(msg)
    if matrix.shape[0] == 0:
        msg = "Hessian must contain at least one parameter."
        raise ValueError(msg)
    if jitter < 0.0:
        msg = "Hessian jitter must be nonnegative."
        raise ValueError(msg)
    if not np.all(np.isfinite(matrix)):
        msg = "Hessian must contain only finite values."
        raise ValueError(msg)

    symmetric = 0.5 * (matrix + matrix.T)
    if jitter > 0.0:
        symmetric = symmetric + np.eye(symmetric.shape[0], dtype=np.float64) * jitter
    try:
        covariance = np.linalg.inv(symmetric)
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as err:
        msg = "Hessian-derived proposal covariance must be positive definite."
        raise ValueError(msg) from err
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        msg = "Hessian-derived proposal covariance must contain only finite values."
        raise ValueError(msg)
    return covariance


def _generate_free_blocks(
    n_parameters: int,
    n_blocks: int,
    *,
    rng: np.random.Generator,
) -> list[list[int]]:
    if n_parameters <= 0:
        msg = "n_parameters must be positive."
        raise ValueError(msg)
    if n_blocks < 1:
        msg = "n_param_blocks must be positive."
        raise ValueError(msg)
    if n_blocks > n_parameters:
        msg = "n_param_blocks cannot exceed the number of parameters."
        raise ValueError(msg)
    shuffled = np.asarray(rng.permutation(np.arange(n_parameters, dtype=np.int64)), dtype=np.int64)
    subset_length = int(np.ceil(n_parameters / n_blocks))
    blocks = []
    for block in range(n_blocks):
        if block < n_blocks - 1:
            block_indices = shuffled[(block * subset_length) : (block + 1) * subset_length]
        else:
            block_indices = shuffled[block * subset_length :]
        block_sorted = sorted(int(index) for index in block_indices)
        if block_sorted:
            blocks.append(block_sorted)
    return blocks


def _logpdf_mvn(
    point: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> float:
    point = np.asarray(point, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if point.shape != mean.shape:
        msg = "The point and mean must have the same shape."
        raise ValueError(msg)
    if covariance.shape[0] != covariance.shape[1]:
        msg = "Covariance must be a square matrix."
        raise ValueError(msg)
    if point.size == 0:
        return 0.0
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as err:
        msg = "Covariance must be positive definite."
        raise ValueError(msg) from err
    delta = point - mean
    solved = np.linalg.solve(cholesky, delta)
    quadratic_form = float(np.dot(solved, solved))
    logdet = float(2.0 * np.sum(np.log(np.diag(cholesky))))
    if np.isinf(logdet):
        return float("-inf")
    return float(-0.5 * (point.size * np.log(2.0 * np.pi) + logdet + quadratic_form))


def _proposal_correction_log_ratio(
    proposal: np.ndarray,
    current: np.ndarray,
    covariance: np.ndarray,
    *,
    alpha: float,
    c: float,
) -> tuple[float, float]:
    if not (0.0 <= alpha <= 1.0):
        msg = "Adaptive alpha must be in [0, 1]."
        raise ValueError(msg)
    if c <= 0.0:
        msg = "Adaptive proposal scaling c must be positive."
        raise ValueError(msg)
    scaled_covariance = covariance * (c * c)
    mixed_mean = proposal
    try:
        log_q0_term = _logpdf_mvn(current, proposal, scaled_covariance)
        log_q1_term = _logpdf_mvn(proposal, current, scaled_covariance)
        log_mid_current = _logpdf_mvn(current, mixed_mean, scaled_covariance)
        log_mid_proposal = _logpdf_mvn(proposal, mixed_mean, scaled_covariance)
    except ValueError:
        return float("-inf"), float("-inf")

    diag_std = np.sqrt(np.clip(np.diag(scaled_covariance), 1.0e-16, None))
    z = (current - proposal) / diag_std
    log_diag = np.sum(-np.log(diag_std * np.sqrt(2.0 * np.pi)) - 0.5 * (z**2))

    if alpha == 0.0:
        q0 = np.exp(np.log((1.0 - alpha) / 2.0) + log_diag)
        q1 = np.exp(np.log((1.0 - alpha) / 2.0) + log_diag)
        return (
            float(np.log(q0)) if q0 > 0 else float("-inf"),
            float(np.log(q1)) if q1 > 0 else float("-inf"),
        )

    log_q0 = np.logaddexp.reduce(
        [
            np.log(alpha) + log_q0_term,
            np.log((1.0 - alpha) / 2.0) + log_diag,
            np.log((1.0 - alpha) / 2.0) + log_mid_current,
        ]
    )
    log_q1 = np.logaddexp.reduce(
        [
            np.log(alpha) + log_q1_term,
            np.log((1.0 - alpha) / 2.0) + log_diag,
            np.log((1.0 - alpha) / 2.0) + log_mid_proposal,
        ]
    )
    return log_q0, log_q1


def metropolis_hastings(
    model: DSGEModel,
    observations: np.ndarray,
    *,
    parameter_names: tuple[str, ...],
    draws: int,
    burnin: int = 0,
    n_blocks: int = 1,
    n_param_blocks: int = 1,
    mhthin: int = 1,
    burnin_blocks: int = 0,
    adaptive_accept: bool = False,
    target_accept: float = 0.25,
    alpha: float = 1.0,
    c: float = 0.5,
    proposal_covariance: np.ndarray | None = None,
    proposal_scale: float = 1.0,
    seed: int | None = None,
    start_date: Any | None = None,
) -> MetropolisHastingsResult:
    if draws <= 0:
        msg = "Metropolis-Hastings draws must be positive."
        raise ValueError(msg)
    if burnin < 0:
        msg = "Metropolis-Hastings burn-in must be nonnegative."
        raise ValueError(msg)
    if n_blocks < 1:
        msg = "Metropolis-Hastings blocks must be positive."
        raise ValueError(msg)
    if n_param_blocks < 1:
        msg = "Metropolis-Hastings parameter blocks must be positive."
        raise ValueError(msg)
    if burnin_blocks < 0 or burnin_blocks > n_blocks:
        msg = "Metropolis-Hastings burn blocks must be between 0 and n_blocks."
        raise ValueError(msg)
    if proposal_scale <= 0.0:
        msg = "proposal_scale must be positive."
        raise ValueError(msg)
    if mhthin < 1:
        msg = "Metropolis-Hastings thinning must be positive."
        raise ValueError(msg)
    if adaptive_accept and not 0.0 <= target_accept <= 1.0:
        msg = "Metropolis-Hastings target_accept must be between 0 and 1."
        raise ValueError(msg)

    original_parameters = dict(model.parameters)
    n_parameters = len(parameter_names)
    if n_param_blocks > n_parameters:
        msg = "n_param_blocks cannot exceed number of estimated parameters."
        raise ValueError(msg)
    current = parameter_estimation_vector(model, parameter_names)
    covariance = _proposal_covariance(
        proposal_covariance,
        n_parameters=n_parameters,
        scale=proposal_scale,
    )
    proposal_scale_adaptive = float(proposal_scale)

    current_log_posterior = _log_posterior_for_estimation_values(
        model,
        observations,
        original_parameters,
        parameter_names,
        current,
        start_date=start_date,
    )
    if not np.isfinite(current_log_posterior):
        msg = "Initial Metropolis-Hastings log posterior is not finite."
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    steps_per_block = draws * mhthin
    if n_blocks == 1 and burnin_blocks == 0:
        steps_per_block += burnin * mhthin
    if n_param_blocks == 1:
        parameter_blocks = [list(range(n_parameters))]
        reblock = False
    else:
        parameter_blocks = []
        reblock = True

    target_draws = draws
    estimation_draws: list[np.ndarray] = []
    parameter_draws: list[np.ndarray] = []
    log_posterior_draws: list[float] = []
    accepted_draws: list[bool] = []
    accepted_total = 0
    proposal_count = 0
    block_acceptance_rate = target_accept

    for block in range(n_blocks):
        if len(estimation_draws) >= target_draws:
            break
        if adaptive_accept:
            proposal_scale_adaptive *= 0.95 + 0.10 * (
                np.exp(16.0 * (block_acceptance_rate - target_accept))
                / (1.0 + np.exp(16.0 * (block_acceptance_rate - target_accept)))
            )
        block_rejections = 0
        block_proposals = 0
        for step in range(1, steps_per_block + 1):
            if len(estimation_draws) >= target_draws:
                break
            if reblock:
                parameter_blocks = _generate_free_blocks(
                    n_parameters,
                    n_param_blocks,
                    rng=rng,
                )
            sweep_accepted = False
            for block_indices in parameter_blocks:
                proposal = current.copy()
                block_indices_array = np.asarray(block_indices, dtype=np.int64)
                block_covariance = covariance[np.ix_(block_indices_array, block_indices_array)]
                block_scale = proposal_scale_adaptive
                block_cholesky = np.linalg.cholesky(block_covariance * (block_scale * block_scale))
                proposal[block_indices_array] = proposal[
                    block_indices_array
                ] + block_cholesky @ rng.standard_normal(block_indices_array.size)

                proposal_log_posterior = _log_posterior_for_estimation_values(
                    model,
                    observations,
                    original_parameters,
                    parameter_names,
                    proposal,
                    start_date=start_date,
                )

                accepted_step = False
                block_proposals += 1
                proposal_count += 1
                if np.isfinite(proposal_log_posterior):
                    log_acceptance = proposal_log_posterior - current_log_posterior
                    if adaptive_accept:
                        log_q0, log_q1 = _proposal_correction_log_ratio(
                            proposal=proposal[block_indices_array],
                            current=current[block_indices_array],
                            covariance=block_covariance,
                            alpha=alpha,
                            c=c,
                        )
                        if np.isfinite(log_q0) and np.isfinite(log_q1):
                            log_acceptance += log_q0 - log_q1
                    if log_acceptance >= 0.0 or np.log(rng.random()) < log_acceptance:
                        current = proposal
                        current_log_posterior = proposal_log_posterior
                        accepted_step = True
                        accepted_total += 1
                        sweep_accepted = True
                if not accepted_step:
                    block_rejections += 1

                if step % mhthin == 0 and block >= burnin_blocks and proposal_count > burnin:
                    if len(estimation_draws) < target_draws:
                        estimation_draws.append(current.copy())
                        parameter_draws.append(
                            _model_values_for_estimation_vector(
                                original_parameters,
                                parameter_names,
                                current,
                            )
                        )
                        log_posterior_draws.append(current_log_posterior)
                        accepted_draws.append(sweep_accepted)
                        if len(estimation_draws) >= target_draws:
                            break

        if adaptive_accept and block_proposals:
            block_acceptance_rate = 1.0 - (block_rejections / block_proposals)

    estimation_draw_array = (
        np.asarray(estimation_draws, dtype=np.float64)
        if estimation_draws
        else np.zeros((0, n_parameters), dtype=np.float64)
    )
    parameter_draw_array = (
        np.asarray(parameter_draws, dtype=np.float64)
        if parameter_draws
        else np.zeros((0, n_parameters), dtype=np.float64)
    )
    log_posterior_array = (
        np.asarray(log_posterior_draws, dtype=np.float64)
        if log_posterior_draws
        else np.zeros(0, dtype=np.float64)
    )
    accepted_array = np.asarray(accepted_draws, dtype=bool)
    acceptance_rate = float(accepted_total / proposal_count) if proposal_count else 0.0

    _set_parameter_estimation_vector(model, original_parameters, parameter_names, current)
    return MetropolisHastingsResult(
        parameter_names=parameter_names,
        estimation_draws=estimation_draw_array,
        parameter_draws=parameter_draw_array,
        log_posterior=log_posterior_array,
        accepted=accepted_array,
        acceptance_rate=acceptance_rate,
        proposal_covariance=covariance,
        seed=seed,
        burnin=burnin,
        n_blocks=n_blocks,
        n_param_blocks=n_param_blocks,
        mhthin=mhthin,
        burnin_blocks=burnin_blocks,
        proposal_scale=proposal_scale,
        adaptive_accept=adaptive_accept,
        target_accept=target_accept,
        alpha=alpha,
        c=c,
    )


def save_sampler_result(
    sampler: MetropolisHastingsResult,
    path: Path | str,
) -> Path:
    validate_sampler_result(sampler)
    destination = Path(path)
    if destination.suffix != ".npz":
        msg = "Sampler results must be written as a .npz archive."
        raise ValueError(msg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        parameter_names=np.asarray(sampler.parameter_names, dtype=str),
        estimation_draws=sampler.estimation_draws,
        parameter_draws=sampler.parameter_draws,
        log_posterior=sampler.log_posterior,
        accepted=sampler.accepted,
        acceptance_rate=np.asarray([sampler.acceptance_rate], dtype=np.float64),
        proposal_covariance=sampler.proposal_covariance,
        n_blocks=np.asarray([sampler.n_blocks], dtype=np.int64),
        n_param_blocks=np.asarray([sampler.n_param_blocks], dtype=np.int64),
        mhthin=np.asarray([sampler.mhthin], dtype=np.int64),
        burnin_blocks=np.asarray([sampler.burnin_blocks], dtype=np.int64),
        proposal_scale=np.asarray([sampler.proposal_scale], dtype=np.float64),
        adaptive_accept=np.asarray([sampler.adaptive_accept], dtype=bool),
        target_accept=np.asarray([sampler.target_accept], dtype=np.float64),
        alpha=np.asarray([sampler.alpha], dtype=np.float64),
        c=np.asarray([sampler.c], dtype=np.float64),
        seed=np.asarray([-1 if sampler.seed is None else sampler.seed], dtype=np.int64),
        burnin=np.asarray([sampler.burnin], dtype=np.int64),
    )
    return destination


def load_sampler_result(path: Path | str) -> MetropolisHastingsResult:
    source = Path(path)
    with np.load(source) as archive:
        required = {
            "parameter_names",
            "estimation_draws",
            "parameter_draws",
            "log_posterior",
            "accepted",
            "acceptance_rate",
            "proposal_covariance",
            "seed",
            "burnin",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            msg = "Sampler archive is missing array(s): " + ", ".join(missing)
            raise KeyError(msg)
        seed_value = int(np.ravel(archive["seed"])[0])
        result = MetropolisHastingsResult(
            parameter_names=tuple(str(name) for name in archive["parameter_names"].tolist()),
            estimation_draws=np.asarray(archive["estimation_draws"], dtype=np.float64),
            parameter_draws=np.asarray(archive["parameter_draws"], dtype=np.float64),
            log_posterior=np.asarray(archive["log_posterior"], dtype=np.float64),
            accepted=np.asarray(archive["accepted"], dtype=bool),
            acceptance_rate=float(np.ravel(archive["acceptance_rate"])[0]),
            proposal_covariance=np.asarray(archive["proposal_covariance"], dtype=np.float64),
            seed=None if seed_value < 0 else seed_value,
            burnin=int(np.ravel(archive["burnin"])[0]),
            n_blocks=int(np.ravel(archive["n_blocks"])[0]) if "n_blocks" in archive else 1,
            n_param_blocks=int(np.ravel(archive["n_param_blocks"])[0])
            if "n_param_blocks" in archive
            else 1,
            mhthin=int(np.ravel(archive["mhthin"])[0]) if "mhthin" in archive else 1,
            burnin_blocks=int(np.ravel(archive["burnin_blocks"])[0])
            if "burnin_blocks" in archive
            else 0,
            proposal_scale=float(np.ravel(archive["proposal_scale"])[0])
            if "proposal_scale" in archive
            else 1.0,
            adaptive_accept=bool(np.ravel(archive["adaptive_accept"])[0])
            if "adaptive_accept" in archive
            else False,
            target_accept=float(np.ravel(archive["target_accept"])[0])
            if "target_accept" in archive
            else 0.25,
            alpha=float(np.ravel(archive["alpha"])[0]) if "alpha" in archive else 1.0,
            c=float(np.ravel(archive["c"])[0]) if "c" in archive else 0.5,
        )
    validate_sampler_result(result)
    return result


def validate_sampler_result(sampler: MetropolisHastingsResult) -> None:
    n_draws = sampler.estimation_draws.shape[0]
    n_parameters = len(sampler.parameter_names)
    if n_parameters == 0:
        msg = "Sampler archive must contain at least one parameter name."
        raise ValueError(msg)
    if sampler.estimation_draws.ndim != 2:
        msg = "Sampler estimation_draws must be a 2D array."
        raise ValueError(msg)
    if sampler.estimation_draws.shape[1] != n_parameters:
        msg = "Sampler estimation_draws column count must match parameter_names."
        raise ValueError(msg)
    if sampler.parameter_draws.shape != sampler.estimation_draws.shape:
        msg = "Sampler parameter_draws must have the same shape as estimation_draws."
        raise ValueError(msg)
    if sampler.log_posterior.shape != (n_draws,):
        msg = f"Sampler log_posterior must have shape {(n_draws,)}."
        raise ValueError(msg)
    if sampler.accepted.shape != (n_draws,):
        msg = f"Sampler accepted must have shape {(n_draws,)}."
        raise ValueError(msg)
    if sampler.proposal_covariance.shape != (n_parameters, n_parameters):
        msg = f"Sampler proposal_covariance must have shape {(n_parameters, n_parameters)}."
        raise ValueError(msg)
    if not np.allclose(sampler.proposal_covariance, sampler.proposal_covariance.T):
        msg = "Sampler proposal_covariance must be symmetric."
        raise ValueError(msg)
    if not np.isfinite(sampler.proposal_scale) or sampler.proposal_scale <= 0.0:
        msg = "Sampler proposal_scale must be positive."
        raise ValueError(msg)
    if sampler.n_blocks < 1:
        msg = "Sampler n_blocks must be positive."
        raise ValueError(msg)
    if sampler.n_param_blocks < 1:
        msg = "Sampler n_param_blocks must be positive."
        raise ValueError(msg)
    if sampler.n_param_blocks > len(sampler.parameter_names):
        msg = "Sampler n_param_blocks cannot exceed number of parameters."
        raise ValueError(msg)
    if sampler.mhthin < 1:
        msg = "Sampler mhthin must be positive."
        raise ValueError(msg)
    if sampler.burnin_blocks < 0 or sampler.burnin_blocks > sampler.n_blocks:
        msg = "Sampler burnin_blocks must be between 0 and n_blocks."
        raise ValueError(msg)
    if sampler.adaptive_accept and not 0.0 <= sampler.target_accept <= 1.0:
        msg = "Sampler target_accept must be between 0 and 1."
        raise ValueError(msg)
    if not 0.0 <= sampler.alpha <= 1.0:
        msg = "Sampler alpha must be in [0, 1]."
        raise ValueError(msg)
    if sampler.c <= 0.0:
        msg = "Sampler c must be positive."
        raise ValueError(msg)
    if not np.isfinite(sampler.acceptance_rate) or not 0.0 <= sampler.acceptance_rate <= 1.0:
        msg = "Sampler acceptance_rate must be finite and between 0 and 1."
        raise ValueError(msg)
    if sampler.burnin < 0:
        msg = "Sampler burnin must be nonnegative."
        raise ValueError(msg)


def sampler_diagnostics(
    sampler: MetropolisHastingsResult,
    *,
    windows: int = 4,
) -> SamplerDiagnostics:
    validate_sampler_result(sampler)
    if windows <= 0:
        msg = "Sampler diagnostics windows must be positive."
        raise ValueError(msg)
    draws = int(sampler.estimation_draws.shape[0])
    if draws <= 0:
        msg = "Sampler diagnostics require at least one retained draw."
        raise ValueError(msg)
    accepted_draws = int(np.count_nonzero(sampler.accepted))
    eigenvalues = np.linalg.eigvalsh(sampler.proposal_covariance)
    min_eigenvalue = float(np.min(eigenvalues))
    max_eigenvalue = float(np.max(eigenvalues))
    if min_eigenvalue <= 0.0:
        condition_number = float("inf")
    else:
        condition_number = float(max_eigenvalue / min_eigenvalue)
    return SamplerDiagnostics(
        parameter_names=sampler.parameter_names,
        draws=draws,
        burnin=sampler.burnin,
        seed=sampler.seed,
        accepted_draws=accepted_draws,
        acceptance_rate=float(sampler.acceptance_rate),
        realized_acceptance_rate=float(accepted_draws / draws) if draws else 0.0,
        acceptance_windows=_acceptance_windows(sampler.accepted, windows=windows),
        proposal_covariance_shape=sampler.proposal_covariance.shape,
        proposal_covariance_min_eigenvalue=min_eigenvalue,
        proposal_covariance_max_eigenvalue=max_eigenvalue,
        proposal_covariance_condition_number=condition_number,
        proposal_covariance_positive_semidefinite=bool(min_eigenvalue >= -1.0e-12),
        log_posterior_mean=float(np.mean(sampler.log_posterior)),
        log_posterior_minimum=float(np.min(sampler.log_posterior)),
        log_posterior_maximum=float(np.max(sampler.log_posterior)),
        parameters=tuple(
            _sampler_parameter_diagnostics(
                name,
                sampler.parameter_draws[:, parameter_index],
            )
            for parameter_index, name in enumerate(sampler.parameter_names)
        ),
    )


def _acceptance_windows(accepted: np.ndarray, *, windows: int) -> tuple[float, ...]:
    if accepted.size == 0:
        return ()
    chunks = np.array_split(np.asarray(accepted, dtype=bool), min(windows, accepted.size))
    return tuple(float(np.mean(chunk)) for chunk in chunks if chunk.size)


def _sampler_parameter_diagnostics(
    name: str,
    draws: np.ndarray,
) -> SamplerParameterDiagnostics:
    values = np.asarray(draws, dtype=np.float64)
    ess, iat = _effective_sample_size(values)
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    if ess <= 0.0 or not np.isfinite(ess):
        mcse = float("inf")
    else:
        mcse = float(std / np.sqrt(ess))
    return SamplerParameterDiagnostics(
        name=name,
        mean=float(np.mean(values)),
        std=std,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        effective_sample_size=ess,
        integrated_autocorrelation_time=iat,
        monte_carlo_standard_error=mcse,
        split_rhat=_split_rhat(values),
    )


def _effective_sample_size(values: np.ndarray) -> tuple[float, float]:
    sample = np.asarray(values, dtype=np.float64)
    n_draws = int(sample.size)
    if n_draws <= 1:
        return float(n_draws), 1.0
    centered = sample - float(np.mean(sample))
    variance = float(np.dot(centered, centered) / n_draws)
    if not np.isfinite(variance) or variance <= np.finfo(np.float64).eps:
        return float(n_draws), 1.0
    autocorrelation_sum = 0.0
    for lag in range(1, n_draws):
        autocorrelation = float(
            np.dot(centered[:-lag], centered[lag:]) / ((n_draws - lag) * variance)
        )
        if not np.isfinite(autocorrelation) or autocorrelation <= 0.0:
            break
        autocorrelation_sum += autocorrelation
    iat = max(1.0, 1.0 + 2.0 * autocorrelation_sum)
    return float(min(n_draws, n_draws / iat)), float(iat)


def _split_rhat(values: np.ndarray) -> float | None:
    sample = np.asarray(values, dtype=np.float64)
    n_draws = int(sample.size)
    if n_draws < 4:
        return None
    half = n_draws // 2
    chains = np.vstack([sample[:half], sample[-half:]])
    chain_variances = np.var(chains, axis=1, ddof=1)
    within_variance = float(np.mean(chain_variances))
    chain_means = np.mean(chains, axis=1)
    between_variance = float(half * np.var(chain_means, ddof=1))
    if not np.isfinite(within_variance) or not np.isfinite(between_variance):
        return None
    if within_variance <= np.finfo(np.float64).eps:
        return 1.0 if between_variance <= np.finfo(np.float64).eps else None
    variance_estimate = ((half - 1.0) / half) * within_variance + between_variance / half
    if variance_estimate < 0.0:
        return None
    return float(np.sqrt(variance_estimate / within_variance))


def _evaluate_log_posterior(
    model: DSGEModel,
    observations: np.ndarray,
    *,
    start_date: Any | None = None,
) -> tuple[float, float, float, KalmanResult]:
    system = compute_system(model)
    kalman = kalman_log_likelihood(
        system,
        observations,
        process_covariances=model_process_covariances(
            model,
            system,
            observations.shape[0],
            start_date=start_date,
        ),
        backend=get_backend(model.runtime),
    )
    log_prior = model_log_prior(model.parameters)
    log_posterior = kalman.log_likelihood + log_prior
    return float(log_posterior), float(kalman.log_likelihood), float(log_prior), kalman


def _negative_log_posterior_objective(
    model: DSGEModel,
    observations: np.ndarray,
    parameter_names: tuple[str, ...],
    *,
    start_date: Any | None = None,
) -> Any:
    original_parameters = dict(model.parameters)

    def objective(values: np.ndarray) -> float:
        log_posterior = _log_posterior_for_estimation_values(
            model,
            observations,
            original_parameters,
            parameter_names,
            np.asarray(values, dtype=np.float64),
            start_date=start_date,
        )
        if not np.isfinite(log_posterior):
            return float("inf")
        return float(-log_posterior)

    return objective


def _log_posterior_for_estimation_values(
    model: DSGEModel,
    observations: np.ndarray,
    original_parameters: dict[str, Parameter],
    parameter_names: tuple[str, ...],
    values: np.ndarray,
    *,
    start_date: Any | None = None,
) -> float:
    try:
        _set_parameter_estimation_vector(
            model,
            original_parameters,
            parameter_names,
            np.asarray(values, dtype=np.float64),
        )
        log_posterior, _, _, _ = _evaluate_log_posterior(
            model,
            observations,
            start_date=start_date,
        )
        if not np.isfinite(log_posterior):
            return float("-inf")
        return float(log_posterior)
    except Exception:
        return float("-inf")


def _set_parameter_estimation_vector(
    model: DSGEModel,
    original_parameters: dict[str, Parameter],
    parameter_names: tuple[str, ...],
    values: np.ndarray,
) -> None:
    if values.shape != (len(parameter_names),):
        msg = f"Parameter vector must have shape {(len(parameter_names),)}."
        raise ValueError(msg)
    for name, value in zip(parameter_names, values, strict=True):
        model.parameters[name] = update_parameter_value(original_parameters[name], float(value))


def _model_values_for_estimation_vector(
    original_parameters: dict[str, Parameter],
    parameter_names: tuple[str, ...],
    values: np.ndarray,
) -> np.ndarray:
    if values.shape != (len(parameter_names),):
        msg = f"Parameter vector must have shape {(len(parameter_names),)}."
        raise ValueError(msg)
    return np.asarray(
        [
            update_parameter_value(original_parameters[name], float(value)).value
            for name, value in zip(parameter_names, values, strict=True)
        ],
        dtype=np.float64,
    )


def _proposal_covariance(
    proposal_covariance: np.ndarray | None,
    *,
    n_parameters: int,
    scale: float,
) -> np.ndarray:
    if proposal_covariance is None:
        covariance = np.eye(n_parameters, dtype=np.float64)
    else:
        covariance = np.asarray(proposal_covariance, dtype=np.float64)
    if covariance.shape != (n_parameters, n_parameters):
        msg = f"Proposal covariance must have shape {(n_parameters, n_parameters)}."
        raise ValueError(msg)
    if not np.allclose(covariance, covariance.T):
        msg = "Proposal covariance must be symmetric."
        raise ValueError(msg)
    scaled = covariance * (scale * scale)
    try:
        np.linalg.cholesky(scaled)
    except np.linalg.LinAlgError as err:
        msg = "Proposal covariance must be positive definite."
        raise ValueError(msg) from err
    return scaled


def _optimization_result(
    parameter_names: tuple[str, ...],
    best: np.ndarray,
    result: OptimizeResult,
) -> OptimizationResult:
    return OptimizationResult(
        parameter_names=parameter_names,
        estimation_values=best.copy(),
        objective_value=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        iterations=_optional_int(getattr(result, "nit", None)),
        function_evaluations=_optional_int(getattr(result, "nfev", None)),
    )


def _optimization_result_from_mode(mode: EstimationModeResult) -> OptimizationResult:
    return OptimizationResult(
        parameter_names=mode.parameter_names,
        estimation_values=mode.estimation_values.copy(),
        objective_value=mode.objective_value,
        success=mode.success,
        message=mode.message,
        iterations=mode.iterations,
        function_evaluations=mode.function_evaluations,
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _parameter_transform(parameter: Parameter) -> TransformName:
    match parameter.transform:
        case "identity" | "untransformed" | "exponential" | "sqrt" | "square_root":
            return cast(TransformName, parameter.transform)
        case _:
            msg = f"Unsupported parameter transform: {parameter.transform}"
            raise ValueError(msg)
