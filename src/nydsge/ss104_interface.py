"""Native ss104 entry point for fixed-point evaluation and bounded mode searches.

This does not certify optimizer convergence, posterior covariance or a posterior
sample. Model-space modes have no transformation Jacobian (unlike MH targets).
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np
from scipy.optimize import minimize

from nydsge.core import NotPortedError
from nydsge.ss104_estimation import (
    evaluate_ss104_posterior,
    ss104_estimation_bounds,
    ss104_estimation_vector,
    ss104_parameter_catalog,
    update_ss104_parameters,
)


def estimate_ss104(
    model: Any,
    data: Any,
    *,
    optimize: bool = False,
    parameter_names: list[str] | tuple[str, ...] | None = None,
    optimizer_method: str = "Nelder-Mead",
    maxiter: int = 100,
    compute_hessian: bool = False,
    hessian_step: float = 1e-4,
    mode: Any = None,
):
    """Return the shared EstimateResult; caller model is unchanged on every exit.

    Data must have explicit quarterly dates, including presample observations.
    Coordinates are the native free parameter-regime labels, e.g. sigma_g@1.
    Saved modes must be explicitly successful before they can be reused here.
    """
    from nydsge.estimate import (
        EstimateResult,
        _optimization_result,
        _optimization_result_from_mode,
        finite_difference_hessian,
        validate_estimation_mode,
    )

    if maxiter <= 0 or not np.isfinite(hessian_step) or hessian_step <= 0:
        raise ValueError("Require positive maxiter and finite positive Hessian step.")
    if compute_hessian and not optimize:
        raise ValueError("compute_hessian requires optimize=True.")
    if optimizer_method not in ("Nelder-Mead", "BFGS", "L-BFGS-B"):
        raise ValueError("ss104 supports native Nelder-Mead, BFGS or L-BFGS-B searches.")
    working = deepcopy(model)
    catalog = ss104_parameter_catalog(working)
    labels = tuple(catalog) if parameter_names is None else tuple(parameter_names)
    if not labels or len(set(labels)) != len(labels) or set(labels) - set(catalog):
        raise ValueError("Use unique free ss104 parameter-regime coordinates.")
    selected = OrderedDict((name, catalog[name]) for name in labels)
    if parameter_names is not None and not optimize and mode is None:
        raise ValueError("parameter_names require optimization or an explicit mode.")
    optimization = None
    hessian = None
    if mode is not None:
        validate_estimation_mode(mode)
        if optimize or tuple(mode.parameter_names) != labels:
            raise ValueError("Saved ss104 mode coordinates differ or optimize=True.")
        if not mode.success:
            raise ValueError("Unconverged saved modes cannot be adopted as validated modes.")
        update_ss104_parameters(working, selected, mode.estimation_values)
        optimization = _optimization_result_from_mode(mode)
        hessian = None if mode.hessian is None else np.array(mode.hessian, copy=True)
    if optimize:
        start = ss104_estimation_vector(selected)

        def objective(values):
            try:
                update_ss104_parameters(working, selected, values)
                posterior, _, _, _ = evaluate_ss104_posterior(working, data)
                return -float(posterior) if np.isfinite(posterior) else float("inf")
            except (ValueError, np.linalg.LinAlgError, OverflowError, NotPortedError):
                return float("inf")

        if not np.isfinite(objective(start)):
            raise ValueError("Initial ss104 posterior is not finite.")
        result = minimize(
            objective,
            start,
            method=optimizer_method,
            bounds=ss104_estimation_bounds(selected) if optimizer_method == "L-BFGS-B" else None,
            options={"maxiter": maxiter},
        )
        best = np.asarray(result.x, dtype=float)
        if not np.isfinite(best).all() or not np.isfinite(result.fun):
            raise ValueError("Native ss104 search returned no finite candidate.")
        optimization = _optimization_result(labels, best, result)
        if compute_hessian:
            hessian = finite_difference_hessian(objective, best, step=hessian_step)
            if not np.isfinite(hessian).all():
                raise ValueError("Nonfinite ss104 finite-difference Hessian.")
        update_ss104_parameters(working, selected, best)
    posterior, likelihood, prior, filtered = evaluate_ss104_posterior(working, data)
    if not np.isfinite([posterior, likelihood, prior]).all():
        raise ValueError("Nonfinite ss104 posterior result.")
    values = {name: float(p.value) for name, p in working.parameters.items()}
    values.update({name: float(p.value) for name, p in ss104_parameter_catalog(working).items()})
    return EstimateResult(
        log_posterior=float(posterior),
        log_likelihood=float(likelihood),
        log_prior=float(prior),
        parameter_values=values,
        kalman=filtered,
        optimization=optimization,
        hessian=hessian,
    )
