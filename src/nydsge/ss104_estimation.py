"""Dated ss104 posterior evaluation and independent parameter-regime updates.

The model-space posterior is optimized in the existing unbounded parameter
coordinates. Unlike an MH target, a mode objective has no transform Jacobian.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from nydsge.core import Parameter
from nydsge.kalman import kalman_log_likelihood, model_process_covariances
from nydsge.parameters import (
    parameter_log_prior,
    transform_to_estimation_space,
    update_parameter_value,
)
from nydsge.regimes import regime_indices_from_dates
from nydsge.solve import compute_system
from nydsge.ss104_workflow import prepare_ss104_observations


def ss104_parameter_catalog(model) -> OrderedDict[str, Parameter]:
    """One coordinate per free parameter regime, with no duplicated baseline prior."""
    if model.subspec != "ss104":
        raise ValueError("The ss104 parameter catalog requires Model1002 ss104.")
    regimes = model.get_setting("parameter_regimes")
    catalog = OrderedDict()
    for name, baseline in model.parameters.items():
        entries = regimes.get(name, {None: baseline})
        for regime, parameter in entries.items():
            if not parameter.fixed and parameter.prior is not None:
                label = name if regime is None else f"{name}@{regime}"
                catalog[label] = parameter
    return catalog


def ss104_estimation_vector(catalog) -> np.ndarray:
    return np.array(
        [
            transform_to_estimation_space(p.value, p.transform, bounds=p.value_bounds)
            for p in catalog.values()
        ]
    )


def ss104_estimation_bounds(catalog):
    """Model bounds expressed in the existing optimization coordinates."""
    result = []
    for p in catalog.values():
        if p.value_bounds is None or p.transform in ("sqrt", "square_root"):
            result.append((None, None))
        elif p.transform == "exponential":
            lower, upper = p.value_bounds
            result.append(
                (
                    float(np.log(lower)) if lower > 0 else None,
                    float(np.log(upper)) if np.isfinite(upper) else None,
                )
            )
        else:
            result.append(p.value_bounds)
    return result


def ss104_projected_gradient(catalog, values, gradient):
    """KKT gradient: a boundary coordinate need not have zero outward derivative."""
    projected = np.asarray(gradient, dtype=float).copy()
    for i, (lower, upper) in enumerate(ss104_estimation_bounds(catalog)):
        if (lower is not None and values[i] <= lower + 1e-8 and projected[i] > 0) or (
            upper is not None and values[i] >= upper - 1e-8 and projected[i] < 0
        ):
            projected[i] = 0
    return projected


def update_ss104_parameters(model, catalog, values) -> None:
    """Update all free coordinates from immutable originals; fixed regimes remain fixed."""
    values = np.asarray(values, dtype=float)
    if values.shape != (len(catalog),) or not np.isfinite(values).all():
        raise ValueError("Invalid ss104 estimation vector.")
    updated = {
        k: update_parameter_value(p, float(x))
        for (k, p), x in zip(catalog.items(), values, strict=True)
    }
    for p in updated.values():
        if not np.isfinite(p.value) or (
            p.value_bounds is not None and not p.value_bounds[0] <= p.value <= p.value_bounds[1]
        ):
            raise ValueError(f"ss104 parameter out of bounds: {p.name}")
    regimes = {name: dict(path) for name, path in model.get_setting("parameter_regimes").items()}
    for label, parameter in updated.items():
        if "@" in label:
            name, regime = label.rsplit("@", 1)
            regimes[name][int(regime)] = parameter
            if int(regime) == 1:
                model.parameters[name] = parameter
        else:
            model.parameters[label] = parameter
    model.set_setting("parameter_regimes", regimes)


def evaluate_ss104_posterior(model, data, *, system=None):
    """Return log posterior, likelihood, prior and filter result, excluding presample scoring."""
    frame = prepare_ss104_observations(model, data)
    forecast_start = pd.Period(model.get_setting("date_forecast_start"), freq="Q")
    if frame.empty or frame.index[-1] >= forecast_start:
        raise ValueError(
            "ss104 estimation requires nonempty historical data before forecast start."
        )
    prior = float(sum(parameter_log_prior(p) for p in ss104_parameter_catalog(model).values()))
    if not np.isfinite(prior):
        raise ValueError("Nonfinite ss104 prior.")
    system = compute_system(model) if system is None else system
    indices = regime_indices_from_dates(model.get_setting("regime_dates"), frame.index)
    presample = int(
        (frame.index < pd.Period(model.get_setting("date_mainsample_start"), freq="Q")).sum()
    )
    filtered = kalman_log_likelihood(
        system,
        frame.to_numpy(dtype=float),
        regime_indices=indices,
        log_likelihood_start=presample,
        process_covariances=model_process_covariances(
            model, system, len(frame), start_date=frame.index[0]
        ),
    )
    return filtered.log_likelihood + prior, filtered.log_likelihood, prior, filtered
