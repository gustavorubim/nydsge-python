"""Native ss104 supplied-parameter and future-innovation forecast ensembles.

These are conditional on fitted terminal states, the supplied data vintage and
regime/policy specification. QQ draws include economic shocks and measurement-state
process innovations (for example corepce_sh, meas_pi_sh and exp_rm_sh). EE
observation noise, initial/terminal-state uncertainty and model uncertainty are
not sampled. Supplied parameter draws are not certified as a posterior.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd

from nydsge.forecast import forecast_linear_system
from nydsge.kalman import model_shock_covariance
from nydsge.parameters import TransformName, transform_to_estimation_space
from nydsge.regimes import RegimeSwitchingSystem, regime_indices_from_dates
from nydsge.solve import compute_system
from nydsge.ss104_estimation import (
    ss104_estimation_vector,
    ss104_parameter_catalog,
    update_ss104_parameters,
)
from nydsge.ss104_workflow import forecast_ss104


def _parameter_model(model, row, names):
    """Validate model-space coordinates and update an isolated model instance."""
    candidate = deepcopy(model)
    catalog = ss104_parameter_catalog(candidate)
    values = ss104_estimation_vector(catalog)
    positions = {name: i for i, name in enumerate(catalog)}
    for name, value in zip(names, row, strict=True):
        if name not in catalog:
            raise ValueError(f"Unknown/free ss104 parameter coordinate: {name}")
        parameter = catalog[name]
        bounds = parameter.value_bounds
        if bounds is not None and not bounds[0] <= value <= bounds[1]:
            raise ValueError(f"Out-of-bounds model-space parameter: {name}")
        values[positions[name]] = transform_to_estimation_space(
            float(value), cast(TransformName, parameter.transform), bounds=bounds
        )
    update_ss104_parameters(candidate, catalog, values)
    return candidate


def _future_shocks(model, system, prediction, rng):
    """Sample dated model-innovation QQ, including measurement-state processes.

    This samples process innovations, not independent EE observation noise.
    Singular and zero innovation variances are retained.
    """
    start = pd.Period(model.get_setting("date_forecast_start"), freq="Q")
    if not isinstance(start, pd.Period):
        raise ValueError("Invalid forecast start quarter.")
    count_cond = 0 if prediction.conditional_states is None else len(prediction.conditional_states)
    dates = pd.period_range(start + count_cond, periods=len(prediction.states), freq="Q")
    path = regime_indices_from_dates(model.get_setting("regime_dates"), dates)
    result = []
    for date, regime in zip(dates, path, strict=True):
        covariance = model_shock_covariance(model, system[regime], date=date)
        eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
        if np.min(eigenvalues) < -1e-10:
            raise ValueError("Forecast shock covariance is not positive semidefinite.")
        root = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
        result.append(root @ rng.standard_normal(len(eigenvalues)))
    return np.asarray(result).reshape(len(dates), system[1, "RRR"].shape[1])


def forecast_ss104_distribution(
    model,
    *,
    parameter_draws=None,
    parameter_names=None,
    parameter_draw_provenance=None,
    draw_future_shocks=False,
    draws=0,
    seed=None,
    shock_samples=None,
    **forecast_kwargs,
):
    """Run one native forecast per supplied row or fixed-parameter innovation draw.

    ``parameter_names`` are explicit free catalog labels (including ``name@regime``),
    in model-space units. Future innovation arrays have shape
    ``(draws, unconditioned_future_quarters, structural_shocks)``. Conditioning
    observations are assimilated independently for each parameter row; future
    innovations begin after the conditional window. Rejected rows fail the whole
    call, never silently change the empirical distribution.
    """
    if not isinstance(draws, (int, np.integer)) or draws < 0:
        raise ValueError("draws must be a nonnegative integer.")
    rows = None if parameter_draws is None else np.asarray(parameter_draws, dtype=np.float64)
    names = tuple(() if parameter_names is None else parameter_names)
    if rows is not None:
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != len(names) or not names:
            raise ValueError("parameter_draws need nonempty rows and explicit matching names.")
        if len(set(names)) != len(names) or not np.isfinite(rows).all():
            raise ValueError("Parameter names must be unique and values finite.")
        if not isinstance(parameter_draw_provenance, str) or not parameter_draw_provenance.strip():
            raise ValueError(
                "Explicit parameter_draw_provenance is required; posterior validity is external."
            )
        if draws not in (0, len(rows)):
            raise ValueError("draws must match supplied parameter rows.")
        count = len(rows)
    else:
        if names or parameter_draw_provenance is not None:
            raise ValueError("Parameter metadata requires parameter_draws.")
        count = draws
    samples = None if shock_samples is None else np.asarray(shock_samples, dtype=np.float64)
    if samples is not None:
        if draw_future_shocks:
            raise ValueError("Use explicit shock_samples or generated future shocks, not both.")
        if samples.ndim != 3 or samples.shape[0] == 0 or not np.isfinite(samples).all():
            raise ValueError("shock_samples must be a finite nonempty 3D array.")
        if count not in (0, len(samples)):
            raise ValueError("Shock and parameter draw counts must match.")
        count = len(samples)
    if count <= 0:
        raise ValueError("Full forecasts require at least one parameter or future-shock draw.")
    # draws alone retains the established high-level full-forecast convention.
    generated = samples is None and (draw_future_shocks or rows is None)
    rng = np.random.default_rng(seed)
    outputs = []
    fixed_system = fixed_baseline = None
    for i in range(count):
        candidate = model if rows is None else _parameter_model(model, rows[i], names)
        if rows is None and fixed_system is not None:
            system, baseline = fixed_system, fixed_baseline
        else:
            system = compute_system(candidate)
            if not isinstance(system, RegimeSwitchingSystem):
                raise TypeError("ss104 requires a regime-switching system.")
            baseline = forecast_ss104(candidate, system=system, **forecast_kwargs)
            if rows is None:
                fixed_system, fixed_baseline = system, baseline
        assert baseline is not None
        shocks = (
            _future_shocks(candidate, system, baseline, rng)
            if generated
            else (samples[i] if samples is not None else None)
        )
        if shocks is not None:
            expected = (len(baseline.states), system[1, "RRR"].shape[1])
            if shocks.shape != expected:
                raise ValueError(f"Future shock samples must have per-draw shape {expected}.")
            initial = forecast_kwargs.get("initial_state")
            if baseline.conditional_states is not None and len(baseline.conditional_states):
                initial = baseline.conditional_states[-1]
            elif baseline.history_states is not None and len(baseline.history_states):
                initial = baseline.history_states[-1]
            if initial is None:
                initial = np.zeros(system[1, "TTT"].shape[0])
            start = pd.Period(candidate.get_setting("date_forecast_start"), freq="Q")
            if not isinstance(start, pd.Period):
                raise ValueError("Invalid forecast start quarter.")
            count_cond = (
                0 if baseline.conditional_states is None else len(baseline.conditional_states)
            )
            dates = pd.period_range(start + count_cond, periods=len(baseline.states), freq="Q")
            path = regime_indices_from_dates(candidate.get_setting("regime_dates"), dates)
            future = forecast_linear_system(
                system,
                initial,
                horizon=len(baseline.states),
                shocks=shocks,
                include_pseudo=True,
                regime_indices=path,
            )
            baseline = replace(
                baseline,
                states=future.states,
                observables=future.observables,
                pseudo_observables=future.pseudo_observables,
            )
        for values in (baseline.states, baseline.observables, baseline.pseudo_observables):
            if values is not None and not np.isfinite(values).all():
                raise ValueError(f"Nonfinite forecast in ensemble row {i}; no rows were dropped.")
        outputs.append(baseline)
    sample_fields = {
        "states": "state_samples",
        "observables": "observable_samples",
        "pseudo_observables": "pseudo_observable_samples",
        "history_states": "history_state_samples",
        "history_observables": "history_observable_samples",
        "history_pseudo_observables": "history_pseudo_observable_samples",
    }
    updates = {}
    for field, sample_field in sample_fields.items():
        if getattr(outputs[0], field) is not None:
            array = np.stack([getattr(output, field) for output in outputs])
            updates[field] = np.mean(array, axis=0)
            updates[sample_field] = array
    for field in ("conditional_states", "conditional_observables"):
        if getattr(outputs[0], field) is not None:
            updates[field] = np.mean([getattr(output, field) for output in outputs], axis=0)
    sources = (("supplied_parameters",) if rows is not None else ()) + (
        ("future_model_innovations",) if generated or samples is not None else ()
    )
    return replace(
        outputs[0],
        **updates,
        log_likelihood=None,
        uncertainty_sources=sources,
        parameter_draw_provenance=parameter_draw_provenance,
    )
