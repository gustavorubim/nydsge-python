from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nydsge.core import DSGEModel, Parameter
from nydsge.estimate import (
    EstimateResult,
    MetropolisHastingsResult,
    evaluate_log_posterior_for_parameter_values,
    sampler_diagnostics,
    validate_sampler_result,
)
from nydsge.forecast import ForecastOutput, MeansBands
from nydsge.kalman import KalmanResult
from nydsge.parameters import parameter_log_prior
from nydsge.solve import CanonicalSolveResult, CanonicalSystem, System

FixtureLabels = dict[str, dict[int, tuple[str, ...]]]


@dataclass(frozen=True)
class ToleranceProfile:
    name: str
    atol: float
    rtol: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArrayComparison:
    name: str
    status: str
    expected_shape: tuple[int, ...] | None
    actual_shape: tuple[int, ...] | None
    max_abs_diff: float | None
    max_rel_diff: float | None
    atol: float
    rtol: float
    max_abs_index: tuple[int, ...] | None = None
    max_abs_label: tuple[str | None, ...] | None = None
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FixtureComparisonReport:
    oracle_dir: Path
    candidate_dir: Path
    comparisons: tuple[ArrayComparison, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "oracle_dir": str(self.oracle_dir),
            "candidate_dir": str(self.candidate_dir),
            "passed": self.passed,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


@dataclass(frozen=True)
class FixtureCoverageReport:
    fixture_dir: Path
    profile: str
    required: tuple[str, ...]
    available: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_dir": str(self.fixture_dir),
            "profile": self.profile,
            "passed": self.passed,
            "required": list(self.required),
            "available": list(self.available),
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class SamplerComparisonReport:
    oracle_sampler: Path
    candidate_sampler: Path
    comparisons: tuple[ArrayComparison, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "oracle_sampler": str(self.oracle_sampler),
            "candidate_sampler": str(self.candidate_sampler),
            "passed": self.passed,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


@dataclass(frozen=True)
class SamplerProposalTraceReport:
    sampler_path: Path
    comparisons: tuple[ArrayComparison, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler_path": str(self.sampler_path),
            "passed": self.passed,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


@dataclass(frozen=True)
class SamplerPosteriorReplayReport:
    sampler_path: Path
    draws: int
    parameter_count: int
    posterior_offset_mean: float | None
    posterior_offset_std: float | None
    comparisons: tuple[ArrayComparison, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler_path": str(self.sampler_path),
            "draws": self.draws,
            "parameter_count": self.parameter_count,
            "posterior_offset_mean": self.posterior_offset_mean,
            "posterior_offset_std": self.posterior_offset_std,
            "passed": self.passed,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


@dataclass(frozen=True)
class SamplerFixtureSummary:
    fixture_path: Path
    parameter_names: tuple[str, ...]
    mhparams_shape: tuple[int, int]
    parameter_axis: int
    draw_axis: int
    draws: int
    parameter_count: int
    covariance_shape: tuple[int, int]
    covariance_source: str
    covariance_min_eigenvalue: float
    covariance_max_eigenvalue: float
    covariance_condition_number: float
    covariance_positive_semidefinite: bool
    input_proposal_covariance_available: bool
    trace_available: bool
    accepted_shape: tuple[int, ...] | None
    log_posterior_shape: tuple[int, ...] | None
    accepted_draws: int | None
    realized_acceptance_rate: float | None
    log_posterior_mean: float | None
    log_posterior_minimum: float | None
    log_posterior_maximum: float | None
    proposal_trace_available: bool
    proposal_parameters_shape: tuple[int, ...] | None
    previous_parameters_shape: tuple[int, ...] | None
    proposal_log_posterior_shape: tuple[int, ...] | None
    previous_log_posterior_shape: tuple[int, ...] | None
    uniform_draw_shape: tuple[int, ...] | None
    log_acceptance_shape: tuple[int, ...] | None
    proposal_log_posterior_minimum: float | None
    proposal_log_posterior_maximum: float | None
    log_acceptance_minimum: float | None
    log_acceptance_maximum: float | None
    metadata: dict[str, Any]
    unavailable_diagnostics: tuple[str, ...]
    unavailable_proposal_diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_path": str(self.fixture_path),
            "parameter_names": list(self.parameter_names),
            "mhparams_shape": list(self.mhparams_shape),
            "parameter_axis": self.parameter_axis,
            "draw_axis": self.draw_axis,
            "draws": self.draws,
            "parameter_count": self.parameter_count,
            "covariance_shape": list(self.covariance_shape),
            "covariance_source": self.covariance_source,
            "covariance_min_eigenvalue": self.covariance_min_eigenvalue,
            "covariance_max_eigenvalue": self.covariance_max_eigenvalue,
            "covariance_condition_number": self.covariance_condition_number,
            "covariance_positive_semidefinite": self.covariance_positive_semidefinite,
            "input_proposal_covariance_available": self.input_proposal_covariance_available,
            "trace_available": self.trace_available,
            "accepted_shape": None if self.accepted_shape is None else list(self.accepted_shape),
            "log_posterior_shape": None
            if self.log_posterior_shape is None
            else list(self.log_posterior_shape),
            "accepted_draws": self.accepted_draws,
            "realized_acceptance_rate": self.realized_acceptance_rate,
            "log_posterior_mean": self.log_posterior_mean,
            "log_posterior_minimum": self.log_posterior_minimum,
            "log_posterior_maximum": self.log_posterior_maximum,
            "proposal_trace_available": self.proposal_trace_available,
            "proposal_parameters_shape": None
            if self.proposal_parameters_shape is None
            else list(self.proposal_parameters_shape),
            "previous_parameters_shape": None
            if self.previous_parameters_shape is None
            else list(self.previous_parameters_shape),
            "proposal_log_posterior_shape": None
            if self.proposal_log_posterior_shape is None
            else list(self.proposal_log_posterior_shape),
            "previous_log_posterior_shape": None
            if self.previous_log_posterior_shape is None
            else list(self.previous_log_posterior_shape),
            "uniform_draw_shape": None
            if self.uniform_draw_shape is None
            else list(self.uniform_draw_shape),
            "log_acceptance_shape": None
            if self.log_acceptance_shape is None
            else list(self.log_acceptance_shape),
            "proposal_log_posterior_minimum": self.proposal_log_posterior_minimum,
            "proposal_log_posterior_maximum": self.proposal_log_posterior_maximum,
            "log_acceptance_minimum": self.log_acceptance_minimum,
            "log_acceptance_maximum": self.log_acceptance_maximum,
            "metadata": self.metadata,
            "unavailable_diagnostics": list(self.unavailable_diagnostics),
            "unavailable_proposal_diagnostics": list(self.unavailable_proposal_diagnostics),
        }


TOLERANCE_PROFILES: dict[str, ToleranceProfile] = {
    "strict": ToleranceProfile(
        name="strict",
        atol=1.0e-10,
        rtol=1.0e-10,
        description="CPU oracle and matrix parity.",
    ),
    "cpu-oracle": ToleranceProfile(
        name="cpu-oracle",
        atol=1.0e-10,
        rtol=1.0e-10,
        description="Alias for strict CPU oracle parity.",
    ),
    "forecast": ToleranceProfile(
        name="forecast",
        atol=1.0e-8,
        rtol=1.0e-8,
        description="Forecast and means/bands parity after solver/filter accumulation.",
    ),
    "accelerator": ToleranceProfile(
        name="accelerator",
        atol=1.0e-5,
        rtol=1.0e-5,
        description="CUDA/MPS/JAX/Torch accelerator parity against NumPy CPU.",
    ),
}


MODEL_SETUP_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "parameters/values",
    "parameters/scaled_values",
    "parameters/fixed",
    "parameters/bounds",
    "steady_state/values",
)

MATRIX_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "canonical/Gamma0",
    "canonical/Gamma1",
    "canonical/C",
    "canonical/Psi",
    "canonical/Pi",
    "transition/TTT",
    "transition/RRR",
    "transition/CCC",
    "transition/eu",
    "system/TTT",
    "system/RRR",
    "system/CCC",
    "system/ZZ",
    "system/DD",
    "system/QQ",
    "system/EE",
)

FORECAST_MODE_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "forecast_mode/states",
    "forecast_mode/observables",
    "meansbands_mode_forecastobs/mean",
    "meansbands_mode_forecastobs/lower",
    "meansbands_mode_forecastobs/upper",
)

FORECAST_MODE_HISTORY_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "forecast_mode/history_observables",
    "meansbands_mode_histobs/mean",
    "meansbands_mode_histobs/lower",
    "meansbands_mode_histobs/upper",
)

FORECAST_FULL_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "forecast_full/observables",
    "forecast_full/observable_samples",
    "meansbands_full_forecastobs/mean",
    "meansbands_full_forecastobs/lower",
    "meansbands_full_forecastobs/upper",
)

FORECAST_FULL_HISTORY_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "forecast_full/history_observables",
    "forecast_full/history_observable_samples",
    "meansbands_full_histobs/mean",
    "meansbands_full_histobs/lower",
    "meansbands_full_histobs/upper",
)

PARAMETER_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "parameters/values",
    "parameters/scaled_values",
    "parameters/fixed",
    "parameters/bounds",
)

STEADY_STATE_FIXTURE_REQUIREMENTS: tuple[str, ...] = ("steady_state/values",)

FINANCIAL_FRICTIONS_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "financial_frictions/inputs",
    "financial_frictions/values",
)

KALMAN_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "kalman/log_likelihood",
    "kalman/predicted_states",
    "kalman/filtered_states",
    "kalman/predicted_covariances",
    "kalman/filtered_covariances",
    "kalman/final_filtered_state",
    "kalman/total_log_likelihood",
)

POSTERIOR_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "posterior/log_posterior",
    "posterior/log_likelihood",
    "posterior/log_prior",
    "posterior/log_likelihood_by_period",
    "posterior/log_prior_by_parameter",
    "posterior/parameter_values",
)

HARD_TARGET_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    *MODEL_SETUP_FIXTURE_REQUIREMENTS,
    *MATRIX_FIXTURE_REQUIREMENTS,
    *POSTERIOR_FIXTURE_REQUIREMENTS,
    "forecast_mode/observables",
    "forecast_mode/history_observables",
    "forecast_full/observables",
    "forecast_full/history_observables",
    "forecast_full/observable_samples",
    "forecast_full/history_observable_samples",
    "meansbands_mode_forecastobs/mean",
    "meansbands_mode_forecastobs/lower",
    "meansbands_mode_forecastobs/upper",
    "meansbands_mode_histobs/mean",
    "meansbands_mode_histobs/lower",
    "meansbands_mode_histobs/upper",
    "meansbands_full_forecastobs/mean",
    "meansbands_full_forecastobs/lower",
    "meansbands_full_forecastobs/upper",
    "meansbands_full_histobs/mean",
    "meansbands_full_histobs/lower",
    "meansbands_full_histobs/upper",
)

SAMPLER_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    "sampler/mhparams",
    "sampler/proposal_covariance",
)

SAMPLER_TRACE_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    *SAMPLER_FIXTURE_REQUIREMENTS,
    "sampler/accepted",
    "sampler/log_posterior",
)

SAMPLER_PROPOSAL_TRACE_FIXTURE_REQUIREMENTS: tuple[str, ...] = (
    *SAMPLER_TRACE_FIXTURE_REQUIREMENTS,
    "sampler/proposal_parameters",
    "sampler/previous_parameters",
    "sampler/proposal_log_posterior",
    "sampler/previous_log_posterior",
    "sampler/uniform_draw",
    "sampler/log_acceptance",
)

FIXTURE_REQUIREMENT_PROFILES: dict[str, tuple[str, ...]] = {
    "parameters": PARAMETER_FIXTURE_REQUIREMENTS,
    "steady-state": STEADY_STATE_FIXTURE_REQUIREMENTS,
    "financial-frictions": FINANCIAL_FRICTIONS_FIXTURE_REQUIREMENTS,
    "kalman": KALMAN_FIXTURE_REQUIREMENTS,
    "posterior": POSTERIOR_FIXTURE_REQUIREMENTS,
    "model-setup": MODEL_SETUP_FIXTURE_REQUIREMENTS,
    "matrix": MATRIX_FIXTURE_REQUIREMENTS,
    "forecast-mode": FORECAST_MODE_FIXTURE_REQUIREMENTS,
    "forecast-mode-history": FORECAST_MODE_HISTORY_FIXTURE_REQUIREMENTS,
    "forecast-full": FORECAST_FULL_FIXTURE_REQUIREMENTS,
    "forecast-full-history": FORECAST_FULL_HISTORY_FIXTURE_REQUIREMENTS,
    "hard-target": HARD_TARGET_FIXTURE_REQUIREMENTS,
    "sampler": SAMPLER_FIXTURE_REQUIREMENTS,
    "sampler-trace": SAMPLER_TRACE_FIXTURE_REQUIREMENTS,
    "sampler-proposal-trace": SAMPLER_PROPOSAL_TRACE_FIXTURE_REQUIREMENTS,
}


def resolve_tolerance_profile(
    profile: str = "strict",
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> ToleranceProfile:
    key = profile.casefold()
    if key not in TOLERANCE_PROFILES:
        valid = ", ".join(sorted(TOLERANCE_PROFILES))
        msg = f"Unknown tolerance profile '{profile}'. Valid profiles: {valid}."
        raise ValueError(msg)
    base = TOLERANCE_PROFILES[key]
    resolved_atol = base.atol if atol is None else atol
    resolved_rtol = base.rtol if rtol is None else rtol
    if resolved_atol < 0.0 or resolved_rtol < 0.0:
        msg = "Tolerance values must be nonnegative."
        raise ValueError(msg)
    return ToleranceProfile(
        name=base.name,
        atol=resolved_atol,
        rtol=resolved_rtol,
        description=base.description,
    )


def required_fixture_arrays(profile: str) -> tuple[str, ...]:
    key = profile.casefold()
    if key not in FIXTURE_REQUIREMENT_PROFILES:
        valid = ", ".join(sorted(FIXTURE_REQUIREMENT_PROFILES))
        msg = f"Unknown fixture coverage profile '{profile}'. Valid profiles: {valid}."
        raise ValueError(msg)
    return FIXTURE_REQUIREMENT_PROFILES[key]


def check_fixture_coverage(
    fixture_dir: Path,
    *,
    profile: str = "hard-target",
) -> FixtureCoverageReport:
    required = required_fixture_arrays(profile)
    arrays = load_fixture_arrays(fixture_dir)
    available = tuple(sorted(arrays))
    available_set = set(available)
    missing = tuple(name for name in required if name not in available_set)
    return FixtureCoverageReport(
        fixture_dir=fixture_dir,
        profile=profile.casefold(),
        required=required,
        available=available,
        missing=missing,
    )


def summarize_sampler_fixture(path: Path | str) -> SamplerFixtureSummary:
    fixture_path = _resolve_sampler_fixture_path(Path(path))
    h5py: Any = import_module("h5py")
    with h5py.File(fixture_path, "r") as handle:
        if "sampler/mhparams" not in handle:
            msg = f"Sampler fixture is missing sampler/mhparams: {fixture_path}"
            raise KeyError(msg)
        if "sampler/proposal_covariance" not in handle:
            msg = f"Sampler fixture is missing sampler/proposal_covariance: {fixture_path}"
            raise KeyError(msg)

        mhparams_shape = tuple(int(value) for value in handle["sampler/mhparams"].shape)
        if len(mhparams_shape) != 2:
            msg = f"sampler/mhparams must be two-dimensional, got {mhparams_shape}."
            raise ValueError(msg)
        covariance = np.asarray(handle["sampler/proposal_covariance"][()], dtype=np.float64)
        covariance_shape = tuple(int(value) for value in covariance.shape)
        if len(covariance_shape) != 2 or covariance_shape[0] != covariance_shape[1]:
            msg = f"sampler/proposal_covariance must be square, got {covariance_shape}."
            raise ValueError(msg)

        parameter_names = _parse_hdf5_name_attr(handle.attrs.get("sampler_parameter_names"))
        if not parameter_names:
            parameter_names = tuple(f"parameter_{index}" for index in range(covariance_shape[0]))
        parameter_count = len(parameter_names)
        metadata = _sampler_hdf5_metadata(handle)
        metadata_draws = _metadata_optional_int(metadata.get("draws"))
        parameter_axis, draw_axis, draws = _infer_sampler_mhparams_axes(
            mhparams_shape,
            parameter_count,
            metadata_draws,
        )
        if metadata_draws is not None and metadata_draws != draws:
            msg = (
                "sampler_draws metadata does not match sampler/mhparams shape: "
                f"{metadata_draws} vs inferred {draws}."
            )
            raise ValueError(msg)
        if covariance_shape != (parameter_count, parameter_count):
            msg = (
                "sampler/proposal_covariance shape does not align with "
                f"sampler_parameter_names: {covariance_shape} vs {parameter_count} names."
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(covariance)):
            msg = "sampler/proposal_covariance must contain only finite values."
            raise ValueError(msg)
        if not np.allclose(covariance, covariance.T, atol=1.0e-12, rtol=1.0e-12):
            msg = "sampler/proposal_covariance must be symmetric."
            raise ValueError(msg)

        eigenvalues = np.linalg.eigvalsh(covariance)
        min_eigenvalue = float(np.min(eigenvalues))
        max_eigenvalue = float(np.max(eigenvalues))
        condition_number = (
            float("inf") if min_eigenvalue <= 0.0 else float(max_eigenvalue / min_eigenvalue)
        )
        covariance_source = str(metadata.get("covariance_source") or "saved_draw_covariance")
        input_proposal_dataset_available = "sampler/input_proposal_covariance" in handle
        input_proposal_attr_available = bool(
            metadata.get("input_proposal_covariance_available", False)
        )
        if input_proposal_attr_available and not input_proposal_dataset_available:
            msg = (
                "sampler_input_proposal_covariance_available is true, but "
                "sampler/input_proposal_covariance is missing."
            )
            raise ValueError(msg)
        accepted = _optional_sampler_trace_vector(handle, "sampler/accepted", draws)
        log_posterior = _optional_sampler_trace_vector(
            handle,
            "sampler/log_posterior",
            draws,
        )
        accepted_draws: int | None = None
        realized_acceptance_rate: float | None = None
        log_posterior_mean: float | None = None
        log_posterior_minimum: float | None = None
        log_posterior_maximum: float | None = None
        if accepted is not None:
            accepted_bool = np.asarray(accepted, dtype=bool)
            accepted_draws = int(np.count_nonzero(accepted_bool))
            realized_acceptance_rate = float(accepted_draws / draws) if draws else 0.0
        if log_posterior is not None:
            log_values = np.asarray(log_posterior, dtype=np.float64)
            if not np.all(np.isfinite(log_values)):
                msg = "sampler/log_posterior must contain only finite values."
                raise ValueError(msg)
            log_posterior_mean = float(np.mean(log_values))
            log_posterior_minimum = float(np.min(log_values))
            log_posterior_maximum = float(np.max(log_values))
        proposal_parameters = _optional_sampler_trace_matrix(
            handle,
            "sampler/proposal_parameters",
            mhparams_shape,
        )
        previous_parameters = _optional_sampler_trace_matrix(
            handle,
            "sampler/previous_parameters",
            mhparams_shape,
        )
        proposal_log_posterior = _optional_sampler_trace_vector(
            handle,
            "sampler/proposal_log_posterior",
            draws,
        )
        previous_log_posterior = _optional_sampler_trace_vector(
            handle,
            "sampler/previous_log_posterior",
            draws,
        )
        uniform_draw = _optional_sampler_trace_vector(handle, "sampler/uniform_draw", draws)
        log_acceptance = _optional_sampler_trace_vector(handle, "sampler/log_acceptance", draws)
        proposal_log_posterior_minimum: float | None = None
        proposal_log_posterior_maximum: float | None = None
        log_acceptance_minimum: float | None = None
        log_acceptance_maximum: float | None = None
        if proposal_log_posterior is not None:
            proposal_log_values = np.asarray(proposal_log_posterior, dtype=np.float64)
            if not np.all(np.isfinite(proposal_log_values)):
                msg = "sampler/proposal_log_posterior must contain only finite values."
                raise ValueError(msg)
            proposal_log_posterior_minimum = float(np.min(proposal_log_values))
            proposal_log_posterior_maximum = float(np.max(proposal_log_values))
        if previous_log_posterior is not None and not np.all(np.isfinite(previous_log_posterior)):
            msg = "sampler/previous_log_posterior must contain only finite values."
            raise ValueError(msg)
        if uniform_draw is not None:
            uniform_values = np.asarray(uniform_draw, dtype=np.float64)
            if not np.all((0.0 <= uniform_values) & (uniform_values <= 1.0)):
                msg = "sampler/uniform_draw must be in [0, 1]."
                raise ValueError(msg)
        if log_acceptance is not None:
            log_acceptance_values = np.asarray(log_acceptance, dtype=np.float64)
            if not np.all(np.isfinite(log_acceptance_values)):
                msg = "sampler/log_acceptance must contain only finite values."
                raise ValueError(msg)
            log_acceptance_minimum = float(np.min(log_acceptance_values))
            log_acceptance_maximum = float(np.max(log_acceptance_values))
        trace_available = accepted is not None and log_posterior is not None
        proposal_trace_values = (
            proposal_parameters,
            previous_parameters,
            proposal_log_posterior,
            previous_log_posterior,
            uniform_draw,
            log_acceptance,
        )
        proposal_trace_available = all(value is not None for value in proposal_trace_values)
        unavailable_diagnostics = tuple(
            name
            for name, values in (
                ("accepted", accepted),
                ("log_posterior", log_posterior),
            )
            if values is None
        )
        unavailable_proposal_diagnostics = tuple(
            name
            for name, values in (
                ("proposal_parameters", proposal_parameters),
                ("previous_parameters", previous_parameters),
                ("proposal_log_posterior", proposal_log_posterior),
                ("previous_log_posterior", previous_log_posterior),
                ("uniform_draw", uniform_draw),
                ("log_acceptance", log_acceptance),
            )
            if values is None
        )
        return SamplerFixtureSummary(
            fixture_path=fixture_path,
            parameter_names=parameter_names,
            mhparams_shape=mhparams_shape,
            parameter_axis=parameter_axis,
            draw_axis=draw_axis,
            draws=draws,
            parameter_count=parameter_count,
            covariance_shape=covariance_shape,
            covariance_source=covariance_source,
            covariance_min_eigenvalue=min_eigenvalue,
            covariance_max_eigenvalue=max_eigenvalue,
            covariance_condition_number=condition_number,
            covariance_positive_semidefinite=bool(min_eigenvalue >= -1.0e-12),
            input_proposal_covariance_available=input_proposal_dataset_available,
            trace_available=trace_available,
            accepted_shape=None if accepted is None else accepted.shape,
            log_posterior_shape=None if log_posterior is None else log_posterior.shape,
            accepted_draws=accepted_draws,
            realized_acceptance_rate=realized_acceptance_rate,
            log_posterior_mean=log_posterior_mean,
            log_posterior_minimum=log_posterior_minimum,
            log_posterior_maximum=log_posterior_maximum,
            proposal_trace_available=proposal_trace_available,
            proposal_parameters_shape=None
            if proposal_parameters is None
            else proposal_parameters.shape,
            previous_parameters_shape=None
            if previous_parameters is None
            else previous_parameters.shape,
            proposal_log_posterior_shape=None
            if proposal_log_posterior is None
            else proposal_log_posterior.shape,
            previous_log_posterior_shape=None
            if previous_log_posterior is None
            else previous_log_posterior.shape,
            uniform_draw_shape=None if uniform_draw is None else uniform_draw.shape,
            log_acceptance_shape=None if log_acceptance is None else log_acceptance.shape,
            proposal_log_posterior_minimum=proposal_log_posterior_minimum,
            proposal_log_posterior_maximum=proposal_log_posterior_maximum,
            log_acceptance_minimum=log_acceptance_minimum,
            log_acceptance_maximum=log_acceptance_maximum,
            metadata=metadata,
            unavailable_diagnostics=unavailable_diagnostics,
            unavailable_proposal_diagnostics=unavailable_proposal_diagnostics,
        )


def load_sampler_fixture_result(path: Path | str) -> MetropolisHastingsResult:
    summary = summarize_sampler_fixture(path)
    if not summary.trace_available:
        missing = ", ".join(summary.unavailable_diagnostics)
        msg = f"Sampler fixture is missing required trace diagnostics: {missing}."
        raise ValueError(msg)

    h5py: Any = import_module("h5py")
    with h5py.File(summary.fixture_path, "r") as handle:
        mhparams = np.asarray(handle["sampler/mhparams"][()], dtype=np.float64)
        parameter_draws = (
            mhparams.T if summary.parameter_axis == 0 else np.asarray(mhparams, dtype=np.float64)
        )
        accepted = np.asarray(handle["sampler/accepted"][()], dtype=bool)
        log_posterior = np.asarray(handle["sampler/log_posterior"][()], dtype=np.float64)
        proposal_covariance = np.asarray(
            handle["sampler/proposal_covariance"][()],
            dtype=np.float64,
        )

    metadata = summary.metadata
    acceptance_rate = _metadata_float_or_default(
        metadata.get("acceptance_rate"),
        summary.realized_acceptance_rate,
        default=0.0,
    )
    result = MetropolisHastingsResult(
        parameter_names=summary.parameter_names,
        estimation_draws=parameter_draws.copy(),
        parameter_draws=parameter_draws,
        log_posterior=log_posterior,
        accepted=accepted,
        acceptance_rate=acceptance_rate,
        proposal_covariance=proposal_covariance,
        seed=_metadata_optional_int(metadata.get("seed")),
        burnin=_metadata_int_or_default(metadata.get("burnin"), default=0),
        n_blocks=_metadata_int_or_default(metadata.get("blocks"), default=1),
        n_param_blocks=_metadata_int_or_default(metadata.get("param_blocks"), default=1),
        mhthin=_metadata_int_or_default(metadata.get("thin"), default=1),
        burnin_blocks=_metadata_int_or_default(metadata.get("burnin"), default=0),
        proposal_scale=_metadata_float_or_default(
            metadata.get("proposal_scale"),
            None,
            default=1.0,
        ),
        adaptive_accept=bool(metadata.get("adaptive_accept", False)),
        target_accept=_metadata_float_or_default(
            metadata.get("target_accept"),
            None,
            default=0.25,
        ),
        alpha=_metadata_float_or_default(metadata.get("alpha"), None, default=1.0),
        c=_metadata_float_or_default(metadata.get("c"), None, default=0.5),
    )
    validate_sampler_result(result)
    return result


def check_sampler_proposal_trace(
    path: Path | str,
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> SamplerProposalTraceReport:
    summary = summarize_sampler_fixture(path)
    if not summary.trace_available:
        missing = ", ".join(summary.unavailable_diagnostics)
        msg = f"Sampler fixture is missing required trace diagnostics: {missing}."
        raise ValueError(msg)
    if not summary.proposal_trace_available:
        missing = ", ".join(summary.unavailable_proposal_diagnostics)
        msg = f"Sampler fixture is missing required proposal trace diagnostics: {missing}."
        raise ValueError(msg)

    h5py: Any = import_module("h5py")
    with h5py.File(summary.fixture_path, "r") as handle:
        mhparams = np.asarray(handle["sampler/mhparams"][()], dtype=np.float64)
        accepted = np.asarray(handle["sampler/accepted"][()], dtype=bool)
        log_posterior = np.asarray(handle["sampler/log_posterior"][()], dtype=np.float64)
        proposal_parameters = np.asarray(
            handle["sampler/proposal_parameters"][()],
            dtype=np.float64,
        )
        previous_parameters = np.asarray(
            handle["sampler/previous_parameters"][()],
            dtype=np.float64,
        )
        proposal_log_posterior = np.asarray(
            handle["sampler/proposal_log_posterior"][()],
            dtype=np.float64,
        )
        previous_log_posterior = np.asarray(
            handle["sampler/previous_log_posterior"][()],
            dtype=np.float64,
        )
        uniform_draw = np.asarray(handle["sampler/uniform_draw"][()], dtype=np.float64)
        log_acceptance = np.asarray(handle["sampler/log_acceptance"][()], dtype=np.float64)

    expected_log_acceptance = proposal_log_posterior - previous_log_posterior
    with np.errstate(divide="ignore"):
        log_uniform = np.log(uniform_draw)
    expected_accepted = log_uniform < np.minimum(0.0, log_acceptance)
    expected_log_posterior = np.where(
        accepted,
        proposal_log_posterior,
        previous_log_posterior,
    )
    accepted_mask = (
        accepted.reshape(1, summary.draws)
        if summary.draw_axis == 1
        else accepted.reshape(summary.draws, 1)
    )
    expected_mhparams = np.where(
        accepted_mask,
        proposal_parameters,
        previous_parameters,
    )
    parameter_labels = {summary.parameter_axis: summary.parameter_names}
    comparisons = (
        compare_arrays(
            "proposal_trace/log_acceptance",
            expected_log_acceptance,
            log_acceptance,
            atol=atol,
            rtol=rtol,
        ),
        compare_arrays(
            "proposal_trace/accepted",
            expected_accepted.astype(np.float64),
            accepted.astype(np.float64),
            atol=0.0,
            rtol=0.0,
        ),
        compare_arrays(
            "proposal_trace/retained_log_posterior",
            expected_log_posterior,
            log_posterior,
            atol=atol,
            rtol=rtol,
        ),
        compare_arrays(
            "proposal_trace/retained_parameters",
            expected_mhparams,
            mhparams,
            atol=atol,
            rtol=rtol,
            labels=parameter_labels,
        ),
    )
    return SamplerProposalTraceReport(
        sampler_path=summary.fixture_path,
        comparisons=comparisons,
    )


def replay_sampler_proposal_posteriors(
    model: DSGEModel,
    observations: np.ndarray,
    path: Path | str,
    *,
    start_date: Any | None = None,
    log_likelihood_start: int = 0,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> SamplerPosteriorReplayReport:
    summary = summarize_sampler_fixture(path)
    if not summary.proposal_trace_available:
        missing = ", ".join(summary.unavailable_proposal_diagnostics)
        msg = f"Sampler fixture is missing required proposal trace diagnostics: {missing}."
        raise ValueError(msg)

    h5py: Any = import_module("h5py")
    with h5py.File(summary.fixture_path, "r") as handle:
        proposal_parameters = np.asarray(
            handle["sampler/proposal_parameters"][()],
            dtype=np.float64,
        )
        previous_parameters = np.asarray(
            handle["sampler/previous_parameters"][()],
            dtype=np.float64,
        )
        proposal_log_posterior = np.asarray(
            handle["sampler/proposal_log_posterior"][()],
            dtype=np.float64,
        )
        previous_log_posterior = np.asarray(
            handle["sampler/previous_log_posterior"][()],
            dtype=np.float64,
        )
        log_acceptance = np.asarray(handle["sampler/log_acceptance"][()], dtype=np.float64)
        proposal_log_likelihood = _optional_sampler_trace_vector(
            handle,
            "sampler/proposal_log_likelihood",
            summary.draws,
        )
        previous_log_likelihood = _optional_sampler_trace_vector(
            handle,
            "sampler/previous_log_likelihood",
            summary.draws,
        )
        proposal_log_prior = _optional_sampler_trace_vector(
            handle,
            "sampler/proposal_log_prior",
            summary.draws,
        )
        previous_log_prior = _optional_sampler_trace_vector(
            handle,
            "sampler/previous_log_prior",
            summary.draws,
        )

    proposal_draws = _sampler_trace_matrix_to_draw_major(proposal_parameters, summary)
    previous_draws = _sampler_trace_matrix_to_draw_major(previous_parameters, summary)
    model_parameter_names = _resolve_model_parameter_names(
        model,
        summary.parameter_names,
    )
    proposal_components = np.asarray(
        [
            _safe_model_value_posterior_components(
                model,
                observations,
                model_parameter_names,
                draw,
                start_date=start_date,
                log_likelihood_start=log_likelihood_start,
            )
            for draw in proposal_draws
        ],
        dtype=np.float64,
    )
    previous_components = np.asarray(
        [
            _safe_model_value_posterior_components(
                model,
                observations,
                model_parameter_names,
                draw,
                start_date=start_date,
                log_likelihood_start=log_likelihood_start,
            )
            for draw in previous_draws
        ],
        dtype=np.float64,
    )
    proposal_replay = proposal_components[:, 0]
    proposal_likelihood_replay = proposal_components[:, 1]
    proposal_prior_replay = proposal_components[:, 2]
    previous_replay = previous_components[:, 0]
    previous_likelihood_replay = previous_components[:, 1]
    previous_prior_replay = previous_components[:, 2]
    posterior_offsets = np.concatenate(
        [
            proposal_replay - proposal_log_posterior,
            previous_replay - previous_log_posterior,
        ]
    )
    finite_offsets = posterior_offsets[np.isfinite(posterior_offsets)]
    offset_mean = float(np.mean(finite_offsets)) if finite_offsets.size else None
    offset_std = float(np.std(finite_offsets)) if finite_offsets.size else None
    draw_labels = {0: _sampler_draw_labels(summary.draws)}
    log_acceptance_atol = _propagated_difference_atol(
        proposal_log_posterior,
        previous_log_posterior,
        atol=atol,
        rtol=rtol,
    )
    comparisons = [
        compare_arrays(
            "proposal_trace/proposal_log_posterior",
            proposal_log_posterior,
            proposal_replay,
            atol=atol,
            rtol=rtol,
            labels=draw_labels,
        ),
        compare_arrays(
            "proposal_trace/previous_log_posterior",
            previous_log_posterior,
            previous_replay,
            atol=atol,
            rtol=rtol,
            labels=draw_labels,
        ),
        compare_arrays(
            "proposal_trace/log_acceptance_from_replay",
            log_acceptance,
            proposal_replay - previous_replay,
            atol=log_acceptance_atol,
            rtol=rtol,
            labels=draw_labels,
        ),
    ]
    if proposal_log_likelihood is not None:
        comparisons.append(
            compare_arrays(
                "proposal_trace/proposal_log_likelihood",
                proposal_log_likelihood,
                proposal_likelihood_replay,
                atol=atol,
                rtol=rtol,
                labels=draw_labels,
            )
        )
    if previous_log_likelihood is not None:
        comparisons.append(
            compare_arrays(
                "proposal_trace/previous_log_likelihood",
                previous_log_likelihood,
                previous_likelihood_replay,
                atol=atol,
                rtol=rtol,
                labels=draw_labels,
            )
        )
    if proposal_log_prior is not None:
        comparisons.append(
            compare_arrays(
                "proposal_trace/proposal_log_prior",
                proposal_log_prior,
                proposal_prior_replay,
                atol=atol,
                rtol=rtol,
                labels=draw_labels,
            )
        )
    if previous_log_prior is not None:
        comparisons.append(
            compare_arrays(
                "proposal_trace/previous_log_prior",
                previous_log_prior,
                previous_prior_replay,
                atol=atol,
                rtol=rtol,
                labels=draw_labels,
            )
        )
    return SamplerPosteriorReplayReport(
        sampler_path=summary.fixture_path,
        draws=summary.draws,
        parameter_count=summary.parameter_count,
        posterior_offset_mean=offset_mean,
        posterior_offset_std=offset_std,
        comparisons=tuple(comparisons),
    )


def compare_sampler_results(
    oracle_sampler: Path,
    candidate_sampler: Path,
    *,
    oracle_result: MetropolisHastingsResult,
    candidate_result: MetropolisHastingsResult,
    windows: int = 4,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
) -> SamplerComparisonReport:
    validate_sampler_result(oracle_result)
    validate_sampler_result(candidate_result)
    oracle_diagnostics = sampler_diagnostics(oracle_result, windows=windows)
    candidate_diagnostics = sampler_diagnostics(candidate_result, windows=windows)
    comparisons: list[ArrayComparison] = []
    if oracle_result.parameter_names != candidate_result.parameter_names:
        comparisons.append(
            ArrayComparison(
                name="parameter_names",
                status="label_mismatch",
                expected_shape=(len(oracle_result.parameter_names),),
                actual_shape=(len(candidate_result.parameter_names),),
                max_abs_diff=None,
                max_rel_diff=None,
                atol=atol,
                rtol=rtol,
                message=(
                    "Sampler parameter names differ: "
                    f"{oracle_result.parameter_names} vs {candidate_result.parameter_names}."
                ),
            )
        )

    comparisons.extend(
        [
            compare_arrays(
                "parameter_draws",
                oracle_result.parameter_draws,
                candidate_result.parameter_draws,
                atol=atol,
                rtol=rtol,
                labels={1: oracle_result.parameter_names},
            ),
            compare_arrays(
                "log_posterior",
                oracle_result.log_posterior,
                candidate_result.log_posterior,
                atol=atol,
                rtol=rtol,
            ),
            compare_arrays(
                "accepted",
                oracle_result.accepted.astype(np.float64),
                candidate_result.accepted.astype(np.float64),
                atol=0.0,
                rtol=0.0,
            ),
            compare_arrays(
                "proposal_covariance",
                oracle_result.proposal_covariance,
                candidate_result.proposal_covariance,
                atol=atol,
                rtol=rtol,
                labels={
                    0: oracle_result.parameter_names,
                    1: oracle_result.parameter_names,
                },
            ),
            compare_arrays(
                "diagnostics/core",
                _sampler_diagnostic_vector(oracle_diagnostics),
                _sampler_diagnostic_vector(candidate_diagnostics),
                atol=atol,
                rtol=rtol,
            ),
            compare_arrays(
                "diagnostics/acceptance_windows",
                np.asarray(oracle_diagnostics.acceptance_windows, dtype=np.float64),
                np.asarray(candidate_diagnostics.acceptance_windows, dtype=np.float64),
                atol=atol,
                rtol=rtol,
            ),
            compare_arrays(
                "diagnostics/parameters",
                _sampler_parameter_diagnostic_matrix(oracle_diagnostics),
                _sampler_parameter_diagnostic_matrix(candidate_diagnostics),
                atol=atol,
                rtol=rtol,
                labels={0: oracle_result.parameter_names},
            ),
        ]
    )
    return SamplerComparisonReport(
        oracle_sampler=oracle_sampler,
        candidate_sampler=candidate_sampler,
        comparisons=tuple(comparisons),
    )


def compare_arrays(
    name: str,
    expected: np.ndarray,
    actual: np.ndarray,
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
    labels: dict[int, tuple[str, ...]] | None = None,
) -> ArrayComparison:
    expected_arr = np.asarray(expected, dtype=np.float64)
    actual_arr = np.asarray(actual, dtype=np.float64)
    if expected_arr.shape != actual_arr.shape:
        return ArrayComparison(
            name=name,
            status="shape_mismatch",
            expected_shape=expected_arr.shape,
            actual_shape=actual_arr.shape,
            max_abs_diff=None,
            max_rel_diff=None,
            atol=atol,
            rtol=rtol,
            message="Array shapes differ.",
        )

    finite_pairs = np.isfinite(expected_arr) & np.isfinite(actual_arr)
    same_special = (~finite_pairs) & (
        (expected_arr == actual_arr) | (np.isnan(expected_arr) & np.isnan(actual_arr))
    )
    diff = np.zeros_like(expected_arr, dtype=np.float64)
    diff[finite_pairs] = np.abs(expected_arr[finite_pairs] - actual_arr[finite_pairs])
    diff[(~finite_pairs) & (~same_special)] = np.inf
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    max_abs_index = _max_abs_index(diff)
    rel = np.zeros_like(expected_arr, dtype=np.float64)
    denominator = np.maximum(np.abs(expected_arr[finite_pairs]), np.finfo(np.float64).eps)
    rel[finite_pairs] = diff[finite_pairs] / denominator
    rel[(~finite_pairs) & (~same_special)] = np.inf
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    passed = bool(np.allclose(expected_arr, actual_arr, atol=atol, rtol=rtol, equal_nan=True))
    return ArrayComparison(
        name=name,
        status="passed" if passed else "failed",
        expected_shape=expected_arr.shape,
        actual_shape=actual_arr.shape,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
        atol=atol,
        rtol=rtol,
        max_abs_index=max_abs_index,
        max_abs_label=_labels_for_index(labels, max_abs_index),
        message="" if passed else "Array values exceed tolerance.",
    )


def compare_fixture_dirs(
    oracle_dir: Path,
    candidate_dir: Path,
    *,
    atol: float = 1.0e-10,
    rtol: float = 1.0e-10,
    array_names: tuple[str, ...] | None = None,
) -> FixtureComparisonReport:
    oracle_arrays = load_fixture_arrays(oracle_dir)
    candidate_arrays = load_fixture_arrays(candidate_dir)
    if array_names is not None:
        included = set(array_names)
        oracle_arrays = {name: value for name, value in oracle_arrays.items() if name in included}
        candidate_arrays = {
            name: value for name, value in candidate_arrays.items() if name in included
        }
    labels = {**load_fixture_labels(candidate_dir), **load_fixture_labels(oracle_dir)}
    comparisons: list[ArrayComparison] = []

    for name, expected in sorted(oracle_arrays.items()):
        if name not in candidate_arrays:
            comparisons.append(
                ArrayComparison(
                    name=name,
                    status="missing_candidate",
                    expected_shape=expected.shape,
                    actual_shape=None,
                    max_abs_diff=None,
                    max_rel_diff=None,
                    atol=atol,
                    rtol=rtol,
                    message="Candidate fixture is missing this array.",
                )
            )
            continue
        comparisons.append(
            compare_arrays(
                name,
                expected,
                candidate_arrays[name],
                atol=atol,
                rtol=rtol,
                labels=labels.get(name),
            )
        )

    for name, actual in sorted(candidate_arrays.items()):
        if name not in oracle_arrays:
            comparisons.append(
                ArrayComparison(
                    name=name,
                    status="extra_candidate",
                    expected_shape=None,
                    actual_shape=actual.shape,
                    max_abs_diff=None,
                    max_rel_diff=None,
                    atol=atol,
                    rtol=rtol,
                    message="Candidate fixture has no oracle counterpart.",
                )
            )

    return FixtureComparisonReport(
        oracle_dir=oracle_dir,
        candidate_dir=candidate_dir,
        comparisons=tuple(comparisons),
    )


def save_system_fixture(
    system: System,
    directory: Path,
    *,
    filename: str = "system.npz",
) -> Path:
    if Path(filename).name != filename:
        msg = "Fixture filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".npz"):
        msg = "System fixtures must be written as .npz archives."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    arrays = {
        "TTT": system.transition.TTT,
        "RRR": system.transition.RRR,
        "CCC": system.transition.CCC,
        "ZZ": system.measurement.ZZ,
        "DD": system.measurement.DD,
        "QQ": system.measurement.QQ,
        "EE": system.measurement.EE,
    }
    if system.pseudo_measurement is not None:
        arrays["ZZ_pseudo"] = system.pseudo_measurement.ZZ_pseudo
        arrays["DD_pseudo"] = system.pseudo_measurement.DD_pseudo
    _write_npz_fixture(path, arrays)
    return path


def save_parameter_fixture(
    parameters: dict[str, Parameter],
    directory: Path,
    *,
    filename: str = "parameters.npz",
) -> Path:
    if Path(filename).name != filename:
        msg = "Fixture filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".npz"):
        msg = "Parameter fixtures must be written as .npz archives."
        raise ValueError(msg)

    parameter_values = tuple(parameters.values())
    bounds = np.full((len(parameter_values), 2), np.nan, dtype=np.float64)
    for row, parameter in enumerate(parameter_values):
        if parameter.value_bounds is not None:
            bounds[row] = np.asarray(parameter.value_bounds, dtype=np.float64)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    _write_npz_fixture(
        path,
        {
            "values": np.asarray([parameter.value for parameter in parameter_values]),
            "scaled_values": np.asarray(
                [parameter.scaled_value for parameter in parameter_values],
                dtype=np.float64,
            ),
            "fixed": np.asarray(
                [1.0 if parameter.fixed else 0.0 for parameter in parameter_values],
                dtype=np.float64,
            ),
            "bounds": bounds,
        },
    )
    return path


def save_steady_state_fixture(
    steady_state: dict[str, float],
    directory: Path,
    *,
    filename: str = "steady_state.npz",
) -> Path:
    if Path(filename).name != filename:
        msg = "Fixture filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".npz"):
        msg = "Steady-state fixtures must be written as .npz archives."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    _write_npz_fixture(
        path,
        {"values": np.asarray(list(steady_state.values()), dtype=np.float64)},
    )
    return path


def save_canonical_fixture(
    canonical: CanonicalSystem,
    directory: Path,
    *,
    filename: str = "canonical.npz",
) -> Path:
    if Path(filename).name != filename:
        msg = "Fixture filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".npz"):
        msg = "Canonical fixtures must be written as .npz archives."
        raise ValueError(msg)

    canonical.validate()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    _write_npz_fixture(
        path,
        {
            "Gamma0": canonical.Gamma0,
            "Gamma1": canonical.Gamma1,
            "C": canonical.C,
            "Psi": canonical.Psi,
            "Pi": canonical.Pi,
        },
    )
    return path


def save_forecast_fixture(
    forecast: ForecastOutput,
    directory: Path,
    *,
    filename: str = "forecast.npz",
) -> Path:
    if Path(filename).name != filename:
        msg = "Fixture filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".npz"):
        msg = "Forecast fixtures must be written as .npz archives."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    _write_npz_fixture(
        path,
        {
            "states": forecast.states,
            "observables": forecast.observables,
            **(
                {}
                if forecast.pseudo_observables is None
                else {"pseudo_observables": forecast.pseudo_observables}
            ),
            **(
                {}
                if forecast.conditional_shocks is None
                else {"conditional_shocks": forecast.conditional_shocks}
            ),
            **(
                {}
                if forecast.conditional_states is None
                else {"conditional_states": forecast.conditional_states}
            ),
            **(
                {}
                if forecast.conditional_observables is None
                else {"conditional_observables": forecast.conditional_observables}
            ),
            **({} if forecast.state_samples is None else {"state_samples": forecast.state_samples}),
            **(
                {}
                if forecast.observable_samples is None
                else {"observable_samples": forecast.observable_samples}
            ),
            **(
                {}
                if forecast.pseudo_observable_samples is None
                else {"pseudo_observable_samples": forecast.pseudo_observable_samples}
            ),
            **(
                {}
                if forecast.history_states is None
                else {"history_states": forecast.history_states}
            ),
            **(
                {}
                if forecast.history_observables is None
                else {"history_observables": forecast.history_observables}
            ),
            **(
                {}
                if forecast.history_pseudo_observables is None
                else {"history_pseudo_observables": forecast.history_pseudo_observables}
            ),
            **(
                {}
                if forecast.history_state_samples is None
                else {"history_state_samples": forecast.history_state_samples}
            ),
            **(
                {}
                if forecast.history_observable_samples is None
                else {"history_observable_samples": forecast.history_observable_samples}
            ),
            **(
                {}
                if forecast.history_pseudo_observable_samples is None
                else {
                    "history_pseudo_observable_samples": forecast.history_pseudo_observable_samples
                }
            ),
        },
    )
    return path


def save_meansbands_fixture(
    meansbands: MeansBands,
    directory: Path,
    *,
    filename: str = "meansbands.npz",
) -> Path:
    if Path(filename).name != filename:
        msg = "Fixture filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".npz"):
        msg = "Means/bands fixtures must be written as .npz archives."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    _write_npz_fixture(
        path,
        {
            "mean": meansbands.mean,
            "lower": meansbands.lower,
            "upper": meansbands.upper,
        },
    )
    return path


def save_kalman_fixture(
    kalman: KalmanResult,
    directory: Path,
    *,
    filename: str = "kalman.npz",
) -> Path:
    if Path(filename).name != filename:
        msg = "Fixture filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".npz"):
        msg = "Kalman fixtures must be written as .npz archives."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    _write_npz_fixture(
        path,
        {
            "log_likelihood": np.asarray(
                kalman.log_likelihood_by_period,
                dtype=np.float64,
            ),
            "predicted_states": kalman.predicted_states,
            "filtered_states": kalman.filtered_states,
            "predicted_covariances": kalman.predicted_covariances,
            "filtered_covariances": kalman.filtered_covariances,
            "final_filtered_state": kalman.final_filtered_state,
            "total_log_likelihood": np.asarray([kalman.log_likelihood], dtype=np.float64),
        },
    )
    return path


def save_posterior_fixture(
    result: EstimateResult,
    parameters: dict[str, Parameter],
    directory: Path,
    *,
    filename: str = "posterior.npz",
) -> Path:
    if Path(filename).name != filename:
        msg = "Fixture filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".npz"):
        msg = "Posterior fixtures must be written as a .npz archive."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    _write_npz_fixture(
        path,
        {
            "log_posterior": np.asarray([result.log_posterior], dtype=np.float64),
            "log_likelihood": np.asarray([result.log_likelihood], dtype=np.float64),
            "log_prior": np.asarray([result.log_prior], dtype=np.float64),
            "log_likelihood_by_period": np.asarray(
                result.kalman.log_likelihood_by_period,
                dtype=np.float64,
            ),
            "log_prior_by_parameter": np.asarray(
                [parameter_log_prior(parameter) for parameter in parameters.values()],
                dtype=np.float64,
            ),
            "parameter_values": np.asarray(
                list(result.parameter_values.values()),
                dtype=np.float64,
            ),
        },
    )
    return path


def load_canonical_fixture(directory: Path) -> CanonicalSystem:
    arrays = load_fixture_arrays(directory)
    return CanonicalSystem(
        Gamma0=_find_fixture_array(arrays, "Gamma0"),
        Gamma1=_find_fixture_array(arrays, "Gamma1"),
        C=np.ravel(_find_fixture_array(arrays, "C")),
        Psi=_find_fixture_array(arrays, "Psi"),
        Pi=_find_fixture_array(arrays, "Pi"),
    )


def save_transition_fixture(
    result: CanonicalSolveResult,
    directory: Path,
    *,
    filename: str = "transition.npz",
) -> Path:
    if Path(filename).name != filename:
        msg = "Fixture filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".npz"):
        msg = "Transition fixtures must be written as .npz archives."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    _write_npz_fixture(
        path,
        {
            "TTT": result.transition.TTT,
            "RRR": result.transition.RRR,
            "CCC": result.transition.CCC,
            "eu": np.asarray(result.eu, dtype=np.int64),
        },
    )
    return path


def save_fixture_manifest(
    directory: Path,
    metadata: dict[str, Any],
    *,
    filename: str = "manifest.json",
) -> Path:
    if Path(filename).name != filename:
        msg = "Manifest filename must not include a directory."
        raise ValueError(msg)
    if not filename.endswith(".json"):
        msg = "Fixture manifest must be written as a .json file."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    output = _merge_manifest(path, metadata)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_fixture_arrays(directory: Path) -> dict[str, np.ndarray]:
    if not directory.exists():
        msg = f"Fixture directory does not exist: {directory}"
        raise FileNotFoundError(msg)
    if not directory.is_dir():
        msg = f"Fixture path is not a directory: {directory}"
        raise NotADirectoryError(msg)

    arrays: dict[str, np.ndarray] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        relative_stem = _relative_stem(directory, path)
        if suffix == ".npy":
            arrays[relative_stem] = np.load(path)
        elif suffix == ".npz":
            with np.load(path) as archive:
                for key in archive.files:
                    arrays[f"{relative_stem}/{key}"] = archive[key]
        elif suffix == ".csv":
            try:
                arrays[relative_stem] = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
            except ValueError:
                continue
        elif suffix in {".h5", ".hdf5"}:
            arrays.update(_load_hdf5_arrays(path))
    return arrays


def load_fixture_labels(directory: Path) -> FixtureLabels:
    if not directory.exists():
        msg = f"Fixture directory does not exist: {directory}"
        raise FileNotFoundError(msg)
    if not directory.is_dir():
        msg = f"Fixture path is not a directory: {directory}"
        raise NotADirectoryError(msg)

    labels: FixtureLabels = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.casefold() in {".h5", ".hdf5"}:
            labels.update(_load_hdf5_labels(path))

    for path in sorted(directory.rglob("manifest.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_labels = raw.get("labels", {})
        if not isinstance(raw_labels, dict):
            continue
        for array_name, axis_labels in raw_labels.items():
            if not isinstance(axis_labels, dict):
                continue
            parsed = _parse_axis_labels(axis_labels)
            if parsed:
                labels[str(array_name)] = parsed
    return labels


def _resolve_sampler_fixture_path(path: Path) -> Path:
    if path.is_file():
        if path.suffix.casefold() not in {".h5", ".hdf5"}:
            msg = f"Sampler fixture path must be an HDF5 file: {path}"
            raise ValueError(msg)
        return path
    if not path.exists():
        msg = f"Sampler fixture path does not exist: {path}"
        raise FileNotFoundError(msg)
    if not path.is_dir():
        msg = f"Sampler fixture path is not a file or directory: {path}"
        raise ValueError(msg)

    h5py: Any = import_module("h5py")
    matches: list[Path] = []
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or candidate.suffix.casefold() not in {".h5", ".hdf5"}:
            continue
        with h5py.File(candidate, "r") as handle:
            if "sampler/mhparams" in handle and "sampler/proposal_covariance" in handle:
                matches.append(candidate)
    if not matches:
        msg = f"No sampler HDF5 fixture found under: {path}"
        raise FileNotFoundError(msg)
    if len(matches) > 1:
        names = ", ".join(str(match) for match in matches)
        msg = f"Multiple sampler HDF5 fixtures found; pass one explicitly: {names}"
        raise ValueError(msg)
    return matches[0]


def _sampler_hdf5_metadata(handle: Any) -> dict[str, Any]:
    keys = (
        "sampler_sampling_method",
        "sampler_draws",
        "sampler_burnin",
        "sampler_blocks",
        "sampler_param_blocks",
        "sampler_thin",
        "sampler_adaptive_accept",
        "sampler_target_accept",
        "sampler_cc",
        "sampler_alpha",
        "sampler_c",
        "sampler_cc0",
        "sampler_calculate_hessian",
        "sampler_reoptimize",
        "sampler_run_csminwel",
        "sampler_proposal_scale",
        "sampler_mode_in",
        "sampler_hessian_in",
        "sampler_covariance_source",
        "sampler_trace_available",
        "sampler_proposal_trace_available",
        "sampler_acceptance_rate",
        "sampler_block_acceptance_rates",
        "sampler_input_proposal_covariance_available",
        "sampler_seed",
    )
    metadata: dict[str, Any] = {}
    for key in keys:
        if key in handle.attrs:
            metadata[key.removeprefix("sampler_")] = _coerce_hdf5_scalar(handle.attrs[key])
    return metadata


def _metadata_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _metadata_int_or_default(value: Any, *, default: int) -> int:
    parsed = _metadata_optional_int(value)
    return default if parsed is None else parsed


def _metadata_float_or_default(
    value: Any,
    fallback: float | None,
    *,
    default: float,
) -> float:
    if value is None:
        return default if fallback is None else float(fallback)
    if isinstance(value, bool):
        return default if fallback is None else float(fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default if fallback is None else float(fallback)


def _sampler_diagnostic_vector(diagnostics: Any) -> np.ndarray:
    return np.asarray(
        [
            diagnostics.draws,
            diagnostics.burnin,
            diagnostics.accepted_draws,
            diagnostics.acceptance_rate,
            diagnostics.realized_acceptance_rate,
            diagnostics.proposal_covariance_min_eigenvalue,
            diagnostics.proposal_covariance_max_eigenvalue,
            diagnostics.proposal_covariance_condition_number,
            1.0 if diagnostics.proposal_covariance_positive_semidefinite else 0.0,
            diagnostics.log_posterior_mean,
            diagnostics.log_posterior_minimum,
            diagnostics.log_posterior_maximum,
        ],
        dtype=np.float64,
    )


def _sampler_parameter_diagnostic_matrix(diagnostics: Any) -> np.ndarray:
    return np.asarray(
        [
            [
                parameter.mean,
                parameter.std,
                parameter.minimum,
                parameter.maximum,
                parameter.effective_sample_size,
                parameter.integrated_autocorrelation_time,
                parameter.monte_carlo_standard_error,
                np.nan if parameter.split_rhat is None else parameter.split_rhat,
            ]
            for parameter in diagnostics.parameters
        ],
        dtype=np.float64,
    )


def _optional_sampler_trace_vector(
    handle: Any,
    name: str,
    expected_draws: int,
) -> np.ndarray | None:
    if name not in handle:
        return None
    values = np.asarray(handle[name][()])
    shape = tuple(int(value) for value in values.shape)
    if shape != (expected_draws,):
        msg = f"{name} must have shape {(expected_draws,)}, got {shape}."
        raise ValueError(msg)
    return values


def _optional_sampler_trace_matrix(
    handle: Any,
    name: str,
    expected_shape: tuple[int, int],
) -> np.ndarray | None:
    if name not in handle:
        return None
    values = np.asarray(handle[name][()], dtype=np.float64)
    shape = tuple(int(value) for value in values.shape)
    if shape != expected_shape:
        msg = f"{name} must have shape {expected_shape}, got {shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    return values


def _sampler_trace_matrix_to_draw_major(
    values: np.ndarray,
    summary: SamplerFixtureSummary,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != summary.mhparams_shape:
        msg = f"Sampler trace matrix must have shape {summary.mhparams_shape}, got {matrix.shape}."
        raise ValueError(msg)
    return matrix.T if summary.parameter_axis == 0 else matrix.copy()


def _resolve_model_parameter_names(
    model: DSGEModel,
    fixture_names: tuple[str, ...],
) -> tuple[str, ...]:
    model_names = tuple(model.parameters)
    if all(name in model.parameters for name in fixture_names):
        return fixture_names
    if len(fixture_names) == len(model_names):
        return model_names
    missing = [name for name in fixture_names if name not in model.parameters]
    msg = "Sampler fixture parameter names do not match model parameters: " + ", ".join(missing)
    raise KeyError(msg)


def _safe_model_value_posterior_components(
    model: DSGEModel,
    observations: np.ndarray,
    parameter_names: tuple[str, ...],
    values: np.ndarray,
    *,
    start_date: Any | None = None,
    log_likelihood_start: int = 0,
) -> np.ndarray:
    try:
        log_posterior, log_likelihood, log_prior, _ = evaluate_log_posterior_for_parameter_values(
            model,
            observations,
            parameter_names,
            values,
            start_date=start_date,
            log_likelihood_start=log_likelihood_start,
            update_fixed_parameters=False,
        )
        if not np.isfinite(log_posterior):
            return np.asarray([float("-inf"), float("-inf"), float("-inf")], dtype=np.float64)
        return np.asarray(
            [log_posterior, log_likelihood, log_prior],
            dtype=np.float64,
        )
    except Exception:
        return np.asarray([float("-inf"), float("-inf"), float("-inf")], dtype=np.float64)


def _sampler_draw_labels(draws: int) -> tuple[str, ...]:
    return tuple(f"draw_{index}" for index in range(draws))


def _propagated_difference_atol(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> float:
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.size == 0 and right_arr.size == 0:
        return atol
    left_scale = float(np.nanmax(np.abs(left_arr))) if left_arr.size else 0.0
    right_scale = float(np.nanmax(np.abs(right_arr))) if right_arr.size else 0.0
    return max(atol, 2.0 * atol + rtol * (left_scale + right_scale))


def _infer_sampler_mhparams_axes(
    shape: tuple[int, int],
    parameter_count: int,
    metadata_draws: int | None = None,
) -> tuple[int, int, int]:
    matches: list[tuple[int, int, int]] = []
    if shape[0] == parameter_count:
        matches.append((0, 1, shape[1]))
    if shape[1] == parameter_count:
        matches.append((1, 0, shape[0]))
    if not matches:
        msg = (
            "sampler/mhparams shape does not align with sampler_parameter_names: "
            f"{shape} vs {parameter_count} names."
        )
        raise ValueError(msg)
    if len(matches) == 1:
        return matches[0]
    if metadata_draws is not None:
        matching_draws = [match for match in matches if match[2] == metadata_draws]
        if len(matching_draws) == 1:
            return matching_draws[0]
    msg = (
        "sampler/mhparams orientation is ambiguous because both axes match "
        f"{parameter_count} sampler parameters; use a non-square fixture or "
        "metadata that disambiguates retained draws."
    )
    raise ValueError(msg)


def _coerce_hdf5_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        text = value.decode()
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            return value.tolist()
        return _coerce_hdf5_scalar(value.item())
    elif isinstance(value, np.generic):
        return _coerce_hdf5_scalar(value.item())
    else:
        text = str(value)

    stripped = text.strip()
    if stripped == "":
        return None
    lowered = stripped.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _load_hdf5_arrays(path: Path) -> dict[str, np.ndarray]:
    h5py: Any = import_module("h5py")
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:

        def visit(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                arrays[name] = np.asarray(obj[()], dtype=np.float64)

        handle.visititems(visit)
    return arrays


def _load_hdf5_labels(path: Path) -> FixtureLabels:
    h5py: Any = import_module("h5py")
    labels: FixtureLabels = {}
    with h5py.File(path, "r") as handle:
        parameter_names = _parse_hdf5_name_attr(handle.attrs.get("parameter_names"))
        if parameter_names:
            for name in (
                "parameters/values",
                "parameters/scaled_values",
                "parameters/fixed",
            ):
                shape = _hdf5_dataset_shape(handle, name, h5py.Dataset)
                if shape is not None and shape[:1] == (len(parameter_names),):
                    labels[name] = {0: parameter_names}

            bounds_shape = _hdf5_dataset_shape(handle, "parameters/bounds", h5py.Dataset)
            if bounds_shape is not None and bounds_shape[:2] == (len(parameter_names), 2):
                labels["parameters/bounds"] = {
                    0: parameter_names,
                    1: ("lower", "upper"),
                }

        steady_state_names = _parse_hdf5_name_attr(handle.attrs.get("steady_state_names"))
        steady_shape = _hdf5_dataset_shape(handle, "steady_state/values", h5py.Dataset)
        if (
            steady_state_names
            and steady_shape is not None
            and steady_shape[:1] == (len(steady_state_names),)
        ):
            labels["steady_state/values"] = {0: steady_state_names}

        financial_input_names = _parse_hdf5_name_attr(
            handle.attrs.get("financial_frictions_input_names")
        )
        financial_case_names = _parse_hdf5_name_attr(
            handle.attrs.get("financial_frictions_case_names")
        )
        financial_function_names = _parse_hdf5_name_attr(
            handle.attrs.get("financial_frictions_function_names")
        )
        financial_inputs_shape = _hdf5_dataset_shape(
            handle, "financial_frictions/inputs", h5py.Dataset
        )
        if (
            financial_case_names
            and financial_input_names
            and financial_inputs_shape is not None
            and financial_inputs_shape[:2]
            == (len(financial_case_names), len(financial_input_names))
        ):
            labels["financial_frictions/inputs"] = {
                0: financial_case_names,
                1: financial_input_names,
            }
        financial_values_shape = _hdf5_dataset_shape(
            handle, "financial_frictions/values", h5py.Dataset
        )
        if (
            financial_case_names
            and financial_function_names
            and financial_values_shape is not None
            and financial_values_shape[:2]
            == (len(financial_case_names), len(financial_function_names))
        ):
            labels["financial_frictions/values"] = {
                0: financial_case_names,
                1: financial_function_names,
            }

        transition_status_shape = _hdf5_dataset_shape(handle, "transition/eu", h5py.Dataset)
        if transition_status_shape == (2,):
            labels["transition/eu"] = {0: ("existence", "uniqueness")}

        endogenous_state_names = _parse_hdf5_name_attr(handle.attrs.get("endogenous_state_names"))
        augmented_state_names = _parse_hdf5_name_attr(handle.attrs.get("augmented_state_names"))
        state_names = _parse_hdf5_name_attr(handle.attrs.get("state_names"))
        if not state_names and endogenous_state_names and augmented_state_names:
            state_names = (*endogenous_state_names, *augmented_state_names)
        shock_names = _parse_hdf5_name_attr(handle.attrs.get("exogenous_shock_names"))
        expected_shock_names = _parse_hdf5_name_attr(handle.attrs.get("expected_shock_names"))
        equation_names = _parse_hdf5_name_attr(handle.attrs.get("equation_names"))
        observable_names = _parse_hdf5_name_attr(handle.attrs.get("observable_names"))
        pseudo_observable_names = _parse_hdf5_name_attr(handle.attrs.get("pseudo_observable_names"))
        if endogenous_state_names and equation_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "canonical/Gamma0",
                (len(equation_names), len(endogenous_state_names)),
                {0: equation_names, 1: endogenous_state_names},
            )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "canonical/Gamma1",
                (len(equation_names), len(endogenous_state_names)),
                {0: equation_names, 1: endogenous_state_names},
            )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "canonical/C",
                (len(equation_names),),
                {0: equation_names},
            )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "transition/TTT",
                (len(endogenous_state_names), len(endogenous_state_names)),
                {0: endogenous_state_names, 1: endogenous_state_names},
            )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "transition/CCC",
                (len(endogenous_state_names),),
                {0: endogenous_state_names},
            )
        if equation_names and shock_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "canonical/Psi",
                (len(equation_names), len(shock_names)),
                {0: equation_names, 1: shock_names},
            )
        if equation_names and expected_shock_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "canonical/Pi",
                (len(equation_names), len(expected_shock_names)),
                {0: equation_names, 1: expected_shock_names},
            )
        if endogenous_state_names and shock_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "transition/RRR",
                (len(endogenous_state_names), len(shock_names)),
                {0: endogenous_state_names, 1: shock_names},
            )
        if state_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "system/TTT",
                (len(state_names), len(state_names)),
                {0: state_names, 1: state_names},
            )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "system/CCC",
                (len(state_names),),
                {0: state_names},
            )
        if state_names and shock_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "system/RRR",
                (len(state_names), len(shock_names)),
                {0: state_names, 1: shock_names},
            )
        if observable_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "system/DD",
                (len(observable_names),),
                {0: observable_names},
            )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "system/EE",
                (len(observable_names), len(observable_names)),
                {0: observable_names, 1: observable_names},
            )
        if observable_names and state_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "system/ZZ",
                (len(observable_names), len(state_names)),
                {0: observable_names, 1: state_names},
            )
        if shock_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "system/QQ",
                (len(shock_names), len(shock_names)),
                {0: shock_names, 1: shock_names},
            )
        if pseudo_observable_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "system/DD_pseudo",
                (len(pseudo_observable_names),),
                {0: pseudo_observable_names},
            )
        if pseudo_observable_names and state_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "system/ZZ_pseudo",
                (len(pseudo_observable_names), len(state_names)),
                {0: pseudo_observable_names, 1: state_names},
            )

        forecast_mode_dates = _parse_hdf5_name_attr(handle.attrs.get("forecast_mode_dates"))
        if forecast_mode_dates and state_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "forecast_mode/states",
                (len(forecast_mode_dates), len(state_names)),
                {0: forecast_mode_dates, 1: state_names},
            )
        if forecast_mode_dates and observable_names:
            forecast_observable_axes = {
                0: forecast_mode_dates,
                1: observable_names,
            }
            for name in (
                "forecast_mode/observables",
                "meansbands_mode_forecastobs/mean",
                "meansbands_mode_forecastobs/lower",
                "meansbands_mode_forecastobs/upper",
            ):
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    name,
                    (len(forecast_mode_dates), len(observable_names)),
                    forecast_observable_axes,
                )
        if forecast_mode_dates and pseudo_observable_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "forecast_mode/pseudo_observables",
                (len(forecast_mode_dates), len(pseudo_observable_names)),
                {0: forecast_mode_dates, 1: pseudo_observable_names},
            )
        if forecast_mode_dates and shock_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "forecast_mode/shocks",
                (len(forecast_mode_dates), len(shock_names)),
                {0: forecast_mode_dates, 1: shock_names},
            )

        forecast_full_dates = _parse_hdf5_name_attr(handle.attrs.get("forecast_full_dates"))
        forecast_full_draws = _parse_hdf5_int_attr(handle.attrs.get("forecast_full_draws"))
        if forecast_full_dates and state_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "forecast_full/states",
                (len(forecast_full_dates), len(state_names)),
                {0: forecast_full_dates, 1: state_names},
            )
        if forecast_full_dates and observable_names:
            forecast_full_observable_axes = {
                0: forecast_full_dates,
                1: observable_names,
            }
            for name in (
                "forecast_full/observables",
                "meansbands_full_forecastobs/mean",
                "meansbands_full_forecastobs/lower",
                "meansbands_full_forecastobs/upper",
            ):
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    name,
                    (len(forecast_full_dates), len(observable_names)),
                    forecast_full_observable_axes,
                )
        if forecast_full_dates and pseudo_observable_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "forecast_full/pseudo_observables",
                (len(forecast_full_dates), len(pseudo_observable_names)),
                {0: forecast_full_dates, 1: pseudo_observable_names},
            )
        if forecast_full_dates and forecast_full_draws is not None:
            draw_labels = tuple(f"draw_{index}" for index in range(forecast_full_draws))
            if state_names:
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    "forecast_full/state_samples",
                    (forecast_full_draws, len(forecast_full_dates), len(state_names)),
                    {0: draw_labels, 1: forecast_full_dates, 2: state_names},
                )
            if observable_names:
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    "forecast_full/observable_samples",
                    (forecast_full_draws, len(forecast_full_dates), len(observable_names)),
                    {0: draw_labels, 1: forecast_full_dates, 2: observable_names},
                )
            if pseudo_observable_names:
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    "forecast_full/pseudo_observable_samples",
                    (
                        forecast_full_draws,
                        len(forecast_full_dates),
                        len(pseudo_observable_names),
                    ),
                    {0: draw_labels, 1: forecast_full_dates, 2: pseudo_observable_names},
                )
            if shock_names:
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    "forecast_full/shock_samples",
                    (forecast_full_draws, len(forecast_full_dates), len(shock_names)),
                    {0: draw_labels, 1: forecast_full_dates, 2: shock_names},
                )

        history_dates = _parse_hdf5_name_attr(handle.attrs.get("history_dates"))
        if history_dates and observable_names:
            history_observable_axes = {
                0: history_dates,
                1: observable_names,
            }
            for name in (
                "forecast_mode/history_observables",
                "forecast_full/history_observables",
                "meansbands_mode_histobs/mean",
                "meansbands_mode_histobs/lower",
                "meansbands_mode_histobs/upper",
                "meansbands_full_histobs/mean",
                "meansbands_full_histobs/lower",
                "meansbands_full_histobs/upper",
            ):
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    name,
                    (len(history_dates), len(observable_names)),
                    history_observable_axes,
                )
            if forecast_full_draws is not None:
                draw_labels = tuple(f"draw_{index}" for index in range(forecast_full_draws))
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    "forecast_full/history_observable_samples",
                    (forecast_full_draws, len(history_dates), len(observable_names)),
                    {0: draw_labels, 1: history_dates, 2: observable_names},
                )
        if history_dates:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "posterior/log_likelihood_by_period",
                (len(history_dates),),
                {0: history_dates},
            )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "kalman/log_likelihood",
                (len(history_dates),),
                {0: history_dates},
            )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "kalman/total_log_likelihood",
                (1,),
                {0: ("total",)},
            )
            if state_names:
                kalman_state_axes = {0: history_dates, 1: state_names}
                for name in ("kalman/predicted_states", "kalman/filtered_states"):
                    _add_hdf5_dataset_labels(
                        labels,
                        handle,
                        h5py.Dataset,
                        name,
                        (len(history_dates), len(state_names)),
                        kalman_state_axes,
                    )
                kalman_covariance_axes = {
                    0: history_dates,
                    1: state_names,
                    2: state_names,
                }
                for name in (
                    "kalman/predicted_covariances",
                    "kalman/filtered_covariances",
                ):
                    _add_hdf5_dataset_labels(
                        labels,
                        handle,
                        h5py.Dataset,
                        name,
                        (len(history_dates), len(state_names), len(state_names)),
                        kalman_covariance_axes,
                    )
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    "kalman/final_filtered_state",
                    (len(state_names),),
                    {0: state_names},
                )
        if parameter_names:
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "posterior/parameter_values",
                (len(parameter_names),),
                {0: parameter_names},
            )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "posterior/log_prior_by_parameter",
                (len(parameter_names),),
                {0: parameter_names},
            )
        sampler_parameter_names = _parse_hdf5_name_attr(handle.attrs.get("sampler_parameter_names"))
        if not sampler_parameter_names:
            sampler_parameter_names = parameter_names
        if sampler_parameter_names:
            mhparams_shape = _hdf5_dataset_shape(
                handle,
                "sampler/mhparams",
                h5py.Dataset,
            )
            if mhparams_shape is not None and len(mhparams_shape) == 2:
                try:
                    parameter_axis, draw_axis, draws = _infer_sampler_mhparams_axes(
                        mhparams_shape,
                        len(sampler_parameter_names),
                        _parse_hdf5_int_attr(handle.attrs.get("sampler_draws")),
                    )
                except ValueError:
                    pass
                else:
                    draw_labels = tuple(f"draw_{index}" for index in range(draws))
                    labels["sampler/mhparams"] = {
                        parameter_axis: sampler_parameter_names,
                        draw_axis: draw_labels,
                    }
                    for matrix_name in (
                        "sampler/proposal_parameters",
                        "sampler/previous_parameters",
                    ):
                        _add_hdf5_dataset_labels(
                            labels,
                            handle,
                            h5py.Dataset,
                            matrix_name,
                            mhparams_shape,
                            {
                                parameter_axis: sampler_parameter_names,
                                draw_axis: draw_labels,
                            },
                        )
                    for trace_name in (
                        "sampler/accepted",
                        "sampler/log_posterior",
                        "sampler/proposal_log_posterior",
                        "sampler/previous_log_posterior",
                        "sampler/proposal_log_likelihood",
                        "sampler/previous_log_likelihood",
                        "sampler/proposal_log_prior",
                        "sampler/previous_log_prior",
                        "sampler/uniform_draw",
                        "sampler/log_acceptance",
                    ):
                        _add_hdf5_dataset_labels(
                            labels,
                            handle,
                            h5py.Dataset,
                            trace_name,
                            (draws,),
                            {0: draw_labels},
                        )
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                "sampler/proposal_covariance",
                (len(sampler_parameter_names), len(sampler_parameter_names)),
                {0: sampler_parameter_names, 1: sampler_parameter_names},
            )
            for covariance_name in (
                "sampler/draw_covariance",
                "sampler/input_proposal_covariance",
            ):
                _add_hdf5_dataset_labels(
                    labels,
                    handle,
                    h5py.Dataset,
                    covariance_name,
                    (len(sampler_parameter_names), len(sampler_parameter_names)),
                    {0: sampler_parameter_names, 1: sampler_parameter_names},
                )
        for name in (
            "posterior/log_posterior",
            "posterior/log_likelihood",
            "posterior/log_prior",
        ):
            _add_hdf5_dataset_labels(
                labels,
                handle,
                h5py.Dataset,
                name,
                (1,),
                {0: ("value",)},
            )
    return labels


def _add_hdf5_dataset_labels(
    labels: FixtureLabels,
    handle: Any,
    dataset_type: type[Any],
    name: str,
    expected_shape: tuple[int, ...],
    axis_labels: dict[int, tuple[str, ...]],
) -> None:
    shape = _hdf5_dataset_shape(handle, name, dataset_type)
    if shape == expected_shape:
        labels[name] = axis_labels


def _hdf5_dataset_shape(
    handle: Any,
    name: str,
    dataset_type: type[Any],
) -> tuple[int, ...] | None:
    try:
        dataset = handle[name]
    except KeyError:
        return None
    if not isinstance(dataset, dataset_type):
        return None
    return tuple(int(item) for item in dataset.shape)


def _parse_hdf5_name_attr(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())

    array = np.asarray(value)
    if array.ndim == 0:
        return _parse_hdf5_name_attr(array.item())

    names: list[str] = []
    for item in array.tolist():
        if isinstance(item, bytes):
            item = item.decode()
        text = str(item).strip()
        if text:
            names.append(text)
    return tuple(names)


def _parse_hdf5_int_attr(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        text = value.strip()
        return None if text == "" else int(text)
    array = np.asarray(value)
    if array.ndim == 0:
        return _parse_hdf5_int_attr(array.item())
    if array.size != 1:
        return None
    return int(array.reshape(-1)[0])


def _merge_manifest(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return metadata
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, dict):
        return metadata
    merged = {**existing, **metadata}
    existing_labels = existing.get("labels", {})
    metadata_labels = metadata.get("labels", {})
    if isinstance(existing_labels, dict) or isinstance(metadata_labels, dict):
        labels: dict[str, Any] = {}
        if isinstance(existing_labels, dict):
            labels.update(existing_labels)
        if isinstance(metadata_labels, dict):
            labels.update(metadata_labels)
        merged["labels"] = labels
    existing_samplers = existing.get("samplers", {})
    metadata_samplers = metadata.get("samplers", {})
    if isinstance(existing_samplers, dict) or isinstance(metadata_samplers, dict):
        samplers: dict[str, Any] = {}
        if isinstance(existing_samplers, dict):
            samplers.update(existing_samplers)
        if isinstance(metadata_samplers, dict):
            samplers.update(metadata_samplers)
        merged["samplers"] = samplers
    existing_shock_samples = existing.get("shock_samples", {})
    metadata_shock_samples = metadata.get("shock_samples", {})
    if isinstance(existing_shock_samples, dict) or isinstance(metadata_shock_samples, dict):
        shock_samples: dict[str, Any] = {}
        if isinstance(existing_shock_samples, dict):
            shock_samples.update(existing_shock_samples)
        if isinstance(metadata_shock_samples, dict):
            shock_samples.update(metadata_shock_samples)
        merged["shock_samples"] = shock_samples
    return merged


def _parse_axis_labels(axis_labels: dict[str, Any]) -> dict[int, tuple[str, ...]]:
    parsed: dict[int, tuple[str, ...]] = {}
    for axis_name, values in axis_labels.items():
        if not axis_name.startswith("axis"):
            continue
        try:
            axis = int(axis_name[4:])
        except ValueError:
            continue
        if not isinstance(values, list):
            continue
        parsed[axis] = tuple(str(value) for value in values)
    return parsed


def _max_abs_index(diff: np.ndarray) -> tuple[int, ...] | None:
    if diff.size == 0 or np.all(np.isnan(diff)):
        return None
    return tuple(int(item) for item in np.unravel_index(np.nanargmax(diff), diff.shape))


def _labels_for_index(
    labels: dict[int, tuple[str, ...]] | None,
    index: tuple[int, ...] | None,
) -> tuple[str | None, ...] | None:
    if labels is None or index is None:
        return None
    resolved: list[str | None] = []
    for axis, coordinate in enumerate(index):
        axis_labels = labels.get(axis)
        resolved.append(
            None
            if axis_labels is None or coordinate >= len(axis_labels)
            else axis_labels[coordinate]
        )
    return None if all(item is None for item in resolved) else tuple(resolved)


def _relative_stem(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    return relative.as_posix()


def _find_fixture_array(arrays: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name in arrays:
        return arrays[name]
    matches = [key for key in arrays if key.endswith(f"/{name}")]
    if not matches:
        msg = f"Fixture array not found: {name}"
        raise KeyError(msg)
    if len(matches) > 1:
        msg = f"Fixture array name is ambiguous for {name}: {', '.join(sorted(matches))}"
        raise ValueError(msg)
    return arrays[matches[0]]


def _write_npz_fixture(path: Path, arrays: dict[str, np.ndarray]) -> None:
    savez = cast(Any, np.savez)
    savez(path, **arrays)
