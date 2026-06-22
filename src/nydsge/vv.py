from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from nydsge.core import Parameter
from nydsge.estimate import EstimateResult
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

    diff = np.abs(expected_arr - actual_arr)
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    max_abs_index = _max_abs_index(diff)
    denominator = np.maximum(np.abs(expected_arr), np.finfo(np.float64).eps)
    rel = diff / denominator
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
            arrays[relative_stem] = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
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
