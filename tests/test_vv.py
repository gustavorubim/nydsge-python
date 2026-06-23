from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from nydsge.cli import app
from nydsge.estimate import (
    MetropolisHastingsResult,
    evaluate_log_posterior_for_parameter_values,
    save_sampler_result,
)
from nydsge.forecast import ForecastOutput, MeansBands
from nydsge.models import Model1002
from nydsge.solve import build_system, solve_canonical
from nydsge.vv import (
    check_fixture_coverage,
    check_sampler_proposal_trace,
    compare_arrays,
    compare_fixture_dirs,
    compare_sampler_results,
    load_canonical_fixture,
    load_fixture_arrays,
    load_fixture_labels,
    load_sampler_fixture_result,
    replay_sampler_proposal_posteriors,
    required_fixture_arrays,
    resolve_tolerance_profile,
    save_canonical_fixture,
    save_fixture_manifest,
    save_forecast_fixture,
    save_meansbands_fixture,
    save_model_metadata_fixture,
    save_parameter_fixture,
    save_steady_state_fixture,
    save_system_fixture,
    save_transition_fixture,
    summarize_sampler_fixture,
)


def test_compare_arrays_passes_within_tolerance() -> None:
    report = compare_arrays(
        "matrix",
        np.array([[1.0, 2.0]]),
        np.array([[1.0, 2.0 + 1.0e-12]]),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert report.passed
    assert report.status == "passed"


def test_resolve_tolerance_profile_defaults_and_overrides() -> None:
    strict = resolve_tolerance_profile("strict")
    assert strict.atol == 1.0e-10
    assert strict.rtol == 1.0e-10

    accelerator = resolve_tolerance_profile("ACCELERATOR")
    assert accelerator.name == "accelerator"
    assert accelerator.atol == 1.0e-5
    assert accelerator.rtol == 1.0e-5

    override = resolve_tolerance_profile("forecast", atol=2.0e-7)
    assert override.atol == 2.0e-7
    assert override.rtol == 1.0e-8


def test_resolve_tolerance_profile_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="Unknown tolerance profile"):
        resolve_tolerance_profile("loose")

    with pytest.raises(ValueError, match="nonnegative"):
        resolve_tolerance_profile("strict", atol=-1.0)


def test_required_fixture_arrays_profiles() -> None:
    model_metadata = required_fixture_arrays("model-metadata")
    parameters = required_fixture_arrays("parameters")
    steady_state = required_fixture_arrays("steady-state")
    financial_frictions = required_fixture_arrays("financial-frictions")
    kalman = required_fixture_arrays("kalman")
    posterior = required_fixture_arrays("posterior")
    model_setup = required_fixture_arrays("model-setup")
    matrix = required_fixture_arrays("matrix")
    forecast_mode = required_fixture_arrays("forecast-mode")
    forecast_mode_history = required_fixture_arrays("forecast-mode-history")
    forecast_full = required_fixture_arrays("forecast-full")
    forecast_full_history = required_fixture_arrays("forecast-full-history")
    hard_target = required_fixture_arrays("hard-target")
    sampler = required_fixture_arrays("sampler")
    sampler_trace = required_fixture_arrays("sampler-trace")
    sampler_proposal_trace = required_fixture_arrays("sampler-proposal-trace")

    assert model_metadata == (
        "metadata/observable_names",
        "metadata/pseudo_observable_names",
    )
    assert parameters == (
        "parameters/values",
        "parameters/scaled_values",
        "parameters/fixed",
        "parameters/bounds",
    )
    assert steady_state == ("steady_state/values",)
    assert financial_frictions == (
        "financial_frictions/inputs",
        "financial_frictions/values",
    )
    assert "kalman/log_likelihood" in kalman
    assert "kalman/filtered_states" in kalman
    assert "kalman/total_log_likelihood" in kalman
    assert "forecast_mode/observables" not in kalman
    assert posterior == (
        "posterior/log_posterior",
        "posterior/log_likelihood",
        "posterior/log_prior",
        "posterior/log_likelihood_by_period",
        "posterior/log_prior_by_parameter",
        "posterior/parameter_values",
    )
    assert "kalman/filtered_states" not in posterior
    assert set(parameters).issubset(set(model_setup))
    assert set(model_metadata).issubset(set(model_setup))
    assert set(steady_state).issubset(set(model_setup))
    assert set(model_setup).isdisjoint(set(matrix))
    assert "canonical/Gamma0" in matrix
    assert "transition/eu" in matrix
    assert "parameters/values" not in matrix
    assert "forecast_full/history_observables" not in matrix
    assert "forecast_mode/observables" in forecast_mode
    assert "meansbands_mode_forecastobs/mean" in forecast_mode
    assert "forecast_full/observables" not in forecast_mode
    assert "forecast_mode/history_observables" in forecast_mode_history
    assert "meansbands_mode_histobs/mean" in forecast_mode_history
    assert "forecast_mode/observables" not in forecast_mode_history
    assert "forecast_full/observables" in forecast_full
    assert "forecast_full/observable_samples" in forecast_full
    assert "meansbands_full_forecastobs/mean" in forecast_full
    assert "forecast_full/history_observable_samples" not in forecast_full
    assert "forecast_full/history_observables" in forecast_full_history
    assert "forecast_full/history_observable_samples" in forecast_full_history
    assert "meansbands_full_histobs/mean" in forecast_full_history
    assert "forecast_full/observable_samples" not in forecast_full_history
    assert set(model_setup).issubset(set(hard_target))
    assert set(matrix).issubset(set(hard_target))
    assert set(posterior).issubset(set(hard_target))
    assert set(forecast_full).issubset(set(hard_target))
    assert set(forecast_full_history).issubset(set(hard_target))
    assert "forecast_mode/observables" in hard_target
    assert "forecast_mode/states" not in hard_target
    assert "forecast_full/history_observables" in hard_target
    assert "forecast_full/observable_samples" in hard_target
    assert "forecast_full/history_observable_samples" in hard_target
    assert sampler == ("sampler/mhparams", "sampler/fixed", "sampler/proposal_covariance")
    assert sampler_trace == (
        "sampler/mhparams",
        "sampler/fixed",
        "sampler/proposal_covariance",
        "sampler/accepted",
        "sampler/log_posterior",
    )
    assert set(sampler).issubset(set(sampler_trace))
    assert sampler_proposal_trace == (
        "sampler/mhparams",
        "sampler/fixed",
        "sampler/proposal_covariance",
        "sampler/accepted",
        "sampler/log_posterior",
        "sampler/proposal_parameters",
        "sampler/previous_parameters",
        "sampler/proposal_log_posterior",
        "sampler/previous_log_posterior",
        "sampler/uniform_draw",
        "sampler/log_acceptance",
    )
    assert set(sampler_trace).issubset(set(sampler_proposal_trace))
    assert set(sampler).issubset(set(hard_target)) is False

    with pytest.raises(ValueError, match="Unknown fixture coverage profile"):
        required_fixture_arrays("small")


def test_compare_arrays_reports_shape_mismatch() -> None:
    report = compare_arrays("matrix", np.ones((1, 2)), np.ones((2, 1)))
    assert not report.passed
    assert report.status == "shape_mismatch"


def test_compare_arrays_treats_matching_nonfinite_values_as_equal() -> None:
    report = compare_arrays(
        "diagnostics/core",
        np.array([np.inf, np.nan, 1.0]),
        np.array([np.inf, np.nan, 1.0 + 1.0e-12]),
    )

    assert report.passed
    assert report.max_abs_diff == pytest.approx(1.0e-12)


def test_compare_arrays_reports_worst_mismatch_coordinate() -> None:
    report = compare_arrays(
        "forecast/observables",
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1.0, 2.0], [10.0, 4.5]]),
    )

    assert not report.passed
    assert report.max_abs_index == (1, 0)
    assert report.max_abs_label is None
    assert report.to_dict()["max_abs_index"] == (1, 0)


def test_compare_arrays_reports_manifest_labels_for_worst_mismatch() -> None:
    report = compare_arrays(
        "forecast/observables",
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1.0, 9.0], [3.1, 4.0]]),
        labels={
            0: ("2018-Q4", "2019-Q1"),
            1: ("obs_gdp", "obs_hours"),
        },
    )

    assert not report.passed
    assert report.max_abs_index == (0, 1)
    assert report.max_abs_label == ("2018-Q4", "obs_hours")


def test_load_fixture_arrays_reads_npz_and_compare_dirs(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    np.savez(oracle / "system.npz", TTT=np.eye(2), CCC=np.zeros(2))
    np.savez(candidate / "system.npz", TTT=np.eye(2), CCC=np.zeros(2))

    arrays = load_fixture_arrays(oracle)
    assert sorted(arrays) == ["system/CCC", "system/TTT"]

    report = compare_fixture_dirs(oracle, candidate)
    assert report.passed
    assert {item.name for item in report.comparisons} == {"system/CCC", "system/TTT"}


def test_load_fixture_arrays_ignores_non_numeric_csv(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    np.savetxt(oracle / "numeric.csv", np.array([[1.0, 2.0]]), delimiter=",")
    pd.DataFrame({"date": ["1960-Q1"], "obs_gdp": [1.0]}).to_csv(
        oracle / "observables.csv",
        index=False,
    )

    arrays = load_fixture_arrays(oracle)

    assert sorted(arrays) == ["numeric"]
    np.testing.assert_allclose(arrays["numeric"], np.array([[1.0, 2.0]]))


def test_load_fixture_arrays_reads_hdf5_dataset_paths_without_file_prefix(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "m1002_ss10.h5"
    with h5py.File(path, "w") as handle:
        handle["system/TTT"] = np.eye(2)
        handle["canonical/Gamma0"] = np.eye(1)

    arrays = load_fixture_arrays(oracle)

    assert sorted(arrays) == ["canonical/Gamma0", "system/TTT"]
    np.testing.assert_allclose(arrays["system/TTT"], np.eye(2))


def test_load_fixture_arrays_reads_case_insensitive_hdf5_suffix(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "M1002_SS10.H5"
    with h5py.File(path, "w") as handle:
        handle["parameters/values"] = np.array([1.0])
        handle.attrs["parameter_names"] = "alpha"

    arrays = load_fixture_arrays(oracle)
    labels = load_fixture_labels(oracle)

    np.testing.assert_allclose(arrays["parameters/values"], np.array([1.0]))
    assert labels["parameters/values"] == {0: ("alpha",)}


def test_load_fixture_labels_reads_julia_hdf5_metadata(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "m1002_ss10.h5"
    with h5py.File(path, "w") as handle:
        handle["parameters/values"] = np.array([1.0, 2.0])
        handle["parameters/scaled_values"] = np.array([1.0, 2.0])
        handle["parameters/fixed"] = np.array([0.0, 1.0])
        handle["parameters/bounds"] = np.array([[0.0, 1.0], [1.0, 2.0]])
        handle["steady_state/values"] = np.array([3.0, 4.0])
        handle["canonical/Gamma0"] = np.zeros((2, 2))
        handle["canonical/Psi"] = np.zeros((2, 1))
        handle["system/ZZ"] = np.zeros((1, 3))
        handle["system/DD_pseudo"] = np.zeros(2)
        handle["system/ZZ_pseudo"] = np.zeros((2, 3))
        handle["transition/eu"] = np.array([1.0, 1.0])
        handle.attrs["parameter_names"] = "alpha,beta"
        handle.attrs["steady_state_names"] = "z_star,rstar"
        handle.attrs["endogenous_state_names"] = "y_t,pi_t"
        handle.attrs["augmented_state_names"] = "y_t1"
        handle.attrs["exogenous_shock_names"] = "g_sh"
        handle.attrs["equation_names"] = "eq_y,eq_pi"
        handle.attrs["observable_names"] = "obs_gdp"
        handle.attrs["pseudo_observable_names"] = "y_t,\u03c0_t"

    labels = load_fixture_labels(oracle)
    arrays = load_fixture_arrays(oracle)

    assert arrays["metadata/observable_names"].shape == (1, len("obs_gdp"))
    assert arrays["metadata/observable_names"][0, 0] == ord("o")
    assert arrays["metadata/pseudo_observable_names"].shape == (2, 3)
    assert arrays["metadata/pseudo_observable_names"][1, 0] == ord("\u03c0")
    assert labels["metadata/observable_names"] == {
        0: ("obs_gdp",),
        1: tuple(f"char_{index}" for index in range(len("obs_gdp"))),
    }
    assert labels["metadata/pseudo_observable_names"] == {
        0: ("y_t", "\u03c0_t"),
        1: ("char_0", "char_1", "char_2"),
    }
    assert labels["parameters/values"] == {0: ("alpha", "beta")}
    assert labels["parameters/scaled_values"] == {0: ("alpha", "beta")}
    assert labels["parameters/fixed"] == {0: ("alpha", "beta")}
    assert labels["parameters/bounds"] == {
        0: ("alpha", "beta"),
        1: ("lower", "upper"),
    }
    assert labels["steady_state/values"] == {0: ("z_star", "rstar")}
    assert labels["canonical/Gamma0"] == {
        0: ("eq_y", "eq_pi"),
        1: ("y_t", "pi_t"),
    }
    assert labels["canonical/Psi"] == {0: ("eq_y", "eq_pi"), 1: ("g_sh",)}
    assert labels["system/ZZ"] == {0: ("obs_gdp",), 1: ("y_t", "pi_t", "y_t1")}
    assert labels["system/DD_pseudo"] == {0: ("y_t", "\u03c0_t")}
    assert labels["system/ZZ_pseudo"] == {
        0: ("y_t", "\u03c0_t"),
        1: ("y_t", "pi_t", "y_t1"),
    }
    assert labels["transition/eu"] == {0: ("existence", "uniqueness")}


def test_load_fixture_labels_reads_sampler_dimensions_from_julia_hdf5_attributes(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "m1002_ss10.h5"
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = np.zeros((2, 3))
        handle["sampler/fixed"] = np.array([0, 1, 0], dtype=np.int8)
        handle["sampler/proposal_covariance"] = np.eye(3)
        handle["sampler/draw_covariance"] = np.eye(3)
        handle["sampler/input_proposal_covariance"] = 2.0 * np.eye(3)
        handle["sampler/accepted"] = np.array([1, 0])
        handle["sampler/log_posterior"] = np.array([-10.0, -9.5])
        handle["sampler/proposal_parameters"] = np.zeros((2, 3))
        handle["sampler/previous_parameters"] = np.ones((2, 3))
        handle["sampler/proposal_log_posterior"] = np.array([-9.9, -9.4])
        handle["sampler/previous_log_posterior"] = np.array([-10.1, -9.6])
        handle["sampler/uniform_draw"] = np.array([0.25, 0.75])
        handle["sampler/log_acceptance"] = np.array([0.2, -0.2])
        handle.attrs["sampler_parameter_names"] = "alpha,beta,gamma"

    labels = load_fixture_labels(oracle)

    assert labels["sampler/mhparams"] == {
        0: ("draw_0", "draw_1"),
        1: ("alpha", "beta", "gamma"),
    }
    assert labels["sampler/proposal_covariance"] == {
        0: ("alpha", "beta", "gamma"),
        1: ("alpha", "beta", "gamma"),
    }
    assert labels["sampler/fixed"] == {0: ("alpha", "beta", "gamma")}
    assert labels["sampler/draw_covariance"] == {
        0: ("alpha", "beta", "gamma"),
        1: ("alpha", "beta", "gamma"),
    }
    assert labels["sampler/input_proposal_covariance"] == {
        0: ("alpha", "beta", "gamma"),
        1: ("alpha", "beta", "gamma"),
    }
    assert labels["sampler/accepted"] == {0: ("draw_0", "draw_1")}
    assert labels["sampler/log_posterior"] == {0: ("draw_0", "draw_1")}
    assert labels["sampler/proposal_parameters"] == {
        0: ("draw_0", "draw_1"),
        1: ("alpha", "beta", "gamma"),
    }
    assert labels["sampler/previous_parameters"] == {
        0: ("draw_0", "draw_1"),
        1: ("alpha", "beta", "gamma"),
    }
    assert labels["sampler/proposal_log_posterior"] == {0: ("draw_0", "draw_1")}
    assert labels["sampler/previous_log_posterior"] == {0: ("draw_0", "draw_1")}
    assert labels["sampler/uniform_draw"] == {0: ("draw_0", "draw_1")}
    assert labels["sampler/log_acceptance"] == {0: ("draw_0", "draw_1")}


def test_load_fixture_labels_reads_parameter_first_sampler_dimensions(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "m1002_ss10.h5"
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = np.zeros((3, 2))
        handle["sampler/proposal_covariance"] = np.eye(3)
        handle.attrs["sampler_parameter_names"] = "alpha,beta,gamma"

    labels = load_fixture_labels(oracle)

    assert labels["sampler/mhparams"] == {
        0: ("alpha", "beta", "gamma"),
        1: ("draw_0", "draw_1"),
    }


def test_summarize_sampler_fixture_reads_julia_hdf5_sampler_metadata(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "m1002_ss10_sampler_smoke.h5"
    covariance = np.diag([1.0, 4.0, 9.0])
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = np.array(
            [
                [0.1, 0.2],
                [1.1, 1.2],
                [2.1, 2.2],
            ],
            dtype=np.float64,
        )
        handle["sampler/fixed"] = np.array([0, 1, 0], dtype=np.int8)
        handle["sampler/proposal_covariance"] = covariance
        handle["sampler/draw_covariance"] = covariance
        handle["sampler/input_proposal_covariance"] = 1.0e-8 * np.eye(3)
        handle["sampler/accepted"] = np.array([1, 0], dtype=np.int8)
        handle["sampler/log_posterior"] = np.array([-12.0, -10.0], dtype=np.float64)
        handle["sampler/proposal_parameters"] = np.array(
            [
                [0.15, 0.25],
                [1.15, 1.25],
                [2.15, 2.25],
            ],
            dtype=np.float64,
        )
        handle["sampler/previous_parameters"] = np.array(
            [
                [0.05, 0.15],
                [1.05, 1.15],
                [2.05, 2.15],
            ],
            dtype=np.float64,
        )
        handle["sampler/proposal_log_posterior"] = np.array([-11.5, -9.5])
        handle["sampler/previous_log_posterior"] = np.array([-12.5, -10.5])
        handle["sampler/uniform_draw"] = np.array([0.25, 0.75])
        handle["sampler/log_acceptance"] = np.array([1.0, -1.0])
        handle.attrs["sampler_parameter_names"] = "alpha,beta,gamma"
        handle.attrs["sampler_draws"] = "2"
        handle.attrs["sampler_burnin"] = 0
        handle.attrs["sampler_proposal_scale"] = "1.0e-8"
        handle.attrs["sampler_covariance_source"] = "saved_draw_covariance"
        handle.attrs["sampler_trace_available"] = "true"
        handle.attrs["sampler_proposal_trace_available"] = "true"
        handle.attrs["sampler_acceptance_rate"] = "0.5"
        handle.attrs["sampler_block_acceptance_rates"] = "0.5"
        handle.attrs["sampler_input_proposal_covariance_available"] = "true"
        handle.attrs["sampler_seed"] = 123

    summary = summarize_sampler_fixture(oracle)

    assert summary.fixture_path == path
    assert summary.parameter_names == ("alpha", "beta", "gamma")
    assert summary.fixed_mask == (False, True, False)
    assert summary.fixed_count == 1
    assert summary.mhparams_shape == (3, 2)
    assert summary.parameter_axis == 0
    assert summary.draw_axis == 1
    assert summary.draws == 2
    assert summary.parameter_count == 3
    assert summary.covariance_shape == (3, 3)
    assert summary.covariance_source == "saved_draw_covariance"
    assert summary.covariance_min_eigenvalue == 1.0
    assert summary.covariance_max_eigenvalue == 9.0
    assert summary.covariance_condition_number == 9.0
    assert summary.covariance_positive_semidefinite is True
    assert summary.input_proposal_covariance_available is True
    assert summary.trace_available is True
    assert summary.accepted_shape == (2,)
    assert summary.log_posterior_shape == (2,)
    assert summary.accepted_draws == 1
    assert summary.realized_acceptance_rate == 0.5
    assert summary.log_posterior_mean == -11.0
    assert summary.log_posterior_minimum == -12.0
    assert summary.log_posterior_maximum == -10.0
    assert summary.proposal_trace_available is True
    assert summary.proposal_parameters_shape == (3, 2)
    assert summary.previous_parameters_shape == (3, 2)
    assert summary.proposal_log_posterior_shape == (2,)
    assert summary.previous_log_posterior_shape == (2,)
    assert summary.uniform_draw_shape == (2,)
    assert summary.log_acceptance_shape == (2,)
    assert summary.proposal_log_posterior_minimum == -11.5
    assert summary.proposal_log_posterior_maximum == -9.5
    assert summary.log_acceptance_minimum == -1.0
    assert summary.log_acceptance_maximum == 1.0
    assert summary.metadata["draws"] == 2
    assert summary.metadata["burnin"] == 0
    assert summary.metadata["proposal_scale"] == 1.0e-8
    assert summary.metadata["trace_available"] is True
    assert summary.metadata["proposal_trace_available"] is True
    assert summary.metadata["acceptance_rate"] == 0.5
    assert summary.metadata["block_acceptance_rates"] == 0.5
    assert summary.metadata["seed"] == 123
    assert summary.unavailable_diagnostics == ()
    assert summary.unavailable_proposal_diagnostics == ()

    payload = summary.to_dict()
    assert payload["fixture_path"] == str(path)
    assert payload["fixed_mask"] == [False, True, False]
    assert payload["fixed_count"] == 1
    assert payload["mhparams_shape"] == [3, 2]
    assert payload["covariance_shape"] == [3, 3]
    assert payload["accepted_shape"] == [2]
    assert payload["log_posterior_shape"] == [2]
    assert payload["proposal_trace_available"] is True
    assert payload["proposal_parameters_shape"] == [3, 2]
    assert payload["uniform_draw_shape"] == [2]
    assert payload["unavailable_diagnostics"] == []
    assert payload["unavailable_proposal_diagnostics"] == []


def test_summarize_sampler_fixture_rejects_ambiguous_square_draws(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    path = oracle / "ambiguous_sampler.h5"
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = np.eye(2)
        handle["sampler/proposal_covariance"] = np.eye(2)
        handle.attrs["sampler_parameter_names"] = "alpha,beta"

    with pytest.raises(ValueError, match="orientation is ambiguous"):
        summarize_sampler_fixture(path)


def test_summarize_sampler_fixture_reports_missing_trace_diagnostics(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sampler.h5"
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = np.zeros((2, 3))
        handle["sampler/proposal_covariance"] = np.eye(2)
        handle.attrs["sampler_parameter_names"] = "alpha,beta"
        handle.attrs["sampler_draws"] = 3

    summary = summarize_sampler_fixture(path)

    assert summary.trace_available is False
    assert summary.accepted_shape is None
    assert summary.log_posterior_shape is None
    assert summary.proposal_trace_available is False
    assert summary.unavailable_diagnostics == ("accepted", "log_posterior")
    assert summary.unavailable_proposal_diagnostics == (
        "proposal_parameters",
        "previous_parameters",
        "proposal_log_posterior",
        "previous_log_posterior",
        "uniform_draw",
        "log_acceptance",
    )


def test_load_sampler_fixture_result_converts_julia_trace_to_sampler_result(tmp_path) -> None:
    path = _write_sampler_trace_hdf5(tmp_path)

    result = load_sampler_fixture_result(path)

    assert result.parameter_names == ("alpha", "beta")
    assert result.parameter_draws.shape == (3, 2)
    np.testing.assert_allclose(
        result.parameter_draws,
        np.array([[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]]),
    )
    np.testing.assert_allclose(result.estimation_draws, result.parameter_draws)
    np.testing.assert_array_equal(result.accepted, np.array([True, False, True]))
    np.testing.assert_allclose(result.log_posterior, np.array([-3.0, -2.5, -2.0]))
    np.testing.assert_allclose(result.proposal_covariance, np.diag([0.5, 2.0]))
    assert result.acceptance_rate == 2 / 3
    assert result.seed == 123
    assert result.burnin == 0
    assert result.n_blocks == 1
    assert result.n_param_blocks == 1
    assert result.mhthin == 1
    assert result.proposal_scale == 1.0e-8


def test_compare_sampler_results_passes_matching_julia_trace_and_python_archive(
    tmp_path,
) -> None:
    oracle_path = _write_sampler_trace_hdf5(tmp_path)
    oracle_result = load_sampler_fixture_result(oracle_path)
    candidate_path = save_sampler_result(oracle_result, tmp_path / "candidate_sampler.npz")

    report = compare_sampler_results(
        oracle_path,
        candidate_path,
        oracle_result=oracle_result,
        candidate_result=oracle_result,
        windows=2,
    )

    assert report.passed
    assert {item.name for item in report.comparisons} == {
        "parameter_draws",
        "log_posterior",
        "accepted",
        "proposal_covariance",
        "diagnostics/core",
        "diagnostics/acceptance_windows",
        "diagnostics/parameters",
    }
    payload = report.to_dict()
    assert payload["oracle_sampler"] == str(oracle_path)
    assert payload["candidate_sampler"] == str(candidate_path)
    assert payload["passed"] is True


def test_check_sampler_proposal_trace_replays_acceptance_bookkeeping(
    tmp_path,
) -> None:
    oracle_path = _write_sampler_trace_hdf5(tmp_path)

    report = check_sampler_proposal_trace(oracle_path)

    assert report.passed
    assert {item.name for item in report.comparisons} == {
        "proposal_trace/log_acceptance",
        "proposal_trace/accepted",
        "proposal_trace/retained_log_posterior",
        "proposal_trace/retained_parameters",
    }
    payload = report.to_dict()
    assert payload["sampler_path"] == str(oracle_path)
    assert payload["passed"] is True


def test_check_sampler_proposal_trace_reports_failed_identity(
    tmp_path,
) -> None:
    h5py = pytest.importorskip("h5py")
    oracle_path = _write_sampler_trace_hdf5(tmp_path)
    with h5py.File(oracle_path, "r+") as handle:
        handle["sampler/log_acceptance"][1] = -0.01

    report = check_sampler_proposal_trace(oracle_path)

    assert not report.passed
    statuses = {item.name: item.status for item in report.comparisons}
    assert statuses["proposal_trace/log_acceptance"] == "failed"
    assert statuses["proposal_trace/accepted"] == "failed"


def test_replay_sampler_proposal_posteriors_matches_saved_trace(
    tmp_path,
) -> None:
    model = Model1002()
    observations = np.zeros((1, len(model.observables)))
    oracle_path = _write_sampler_posterior_replay_hdf5(tmp_path, model, observations)

    report = replay_sampler_proposal_posteriors(model, observations, oracle_path)

    assert report.passed
    assert report.draws == 2
    assert report.parameter_count == len(model.parameters)
    assert {item.name for item in report.comparisons} == {
        "proposal_trace/proposal_log_posterior",
        "proposal_trace/previous_log_posterior",
        "proposal_trace/log_acceptance_from_replay",
        "proposal_trace/proposal_log_likelihood",
        "proposal_trace/previous_log_likelihood",
        "proposal_trace/proposal_log_prior",
        "proposal_trace/previous_log_prior",
    }
    payload = report.to_dict()
    assert payload["sampler_path"] == str(oracle_path)
    assert payload["passed"] is True


def test_check_fixture_coverage_reports_missing_required_arrays(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    np.savez(oracle / "canonical.npz", Gamma0=np.eye(1))

    report = check_fixture_coverage(oracle, profile="matrix")

    assert not report.passed
    assert "canonical/Gamma0" in report.available
    assert "canonical/Gamma1" in report.missing


def test_check_fixture_coverage_passes_complete_matrix_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _write_required_arrays(oracle, required_fixture_arrays("matrix"))

    report = check_fixture_coverage(oracle, profile="matrix")

    assert report.passed
    assert report.missing == ()


def test_check_fixture_coverage_passes_complete_model_metadata_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _write_required_arrays(oracle, required_fixture_arrays("model-metadata"))

    report = check_fixture_coverage(oracle, profile="model-metadata")

    assert report.passed
    assert report.missing == ()


def test_check_fixture_coverage_passes_complete_sampler_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _write_required_arrays(oracle, required_fixture_arrays("sampler"))

    report = check_fixture_coverage(oracle, profile="sampler")

    assert report.passed
    assert report.missing == ()


def test_check_fixture_coverage_passes_complete_sampler_trace_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _write_required_arrays(oracle, required_fixture_arrays("sampler-trace"))

    report = check_fixture_coverage(oracle, profile="sampler-trace")

    assert report.passed
    assert report.missing == ()


def test_check_fixture_coverage_passes_complete_sampler_proposal_trace_profile(
    tmp_path,
) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _write_required_arrays(oracle, required_fixture_arrays("sampler-proposal-trace"))

    report = check_fixture_coverage(oracle, profile="sampler-proposal-trace")

    assert report.passed
    assert report.missing == ()


def test_save_canonical_fixture_writes_reloadable_archive(tmp_path) -> None:
    canonical_dir = tmp_path / "candidate"
    canonical = load_canonical_fixture(_write_canonical_fixture(tmp_path / "source"))

    path = save_canonical_fixture(canonical, canonical_dir)
    arrays = load_fixture_arrays(canonical_dir)

    assert path == canonical_dir / "canonical.npz"
    assert sorted(arrays) == [
        "canonical/C",
        "canonical/Gamma0",
        "canonical/Gamma1",
        "canonical/Pi",
        "canonical/Psi",
    ]
    np.testing.assert_allclose(arrays["canonical/Gamma0"], np.eye(1))


def test_save_fixture_manifest_is_ignored_by_array_loader(tmp_path) -> None:
    directory = tmp_path / "candidate"
    path = save_fixture_manifest(directory, {"kind": "test", "arrays": ["A"]})
    np.save(directory / "A.npy", np.eye(1))

    arrays = load_fixture_arrays(directory)

    assert path == directory / "manifest.json"
    assert sorted(arrays) == ["A"]


def test_load_fixture_labels_reads_manifest_metadata(tmp_path) -> None:
    directory = tmp_path / "candidate"
    save_fixture_manifest(
        directory,
        {
            "labels": {
                "forecast/observables": {
                    "axis0": ["2018-Q4"],
                    "axis1": ["obs_gdp", "obs_hours"],
                }
            }
        },
    )

    labels = load_fixture_labels(directory)

    assert labels == {"forecast/observables": {0: ("2018-Q4",), 1: ("obs_gdp", "obs_hours")}}


def test_save_fixture_manifest_merges_existing_labels(tmp_path) -> None:
    directory = tmp_path / "candidate"
    save_fixture_manifest(
        directory,
        {
            "kind": "first",
            "labels": {
                "system/TTT": {
                    "axis0": ["a"],
                    "axis1": ["b"],
                }
            },
        },
    )
    save_fixture_manifest(
        directory,
        {
            "kind": "second",
            "labels": {
                "forecast/observables": {
                    "axis0": ["2018-Q4"],
                    "axis1": ["obs_gdp"],
                }
            },
        },
    )

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["kind"] == "second"
    assert sorted(manifest["labels"]) == ["forecast/observables", "system/TTT"]
    assert load_fixture_labels(directory) == {
        "forecast/observables": {0: ("2018-Q4",), 1: ("obs_gdp",)},
        "system/TTT": {0: ("a",), 1: ("b",)},
    }


def test_save_system_fixture_writes_reloadable_archive(tmp_path) -> None:
    system = build_system(
        TTT=np.eye(2),
        RRR=np.ones((2, 1)),
        CCC=np.array([1.0, 2.0]),
        ZZ=np.eye(2),
        DD=np.zeros(2),
        QQ=np.eye(1),
        EE=np.eye(2),
        ZZ_pseudo=np.ones((1, 2)),
        DD_pseudo=np.zeros(1),
    )

    path = save_system_fixture(system, tmp_path)
    arrays = load_fixture_arrays(tmp_path)

    assert path == tmp_path / "system.npz"
    assert sorted(arrays) == [
        "system/CCC",
        "system/DD",
        "system/DD_pseudo",
        "system/EE",
        "system/QQ",
        "system/RRR",
        "system/TTT",
        "system/ZZ",
        "system/ZZ_pseudo",
    ]
    np.testing.assert_allclose(arrays["system/CCC"], np.array([1.0, 2.0]))


def test_save_parameter_fixture_writes_numeric_parameter_arrays(tmp_path) -> None:
    model = Model1002()

    path = save_parameter_fixture(model.parameters, tmp_path)
    arrays = load_fixture_arrays(tmp_path)

    assert path == tmp_path / "parameters.npz"
    assert arrays["parameters/values"].shape == (95,)
    assert arrays["parameters/scaled_values"].shape == (95,)
    assert arrays["parameters/fixed"].shape == (95,)
    assert arrays["parameters/bounds"].shape == (95, 2)
    alpha_row = list(model.parameters).index("alpha")
    assert arrays["parameters/values"][alpha_row] == model.parameters["alpha"].value
    assert arrays["parameters/scaled_values"][alpha_row] == model.parameters["alpha"].scaled_value
    assert arrays["parameters/fixed"][alpha_row] == model.parameters["alpha"].fixed


def test_save_steady_state_fixture_writes_numeric_values(tmp_path) -> None:
    model = Model1002()
    steady_state = model.steadystate()

    path = save_steady_state_fixture(steady_state, tmp_path)
    arrays = load_fixture_arrays(tmp_path)

    assert path == tmp_path / "steady_state.npz"
    assert arrays["steady_state/values"].shape == (22,)
    z_star_row = list(steady_state).index("z_star")
    assert arrays["steady_state/values"][z_star_row] == steady_state["z_star"]


def test_save_forecast_fixture_writes_reloadable_archive(tmp_path) -> None:
    forecast = ForecastOutput(
        states=np.ones((2, 3)),
        observables=np.zeros((2, 1)),
    )

    path = save_forecast_fixture(forecast, tmp_path)
    arrays = load_fixture_arrays(tmp_path)

    assert path == tmp_path / "forecast.npz"
    assert sorted(arrays) == ["forecast/observables", "forecast/states"]
    np.testing.assert_allclose(arrays["forecast/states"], np.ones((2, 3)))


def test_save_forecast_fixture_includes_optional_history_outputs(tmp_path) -> None:
    forecast = ForecastOutput(
        states=np.ones((2, 3)),
        observables=np.zeros((2, 1)),
        pseudo_observables=4.0 * np.ones((2, 2)),
        history_states=2.0 * np.ones((4, 3)),
        history_observables=3.0 * np.ones((4, 1)),
        history_pseudo_observables=5.0 * np.ones((4, 2)),
    )

    path = save_forecast_fixture(forecast, tmp_path)
    arrays = load_fixture_arrays(tmp_path)

    assert path == tmp_path / "forecast.npz"
    assert sorted(arrays) == [
        "forecast/history_observables",
        "forecast/history_pseudo_observables",
        "forecast/history_states",
        "forecast/observables",
        "forecast/pseudo_observables",
        "forecast/states",
    ]
    np.testing.assert_allclose(arrays["forecast/history_states"], 2.0 * np.ones((4, 3)))
    np.testing.assert_allclose(arrays["forecast/pseudo_observables"], 4.0 * np.ones((2, 2)))


def test_save_forecast_fixture_includes_conditional_outputs(tmp_path) -> None:
    forecast = ForecastOutput(
        states=np.ones((2, 3)),
        observables=np.zeros((2, 1)),
        conditional_shocks=2.0 * np.ones((1, 4)),
        conditional_states=3.0 * np.ones((1, 3)),
        conditional_observables=4.0 * np.ones((1, 1)),
    )

    path = save_forecast_fixture(forecast, tmp_path)
    arrays = load_fixture_arrays(tmp_path)

    assert path == tmp_path / "forecast.npz"
    assert "forecast/conditional_shocks" in arrays
    assert "forecast/conditional_states" in arrays
    assert "forecast/conditional_observables" in arrays
    np.testing.assert_allclose(arrays["forecast/conditional_shocks"], 2.0 * np.ones((1, 4)))


def test_save_forecast_fixture_includes_sample_outputs(tmp_path) -> None:
    forecast = ForecastOutput(
        states=np.ones((2, 3)),
        observables=np.zeros((2, 1)),
        pseudo_observables=4.0 * np.ones((2, 2)),
        state_samples=np.ones((5, 2, 3)),
        observable_samples=2.0 * np.ones((5, 2, 1)),
        pseudo_observable_samples=3.0 * np.ones((5, 2, 2)),
        history_state_samples=4.0 * np.ones((5, 4, 3)),
        history_observable_samples=5.0 * np.ones((5, 4, 1)),
        history_pseudo_observable_samples=6.0 * np.ones((5, 4, 2)),
    )

    path = save_forecast_fixture(forecast, tmp_path)
    arrays = load_fixture_arrays(tmp_path)

    assert path == tmp_path / "forecast.npz"
    assert arrays["forecast/state_samples"].shape == (5, 2, 3)
    assert arrays["forecast/observable_samples"].shape == (5, 2, 1)
    assert arrays["forecast/pseudo_observable_samples"].shape == (5, 2, 2)
    assert arrays["forecast/history_state_samples"].shape == (5, 4, 3)
    assert arrays["forecast/history_observable_samples"].shape == (5, 4, 1)
    assert arrays["forecast/history_pseudo_observable_samples"].shape == (5, 4, 2)


def test_save_meansbands_fixture_writes_reloadable_archive(tmp_path) -> None:
    bands = MeansBands(
        mean=np.ones((2, 3)),
        lower=np.zeros((2, 3)),
        upper=2.0 * np.ones((2, 3)),
    )

    path = save_meansbands_fixture(bands, tmp_path)
    arrays = load_fixture_arrays(tmp_path)

    assert path == tmp_path / "meansbands.npz"
    assert sorted(arrays) == ["meansbands/lower", "meansbands/mean", "meansbands/upper"]
    np.testing.assert_allclose(arrays["meansbands/upper"], 2.0 * np.ones((2, 3)))


def test_load_canonical_and_save_transition_fixtures(tmp_path) -> None:
    canonical_dir = _write_canonical_fixture(tmp_path / "canonical")
    transition_dir = tmp_path / "transition"

    canonical = load_canonical_fixture(canonical_dir)
    result = solve_canonical(canonical)
    path = save_transition_fixture(result, transition_dir)
    arrays = load_fixture_arrays(transition_dir)

    assert path == transition_dir / "transition.npz"
    assert sorted(arrays) == [
        "transition/CCC",
        "transition/RRR",
        "transition/TTT",
        "transition/eu",
    ]
    np.testing.assert_allclose(arrays["transition/TTT"], np.array([[0.5]]))
    np.testing.assert_allclose(arrays["transition/eu"], np.array([1, 1]))


def test_compare_fixture_dirs_reports_missing_and_extra(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    np.save(oracle / "Gamma0.npy", np.eye(1))
    np.save(candidate / "Gamma1.npy", np.eye(1))

    report = compare_fixture_dirs(oracle, candidate)
    statuses = {item.name: item.status for item in report.comparisons}
    assert statuses == {"Gamma0": "missing_candidate", "Gamma1": "extra_candidate"}


def test_compare_fixture_dirs_uses_manifest_labels_for_failures(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    np.savez(oracle / "forecast.npz", observables=np.array([[1.0, 2.0]]))
    np.savez(candidate / "forecast.npz", observables=np.array([[1.0, 3.5]]))
    save_fixture_manifest(
        candidate,
        {
            "labels": {
                "forecast/observables": {
                    "axis0": ["2018-Q4"],
                    "axis1": ["obs_gdp", "obs_hours"],
                }
            }
        },
    )

    report = compare_fixture_dirs(oracle, candidate)
    comparison = report.comparisons[0]

    assert comparison.status == "failed"
    assert comparison.max_abs_index == (0, 1)
    assert comparison.max_abs_label == ("2018-Q4", "obs_hours")


def test_compare_fixture_dirs_uses_julia_hdf5_labels_for_parameter_failures(
    tmp_path,
) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    with h5py.File(oracle / "m1002_ss10.h5", "w") as handle:
        handle["parameters/values"] = np.array([1.0, 3.0])
        handle.attrs["parameter_names"] = "alpha,beta"
    np.savez(candidate / "parameters.npz", values=np.array([1.0, 2.0]))

    report = compare_fixture_dirs(oracle, candidate)
    comparison = report.comparisons[0]

    assert comparison.status == "failed"
    assert comparison.name == "parameters/values"
    assert comparison.max_abs_index == (1,)
    assert comparison.max_abs_label == ("beta",)


def test_load_fixture_labels_reads_julia_hdf5_forecast_labels(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    with h5py.File(oracle / "m1002_ss10.h5", "w") as handle:
        handle.attrs["forecast_mode_dates"] = "2018-Q4,2019-Q1"
        handle.attrs["forecast_full_dates"] = "2018-Q4,2019-Q1"
        handle.attrs["forecast_full_draws"] = "3"
        handle.attrs["history_dates"] = "2018-Q2,2018-Q3"
        handle.attrs["parameter_names"] = "alpha,beta"
        handle.attrs["state_names"] = "y_t,z_t"
        handle.attrs["observable_names"] = "obs_gdp,obs_nominalrate"
        handle.attrs["pseudo_observable_names"] = "y_t,OutputGap"
        handle.attrs["exogenous_shock_names"] = "rm_sh,tfp_sh"
        handle["forecast_mode/states"] = np.zeros((2, 2))
        handle["forecast_mode/observables"] = np.zeros((2, 2))
        handle["forecast_mode/pseudo_observables"] = np.zeros((2, 2))
        handle["forecast_mode/shocks"] = np.zeros((2, 2))
        handle["forecast_full/states"] = np.zeros((2, 2))
        handle["forecast_full/observables"] = np.zeros((2, 2))
        handle["forecast_full/pseudo_observables"] = np.zeros((2, 2))
        handle["forecast_full/state_samples"] = np.zeros((3, 2, 2))
        handle["forecast_full/observable_samples"] = np.zeros((3, 2, 2))
        handle["forecast_full/pseudo_observable_samples"] = np.zeros((3, 2, 2))
        handle["forecast_full/shock_samples"] = np.zeros((3, 2, 2))
        handle["forecast_mode/history_observables"] = np.zeros((2, 2))
        handle["forecast_full/history_observables"] = np.zeros((2, 2))
        handle["forecast_full/history_observable_samples"] = np.zeros((3, 2, 2))
        handle["meansbands_mode_forecastobs/mean"] = np.zeros((2, 2))
        handle["meansbands_mode_forecastobs/lower"] = np.zeros((2, 2))
        handle["meansbands_mode_forecastobs/upper"] = np.zeros((2, 2))
        handle["meansbands_mode_histobs/mean"] = np.zeros((2, 2))
        handle["meansbands_mode_histobs/lower"] = np.zeros((2, 2))
        handle["meansbands_mode_histobs/upper"] = np.zeros((2, 2))
        handle["meansbands_full_forecastobs/mean"] = np.zeros((2, 2))
        handle["meansbands_full_forecastobs/lower"] = np.zeros((2, 2))
        handle["meansbands_full_forecastobs/upper"] = np.zeros((2, 2))
        handle["meansbands_full_histobs/mean"] = np.zeros((2, 2))
        handle["meansbands_full_histobs/lower"] = np.zeros((2, 2))
        handle["meansbands_full_histobs/upper"] = np.zeros((2, 2))
        handle["kalman/log_likelihood"] = np.zeros(2)
        handle["kalman/predicted_states"] = np.zeros((2, 2))
        handle["kalman/filtered_states"] = np.zeros((2, 2))
        handle["kalman/predicted_covariances"] = np.zeros((2, 2, 2))
        handle["kalman/filtered_covariances"] = np.zeros((2, 2, 2))
        handle["kalman/final_filtered_state"] = np.zeros(2)
        handle["kalman/total_log_likelihood"] = np.zeros(1)
        handle["posterior/log_posterior"] = np.zeros(1)
        handle["posterior/log_likelihood"] = np.zeros(1)
        handle["posterior/log_prior"] = np.zeros(1)
        handle["posterior/log_likelihood_by_period"] = np.zeros(2)
        handle["posterior/log_prior_by_parameter"] = np.zeros(2)
        handle["posterior/parameter_values"] = np.zeros(2)

    labels = load_fixture_labels(oracle)

    assert labels["forecast_mode/states"] == {
        0: ("2018-Q4", "2019-Q1"),
        1: ("y_t", "z_t"),
    }
    assert labels["forecast_mode/observables"][1] == (
        "obs_gdp",
        "obs_nominalrate",
    )
    assert labels["forecast_mode/shocks"][1] == ("rm_sh", "tfp_sh")
    assert labels["forecast_full/observables"][0] == (
        "2018-Q4",
        "2019-Q1",
    )
    assert labels["forecast_full/observable_samples"] == {
        0: ("draw_0", "draw_1", "draw_2"),
        1: ("2018-Q4", "2019-Q1"),
        2: ("obs_gdp", "obs_nominalrate"),
    }
    assert labels["forecast_full/shock_samples"][2] == ("rm_sh", "tfp_sh")
    assert labels["meansbands_full_forecastobs/upper"][1] == (
        "obs_gdp",
        "obs_nominalrate",
    )
    assert labels["meansbands_mode_forecastobs/mean"][0] == (
        "2018-Q4",
        "2019-Q1",
    )
    assert labels["forecast_mode/history_observables"] == {
        0: ("2018-Q2", "2018-Q3"),
        1: ("obs_gdp", "obs_nominalrate"),
    }
    assert labels["forecast_full/history_observables"] == {
        0: ("2018-Q2", "2018-Q3"),
        1: ("obs_gdp", "obs_nominalrate"),
    }
    assert labels["forecast_full/history_observable_samples"] == {
        0: ("draw_0", "draw_1", "draw_2"),
        1: ("2018-Q2", "2018-Q3"),
        2: ("obs_gdp", "obs_nominalrate"),
    }
    assert labels["meansbands_mode_histobs/mean"][0] == (
        "2018-Q2",
        "2018-Q3",
    )
    assert labels["meansbands_full_histobs/lower"][1] == (
        "obs_gdp",
        "obs_nominalrate",
    )
    assert labels["kalman/log_likelihood"] == {0: ("2018-Q2", "2018-Q3")}
    assert labels["kalman/filtered_states"] == {
        0: ("2018-Q2", "2018-Q3"),
        1: ("y_t", "z_t"),
    }
    assert labels["kalman/filtered_covariances"] == {
        0: ("2018-Q2", "2018-Q3"),
        1: ("y_t", "z_t"),
        2: ("y_t", "z_t"),
    }
    assert labels["kalman/final_filtered_state"] == {0: ("y_t", "z_t")}
    assert labels["kalman/total_log_likelihood"] == {0: ("total",)}
    assert labels["posterior/log_likelihood_by_period"] == {0: ("2018-Q2", "2018-Q3")}
    assert labels["posterior/log_prior_by_parameter"] == {0: ("alpha", "beta")}
    assert labels["posterior/parameter_values"] == {0: ("alpha", "beta")}
    assert labels["posterior/log_posterior"] == {0: ("value",)}


def test_compare_fixture_dirs_can_filter_to_required_arrays(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    np.save(oracle / "A.npy", np.array([1.0]))
    np.save(oracle / "oracle_only.npy", np.array([2.0]))
    np.save(candidate / "A.npy", np.array([1.0]))
    np.save(candidate / "candidate_only.npy", np.array([3.0]))

    report = compare_fixture_dirs(oracle, candidate, array_names=("A",))

    assert report.passed
    assert [item.name for item in report.comparisons] == ["A"]


def test_compare_fixture_dirs_compares_hdf5_model_metadata_attributes(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    model = Model1002()
    with h5py.File(oracle / "m1002_ss10.h5", "w") as handle:
        handle.attrs["observable_names"] = ",".join(model.observables)
        handle.attrs["pseudo_observable_names"] = ",".join(model.pseudo_observables)
    save_model_metadata_fixture(model, candidate)

    report = compare_fixture_dirs(
        oracle,
        candidate,
        array_names=required_fixture_arrays("model-metadata"),
    )

    assert report.passed
    assert {item.name for item in report.comparisons} == {
        "metadata/observable_names",
        "metadata/pseudo_observable_names",
    }


def test_vv_compare_cli_success_and_failure(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    np.save(oracle / "A.npy", np.ones((1, 1)))
    np.save(candidate / "A.npy", np.ones((1, 1)))

    runner = CliRunner()
    success = runner.invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle),
            "--candidate-dir",
            str(candidate),
            "--json",
        ],
    )
    assert success.exit_code == 0
    assert '"passed": true' in success.stdout

    np.save(candidate / "A.npy", np.zeros((1, 1)))
    failure = runner.invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle),
            "--candidate-dir",
            str(candidate),
        ],
    )
    assert failure.exit_code == 1


def test_vv_compare_cli_uses_tolerance_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    np.save(oracle / "A.npy", np.ones((1, 1)))
    np.save(candidate / "A.npy", np.ones((1, 1)) + 1.0e-6)

    runner = CliRunner()
    strict = runner.invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle),
            "--candidate-dir",
            str(candidate),
            "--tolerance-profile",
            "strict",
            "--json",
        ],
    )
    assert strict.exit_code == 1
    assert '"name": "strict"' in strict.stdout

    accelerator = runner.invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle),
            "--candidate-dir",
            str(candidate),
            "--tolerance-profile",
            "accelerator",
            "--json",
        ],
    )
    assert accelerator.exit_code == 0
    assert '"name": "accelerator"' in accelerator.stdout
    assert '"passed": true' in accelerator.stdout


def test_vv_compare_cli_accepts_tolerance_overrides(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    np.save(oracle / "A.npy", np.ones((1, 1)))
    np.save(candidate / "A.npy", np.ones((1, 1)) + 1.0e-6)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle),
            "--candidate-dir",
            str(candidate),
            "--atol",
            "1e-5",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"atol": 1e-05' in result.stdout


def test_vv_compare_cli_rejects_unknown_tolerance_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    np.save(oracle / "A.npy", np.ones((1, 1)))
    np.save(candidate / "A.npy", np.ones((1, 1)))

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle),
            "--candidate-dir",
            str(candidate),
            "--tolerance-profile",
            "loose",
        ],
    )

    assert result.exit_code == 2
    assert "Unknown tolerance profile" in result.stdout


def test_vv_compare_cli_reports_worst_labeled_failure(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    np.savez(oracle / "forecast.npz", observables=np.array([[1.0, 2.0]]))
    np.savez(candidate / "forecast.npz", observables=np.array([[1.0, 3.5]]))
    save_fixture_manifest(
        candidate,
        {
            "labels": {
                "forecast/observables": {
                    "axis0": ["2018-Q4"],
                    "axis1": ["obs_gdp", "obs_hours"],
                }
            }
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle),
            "--candidate-dir",
            str(candidate),
            "--json",
        ],
    )

    assert result.exit_code == 1
    assert '"max_abs_label": [' in result.stdout
    assert '"2018-Q4"' in result.stdout
    assert '"obs_hours"' in result.stdout


def test_vv_compare_cli_can_filter_to_coverage_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    _write_required_arrays(oracle, required_fixture_arrays("matrix"))
    _write_required_arrays(candidate, required_fixture_arrays("matrix"))
    np.save(candidate / "extra.npy", np.array([1.0]))

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle),
            "--candidate-dir",
            str(candidate),
            "--profile",
            "matrix",
            "--json",
        ],
    )

    payload = json.loads(result.stdout)
    assert result.exit_code == 0
    assert payload["coverage_profile"] == "matrix"
    assert payload["passed"] is True
    assert all(item["name"] != "extra" for item in payload["comparisons"])


def test_vv_oracle_coverage_cli_reports_missing_arrays(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    np.savez(oracle / "canonical.npz", Gamma0=np.eye(1))

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "oracle-coverage",
            "--oracle-dir",
            str(oracle),
            "--profile",
            "matrix",
            "--json",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert "canonical/Gamma1" in payload["missing"]


def test_vv_oracle_coverage_cli_passes_complete_matrix_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _write_required_arrays(oracle, required_fixture_arrays("matrix"))

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "oracle-coverage",
            "--oracle-dir",
            str(oracle),
            "--profile",
            "matrix",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["missing"] == []


def test_vv_oracle_coverage_cli_passes_financial_frictions_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    oracle.mkdir()
    _write_required_arrays(oracle, required_fixture_arrays("financial-frictions"))

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "oracle-coverage",
            "--oracle-dir",
            str(oracle),
            "--profile",
            "financial-frictions",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["missing"] == []


def test_vv_compare_cli_compares_financial_frictions_profile(tmp_path) -> None:
    oracle = tmp_path / "oracle"
    candidate = tmp_path / "candidate"
    oracle.mkdir()
    candidate.mkdir()
    _write_required_arrays(oracle, required_fixture_arrays("financial-frictions"))
    _write_required_arrays(candidate, required_fixture_arrays("financial-frictions"))

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle),
            "--candidate-dir",
            str(candidate),
            "--profile",
            "financial-frictions",
            "--tolerance-profile",
            "strict",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["coverage_profile"] == "financial-frictions"
    assert len(payload["comparisons"]) == 2
    assert payload["comparisons"][0]["name"] == "financial_frictions/inputs"
    assert payload["comparisons"][1]["name"] == "financial_frictions/values"


def test_vv_financial_frictions_oracle_coverage_uses_committed_fixture() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    oracle_dir = repo_root / "tests" / "fixtures" / "oracle"
    result = CliRunner().invoke(
        app,
        [
            "vv",
            "oracle-coverage",
            "--oracle-dir",
            str(oracle_dir),
            "--profile",
            "financial-frictions",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["profile"] == "financial-frictions"
    assert payload["missing"] == []


def test_vv_compare_cli_compares_committed_financial_fixtures() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    oracle_dir = repo_root / "tests" / "fixtures" / "oracle"
    candidate_dir = repo_root / "tests" / "fixtures" / "candidate"
    result = CliRunner().invoke(
        app,
        [
            "vv",
            "compare",
            "--oracle-dir",
            str(oracle_dir),
            "--candidate-dir",
            str(candidate_dir),
            "--profile",
            "financial-frictions",
            "--tolerance-profile",
            "strict",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["coverage_profile"] == "financial-frictions"
    assert len(payload["comparisons"]) == 2
    assert payload["comparisons"][0]["name"] == "financial_frictions/inputs"
    assert payload["comparisons"][1]["name"] == "financial_frictions/values"


def test_vv_solve_canonical_cli_writes_transition_fixture(tmp_path) -> None:
    canonical_dir = tmp_path / "canonical"
    output_dir = tmp_path / "candidate"
    canonical_dir.mkdir()
    np.savez(
        canonical_dir / "canonical.npz",
        Gamma0=np.eye(1),
        Gamma1=np.array([[0.25]]),
        C=np.array([2.0]),
        Psi=np.ones((1, 1)),
        Pi=np.zeros((1, 0)),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "vv",
            "solve-canonical",
            "--input-dir",
            str(canonical_dir),
            "--output-dir",
            str(output_dir),
            "--method",
            "gensys",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"method": "gensys"' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    np.testing.assert_allclose(arrays["transition/TTT"], np.array([[0.25]]))


def test_vv_export_system_cli_writes_model1002_candidate_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    model = Model1002()

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-system",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"TTT": [' in result.stdout
    assert '"ZZ_pseudo": [' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["system/TTT"].shape == (84, 84)
    assert arrays["system/RRR"].shape == (84, 24)
    assert arrays["system/ZZ"].shape == (19, 84)
    assert arrays["system/ZZ_pseudo"].shape == (21, 84)
    assert arrays["system/DD_pseudo"].shape == (21,)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels"]["system/TTT"]["axis0"] == list(
        model.indexes.endogenous_states
    ) + list(model.indexes.endogenous_states_augmented)
    assert manifest["labels"]["system/RRR"]["axis1"] == list(model.indexes.exogenous_shocks)
    assert manifest["labels"]["system/ZZ"]["axis0"] == list(model.observables)
    assert manifest["labels"]["system/ZZ_pseudo"]["axis0"] == list(model.pseudo_observables)


def test_vv_export_system_cli_records_oracle_matching_settings(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-system",
            "--output-dir",
            str(output_dir),
            "--data-vintage",
            "190101",
            "--forecast-start",
            "2020-Q2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"data_vintage": "190101"' in result.stdout
    assert '"forecast_start": "2020-Q2"' in result.stdout
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_vintage"] == "190101"
    assert manifest["forecast_start"] == "2020-Q2"


def test_vv_export_parameters_cli_writes_model1002_candidate_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    model = Model1002()

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-parameters",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"parameters": 95' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["parameters/values"].shape == (95,)
    assert arrays["parameters/scaled_values"].shape == (95,)
    assert arrays["parameters/fixed"].shape == (95,)
    assert arrays["parameters/bounds"].shape == (95, 2)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parameter_count"] == 95
    assert manifest["labels"]["parameters/values"]["axis0"] == list(model.parameters)
    assert manifest["labels"]["parameters/bounds"]["axis1"] == ["lower", "upper"]
    alpha_metadata = manifest["parameter_metadata"][list(model.parameters).index("alpha")]
    assert alpha_metadata["name"] == "alpha"
    assert alpha_metadata["description"] == "Capital share in production."
    assert alpha_metadata["category"] == "structural"
    assert alpha_metadata["regime"] == "baseline"
    assert alpha_metadata["prior"]["name"] == "normal"
    assert alpha_metadata["prior"]["mean"] == 0.3
    assert alpha_metadata["prior"]["std"] == 0.05


def test_vv_export_steady_state_cli_writes_model1002_candidate_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    model = Model1002()
    steady_state = model.steadystate()

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-steady-state",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"steady_state": 22' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["steady_state/values"].shape == (22,)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["steady_state_count"] == 22
    assert manifest["labels"]["steady_state/values"]["axis0"] == list(steady_state)


def test_vv_export_matrices_cli_writes_model1002_candidate_fixtures(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    model = Model1002()

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-matrices",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"canonical_shape": [' in result.stdout
    assert (output_dir / "manifest.json").exists()
    arrays = load_fixture_arrays(output_dir)
    observable_width = max(len(name) for name in model.observables)
    pseudo_width = max(len(name) for name in model.pseudo_observables)
    assert arrays["parameters/values"].shape == (95,)
    assert arrays["parameters/scaled_values"].shape == (95,)
    assert arrays["metadata/observable_names"].shape == (19, observable_width)
    assert arrays["metadata/pseudo_observable_names"].shape == (21, pseudo_width)
    assert arrays["steady_state/values"].shape == (22,)
    assert arrays["canonical/Gamma0"].shape == (68, 68)
    assert arrays["canonical/Gamma1"].shape == (68, 68)
    assert arrays["canonical/C"].shape == (68,)
    assert arrays["canonical/Psi"].shape == (68, 24)
    assert arrays["canonical/Pi"].shape == (68, 13)
    assert arrays["transition/TTT"].shape == (68, 68)
    assert arrays["transition/eu"].shape == (2,)
    assert arrays["system/TTT"].shape == (84, 84)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels"]["parameters/values"]["axis0"] == list(model.parameters)
    assert manifest["labels"]["metadata/observable_names"]["axis0"] == list(model.observables)
    assert manifest["labels"]["metadata/pseudo_observable_names"]["axis0"] == list(
        model.pseudo_observables
    )
    assert manifest["labels"]["steady_state/values"]["axis0"] == list(model.steadystate())
    assert manifest["labels"]["canonical/Gamma0"]["axis0"] == list(
        model.indexes.equilibrium_conditions
    )
    assert manifest["labels"]["canonical/Gamma0"]["axis1"] == list(model.indexes.endogenous_states)
    assert manifest["labels"]["canonical/Psi"]["axis1"] == list(model.indexes.exogenous_shocks)
    assert manifest["labels"]["canonical/Pi"]["axis1"] == list(model.indexes.expected_shocks)
    assert manifest["labels"]["transition/TTT"]["axis0"] == list(model.indexes.endogenous_states)
    assert manifest["labels"]["transition/eu"]["axis0"] == ["existence", "uniqueness"]
    assert manifest["labels"]["system/TTT"]["axis0"] == list(
        model.indexes.endogenous_states
    ) + list(model.indexes.endogenous_states_augmented)
    assert manifest["shapes"]["canonical"]["Gamma0"] == [68, 68]
    assert manifest["shapes"]["metadata"]["observable_names"] == [19, observable_width]
    assert manifest["shapes"]["metadata"]["pseudo_observable_names"] == [21, pseudo_width]
    assert manifest["shapes"]["transition"]["TTT"] == [68, 68]
    assert manifest["shapes"]["transition"]["eu"] == [2]
    assert manifest["shapes"]["system"]["TTT"] == [84, 84]


def test_julia_oracle_export_script_writes_transition_status_fixture() -> None:
    script = Path("tools/oracle_julia/export_model1002.jl").read_text(encoding="utf-8")

    assert '"transition/eu"' in script
    assert 'write_dataset(file, "transition/eu", Int64.(transition_eu))' in script


def test_julia_oracle_export_script_writes_financial_frictions_fixture() -> None:
    script = Path("tools/oracle_julia/export_model1002.jl").read_text(encoding="utf-8")

    assert '"include-financial-frictions" => "false"' in script
    assert '"financial_frictions/inputs"' in script
    assert '"financial_frictions/values"' in script


def test_julia_oracle_export_script_includes_sampler_flags() -> None:
    script = Path("tools/oracle_julia/export_model1002.jl").read_text(encoding="utf-8")

    assert '"include-sampler" => "false"' in script
    assert '"sampler-seed"' in script
    assert '"sampler-draws"' in script
    assert '"sampler-blocks"' in script
    assert '"sampler-param-blocks"' in script
    assert '"sampler-thin"' in script
    assert '"sampler-adaptive-accept"' in script
    assert '"sampler-target-accept"' in script
    assert '"sampler-cc"' in script
    assert '"sampler-alpha"' in script
    assert '"sampler-reoptimize"' in script
    assert '"sampler-run-csminwel"' in script
    assert '"sampler-proposal-scale"' in script
    assert '"sampler-mode-in"' in script
    assert '"sampler-hessian-in"' in script
    assert 'Symbol("mh_", string(Char(0x03b1)))' in script
    assert '"sampler/fixed"' in script
    assert '"sampler/proposal_parameters"' in script
    assert '"sampler/previous_parameters"' in script
    assert '"sampler/proposal_log_posterior"' in script
    assert '"sampler/previous_log_posterior"' in script
    assert '"sampler/uniform_draw"' in script
    assert '"sampler/log_acceptance"' in script


def test_vv_export_matrices_cli_records_oracle_matching_settings(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-matrices",
            "--output-dir",
            str(output_dir),
            "--data-vintage",
            "190101",
            "--forecast-start",
            "2020-Q2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"data_vintage": "190101"' in result.stdout
    assert '"forecast_start": "2020-Q2"' in result.stdout
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_vintage"] == "190101"
    assert manifest["forecast_start"] == "2020-Q2"


def test_vv_export_financial_frictions_cli_writes_helper_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-financial-frictions",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["input_shape"] == [3, 3]
    assert payload["values_shape"] == [3, 16]
    assert payload["functions"][-1] == "zeta_spb"
    arrays = load_fixture_arrays(output_dir)
    assert arrays["financial_frictions/inputs"].shape == (3, 3)
    assert arrays["financial_frictions/values"].shape == (3, 16)
    assert np.isclose(arrays["financial_frictions/values"][0, 0], 0.2620745217846454)
    assert np.isclose(arrays["financial_frictions/values"][0, -1], 0.055975462717559314)
    labels = load_fixture_labels(output_dir)
    assert labels["financial_frictions/inputs"][1] == ("z", "sigma", "spr")
    assert labels["financial_frictions/values"][1][-1] == "zeta_spb"


def test_vv_export_financial_frictions_preserves_existing_manifest_metadata(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    save_fixture_manifest(
        output_dir,
        {
            "kind": "model1002_candidate",
            "shapes": {"canonical": {"Gamma0": [1, 1]}},
        },
    )

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-financial-frictions",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert result.exit_code == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "model1002_candidate"
    assert manifest["shapes"]["canonical"]["Gamma0"] == [1, 1]
    assert manifest["shapes"]["financial_frictions"]["inputs"] == [3, 3]
    assert manifest["shapes"]["financial_frictions"]["values"] == [3, 16]


def test_vv_export_suite_cli_writes_standard_candidate_suite(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-suite",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["comparison"]["status"] == "skipped"
    arrays = load_fixture_arrays(output_dir)
    assert arrays["parameters/values"].shape == (95,)
    assert arrays["steady_state/values"].shape == (22,)
    assert arrays["canonical/Gamma0"].shape == (68, 68)
    assert arrays["system/TTT"].shape == (84, 84)
    assert arrays["forecast_mode/observables"].shape == (2, 19)
    assert arrays["meansbands_mode_forecastobs/mean"].shape == (2, 19)
    assert "posterior/log_posterior" not in arrays
    assert "forecast_full/observables" not in arrays
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "forecast_mode/observables" in manifest["labels"]
    assert "meansbands_mode_forecastobs/mean" in manifest["labels"]


def test_vv_export_suite_cli_can_include_full_and_history_artifacts(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-suite",
            "--output-dir",
            str(output_dir),
            "--data",
            str(data_path),
            "--horizon",
            "2",
            "--full-draws",
            "2",
            "--seed",
            "4",
            "--json",
        ],
    )

    assert result.exit_code == 0
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast_mode/history_observables"].shape == (2, 19)
    assert arrays["meansbands_mode_histobs/mean"].shape == (2, 19)
    assert arrays["forecast_full/observables"].shape == (2, 19)
    assert arrays["forecast_full/history_observables"].shape == (2, 19)
    assert arrays["forecast_full/observable_samples"].shape == (2, 2, 19)
    assert arrays["forecast_full/history_observable_samples"].shape == (2, 2, 19)
    assert arrays["meansbands_full_forecastobs/mean"].shape == (2, 19)
    assert arrays["meansbands_full_histobs/mean"].shape == (2, 19)
    assert arrays["posterior/log_posterior"].shape == (1,)
    assert arrays["posterior/log_likelihood_by_period"].shape == (2,)
    assert arrays["posterior/log_prior_by_parameter"].shape == (len(Model1002().parameters),)
    labels = load_fixture_labels(output_dir)
    assert labels["posterior/log_likelihood_by_period"][0] == ("2018-Q2", "2018-Q3")


def test_vv_export_suite_cli_can_allow_empty_data_columns(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    data_path = _write_observable_csv(tmp_path, periods=2)
    data = pd.read_csv(data_path)
    data["obs_longrate"] = np.nan
    data.to_csv(data_path, index=False)

    rejected = CliRunner().invoke(
        app,
        [
            "vv",
            "export-suite",
            "--output-dir",
            str(output_dir),
            "--data",
            str(data_path),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert rejected.exit_code == 2
    assert "empty columns" in rejected.stdout

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-suite",
            "--output-dir",
            str(output_dir),
            "--data",
            str(data_path),
            "--horizon",
            "2",
            "--allow-empty-data-columns",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["allow_empty_data_columns"] is True
    arrays = load_fixture_arrays(output_dir)
    longrate_index = list(Model1002().observables).index("obs_longrate")
    assert np.isnan(arrays["forecast_mode/history_observables"][:, longrate_index]).all()


def test_vv_export_kalman_cli_writes_filter_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-kalman",
            "--output-dir",
            str(output_dir),
            "--data",
            str(data_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["periods"] == 2
    arrays = load_fixture_arrays(output_dir)
    assert arrays["kalman/log_likelihood"].shape == (2,)
    assert arrays["kalman/filtered_states"].shape == (2, 84)
    assert arrays["kalman/filtered_covariances"].shape == (2, 84, 84)
    assert arrays["kalman/final_filtered_state"].shape == (84,)
    labels = load_fixture_labels(output_dir)
    assert labels["kalman/filtered_states"][0] == ("2018-Q2", "2018-Q3")
    assert labels["kalman/filtered_states"][1][0] == "y_t"


def test_vv_export_posterior_cli_writes_decomposition_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-posterior",
            "--output-dir",
            str(output_dir),
            "--data",
            str(data_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["periods"] == 2
    assert payload["parameters"] == len(Model1002().parameters)
    assert np.isclose(
        payload["log_posterior"],
        payload["log_likelihood"] + payload["log_prior"],
    )
    arrays = load_fixture_arrays(output_dir)
    assert arrays["posterior/log_posterior"].shape == (1,)
    assert arrays["posterior/log_likelihood"].shape == (1,)
    assert arrays["posterior/log_prior"].shape == (1,)
    assert arrays["posterior/log_likelihood_by_period"].shape == (2,)
    assert arrays["posterior/log_prior_by_parameter"].shape == (len(Model1002().parameters),)
    assert arrays["posterior/parameter_values"].shape == (len(Model1002().parameters),)
    labels = load_fixture_labels(output_dir)
    assert labels["posterior/log_likelihood_by_period"][0] == ("2018-Q2", "2018-Q3")
    assert labels["posterior/log_prior_by_parameter"][0][0] == "alpha"
    assert labels["posterior/parameter_values"][0][0] == "alpha"


def test_vv_export_suite_cli_can_compare_against_existing_oracle_dir(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-suite",
            "--output-dir",
            str(output_dir),
            "--oracle-dir",
            str(output_dir),
            "--horizon",
            "1",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["comparison"]["status"] == "passed"


def test_vv_exports_preserve_manifest_labels_when_reusing_output_dir(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    runner = CliRunner()

    matrices = runner.invoke(
        app,
        [
            "vv",
            "export-matrices",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )
    forecast = runner.invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert matrices.exit_code == 0
    assert forecast.exit_code == 0
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "canonical/Gamma0" in manifest["labels"]
    assert "system/TTT" in manifest["labels"]
    assert "forecast/observables" in manifest["labels"]
    labels = load_fixture_labels(output_dir)
    assert "canonical/Gamma0" in labels
    assert "forecast/observables" in labels


def test_vv_export_forecast_cli_writes_model1002_candidate_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"observables_shape": [' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast/states"].shape == (2, 84)
    assert arrays["forecast/observables"].shape == (2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels"]["forecast/observables"]["axis0"] == ["2018-Q4", "2019-Q1"]
    assert manifest["labels"]["forecast/observables"]["axis1"] == list(Model1002().observables)
    expected_state_labels = list(Model1002().indexes.endogenous_states) + list(
        Model1002().indexes.endogenous_states_augmented
    )
    assert manifest["labels"]["forecast/states"]["axis1"] == expected_state_labels


def test_vv_export_forecast_cli_can_write_zlb_conditioning_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "4",
            "--zlb-rates",
            "0.25,-0.10,0.75",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    n_shocks = len(Model1002().indexes.exogenous_shocks)
    assert payload["cond_type"] == "full"
    assert payload["conditional_shocks_shape"] == [3, n_shocks]
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast/conditional_shocks"].shape == (3, n_shocks)
    assert arrays["forecast/conditional_states"].shape == (3, 84)
    assert arrays["forecast/conditional_observables"].shape == (3, 19)
    assert arrays["forecast/states"].shape == (1, 84)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels"]["forecast/conditional_observables"]["axis0"] == [
        "2018-Q4",
        "2019-Q1",
        "2019-Q2",
    ]
    assert manifest["labels"]["forecast/observables"]["axis0"] == ["2019-Q3"]


def test_vv_export_forecast_cli_labels_custom_fixture_stem(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "2",
            "--filename",
            "forecast_mode.npz",
            "--json",
        ],
    )

    assert result.exit_code == 0
    arrays = load_fixture_arrays(output_dir)
    assert "forecast_mode/observables" in arrays
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "forecast_mode/observables" in manifest["labels"]
    assert "forecast/observables" not in manifest["labels"]


def test_vv_export_forecast_cli_uses_forecast_start_setting_for_labels(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--forecast-start",
            "2020-Q2",
            "--horizon",
            "3",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"forecast_start": "2020-Q2"' in result.stdout
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["forecast_start"] == "2020-Q2"
    assert manifest["labels"]["forecast/observables"]["axis0"] == [
        "2020-Q2",
        "2020-Q3",
        "2020-Q4",
    ]


def test_vv_export_forecast_cli_can_write_transformed_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "2",
            "--transformed",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"transformed": true' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast/observables"].shape == (2, 19)


def test_vv_export_forecast_cli_can_write_full_sample_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--input-type",
            "full",
            "--draws",
            "2",
            "--seed",
            "3",
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"observable_samples_shape": [' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast/state_samples"].shape == (2, 2, 84)
    assert arrays["forecast/observable_samples"].shape == (2, 2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels"]["forecast/observable_samples"]["axis0"] == ["draw_0", "draw_1"]
    assert manifest["labels"]["forecast/observable_samples"]["axis1"] == ["2018-Q4", "2019-Q1"]


def test_vv_export_forecast_cli_can_write_full_fixture_from_shock_samples(
    tmp_path,
) -> None:
    output_dir = tmp_path / "candidate"
    shock_samples_path = _write_shock_samples_archive(tmp_path, horizon=2, julia_order=True)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--input-type",
            "full",
            "--shock-samples",
            str(shock_samples_path),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"shock_samples_shape": [' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast/state_samples"].shape == (2, 2, 84)
    assert arrays["forecast/observable_samples"].shape == (2, 2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    shock_samples = manifest["shock_samples"]["forecast"]
    assert shock_samples["source_path"] == str(shock_samples_path)
    assert shock_samples["shape"] == [
        2,
        len(Model1002().indexes.exogenous_shocks),
        2,
    ]


def test_vv_export_forecast_cli_can_write_sampler_draw_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    sampler_path = _write_sampler_archive(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--input-type",
            "full",
            "--sampler-draws",
            str(sampler_path),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"sampler_draws":' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast/state_samples"].shape == (2, 2, 84)
    assert arrays["forecast/observable_samples"].shape == (2, 2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    sampler = manifest["samplers"]["forecast"]
    assert sampler["source_path"] == str(sampler_path)
    assert sampler["parameter_names"] == ["alpha"]
    assert sampler["parameter_draws_shape"] == [2, 1]
    assert sampler["estimation_draws_shape"] == [2, 1]
    assert sampler["draws"] == 2
    assert sampler["parameter_count"] == 1
    assert sampler["acceptance_rate"] == 1.0
    assert sampler["burnin"] == 0
    assert sampler["seed"] == 11


def test_vv_export_forecast_cli_can_write_filter_backed_history_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "1",
            "--data",
            str(data_path),
            "--include-history",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"history_observables_shape": [' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast/history_states"].shape == (2, 84)
    assert arrays["forecast/history_observables"].shape == (2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels"]["forecast/history_observables"]["axis0"] == [
        "2018-Q2",
        "2018-Q3",
    ]
    assert manifest["labels"]["forecast/history_observables"]["axis1"] == list(
        Model1002().observables
    )


def test_vv_export_forecast_cli_can_write_smoothed_history_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "1",
            "--data",
            str(data_path),
            "--include-history",
            "--history-method",
            "smoothed",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"history_method": "smoothed"' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast/history_states"].shape == (2, 84)
    assert arrays["forecast/history_observables"].shape == (2, 19)


def test_vv_export_forecast_cli_can_write_pseudo_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-forecast",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "1",
            "--data",
            str(data_path),
            "--include-history",
            "--include-pseudo",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"pseudo_observables_shape": [' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast/pseudo_observables"].shape == (1, 21)
    assert arrays["forecast/history_pseudo_observables"].shape == (2, 21)


def test_vv_export_meansbands_cli_writes_model1002_candidate_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-meansbands",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"mean_shape": [' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["meansbands/mean"].shape == (2, 19)
    assert arrays["meansbands/lower"].shape == (2, 19)
    assert arrays["meansbands/upper"].shape == (2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels"]["meansbands/mean"]["axis0"] == ["2018-Q4", "2019-Q1"]
    assert manifest["labels"]["meansbands/mean"]["axis1"] == list(Model1002().observables)


def test_vv_export_suite_cli_records_sampler_draw_provenance(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    sampler_path = _write_sampler_archive(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-suite",
            "--output-dir",
            str(output_dir),
            "--sampler-draws",
            str(sampler_path),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast_full/observable_samples"].shape == (2, 2, 19)
    assert arrays["meansbands_full_forecastobs/mean"].shape == (2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    samplers = manifest["samplers"]
    assert samplers["forecast_full"]["source_path"] == str(sampler_path)
    assert samplers["forecast_full"]["parameter_names"] == ["alpha"]
    assert samplers["forecast_full"]["parameter_draws_shape"] == [2, 1]
    assert samplers["meansbands_full_forecastobs"]["source_path"] == str(sampler_path)
    assert samplers["meansbands_full_forecastobs"]["draws"] == 2


def test_vv_export_suite_cli_records_shock_sample_provenance(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    shock_samples_path = _write_shock_samples_archive(tmp_path, horizon=2)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-suite",
            "--output-dir",
            str(output_dir),
            "--shock-samples",
            str(shock_samples_path),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    arrays = load_fixture_arrays(output_dir)
    assert arrays["forecast_full/observable_samples"].shape == (2, 2, 19)
    assert arrays["meansbands_full_forecastobs/mean"].shape == (2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    shock_samples = manifest["shock_samples"]
    assert shock_samples["forecast_full"]["source_path"] == str(shock_samples_path)
    assert shock_samples["forecast_full"]["shape"] == [
        2,
        2,
        len(Model1002().indexes.exogenous_shocks),
    ]
    assert shock_samples["meansbands_full_forecastobs"]["draws"] == 2


def test_vv_export_meansbands_cli_uses_forecast_start_setting_for_labels(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-meansbands",
            "--output-dir",
            str(output_dir),
            "--forecast-start",
            "2021-Q3",
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"forecast_start": "2021-Q3"' in result.stdout
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["forecast_start"] == "2021-Q3"
    assert manifest["labels"]["meansbands/mean"]["axis0"] == ["2021-Q3", "2021-Q4"]


def test_vv_export_meansbands_cli_can_write_transformed_fixture(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-meansbands",
            "--output-dir",
            str(output_dir),
            "--horizon",
            "2",
            "--transformed",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"transformed": true' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["meansbands/mean"].shape == (2, 19)


def test_vv_export_meansbands_cli_can_write_sampler_draw_bands(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    sampler_path = _write_sampler_archive(tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-meansbands",
            "--output-dir",
            str(output_dir),
            "--input-type",
            "full",
            "--sampler-draws",
            str(sampler_path),
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"sampler_draws":' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["meansbands/mean"].shape == (2, 19)
    assert arrays["meansbands/lower"].shape == (2, 19)
    assert arrays["meansbands/upper"].shape == (2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    sampler = manifest["samplers"]["meansbands"]
    assert sampler["source_path"] == str(sampler_path)
    assert sampler["parameter_names"] == ["alpha"]
    assert sampler["parameter_draws_shape"] == [2, 1]
    assert sampler["proposal_covariance_shape"] == [1, 1]
    assert sampler["draws"] == 2


def test_vv_export_meansbands_cli_can_write_histobs_bands(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-meansbands",
            "--output-dir",
            str(output_dir),
            "--source",
            "histobs",
            "--data",
            str(data_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"source": "histobs"' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["meansbands/mean"].shape == (2, 19)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels"]["meansbands/mean"]["axis0"] == ["2018-Q2", "2018-Q3"]


def test_vv_export_meansbands_cli_can_write_smoothed_histobs_bands(tmp_path) -> None:
    output_dir = tmp_path / "candidate"
    data_path = _write_observable_csv(tmp_path, periods=2)

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-meansbands",
            "--output-dir",
            str(output_dir),
            "--source",
            "histobs",
            "--data",
            str(data_path),
            "--history-method",
            "smoothed",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"history_method": "smoothed"' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["meansbands/mean"].shape == (2, 19)


def test_vv_export_meansbands_cli_can_write_pseudo_bands(tmp_path) -> None:
    output_dir = tmp_path / "candidate"

    result = CliRunner().invoke(
        app,
        [
            "vv",
            "export-meansbands",
            "--output-dir",
            str(output_dir),
            "--source",
            "forecastpseudo",
            "--horizon",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0
    assert '"source": "forecastpseudo"' in result.stdout
    arrays = load_fixture_arrays(output_dir)
    assert arrays["meansbands/mean"].shape == (2, 21)


def _write_observable_csv(tmp_path: Path, *, periods: int) -> Path:
    model = Model1002()
    path = tmp_path / "observables.csv"
    pd.DataFrame(
        {
            "date": _pre_forecast_quarters(periods),
            **{name: [0.0] * periods for name in model.observables},
        }
    ).to_csv(
        path,
        index=False,
    )
    return path


def _pre_forecast_quarters(periods: int) -> list[str]:
    return [f"2018-Q{quarter}" for quarter in range(4 - periods, 4)]


def _write_sampler_trace_hdf5(tmp_path: Path) -> Path:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sampler_trace.h5"
    mhparams = np.array(
        [[0.1, 1.1, 2.1], [0.2, 1.2, 2.2]],
        dtype=np.float64,
    )
    proposal_parameters = np.array(
        [[0.1, 1.3, 2.1], [0.2, 1.4, 2.2]],
        dtype=np.float64,
    )
    previous_parameters = np.array(
        [[0.0, 1.1, 2.0], [0.1, 1.2, 2.1]],
        dtype=np.float64,
    )
    proposal_log_posterior = np.array([-3.0, -3.2, -2.0], dtype=np.float64)
    previous_log_posterior = np.array([-3.4, -2.5, -2.2], dtype=np.float64)
    log_acceptance = proposal_log_posterior - previous_log_posterior
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = mhparams
        handle["sampler/proposal_covariance"] = np.diag([0.5, 2.0])
        handle["sampler/accepted"] = np.array([1, 0, 1], dtype=np.int8)
        handle["sampler/log_posterior"] = np.array([-3.0, -2.5, -2.0])
        handle["sampler/proposal_parameters"] = proposal_parameters
        handle["sampler/previous_parameters"] = previous_parameters
        handle["sampler/proposal_log_posterior"] = proposal_log_posterior
        handle["sampler/previous_log_posterior"] = previous_log_posterior
        handle["sampler/uniform_draw"] = np.array([0.2, 0.8, 0.9], dtype=np.float64)
        handle["sampler/log_acceptance"] = log_acceptance
        handle.attrs["sampler_parameter_names"] = "alpha,beta"
        handle.attrs["sampler_draws"] = 3
        handle.attrs["sampler_blocks"] = 1
        handle.attrs["sampler_param_blocks"] = 1
        handle.attrs["sampler_thin"] = 1
        handle.attrs["sampler_burnin"] = 0
        handle.attrs["sampler_proposal_scale"] = "1.0e-8"
        handle.attrs["sampler_acceptance_rate"] = "0.6666666666666666"
        handle.attrs["sampler_trace_available"] = "true"
        handle.attrs["sampler_proposal_trace_available"] = "true"
        handle.attrs["sampler_seed"] = 123
    return path


def _write_sampler_posterior_replay_hdf5(
    tmp_path: Path,
    model: Model1002,
    observations: np.ndarray,
) -> Path:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "sampler_posterior_replay.h5"
    parameter_names = tuple(model.parameters)
    current_values = np.asarray(
        [parameter.value for parameter in model.parameters.values()],
        dtype=np.float64,
    )
    proposal_draws = np.vstack([current_values, current_values])
    previous_draws = np.vstack([current_values, current_values])
    fixed_index = next(
        index for index, parameter in enumerate(model.parameters.values()) if parameter.fixed
    )
    proposal_draws[0, fixed_index] += 1.0e-3
    proposal_components = np.asarray(
        [
            evaluate_log_posterior_for_parameter_values(
                model,
                observations,
                parameter_names,
                draw,
                update_fixed_parameters=False,
            )[:3]
            for draw in proposal_draws
        ],
        dtype=np.float64,
    )
    previous_components = np.asarray(
        [
            evaluate_log_posterior_for_parameter_values(
                model,
                observations,
                parameter_names,
                draw,
                update_fixed_parameters=False,
            )[:3]
            for draw in previous_draws
        ],
        dtype=np.float64,
    )
    proposal_log_posterior = proposal_components[:, 0]
    proposal_log_likelihood = proposal_components[:, 1]
    proposal_log_prior = proposal_components[:, 2]
    previous_log_posterior = previous_components[:, 0]
    previous_log_likelihood = previous_components[:, 1]
    previous_log_prior = previous_components[:, 2]
    fixed_mask = np.asarray(
        [parameter.fixed for parameter in model.parameters.values()],
        dtype=np.int8,
    )
    with h5py.File(path, "w") as handle:
        handle["sampler/mhparams"] = proposal_draws.T
        handle["sampler/fixed"] = fixed_mask
        handle["sampler/proposal_covariance"] = np.eye(len(parameter_names), dtype=np.float64)
        handle["sampler/accepted"] = np.array([1, 1], dtype=np.int8)
        handle["sampler/log_posterior"] = proposal_log_posterior
        handle["sampler/proposal_parameters"] = proposal_draws.T
        handle["sampler/previous_parameters"] = previous_draws.T
        handle["sampler/proposal_log_posterior"] = proposal_log_posterior
        handle["sampler/previous_log_posterior"] = previous_log_posterior
        handle["sampler/proposal_log_likelihood"] = proposal_log_likelihood
        handle["sampler/previous_log_likelihood"] = previous_log_likelihood
        handle["sampler/proposal_log_prior"] = proposal_log_prior
        handle["sampler/previous_log_prior"] = previous_log_prior
        handle["sampler/uniform_draw"] = np.array([0.2, 0.3], dtype=np.float64)
        handle["sampler/log_acceptance"] = proposal_log_posterior - previous_log_posterior
        handle.attrs["sampler_parameter_names"] = ",".join(parameter_names)
        handle.attrs["sampler_draws"] = 2
        handle.attrs["sampler_blocks"] = 1
        handle.attrs["sampler_param_blocks"] = 1
        handle.attrs["sampler_thin"] = 1
        handle.attrs["sampler_burnin"] = 0
        handle.attrs["sampler_trace_available"] = "true"
        handle.attrs["sampler_proposal_trace_available"] = "true"
    return path


def _write_sampler_archive(tmp_path: Path) -> Path:
    model = Model1002()
    alpha = model.parameters["alpha"].value
    sampler = MetropolisHastingsResult(
        parameter_names=("alpha",),
        estimation_draws=np.zeros((2, 1), dtype=np.float64),
        parameter_draws=np.array([[alpha], [alpha + 1.0e-3]], dtype=np.float64),
        log_posterior=np.zeros(2, dtype=np.float64),
        accepted=np.array([True, True]),
        acceptance_rate=1.0,
        proposal_covariance=np.eye(1, dtype=np.float64),
        seed=11,
        burnin=0,
    )
    return save_sampler_result(sampler, tmp_path / "sampler.npz")


def _write_shock_samples_archive(
    tmp_path: Path,
    *,
    horizon: int,
    julia_order: bool = False,
) -> Path:
    n_shocks = len(Model1002().indexes.exogenous_shocks)
    path = tmp_path / "shock_samples.npz"
    shock_samples = np.zeros((2, horizon, n_shocks), dtype=np.float64)
    shock_samples[1, 0, 0] = 0.25
    if julia_order:
        shock_samples = np.transpose(shock_samples, (0, 2, 1))
    np.savez(path, shock_samples=shock_samples)
    return path


def _write_required_arrays(directory: Path, names: tuple[str, ...]) -> None:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for name in names:
        stem, key = name.split("/", maxsplit=1)
        grouped.setdefault(stem, {})[key] = np.zeros((1, 1), dtype=np.float64)
    for stem, arrays in grouped.items():
        savez = cast(Any, np.savez)
        savez(directory / f"{stem}.npz", **arrays)


def _write_canonical_fixture(directory: Path) -> Path:
    directory.mkdir()
    np.savez(
        directory / "canonical.npz",
        Gamma0=np.eye(1),
        Gamma1=np.array([[0.5]]),
        C=np.array([1.0]),
        Psi=np.ones((1, 1)),
        Pi=np.zeros((1, 0)),
    )
    return directory
