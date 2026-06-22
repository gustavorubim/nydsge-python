from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nydsge.core import NotPortedError
from nydsge.data import df_to_matrix
from nydsge.estimate import MetropolisHastingsResult
from nydsge.forecast import (
    ForecastOutput,
    MeansBands,
    build_zlb_conditional_observations,
    compute_meansbands,
    forecast_linear_system,
    forecast_linear_system_samples,
    forecast_one,
    forecast_parameter_draws,
    meansbands_from_forecast,
    meansbands_from_samples,
    reverse_transform_forecast,
    reverse_transform_meansbands,
    solve_shocks_for_observable_targets,
)
from nydsge.kalman import (
    kalman_log_likelihood,
    model_process_covariances,
    smooth_kalman_result,
)
from nydsge.models import Model1002
from nydsge.runtime import RuntimeConfig
from nydsge.solve import Measurement, PseudoMeasurement, System, Transition, compute_system


def test_forecast_linear_system() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.5]),
        ),
        measurement=Measurement(
            ZZ=np.array([[2.0]]),
            DD=np.array([1.0]),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
    )
    output = forecast_linear_system(
        system,
        np.array([0.0]),
        horizon=2,
        shocks=np.array([[1.0], [0.0]]),
    )
    assert output.states.tolist() == [[1.5], [2.0]]
    assert output.observables.tolist() == [[4.0], [5.0]]


def test_forecast_linear_system_pads_and_truncates_shock_paths() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
    )

    padded = forecast_linear_system(
        system,
        np.array([0.0]),
        horizon=3,
        shocks=np.array([[1.0]]),
    )
    truncated = forecast_linear_system(
        system,
        np.array([0.0]),
        horizon=2,
        shocks=np.array([[1.0], [2.0], [100.0]]),
    )

    np.testing.assert_allclose(padded.states[:, 0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(truncated.states[:, 0], [1.0, 3.0])


def test_forecast_linear_system_accepts_julia_shock_orientation() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0, 2.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.eye(2),
            EE=np.eye(1),
        ),
    )

    output = forecast_linear_system(
        system,
        np.array([0.0]),
        horizon=3,
        shocks=np.array(
            [
                [1.0, 2.0, 0.0],
                [2.0, 4.0, 0.0],
            ]
        ),
    )

    np.testing.assert_allclose(output.states[:, 0], [5.0, 15.0, 15.0])


def test_forecast_linear_system_can_include_pseudo_observables() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[0.0]]),
            CCC=np.array([1.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
        pseudo_measurement=PseudoMeasurement(
            ZZ_pseudo=np.array([[2.0]]),
            DD_pseudo=np.array([3.0]),
        ),
    )

    output = forecast_linear_system(system, np.array([0.0]), horizon=2, include_pseudo=True)

    assert output.pseudo_observables is not None
    np.testing.assert_allclose(output.pseudo_observables, np.array([[5.0], [7.0]]))


def test_forecast_linear_system_samples_are_reproducible() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[2.0]]),
            DD=np.array([1.0]),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
    )

    first = forecast_linear_system_samples(system, np.array([0.0]), horizon=3, draws=4, seed=11)
    second = forecast_linear_system_samples(system, np.array([0.0]), horizon=3, draws=4, seed=11)

    assert first.state_samples is not None
    assert first.observable_samples is not None
    assert second.state_samples is not None
    assert first.state_samples.shape == (4, 3, 1)
    assert first.observable_samples.shape == (4, 3, 1)
    np.testing.assert_allclose(first.state_samples, second.state_samples)
    np.testing.assert_allclose(first.states, first.state_samples.mean(axis=0))
    np.testing.assert_allclose(first.observables, first.observable_samples.mean(axis=0))


def test_forecast_linear_system_samples_accept_explicit_shock_samples() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0, 2.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.eye(2),
            EE=np.eye(1),
        ),
    )
    shock_samples = np.array(
        [
            [[1.0, 2.0], [0.0, 1.0], [0.0, 0.0]],
            [[-1.0, 0.0], [2.0, 0.0], [0.0, 0.0]],
        ]
    )

    output = forecast_linear_system_samples(
        system,
        np.array([0.0]),
        horizon=3,
        draws=2,
        shock_samples=shock_samples,
    )
    julia_order_output = forecast_linear_system_samples(
        system,
        np.array([0.0]),
        horizon=3,
        draws=2,
        shock_samples=np.transpose(shock_samples, (0, 2, 1)),
    )

    assert output.state_samples is not None
    assert output.observable_samples is not None
    assert julia_order_output.state_samples is not None
    np.testing.assert_allclose(output.state_samples[:, :, 0], [[5.0, 7.0, 7.0], [-1.0, 1.0, 1.0]])
    np.testing.assert_allclose(output.states, output.state_samples.mean(axis=0))
    np.testing.assert_allclose(output.observables, output.observable_samples.mean(axis=0))
    np.testing.assert_allclose(julia_order_output.state_samples, output.state_samples)


def test_forecast_linear_system_samples_reject_draw_mismatch() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
    )

    with pytest.raises(ValueError, match="explicit shock sample draws"):
        forecast_linear_system_samples(
            system,
            np.array([0.0]),
            horizon=2,
            draws=3,
            shock_samples=np.zeros((2, 2, 1)),
        )


def test_solve_shocks_for_observable_targets_matches_exact_linear_targets() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
    )

    result = solve_shocks_for_observable_targets(
        system,
        np.array([0.0]),
        np.array([[2.0], [5.0]]),
    )

    np.testing.assert_allclose(result.shocks, np.array([[2.0], [3.0]]))
    np.testing.assert_allclose(result.observables, np.array([[2.0], [5.0]]))
    np.testing.assert_allclose(result.residuals, np.zeros((2, 1)), atol=1.0e-12)
    assert result.max_abs_error <= 1.0e-12
    assert result.rank == 2


def test_solve_shocks_for_observable_targets_ignores_missing_targets() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
    )

    result = solve_shocks_for_observable_targets(
        system,
        np.array([0.0]),
        np.array([[np.nan], [4.0]]),
    )

    assert np.isnan(result.residuals[0, 0])
    np.testing.assert_allclose(result.observables[1], np.array([4.0]))


def test_build_zlb_conditional_observations_fills_policy_path_and_anticipated_rates() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 2})

    observations = build_zlb_conditional_observations(
        model,
        [0.25, -0.1, 1.0],
        floor=0.0,
    )

    nominal = list(model.observables).index("obs_nominalrate")
    anticipated_1 = list(model.observables).index("obs_nominalrate1")
    anticipated_2 = list(model.observables).index("obs_nominalrate2")

    assert observations.shape == (3, len(model.observables))
    np.testing.assert_allclose(observations[:, nominal], np.array([0.0625, 0.0, 0.25]))
    np.testing.assert_allclose(observations[:2, anticipated_1], np.array([0.0, 0.25]))
    assert np.isnan(observations[2, anticipated_1])
    np.testing.assert_allclose(observations[0, anticipated_2], np.array(0.25))
    assert np.isnan(observations[1, anticipated_2])


def test_build_zlb_conditional_observations_can_use_model_units() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})

    observations = build_zlb_conditional_observations(
        model,
        [-0.25, 0.5],
        floor=0.1,
        rate_units="model",
    )

    nominal = list(model.observables).index("obs_nominalrate")
    np.testing.assert_allclose(observations[:, nominal], np.array([0.1, 0.5]))


def test_forecast_linear_system_validates_shapes() -> None:
    system = System(
        transition=Transition(
            TTT=np.eye(2),
            RRR=np.ones((2, 1)),
            CCC=np.zeros(2),
        ),
        measurement=Measurement(
            ZZ=np.ones((1, 2)),
            DD=np.zeros(1),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
    )

    with pytest.raises(ValueError, match="Initial state"):
        forecast_linear_system(system, np.zeros(1), horizon=1)

    with pytest.raises(ValueError, match="Shocks"):
        forecast_linear_system(system, np.zeros(2), horizon=2, shocks=np.zeros(1))

    with pytest.raises(ValueError, match="structural shocks"):
        forecast_linear_system(system, np.zeros(2), horizon=2, shocks=np.zeros((3, 4)))


def test_forecast_one_model1002_ss10_mode_none_smoke() -> None:
    model = Model1002(settings={"forecast_horizon": 3})

    output = forecast_one(model, "mode", "none", ["forecastobs"])

    assert output.states.shape == (3, 84)
    assert output.observables.shape == (3, 19)
    assert np.isfinite(output.states).all()
    assert np.isfinite(output.observables).all()


def test_forecast_one_model1002_uses_explicit_numpy_backend() -> None:
    model = Model1002(
        runtime=RuntimeConfig(backend="numpy", device="cpu"),
        settings={"forecast_horizon": 2},
    )

    output = forecast_one(model, "mode", "none", ["forecastobs", "forecastpseudo"])

    assert output.states.shape == (2, 84)
    assert output.pseudo_observables is not None
    assert output.pseudo_observables.shape == (2, 21)


def test_forecast_one_model1002_mode_accepts_short_explicit_shock_path() -> None:
    model = Model1002(settings={"forecast_horizon": 3})

    output = forecast_one(
        model,
        "mode",
        "none",
        ["forecastobs"],
        shocks=np.zeros((1, len(model.indexes.exogenous_shocks))),
    )

    assert output.states.shape == (3, 84)
    assert output.observables.shape == (3, 19)


def test_forecast_one_model1002_full_generates_sample_paths() -> None:
    model = Model1002(settings={"forecast_horizon": 2})

    output = forecast_one(model, "full", "none", ["forecastobs", "forecastpseudo"], draws=3, seed=4)

    assert output.state_samples is not None
    assert output.observable_samples is not None
    assert output.pseudo_observables is not None
    assert output.pseudo_observable_samples is not None
    assert output.state_samples.shape == (3, 2, 84)
    assert output.observable_samples.shape == (3, 2, 19)
    assert output.pseudo_observable_samples.shape == (3, 2, 21)
    np.testing.assert_allclose(output.observables, output.observable_samples.mean(axis=0))


def test_forecast_one_model1002_full_accepts_explicit_shock_samples() -> None:
    model = Model1002(settings={"forecast_horizon": 2})
    n_shocks = len(model.indexes.exogenous_shocks)
    shock_samples = np.zeros((2, 2, n_shocks), dtype=np.float64)
    shock_samples[1, 0, 0] = 0.25

    output = forecast_one(
        model,
        "full",
        "none",
        ["forecastobs"],
        shock_samples=shock_samples,
    )
    julia_order_output = forecast_one(
        model,
        "full",
        "none",
        ["forecastobs"],
        shock_samples=np.transpose(shock_samples, (0, 2, 1)),
    )

    assert output.state_samples is not None
    assert output.observable_samples is not None
    assert julia_order_output.state_samples is not None
    assert output.state_samples.shape == (2, 2, 84)
    assert output.observable_samples.shape == (2, 2, 19)
    np.testing.assert_allclose(output.states, output.state_samples.mean(axis=0))
    np.testing.assert_allclose(output.observables, output.observable_samples.mean(axis=0))
    np.testing.assert_allclose(julia_order_output.state_samples, output.state_samples)


def test_forecast_one_model1002_full_repeats_deterministic_history_samples() -> None:
    model = Model1002(settings={"forecast_horizon": 2})
    data = np.zeros((2, len(model.observables)))
    data[0, 0] = 1.25
    data[1, 3] = -0.5

    output = forecast_one(
        model,
        "full",
        "none",
        ["histobs", "histstates", "forecastobs"],
        data=data,
        draws=3,
        seed=4,
    )

    assert output.history_states is not None
    assert output.history_observables is not None
    assert output.history_state_samples is not None
    assert output.history_observable_samples is not None
    assert output.history_state_samples.shape == (3, 2, 84)
    assert output.history_observable_samples.shape == (3, 2, 19)
    np.testing.assert_allclose(output.history_states, output.history_state_samples.mean(axis=0))
    np.testing.assert_allclose(output.history_observables, data)
    np.testing.assert_allclose(
        output.history_observables,
        output.history_observable_samples.mean(axis=0),
    )


def test_forecast_one_model1002_full_requires_draws() -> None:
    model = Model1002(settings={"forecast_horizon": 1})

    with pytest.raises(ValueError, match="draws"):
        forecast_one(model, "full", "none", ["forecastobs"])


def test_forecast_parameter_draws_uses_sampler_values_and_restores_model() -> None:
    model = Model1002(settings={"forecast_horizon": 2})
    sampler = _sampler_for_alpha_draws(model)
    original_alpha = model.parameters["alpha"].value
    original_steady_state = dict(model.steady_state)

    output = forecast_parameter_draws(model, sampler, horizon=2, include_pseudo=True)
    pseudo_observables = output.pseudo_observables
    pseudo_observable_samples = output.pseudo_observable_samples

    assert output.state_samples is not None
    assert output.observable_samples is not None
    assert pseudo_observables is not None
    assert pseudo_observable_samples is not None
    assert output.state_samples.shape == (2, 2, 84)
    assert output.observable_samples.shape == (2, 2, 19)
    assert pseudo_observable_samples.shape == (2, 2, 21)
    np.testing.assert_allclose(output.states, output.state_samples.mean(axis=0))
    np.testing.assert_allclose(output.observables, output.observable_samples.mean(axis=0))
    np.testing.assert_allclose(
        pseudo_observables,
        pseudo_observable_samples.mean(axis=0),
    )
    assert model.parameters["alpha"].value == original_alpha
    assert dict(model.steady_state) == original_steady_state


def test_forecast_one_model1002_full_can_use_sampler_draws() -> None:
    model = Model1002(settings={"forecast_horizon": 2})
    sampler = _sampler_for_alpha_draws(model)

    output = forecast_one(
        model,
        "full",
        "none",
        ["forecastobs", "forecastpseudo"],
        sampler=sampler,
    )

    assert output.observable_samples is not None
    assert output.pseudo_observable_samples is not None
    assert output.observable_samples.shape == (2, 2, 19)
    assert output.pseudo_observable_samples.shape == (2, 2, 21)


def test_forecast_one_model1002_full_rejects_sampler_with_shock_samples() -> None:
    model = Model1002(settings={"forecast_horizon": 2})
    sampler = _sampler_for_alpha_draws(model)
    n_shocks = len(model.indexes.exogenous_shocks)

    with pytest.raises(ValueError, match="structural shock samples"):
        forecast_one(
            model,
            "full",
            "none",
            ["forecastobs"],
            shock_samples=np.zeros((2, 2, n_shocks)),
            sampler=sampler,
        )


def test_forecast_one_model1002_full_sampler_can_use_history_outputs() -> None:
    model = Model1002(settings={"forecast_horizon": 1})
    sampler = _sampler_for_alpha_draws(model)
    data = np.zeros((2, len(model.observables)))

    output = forecast_one(
        model,
        "full",
        "none",
        ["histobs", "histstates", "histpseudo", "forecastobs", "forecastpseudo"],
        data=data,
        sampler=sampler,
    )

    assert output.history_states is not None
    assert output.history_observables is not None
    assert output.history_pseudo_observables is not None
    assert output.history_state_samples is not None
    assert output.history_observable_samples is not None
    assert output.history_pseudo_observable_samples is not None
    assert output.history_state_samples.shape == (2, 2, 84)
    assert output.history_observable_samples.shape == (2, 2, 19)
    assert output.history_pseudo_observable_samples.shape == (2, 2, 21)
    np.testing.assert_allclose(output.history_states, output.history_state_samples.mean(axis=0))
    np.testing.assert_allclose(
        output.history_observables,
        output.history_observable_samples.mean(axis=0),
    )


def test_forecast_one_model1002_full_sampler_can_use_conditional_history() -> None:
    model = Model1002(
        settings={
            "date_forecast_start": "2018-Q4",
            "forecast_horizon": 4,
        }
    )
    sampler = _sampler_for_alpha_draws(model)
    data = _dated_observable_frame(model, ["2018-Q3", "2018-Q4", "2019-Q1"])

    output = forecast_one(
        model,
        "full",
        "semi",
        ["histobs", "histstates", "forecastobs"],
        data=data,
        sampler=sampler,
    )

    assert output.history_states is not None
    assert output.history_observables is not None
    assert output.history_state_samples is not None
    assert output.history_observable_samples is not None
    assert output.history_states.shape == (1, 84)
    assert output.history_observables.shape == (1, 19)
    assert output.history_state_samples.shape == (2, 1, 84)
    assert output.history_observable_samples.shape == (2, 1, 19)
    assert output.state_samples is not None
    assert output.observable_samples is not None
    assert output.state_samples.shape == (2, 2, 84)
    assert output.observable_samples.shape == (2, 2, 19)
    assert output.log_likelihood is not None


def test_forecast_one_model1002_can_include_pseudo_observables() -> None:
    model = Model1002(settings={"forecast_horizon": 2})
    data = np.zeros((2, len(model.observables)))

    output = forecast_one(
        model,
        "mode",
        "none",
        ["forecastobs", "forecastpseudo", "histpseudo"],
        data=data,
    )

    assert output.pseudo_observables is not None
    assert output.pseudo_observables.shape == (2, 21)
    assert output.history_pseudo_observables is not None
    assert output.history_pseudo_observables.shape == (2, 21)
    assert np.isfinite(output.pseudo_observables).all()


def test_forecast_one_model1002_uses_filtered_history_as_forecast_start() -> None:
    model = Model1002(settings={"forecast_horizon": 2})
    data = np.zeros((2, len(model.observables)))
    data[0, 0] = 1.25
    data[1, 3] = -0.5

    output = forecast_one(
        model, "mode", "none", ["histobs", "histstates", "forecastobs"], data=data
    )

    assert output.history_states is not None
    assert output.history_observables is not None
    assert output.history_states.shape == (2, 84)
    assert output.history_observables.shape == (2, 19)
    np.testing.assert_allclose(output.history_observables, data)
    assert output.log_likelihood is not None
    assert np.isfinite(output.log_likelihood)
    system = compute_system(model)
    expected_first_forecast_state = (
        system.transition.TTT @ output.history_states[-1] + system.transition.CCC
    )
    np.testing.assert_allclose(output.states[0], expected_first_forecast_state)


def test_forecast_one_model1002_can_use_smoothed_history() -> None:
    model = Model1002(settings={"forecast_horizon": 2})
    data = np.zeros((2, len(model.observables)))
    data[0, 0] = 1.25
    data[1, 3] = -0.5

    output = forecast_one(
        model,
        "mode",
        "none",
        ["histobs", "histstates", "forecastobs"],
        data=data,
        history_method="smoothed",
    )

    system = compute_system(model)
    filtered = kalman_log_likelihood(
        system,
        data,
        process_covariances=model_process_covariances(model, system, data.shape[0]),
    )
    expected = smooth_kalman_result(system, filtered)

    assert output.history_states is not None
    assert output.history_observables is not None
    assert expected.smoothed_states is not None
    np.testing.assert_allclose(output.history_states, expected.smoothed_states)
    np.testing.assert_allclose(output.history_observables, data)
    expected_first_forecast_state = (
        system.transition.TTT @ output.history_states[-1] + system.transition.CCC
    )
    np.testing.assert_allclose(output.states[0], expected_first_forecast_state)


def test_forecast_one_model1002_semiconditional_data_reduces_horizon() -> None:
    model = Model1002(
        settings={
            "date_forecast_start": "2018-Q4",
            "forecast_horizon": 4,
        }
    )
    data = _dated_observable_frame(
        model,
        ["2018-Q3", "2018-Q4", "2019-Q1"],
    )

    output = forecast_one(
        model,
        "mode",
        "semi",
        ["histobs", "histstates", "forecastobs"],
        data=data,
    )

    assert output.history_states is not None
    assert output.history_observables is not None
    assert output.history_states.shape == (1, 84)
    assert output.history_observables.shape == (1, 19)
    assert output.states.shape == (2, 84)
    system = compute_system(model)
    combined = np.vstack(
        [
            df_to_matrix(model, data, in_sample=True),
            df_to_matrix(model, data, in_sample=False),
        ]
    )
    filtered = kalman_log_likelihood(
        system,
        combined,
        process_covariances=model_process_covariances(
            model,
            system,
            combined.shape[0],
            start_date="2018-Q3",
        ),
    )
    expected_first_forecast_state = (
        system.transition.TTT @ filtered.filtered_states[-1] + system.transition.CCC
    )
    np.testing.assert_allclose(output.states[0], expected_first_forecast_state)


def test_forecast_one_model1002_conditional_matrix_uses_explicit_period_count() -> None:
    model = Model1002(settings={"forecast_horizon": 3})
    data = np.zeros((2, len(model.observables)))

    output = forecast_one(
        model,
        "mode",
        "full",
        ["histstates", "forecastobs"],
        data=data,
        conditional_periods=1,
    )

    assert output.history_states is not None
    assert output.history_states.shape == (1, 84)
    assert output.states.shape == (2, 84)


def test_forecast_one_model1002_full_conditioning_solves_conditional_shocks() -> None:
    model = Model1002(settings={"forecast_horizon": 3})
    data = np.full((2, len(model.observables)), np.nan, dtype=np.float64)
    data[1, 0] = 0.25

    output = forecast_one(
        model,
        "mode",
        "full",
        ["forecastobs"],
        data=data,
        conditional_periods=1,
    )

    assert output.conditional_shocks is not None
    assert output.conditional_states is not None
    assert output.conditional_observables is not None
    assert output.conditional_shocks.shape == (1, len(model.indexes.exogenous_shocks))
    assert output.conditional_states.shape == (1, 84)
    assert output.conditional_observables.shape == (1, len(model.observables))
    assert output.states.shape == (2, 84)
    assert np.isfinite(output.conditional_observables[0, 0])


def test_forecast_one_model1002_conditional_data_validates_periods() -> None:
    model = Model1002(settings={"forecast_horizon": 1})
    data = _dated_observable_frame(model, ["2018-Q4", "2019-Q1"])

    with pytest.raises(ValueError, match="cannot exceed"):
        forecast_one(model, "mode", "semi", ["forecastobs"], data=data)

    with pytest.raises(ValueError, match="conditional_periods"):
        forecast_one(
            model,
            "mode",
            "semi",
            ["forecastobs"],
            data=np.zeros((2, len(model.observables))),
        )


def test_forecast_one_model1002_rejects_unknown_history_method() -> None:
    model = Model1002(settings={"forecast_horizon": 1})

    with pytest.raises(ValueError, match="History method"):
        forecast_one(model, "mode", "none", ["forecastobs"], history_method="bad")


def test_forecast_one_model1002_rejects_unported_modes() -> None:
    model = Model1002()

    with pytest.raises(NotPortedError, match="input_type"):
        forecast_one(model, "other", "none", ["forecastobs"], horizon=1)

    with pytest.raises(NotPortedError, match="Conditioning type"):
        forecast_one(model, "mode", "bad", ["forecastobs"], horizon=1)


def test_meansbands_from_forecast_is_deterministic() -> None:
    forecast = ForecastOutput(
        states=np.ones((2, 3)),
        observables=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    bands = meansbands_from_forecast(forecast)

    np.testing.assert_allclose(bands.mean, forecast.observables)
    np.testing.assert_allclose(bands.lower, forecast.observables)
    np.testing.assert_allclose(bands.upper, forecast.observables)


def test_meansbands_from_forecast_uses_samples_when_available() -> None:
    forecast = ForecastOutput(
        states=np.zeros((2, 1)),
        observables=np.zeros((2, 1)),
        observable_samples=np.array([[[1.0], [2.0]], [[3.0], [4.0]]]),
    )

    bands = meansbands_from_forecast(
        forecast,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    np.testing.assert_allclose(bands.mean, np.array([[2.0], [3.0]]))
    np.testing.assert_allclose(bands.lower, np.array([[1.0], [2.0]]))
    np.testing.assert_allclose(bands.upper, np.array([[3.0], [4.0]]))


def test_meansbands_from_samples_computes_quantiles() -> None:
    samples = np.array(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
        ]
    )

    bands = meansbands_from_samples(samples, lower_quantile=0.0, upper_quantile=1.0)

    np.testing.assert_allclose(bands.mean, np.array([[3.0], [4.0]]))
    np.testing.assert_allclose(bands.lower, np.array([[1.0], [2.0]]))
    np.testing.assert_allclose(bands.upper, np.array([[5.0], [6.0]]))


def test_meansbands_from_samples_validates_shape_and_quantiles() -> None:
    with pytest.raises(ValueError, match="Samples"):
        meansbands_from_samples(np.ones((2, 3)))

    with pytest.raises(ValueError, match="Quantiles"):
        meansbands_from_samples(np.ones((2, 3, 1)), lower_quantile=0.9, upper_quantile=0.1)


def test_compute_meansbands_model1002_ss10_mode_none_smoke() -> None:
    model = Model1002(settings={"forecast_horizon": 2})

    bands = compute_meansbands(model, "mode", "none", ["forecastobs"])

    assert bands.mean.shape == (2, 19)
    np.testing.assert_allclose(bands.lower, bands.mean)
    np.testing.assert_allclose(bands.upper, bands.mean)


def test_compute_meansbands_model1002_full_uses_sample_quantiles() -> None:
    model = Model1002(settings={"forecast_horizon": 2})

    bands = compute_meansbands(
        model,
        "full",
        "none",
        ["forecastobs"],
        draws=4,
        seed=2,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert bands.mean.shape == (2, 19)
    assert np.all(bands.lower <= bands.upper)


def test_compute_meansbands_model1002_full_can_use_sampler_draws() -> None:
    model = Model1002(settings={"forecast_horizon": 2})
    sampler = _sampler_for_alpha_draws(model)

    bands = compute_meansbands(
        model,
        "full",
        "none",
        ["forecastobs"],
        sampler=sampler,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert bands.mean.shape == (2, 19)
    assert np.all(bands.lower <= bands.upper)


def test_compute_meansbands_model1002_full_sampler_can_use_history_samples() -> None:
    model = Model1002(settings={"forecast_horizon": 1})
    sampler = _sampler_for_alpha_draws(model)
    data = np.zeros((2, len(model.observables)))

    bands = compute_meansbands(
        model,
        "full",
        "none",
        ["histobs"],
        source="histobs",
        data=data,
        sampler=sampler,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert bands.mean.shape == (2, 19)
    assert np.all(bands.lower <= bands.upper)


def test_compute_meansbands_model1002_full_sampler_can_use_conditional_history() -> None:
    model = Model1002(
        settings={
            "date_forecast_start": "2018-Q4",
            "forecast_horizon": 4,
        }
    )
    sampler = _sampler_for_alpha_draws(model)
    data = _dated_observable_frame(model, ["2018-Q3", "2018-Q4", "2019-Q1"])

    history_bands = compute_meansbands(
        model,
        "full",
        "full",
        ["histobs"],
        source="histobs",
        data=data,
        sampler=sampler,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )
    forecast_bands = compute_meansbands(
        model,
        "full",
        "full",
        ["forecastobs"],
        source="forecastobs",
        data=data,
        sampler=sampler,
        lower_quantile=0.0,
        upper_quantile=1.0,
    )

    assert history_bands.mean.shape == (1, 19)
    assert forecast_bands.mean.shape == (2, 19)
    assert np.all(history_bands.lower <= history_bands.upper)
    assert np.all(forecast_bands.lower <= forecast_bands.upper)


def test_compute_meansbands_can_use_pseudo_observable_source() -> None:
    model = Model1002(settings={"forecast_horizon": 2})

    bands = compute_meansbands(
        model,
        "mode",
        "none",
        ["forecastpseudo"],
        source="forecastpseudo",
    )

    assert bands.mean.shape == (2, 21)
    np.testing.assert_allclose(bands.lower, bands.mean)
    np.testing.assert_allclose(bands.upper, bands.mean)


def test_compute_meansbands_can_use_filter_backed_history_source() -> None:
    model = Model1002(settings={"forecast_horizon": 1})
    data = np.zeros((2, len(model.observables)))

    bands = compute_meansbands(model, "mode", "none", ["histobs"], source="histobs", data=data)

    assert bands.mean.shape == (2, 19)
    np.testing.assert_allclose(bands.lower, bands.mean)
    np.testing.assert_allclose(bands.upper, bands.mean)


def test_compute_meansbands_can_use_smoothed_history_source() -> None:
    model = Model1002(settings={"forecast_horizon": 1})
    data = np.zeros((2, len(model.observables)))

    bands = compute_meansbands(
        model,
        "mode",
        "none",
        ["histobs"],
        source="histobs",
        data=data,
        history_method="smoothed",
    )

    assert bands.mean.shape == (2, 19)
    np.testing.assert_allclose(bands.lower, bands.mean)
    np.testing.assert_allclose(bands.upper, bands.mean)


def test_compute_meansbands_can_use_conditional_forecast_source() -> None:
    model = Model1002(
        settings={
            "date_forecast_start": "2018-Q4",
            "forecast_horizon": 3,
        }
    )
    data = _dated_observable_frame(model, ["2018-Q3", "2018-Q4"])

    bands = compute_meansbands(
        model,
        "mode",
        "full",
        ["forecastobs"],
        data=data,
    )

    assert bands.mean.shape == (2, 19)
    np.testing.assert_allclose(bands.lower, bands.mean)
    np.testing.assert_allclose(bands.upper, bands.mean)


def test_reverse_transform_forecast_preserves_states_and_transforms_observables() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    observables = np.zeros((1, len(model.observables)))
    nominal_rate_column = list(model.observables).index("obs_nominalrate")
    observables[0, nominal_rate_column] = 0.25
    pseudo_observables = np.zeros((1, len(model.pseudo_observables)))
    natural_rate_column = list(model.pseudo_observables).index("NaturalRate")
    pseudo_observables[0, natural_rate_column] = 0.25
    forecast = ForecastOutput(
        states=np.ones((1, 2)),
        observables=observables,
        pseudo_observables=pseudo_observables,
        history_states=np.ones((1, 2)),
        history_observables=observables,
        history_pseudo_observables=pseudo_observables,
        history_state_samples=np.ones((1, 1, 2)),
        history_observable_samples=observables.reshape(1, *observables.shape),
        history_pseudo_observable_samples=pseudo_observables.reshape(
            1,
            *pseudo_observables.shape,
        ),
        log_likelihood=-1.0,
    )

    transformed = reverse_transform_forecast(model, forecast)

    np.testing.assert_allclose(transformed.states, forecast.states)
    assert transformed.states is not forecast.states
    assert transformed.observables[0, nominal_rate_column] == 1.0
    assert transformed.pseudo_observables is not None
    assert transformed.pseudo_observables[0, natural_rate_column] == 1.0
    assert transformed.history_observables is not None
    assert transformed.history_observables[0, nominal_rate_column] == 1.0
    assert transformed.history_pseudo_observables is not None
    assert transformed.history_pseudo_observables[0, natural_rate_column] == 1.0
    assert transformed.history_observable_samples is not None
    assert transformed.history_observable_samples[0, 0, nominal_rate_column] == 1.0
    assert transformed.history_pseudo_observable_samples is not None
    assert transformed.history_pseudo_observable_samples[0, 0, natural_rate_column] == 1.0
    assert transformed.log_likelihood == -1.0


def test_reverse_transform_meansbands_transforms_each_band() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    nominal_rate_column = list(model.observables).index("obs_nominalrate")
    shape = (1, len(model.observables))
    mean = np.zeros(shape)
    lower = np.zeros(shape)
    upper = np.zeros(shape)
    mean[0, nominal_rate_column] = 0.25
    lower[0, nominal_rate_column] = 0.1
    upper[0, nominal_rate_column] = 0.5

    transformed = reverse_transform_meansbands(
        model,
        MeansBands(mean=mean, lower=lower, upper=upper),
    )

    assert transformed.mean[0, nominal_rate_column] == 1.0
    assert transformed.lower[0, nominal_rate_column] == 0.4
    assert transformed.upper[0, nominal_rate_column] == 2.0


def test_reverse_transform_meansbands_transforms_pseudo_source() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 0})
    natural_rate_column = list(model.pseudo_observables).index("NaturalRate")
    shape = (1, len(model.pseudo_observables))
    mean = np.zeros(shape)
    lower = np.zeros(shape)
    upper = np.zeros(shape)
    mean[0, natural_rate_column] = 0.25
    lower[0, natural_rate_column] = 0.1
    upper[0, natural_rate_column] = 0.5

    transformed = reverse_transform_meansbands(
        model,
        MeansBands(mean=mean, lower=lower, upper=upper),
        source="forecastpseudo",
    )

    assert transformed.mean[0, natural_rate_column] == 1.0
    assert transformed.lower[0, natural_rate_column] == 0.4
    assert transformed.upper[0, natural_rate_column] == 2.0


def _sampler_for_alpha_draws(model: Model1002) -> MetropolisHastingsResult:
    alpha = model.parameters["alpha"].value
    return MetropolisHastingsResult(
        parameter_names=("alpha",),
        estimation_draws=np.zeros((2, 1), dtype=np.float64),
        parameter_draws=np.array([[alpha], [alpha + 1.0e-3]], dtype=np.float64),
        log_posterior=np.zeros(2, dtype=np.float64),
        accepted=np.array([True, True]),
        acceptance_rate=1.0,
        proposal_covariance=np.eye(1, dtype=np.float64),
        seed=1,
        burnin=0,
    )


def _dated_observable_frame(model: Model1002, dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            **{name: np.zeros(len(dates), dtype=np.float64) for name in model.observables},
        }
    )
