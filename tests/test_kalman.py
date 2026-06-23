from __future__ import annotations

import numpy as np
import pytest

from nydsge.backends import NumpyBackend
from nydsge.kalman import (
    kalman_log_likelihood,
    model_process_covariances,
    observables_from_states,
    smooth_kalman_result,
)
from nydsge.models import Model1002
from nydsge.solve import Measurement, System, Transition, compute_system


def test_kalman_log_likelihood_univariate_known_value() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[0.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.array([[1.0]]),
            EE=np.array([[1.0]]),
        ),
    )
    result = kalman_log_likelihood(
        system,
        np.array([[0.0]]),
        initial_state=np.array([0.0]),
        initial_covariance=np.array([[0.0]]),
    )
    expected = -0.5 * np.log(2.0 * np.pi)
    assert np.isclose(result.log_likelihood, expected)
    np.testing.assert_allclose(result.log_likelihood_by_period, np.array([expected]))
    assert np.isclose(result.log_likelihood_by_period.sum(), result.log_likelihood)
    assert result.filtered_states.tolist() == [[0.0]]


def test_kalman_can_filter_presample_but_score_main_sample_only() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[0.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.array([[1.0]]),
            EE=np.array([[1.0]]),
        ),
    )

    full = kalman_log_likelihood(
        system,
        np.array([[1.0], [0.0]]),
        initial_state=np.array([0.0]),
        initial_covariance=np.array([[0.0]]),
    )
    main_only = kalman_log_likelihood(
        system,
        np.array([[1.0], [0.0]]),
        initial_state=np.array([0.0]),
        initial_covariance=np.array([[0.0]]),
        log_likelihood_start=1,
    )

    assert full.log_likelihood_by_period.shape == (2,)
    assert np.isclose(main_only.log_likelihood, full.log_likelihood_by_period[1])
    assert main_only.filtered_states.shape == full.filtered_states.shape


def test_kalman_skips_missing_observations() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.5]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.array([[0.25]]),
            EE=np.array([[1.0]]),
        ),
    )
    result = kalman_log_likelihood(
        system,
        np.array([[np.nan], [1.0]]),
        initial_state=np.array([0.0]),
        initial_covariance=np.array([[0.0]]),
    )
    assert result.predicted_states[:, 0].tolist() == [0.5, 1.0]
    assert np.isfinite(result.log_likelihood)
    np.testing.assert_allclose(result.final_filtered_state, result.filtered_states[-1])


def test_kalman_accepts_process_covariances_by_period() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[0.0]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.array([[1.0]]),
            EE=np.array([[1.0]]),
        ),
    )

    result = kalman_log_likelihood(
        system,
        np.array([[np.nan], [np.nan]]),
        initial_state=np.array([0.0]),
        initial_covariance=np.array([[0.0]]),
        process_covariances=np.array([[[0.0]], [[2.0]]]),
    )

    np.testing.assert_allclose(result.predicted_covariances[:, 0, 0], np.array([0.0, 2.0]))
    np.testing.assert_allclose(result.filtered_covariances[:, 0, 0], np.array([0.0, 2.0]))


def test_kalman_process_covariances_affect_default_initialization() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[0.5]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.array([[1.0]]),
            EE=np.array([[1.0]]),
        ),
    )

    default = kalman_log_likelihood(system, np.array([[np.nan]]))
    regime = kalman_log_likelihood(
        system,
        np.array([[np.nan]]),
        process_covariances=np.array([[[0.0]]]),
    )

    assert default.predicted_covariances[0, 0, 0] > 0.0
    np.testing.assert_allclose(regime.predicted_covariances[0, 0, 0], 0.0)


def test_model1002_process_covariances_zero_anticipated_shocks_before_zlb() -> None:
    model = Model1002()
    system = compute_system(model)

    covariances = model_process_covariances(model, system, 3, start_date="2008-Q3")

    assert covariances is not None
    assert covariances.shape == (3, system.transition.TTT.shape[0], system.transition.TTT.shape[0])
    post_zlb = system.transition.RRR @ system.measurement.QQ @ system.transition.RRR.T
    np.testing.assert_allclose(covariances[1], post_zlb)
    np.testing.assert_allclose(covariances[2], post_zlb)
    assert np.max(np.abs(covariances[0] - post_zlb)) > 0.0


def test_kalman_backend_path_matches_default_numpy() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[0.9, 0.1], [0.0, 0.8]]),
            RRR=np.array([[1.0], [0.5]]),
            CCC=np.array([0.1, -0.2]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0, 0.5], [0.0, 1.0]]),
            DD=np.array([0.2, -0.1]),
            QQ=np.array([[0.25]]),
            EE=np.eye(2),
        ),
    )
    data = np.array([[0.0, 1.0], [np.nan, 0.5], [0.2, -0.1]])

    default = kalman_log_likelihood(system, data)
    backend = kalman_log_likelihood(system, data, backend=NumpyBackend())

    assert np.isclose(backend.log_likelihood, default.log_likelihood)
    np.testing.assert_allclose(backend.filtered_states, default.filtered_states)
    np.testing.assert_allclose(backend.filtered_covariances, default.filtered_covariances)
    np.testing.assert_allclose(backend.predicted_states, default.predicted_states)
    np.testing.assert_allclose(backend.predicted_covariances, default.predicted_covariances)


def test_kalman_default_initialization_uses_stationary_state() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[0.5]]),
            RRR=np.array([[0.0]]),
            CCC=np.array([2.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.array([[1.0]]),
            EE=np.array([[1.0]]),
        ),
    )

    result = kalman_log_likelihood(system, np.array([[np.nan]]))

    np.testing.assert_allclose(result.predicted_states, np.array([[4.0]]))
    np.testing.assert_allclose(result.filtered_states, np.array([[4.0]]))


def test_smooth_kalman_result_runs_rts_backward_pass() -> None:
    system = System(
        transition=Transition(
            TTT=np.array([[1.0]]),
            RRR=np.array([[1.0]]),
            CCC=np.array([0.0]),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0]]),
            DD=np.array([0.0]),
            QQ=np.array([[1.0]]),
            EE=np.array([[1.0]]),
        ),
    )
    filtered = kalman_log_likelihood(
        system,
        np.array([[0.0], [3.0]]),
        initial_state=np.array([0.0]),
        initial_covariance=np.array([[1.0]]),
    )

    smoothed = smooth_kalman_result(system, filtered)

    assert smoothed.smoothed_states is not None
    assert smoothed.smoothed_covariances is not None
    assert smoothed.smoothed_states.shape == filtered.filtered_states.shape
    assert smoothed.smoothed_covariances.shape == filtered.filtered_covariances.shape
    np.testing.assert_allclose(smoothed.smoothed_states[-1], filtered.filtered_states[-1])
    np.testing.assert_allclose(smoothed.final_smoothed_state, filtered.filtered_states[-1])
    assert smoothed.smoothed_states[0, 0] > filtered.filtered_states[0, 0]
    assert smoothed.smoothed_covariances[0, 0, 0] <= filtered.filtered_covariances[0, 0, 0]


def test_observables_from_states_applies_measurement_equation() -> None:
    system = System(
        transition=Transition(
            TTT=np.eye(2),
            RRR=np.ones((2, 1)),
            CCC=np.zeros(2),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0, 2.0]]),
            DD=np.array([0.5]),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
    )

    values = observables_from_states(system, np.array([[1.0, 2.0], [3.0, 4.0]]))

    np.testing.assert_allclose(values, np.array([[5.5], [11.5]]))


def test_observables_from_states_validates_shape() -> None:
    system = System(
        transition=Transition(
            TTT=np.eye(2),
            RRR=np.ones((2, 1)),
            CCC=np.zeros(2),
        ),
        measurement=Measurement(
            ZZ=np.array([[1.0, 2.0]]),
            DD=np.array([0.0]),
            QQ=np.eye(1),
            EE=np.eye(1),
        ),
    )

    with pytest.raises(ValueError, match="shape"):
        observables_from_states(system, np.ones(2))

    with pytest.raises(ValueError, match="columns"):
        observables_from_states(system, np.ones((1, 1)))
