from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nydsge.backends import ArrayBackend, NumpyBackend
from nydsge.solve import System


@dataclass(frozen=True)
class KalmanResult:
    log_likelihood: float
    log_likelihood_by_period: np.ndarray
    filtered_states: np.ndarray
    filtered_covariances: np.ndarray
    predicted_states: np.ndarray
    predicted_covariances: np.ndarray
    smoothed_states: np.ndarray | None = None
    smoothed_covariances: np.ndarray | None = None

    @property
    def final_filtered_state(self) -> np.ndarray:
        if self.filtered_states.shape[0] == 0:
            msg = "Kalman result has no filtered states."
            raise ValueError(msg)
        return self.filtered_states[-1].copy()

    @property
    def final_smoothed_state(self) -> np.ndarray:
        if self.smoothed_states is None:
            msg = "Kalman result does not include smoothed states."
            raise ValueError(msg)
        if self.smoothed_states.shape[0] == 0:
            msg = "Kalman result has no smoothed states."
            raise ValueError(msg)
        return self.smoothed_states[-1].copy()


def kalman_log_likelihood(
    system: System,
    data: np.ndarray,
    *,
    initial_state: np.ndarray | None = None,
    initial_covariance: np.ndarray | None = None,
    process_covariances: np.ndarray | None = None,
    backend: ArrayBackend | None = None,
    log_likelihood_start: int = 0,
) -> KalmanResult:
    array_backend = backend or NumpyBackend()
    observations = np.asarray(data, dtype=np.float64)
    if observations.ndim != 2:
        msg = "Kalman data must be a 2D array shaped as periods x observables."
        raise ValueError(msg)

    transition = system.transition
    measurement = system.measurement
    n_periods = observations.shape[0]
    n_states = transition.TTT.shape[0]
    if log_likelihood_start < 0 or log_likelihood_start > n_periods:
        msg = "log_likelihood_start must be between 0 and the number of periods."
        raise ValueError(msg)
    if observations.shape[1] != measurement.ZZ.shape[0]:
        msg = (
            "Data observable count does not match measurement matrix rows: "
            f"{observations.shape[1]} != {measurement.ZZ.shape[0]}"
        )
        raise ValueError(msg)

    process_covariance_by_period = None
    if process_covariances is not None:
        process_covariance_by_period = np.asarray(process_covariances, dtype=np.float64)
        expected_process_shape = (n_periods, n_states, n_states)
        if process_covariance_by_period.shape != expected_process_shape:
            msg = (
                "Process covariances must have shape "
                f"{expected_process_shape}; got {process_covariance_by_period.shape}."
            )
            raise ValueError(msg)

    default_state, default_covariance = _default_initial_state_covariance(
        system,
        process_covariance=(
            None if process_covariance_by_period is None else process_covariance_by_period[0]
        ),
    )
    state = default_state if initial_state is None else np.asarray(initial_state, dtype=np.float64)
    covariance = (
        default_covariance
        if initial_covariance is None
        else np.asarray(initial_covariance, dtype=np.float64)
    )
    if state.shape != (n_states,):
        msg = f"Initial state must have shape {(n_states,)}."
        raise ValueError(msg)
    if covariance.shape != (n_states, n_states):
        msg = f"Initial covariance must have shape {(n_states, n_states)}."
        raise ValueError(msg)

    backend_ttt = array_backend.array(transition.TTT)
    backend_rrr = array_backend.array(transition.RRR)
    backend_ccc = array_backend.array(transition.CCC)
    backend_state = array_backend.array(state)
    backend_covariance = array_backend.array(covariance)
    if process_covariance_by_period is None:
        process_covariance = array_backend.matmul(
            array_backend.matmul(backend_rrr, array_backend.array(measurement.QQ)),
            array_backend.transpose(backend_rrr),
        )
    else:
        process_covariance = None
    filtered_states = np.zeros((n_periods, n_states), dtype=np.float64)
    predicted_states = np.zeros((n_periods, n_states), dtype=np.float64)
    filtered_covariances = np.zeros((n_periods, n_states, n_states), dtype=np.float64)
    predicted_covariances = np.zeros((n_periods, n_states, n_states), dtype=np.float64)
    log_likelihood_by_period = np.zeros(n_periods, dtype=np.float64)
    log_likelihood = 0.0

    for period, observed in enumerate(observations):
        predicted_state = array_backend.matmul(backend_ttt, backend_state) + backend_ccc
        predicted_covariance = array_backend.matmul(
            array_backend.matmul(backend_ttt, backend_covariance),
            array_backend.transpose(backend_ttt),
        ) + (
            array_backend.array(process_covariance_by_period[period])
            if process_covariance_by_period is not None
            else process_covariance
        )
        predicted_states[period] = array_backend.as_numpy(predicted_state)
        predicted_covariances[period] = array_backend.as_numpy(predicted_covariance)

        mask = ~np.isnan(observed)
        if np.any(mask):
            z_t = array_backend.array(measurement.ZZ[mask, :])
            d_t = array_backend.array(measurement.DD[mask])
            e_t = array_backend.array(measurement.EE[np.ix_(mask, mask)])
            innovation = array_backend.array(observed[mask]) - (
                array_backend.matmul(z_t, predicted_state) + d_t
            )
            innovation_covariance = (
                array_backend.matmul(
                    array_backend.matmul(z_t, predicted_covariance),
                    array_backend.transpose(z_t),
                )
                + e_t
            )
            period_log_likelihood = _normal_logpdf(
                innovation,
                innovation_covariance,
                array_backend,
            )
            log_likelihood_by_period[period] = period_log_likelihood
            if period >= log_likelihood_start:
                log_likelihood += period_log_likelihood
            gain = array_backend.transpose(
                array_backend.solve(
                    array_backend.transpose(innovation_covariance),
                    array_backend.matmul(z_t, array_backend.transpose(predicted_covariance)),
                )
            )
            backend_state = predicted_state + array_backend.matmul(gain, innovation)
            backend_covariance = predicted_covariance - array_backend.matmul(
                array_backend.matmul(gain, z_t),
                predicted_covariance,
            )
            backend_covariance = _symmetrize(backend_covariance, array_backend)
        else:
            backend_state = predicted_state
            backend_covariance = predicted_covariance

        filtered_states[period] = array_backend.as_numpy(backend_state)
        filtered_covariances[period] = array_backend.as_numpy(backend_covariance)

    return KalmanResult(
        log_likelihood=float(log_likelihood),
        log_likelihood_by_period=log_likelihood_by_period,
        filtered_states=filtered_states,
        filtered_covariances=filtered_covariances,
        predicted_states=predicted_states,
        predicted_covariances=predicted_covariances,
    )


def model_process_covariances(
    model: Any,
    system: System,
    periods: int,
    *,
    start_date: Any | None = None,
) -> np.ndarray | None:
    """Return Julia-compatible Model1002 process covariances when regimes matter."""
    if periods < 0:
        msg = "Process covariance periods must be nonnegative."
        raise ValueError(msg)
    if periods == 0:
        return None
    if getattr(model, "spec", None) != "m1002":
        return None
    n_anticipated = int(model.get_setting("n_mon_anticipated_shocks", 0) or 0)
    if n_anticipated <= 0:
        return None
    zlb_start = model.get_setting("date_zlb_start", None)
    if zlb_start is None:
        return None
    start = start_date
    if start is None:
        start = model.get_setting("date_mainsample_start", None)
    if start is None:
        start = model.get_setting("date_presample_start", None)
    if start is None:
        return None

    start_index = _quarter_to_index(start)
    zlb_index = _quarter_to_index(zlb_start)
    period_indexes = start_index + np.arange(periods, dtype=np.int64)
    pre_zlb = period_indexes < zlb_index
    if not np.any(pre_zlb):
        return None

    transition = system.transition
    measurement = system.measurement
    post_zlb_process = transition.RRR @ measurement.QQ @ transition.RRR.T
    pre_zlb_qq = measurement.QQ.copy()
    anticipated_indexes = _anticipated_policy_shock_indexes(model)
    if not anticipated_indexes:
        return None
    pre_zlb_qq[anticipated_indexes, :] = 0.0
    pre_zlb_qq[:, anticipated_indexes] = 0.0
    pre_zlb_process = transition.RRR @ pre_zlb_qq @ transition.RRR.T

    covariances = np.repeat(post_zlb_process[np.newaxis, :, :], periods, axis=0)
    covariances[pre_zlb] = pre_zlb_process
    return covariances


def _anticipated_policy_shock_indexes(model: Any) -> list[int]:
    shock_names = list(model.indexes.exogenous_shocks)
    base_name = str(model.get_setting("monetary_policy_shock", "rm_sh"))
    prefix = f"{base_name}l"
    n_anticipated = int(model.get_setting("n_mon_anticipated_shocks", 0) or 0)
    expected = {f"{prefix}{index}" for index in range(1, n_anticipated + 1)}
    return [index for index, name in enumerate(shock_names) if str(name) in expected]


def _quarter_to_index(value: Any) -> int:
    text = str(value).strip()
    if "-Q" in text.upper():
        year_text, quarter_text = text.upper().split("-Q", 1)
        return int(year_text) * 4 + int(quarter_text[0])
    parsed = np.datetime64(text, "D").astype(object)
    quarter = (int(parsed.month) - 1) // 3 + 1
    return int(parsed.year) * 4 + quarter


def _default_initial_state_covariance(
    system: System,
    *,
    process_covariance: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    transition = system.transition
    measurement = system.measurement
    ttt = np.asarray(transition.TTT, dtype=np.float64)
    rrr = np.asarray(transition.RRR, dtype=np.float64)
    ccc = np.asarray(transition.CCC, dtype=np.float64)
    qq = np.asarray(measurement.QQ, dtype=np.float64)
    n_states = ttt.shape[0]
    eigenvalues = np.linalg.eigvals(ttt)
    if np.all(np.abs(eigenvalues) < 1.0):
        state = np.linalg.solve(np.eye(n_states, dtype=np.float64) - ttt, ccc)
        state_covariance = (
            np.asarray(process_covariance, dtype=np.float64)
            if process_covariance is not None
            else rrr @ qq @ rrr.T
        )
        covariance = _solve_discrete_lyapunov_doubling(ttt, state_covariance)
        return state, covariance
    return ccc.copy(), np.eye(n_states, dtype=np.float64) * 1.0e6


def _solve_discrete_lyapunov_doubling(
    transition: np.ndarray,
    covariance: np.ndarray,
    *,
    max_iterations: int = 50,
) -> np.ndarray:
    alpha = np.asarray(transition, dtype=np.float64).copy()
    gamma = np.asarray(covariance, dtype=np.float64).copy()
    for _ in range(max_iterations):
        next_alpha = alpha @ alpha
        next_gamma = gamma + alpha @ gamma @ alpha.T
        if np.max(np.abs(next_gamma - gamma)) <= 1.0e-15:
            return next_gamma
        alpha = next_alpha
        gamma = next_gamma
    msg = "Exceeded maximum iterations while solving default Kalman covariance."
    raise ValueError(msg)


def smooth_kalman_result(system: System, result: KalmanResult) -> KalmanResult:
    """Run a Rauch-Tung-Striebel smoother from stored Kalman filter arrays."""
    filtered_states = np.asarray(result.filtered_states, dtype=np.float64)
    filtered_covariances = np.asarray(result.filtered_covariances, dtype=np.float64)
    predicted_states = np.asarray(result.predicted_states, dtype=np.float64)
    predicted_covariances = np.asarray(result.predicted_covariances, dtype=np.float64)
    _validate_smoother_shapes(
        system,
        filtered_states=filtered_states,
        filtered_covariances=filtered_covariances,
        predicted_states=predicted_states,
        predicted_covariances=predicted_covariances,
    )

    smoothed_states = filtered_states.copy()
    smoothed_covariances = filtered_covariances.copy()
    transition = np.asarray(system.transition.TTT, dtype=np.float64)

    for period in range(filtered_states.shape[0] - 2, -1, -1):
        predicted_covariance_next = predicted_covariances[period + 1]
        smoother_gain = _smoother_gain(
            filtered_covariances[period],
            transition,
            predicted_covariance_next,
        )
        state_revision = smoothed_states[period + 1] - predicted_states[period + 1]
        covariance_revision = smoothed_covariances[period + 1] - predicted_covariance_next
        smoothed_states[period] = filtered_states[period] + smoother_gain @ state_revision
        smoothed_covariances[period] = filtered_covariances[period] + (
            smoother_gain @ covariance_revision @ smoother_gain.T
        )
        smoothed_covariances[period] = 0.5 * (
            smoothed_covariances[period] + smoothed_covariances[period].T
        )

    return KalmanResult(
        log_likelihood=result.log_likelihood,
        log_likelihood_by_period=result.log_likelihood_by_period,
        filtered_states=result.filtered_states,
        filtered_covariances=result.filtered_covariances,
        predicted_states=result.predicted_states,
        predicted_covariances=result.predicted_covariances,
        smoothed_states=smoothed_states,
        smoothed_covariances=smoothed_covariances,
    )


def observables_from_states(system: System, states: np.ndarray) -> np.ndarray:
    state_array = np.asarray(states, dtype=np.float64)
    if state_array.ndim != 2:
        msg = "States must have shape (periods, states)."
        raise ValueError(msg)
    measurement = system.measurement
    n_states = measurement.ZZ.shape[1]
    if state_array.shape[1] != n_states:
        msg = f"States must have {n_states} columns."
        raise ValueError(msg)
    return state_array @ measurement.ZZ.T + measurement.DD


def pseudo_observables_from_states(system: System, states: np.ndarray) -> np.ndarray:
    state_array = np.asarray(states, dtype=np.float64)
    if state_array.ndim != 2:
        msg = "States must have shape (periods, states)."
        raise ValueError(msg)
    if system.pseudo_measurement is None:
        msg = "System does not include pseudo-measurement matrices."
        raise ValueError(msg)
    pseudo_measurement = system.pseudo_measurement
    n_states = pseudo_measurement.ZZ_pseudo.shape[1]
    if state_array.shape[1] != n_states:
        msg = f"States must have {n_states} columns."
        raise ValueError(msg)
    return state_array @ pseudo_measurement.ZZ_pseudo.T + pseudo_measurement.DD_pseudo


def _normal_logpdf(
    innovation: np.ndarray,
    covariance: np.ndarray,
    backend: ArrayBackend,
) -> float:
    sign, logdet = backend.slogdet(covariance)
    if backend.scalar(sign) <= 0:
        msg = "Innovation covariance must be positive definite."
        raise np.linalg.LinAlgError(msg)
    solved = backend.solve(covariance, innovation)
    quadratic = backend.scalar(backend.matmul(innovation, solved))
    return -0.5 * (innovation.size * np.log(2.0 * np.pi) + backend.scalar(logdet) + quadratic)


def _symmetrize(matrix: np.ndarray, backend: ArrayBackend) -> np.ndarray:
    return 0.5 * (matrix + backend.transpose(matrix))


def _smoother_gain(
    filtered_covariance: np.ndarray,
    transition: np.ndarray,
    predicted_covariance_next: np.ndarray,
) -> np.ndarray:
    cross_covariance = filtered_covariance @ transition.T
    symmetric = 0.5 * (predicted_covariance_next + predicted_covariance_next.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    max_eigenvalue = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    cutoff = max(max_eigenvalue, 0.0) * 1.0e-10
    if max_eigenvalue > 0.0 and float(eigenvalues[0]) > cutoff:
        # Well-conditioned predicted covariance: the direct solve preserves the
        # existing smoother behavior exactly.
        try:
            gain = np.linalg.solve(symmetric.T, cross_covariance.T).T
            if np.all(np.isfinite(gain)):
                return gain
        except np.linalg.LinAlgError:
            pass
    # Deterministic states (e.g. lag/augmented identities) give zero-variance
    # directions that make the predicted covariance singular and the direct solve
    # numerically explosive. Use a regularized symmetric pseudo-inverse that zeros
    # the near-null directions (no smoothing information) for a stable gain.
    inverse_eigenvalues = np.zeros_like(eigenvalues)
    retained = eigenvalues > cutoff
    inverse_eigenvalues[retained] = 1.0 / eigenvalues[retained]
    pseudo_inverse = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
    return cross_covariance @ pseudo_inverse


def _validate_smoother_shapes(
    system: System,
    *,
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    predicted_states: np.ndarray,
    predicted_covariances: np.ndarray,
) -> None:
    n_periods = filtered_states.shape[0]
    n_states = system.transition.TTT.shape[0]
    if filtered_states.ndim != 2 or filtered_states.shape[1] != n_states:
        msg = f"Filtered states must have shape (periods, {n_states})."
        raise ValueError(msg)
    expected_covariance_shape = (n_periods, n_states, n_states)
    if filtered_covariances.shape != expected_covariance_shape:
        msg = f"Filtered covariances must have shape {expected_covariance_shape}."
        raise ValueError(msg)
    if predicted_states.shape != filtered_states.shape:
        msg = f"Predicted states must have shape {filtered_states.shape}."
        raise ValueError(msg)
    if predicted_covariances.shape != expected_covariance_shape:
        msg = f"Predicted covariances must have shape {expected_covariance_shape}."
        raise ValueError(msg)
