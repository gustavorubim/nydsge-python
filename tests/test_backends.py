from __future__ import annotations

import numpy as np
import pytest

from nydsge.backends import ArrayBackend, JaxBackend, TorchBackend, get_backend
from nydsge.runtime import RuntimeConfig, UnsupportedRuntimeError


def test_get_backend_returns_numpy_reference() -> None:
    backend = get_backend(RuntimeConfig(backend="numpy", device="cpu"))

    assert backend.name == "numpy"
    assert backend.device == "cpu"
    _assert_backend_contract(backend)


def test_torch_cpu_backend_contract_when_available() -> None:
    pytest.importorskip("torch")
    try:
        RuntimeConfig(backend="torch", device="cpu").resolve()
    except UnsupportedRuntimeError as err:
        pytest.skip(str(err))

    _assert_backend_contract(TorchBackend(device="cpu"))


def test_jax_cpu_backend_contract_when_available() -> None:
    pytest.importorskip("jax")
    try:
        RuntimeConfig(backend="jax", device="cpu").resolve()
    except UnsupportedRuntimeError as err:
        pytest.skip(str(err))

    _assert_backend_contract(JaxBackend(device="cpu"))


def _assert_backend_contract(backend: ArrayBackend) -> None:
    matrix = np.array([[2.0, 0.5], [0.5, 1.0]])
    vector = np.array([1.0, 2.0])

    np.testing.assert_allclose(backend.as_numpy(backend.array(vector)), vector)
    np.testing.assert_allclose(backend.as_numpy(backend.zeros((2, 2))), np.zeros((2, 2)))
    np.testing.assert_allclose(backend.as_numpy(backend.eye(2)), np.eye(2))
    np.testing.assert_allclose(backend.as_numpy(backend.matmul(matrix, vector)), matrix @ vector)
    np.testing.assert_allclose(backend.as_numpy(backend.transpose(matrix)), matrix.T)
    np.testing.assert_allclose(
        backend.as_numpy(backend.solve(matrix, vector)), np.linalg.solve(matrix, vector)
    )

    sign, logdet = backend.slogdet(matrix)
    expected_sign, expected_logdet = np.linalg.slogdet(matrix)
    assert backend.scalar(sign) == expected_sign
    assert np.isclose(backend.scalar(logdet), expected_logdet)
