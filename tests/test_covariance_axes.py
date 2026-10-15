"""Asymmetric markers expose covariance-axis errors hidden by exact symmetry."""

import numpy as np
import pytest

from tools.oracle_julia.compare_ss104 import julia_covariances


def test_julia_covariance_cube_restores_both_matrix_axes():
    julia = np.arange(12).reshape((2, 2, 3), order="F")
    raw_h5py = julia.transpose(2, 1, 0)
    expected = julia.transpose(2, 0, 1)
    assert not np.array_equal(raw_h5py, expected)
    np.testing.assert_array_equal(julia_covariances(raw_h5py), expected)
    # The oracle's asymmetry is preserved, never averaged away.
    assert julia_covariances(raw_h5py)[0, 0, 1] != julia_covariances(raw_h5py)[0, 1, 0]
    with pytest.raises(ValueError, match="covariance cube"):
        julia_covariances(np.zeros((2, 3)))
