from __future__ import annotations

import numpy as np
import pytest

from nydsge.core import NotPortedError
from nydsge.solve import (
    CanonicalSystem,
    build_system,
    build_system_from_canonical,
    gensys,
    solve_canonical,
)


def test_canonical_system_validation() -> None:
    canonical = CanonicalSystem(
        Gamma0=np.eye(2),
        Gamma1=np.eye(2),
        C=np.zeros(2),
        Psi=np.ones((2, 1)),
        Pi=np.ones((2, 1)),
    )
    canonical.validate()


def test_canonical_system_validation_rejects_nonfinite_values() -> None:
    canonical = CanonicalSystem(
        Gamma0=np.array([[np.nan]]),
        Gamma1=np.eye(1),
        C=np.zeros(1),
        Psi=np.ones((1, 1)),
        Pi=np.zeros((1, 0)),
    )

    with pytest.raises(ValueError, match="Gamma0.*finite"):
        canonical.validate()


def test_solve_canonical_direct_inverts_gamma0() -> None:
    canonical = CanonicalSystem(
        Gamma0=np.array([[2.0, 0.0], [0.0, 4.0]]),
        Gamma1=np.array([[1.0, 2.0], [3.0, 4.0]]),
        C=np.array([2.0, 8.0]),
        Psi=np.array([[2.0], [8.0]]),
        Pi=np.zeros((2, 0)),
    )

    result = solve_canonical(canonical)

    assert result.eu == (1, 1)
    assert result.method == "direct"
    np.testing.assert_allclose(
        result.transition.TTT, np.linalg.solve(canonical.Gamma0, canonical.Gamma1)
    )
    np.testing.assert_allclose(
        result.transition.RRR, np.linalg.solve(canonical.Gamma0, canonical.Psi)
    )
    np.testing.assert_allclose(
        result.transition.CCC, np.linalg.solve(canonical.Gamma0, canonical.C)
    )


def test_gensys_matches_direct_for_stable_system_without_jumps() -> None:
    canonical = CanonicalSystem(
        Gamma0=np.eye(2),
        Gamma1=np.diag([0.5, 0.2]),
        C=np.array([1.0, 2.0]),
        Psi=np.ones((2, 1)),
        Pi=np.zeros((2, 0)),
    )

    result = gensys(canonical)

    assert result.eu == (1, 1)
    assert result.method == "gensys"
    np.testing.assert_allclose(result.transition.TTT, canonical.Gamma1)
    np.testing.assert_allclose(result.transition.RRR, canonical.Psi)
    np.testing.assert_allclose(result.transition.CCC, canonical.C)


def test_solve_canonical_auto_uses_gensys_for_expectational_terms() -> None:
    canonical = CanonicalSystem(
        Gamma0=np.eye(1),
        Gamma1=np.array([[1.2]]),
        C=np.zeros(1),
        Psi=np.ones((1, 1)),
        Pi=np.ones((1, 1)),
    )

    result = solve_canonical(canonical)

    assert result.eu == (1, 1)
    assert result.method == "gensys"
    np.testing.assert_allclose(result.transition.TTT, np.zeros((1, 1)))


def test_gensys_reports_coincident_zeros() -> None:
    canonical = CanonicalSystem(
        Gamma0=np.zeros((1, 1)),
        Gamma1=np.zeros((1, 1)),
        C=np.zeros(1),
        Psi=np.zeros((1, 1)),
        Pi=np.zeros((1, 0)),
    )

    result = gensys(canonical)

    assert result.eu == (-2, -2)
    assert result.transition.TTT.shape == (0, 0)


def test_solve_canonical_direct_allows_zero_pi_columns() -> None:
    canonical = CanonicalSystem(
        Gamma0=np.eye(1),
        Gamma1=np.ones((1, 1)),
        C=np.zeros(1),
        Psi=np.ones((1, 1)),
        Pi=np.zeros((1, 1)),
    )

    result = solve_canonical(canonical, method="direct")

    np.testing.assert_allclose(result.transition.TTT, np.ones((1, 1)))


def test_solve_canonical_direct_rejects_expectational_terms() -> None:
    canonical = CanonicalSystem(
        Gamma0=np.eye(1),
        Gamma1=np.ones((1, 1)),
        C=np.zeros(1),
        Psi=np.ones((1, 1)),
        Pi=np.ones((1, 1)),
    )

    with pytest.raises(NotPortedError, match="QZ/gensys"):
        solve_canonical(canonical, method="direct")


def test_solve_canonical_direct_rejects_nonsquare_gamma0() -> None:
    canonical = CanonicalSystem(
        Gamma0=np.ones((1, 2)),
        Gamma1=np.ones((1, 2)),
        C=np.zeros(1),
        Psi=np.ones((1, 1)),
        Pi=np.zeros((1, 0)),
    )

    with pytest.raises(ValueError, match="square Gamma0"):
        solve_canonical(canonical)


def test_build_system_from_canonical_solves_transition() -> None:
    canonical = CanonicalSystem(
        Gamma0=2.0 * np.eye(2),
        Gamma1=np.eye(2),
        C=np.array([2.0, 4.0]),
        Psi=np.ones((2, 1)),
        Pi=np.zeros((2, 0)),
    )

    system = build_system_from_canonical(
        canonical=canonical,
        ZZ=np.eye(2),
        DD=np.zeros(2),
        QQ=np.eye(1),
        EE=np.eye(2),
    )

    np.testing.assert_allclose(system["TTT"], 0.5 * np.eye(2))
    np.testing.assert_allclose(system["CCC"], np.array([1.0, 2.0]))


def test_build_system_validates_shapes() -> None:
    system = build_system(
        TTT=np.eye(2),
        RRR=np.ones((2, 1)),
        CCC=np.zeros(2),
        ZZ=np.ones((1, 2)),
        DD=np.zeros(1),
        QQ=np.eye(1),
        EE=np.eye(1),
    )
    assert system["TTT"].shape == (2, 2)
    assert system["ZZ"].shape == (1, 2)


def test_build_system_rejects_bad_measurement_shape() -> None:
    with pytest.raises(ValueError, match="ZZ"):
        build_system(
            TTT=np.eye(2),
            RRR=np.ones((2, 1)),
            CCC=np.zeros(2),
            ZZ=np.ones((1, 3)),
            DD=np.zeros(1),
            QQ=np.eye(1),
            EE=np.eye(1),
        )


def test_build_system_rejects_nonfinite_matrices() -> None:
    with pytest.raises(ValueError, match="TTT.*finite"):
        build_system(
            TTT=np.array([[np.inf]]),
            RRR=np.ones((1, 1)),
            CCC=np.zeros(1),
            ZZ=np.ones((1, 1)),
            DD=np.zeros(1),
            QQ=np.eye(1),
            EE=np.eye(1),
        )
