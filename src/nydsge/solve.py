from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from scipy import linalg

from nydsge.core import DSGEModel, NotPortedError

CanonicalSolveMethod = Literal["auto", "direct", "gensys"]


@dataclass(frozen=True)
class CanonicalSystem:
    Gamma0: np.ndarray
    Gamma1: np.ndarray
    C: np.ndarray
    Psi: np.ndarray
    Pi: np.ndarray

    def validate(self) -> None:
        n_equations, n_states = self.Gamma0.shape
        if self.Gamma1.shape != (n_equations, n_states):
            msg = "Gamma1 must have the same shape as Gamma0."
            raise ValueError(msg)
        if self.C.shape != (n_equations,):
            msg = f"C must have shape {(n_equations,)}."
            raise ValueError(msg)
        if self.Psi.shape[0] != n_equations:
            msg = "Psi must have one row per equilibrium condition."
            raise ValueError(msg)
        if self.Pi.shape[0] != n_equations:
            msg = "Pi must have one row per equilibrium condition."
            raise ValueError(msg)
        _validate_finite_array("Gamma0", self.Gamma0)
        _validate_finite_array("Gamma1", self.Gamma1)
        _validate_finite_array("C", self.C)
        _validate_finite_array("Psi", self.Psi)
        _validate_finite_array("Pi", self.Pi)


@dataclass(frozen=True)
class Transition:
    TTT: np.ndarray
    RRR: np.ndarray
    CCC: np.ndarray


@dataclass(frozen=True)
class CanonicalSolveResult:
    transition: Transition
    eu: tuple[int, int]
    method: str


@dataclass(frozen=True)
class Measurement:
    ZZ: np.ndarray
    DD: np.ndarray
    QQ: np.ndarray
    EE: np.ndarray


@dataclass(frozen=True)
class PseudoMeasurement:
    ZZ_pseudo: np.ndarray
    DD_pseudo: np.ndarray


@dataclass(frozen=True)
class System:
    transition: Transition
    measurement: Measurement
    pseudo_measurement: PseudoMeasurement | None = None

    def __getitem__(self, name: str) -> np.ndarray:
        if hasattr(self.transition, name):
            return getattr(self.transition, name)
        if hasattr(self.measurement, name):
            return getattr(self.measurement, name)
        if self.pseudo_measurement is not None and hasattr(self.pseudo_measurement, name):
            return getattr(self.pseudo_measurement, name)
        msg = f"Unknown system matrix: {name}"
        raise KeyError(msg)


def build_system(
    *,
    TTT: np.ndarray,
    RRR: np.ndarray,
    CCC: np.ndarray,
    ZZ: np.ndarray,
    DD: np.ndarray,
    QQ: np.ndarray,
    EE: np.ndarray,
    ZZ_pseudo: np.ndarray | None = None,
    DD_pseudo: np.ndarray | None = None,
) -> System:
    transition = Transition(
        TTT=np.asarray(TTT, dtype=np.float64),
        RRR=np.asarray(RRR, dtype=np.float64),
        CCC=np.asarray(CCC, dtype=np.float64),
    )
    measurement = Measurement(
        ZZ=np.asarray(ZZ, dtype=np.float64),
        DD=np.asarray(DD, dtype=np.float64),
        QQ=np.asarray(QQ, dtype=np.float64),
        EE=np.asarray(EE, dtype=np.float64),
    )
    pseudo_measurement = None
    if ZZ_pseudo is not None or DD_pseudo is not None:
        if ZZ_pseudo is None or DD_pseudo is None:
            msg = "ZZ_pseudo and DD_pseudo must be provided together."
            raise ValueError(msg)
        pseudo_measurement = PseudoMeasurement(
            ZZ_pseudo=np.asarray(ZZ_pseudo, dtype=np.float64),
            DD_pseudo=np.asarray(DD_pseudo, dtype=np.float64),
        )
    system = System(
        transition=transition,
        measurement=measurement,
        pseudo_measurement=pseudo_measurement,
    )
    validate_system(system)
    return system


def build_system_from_canonical(
    *,
    canonical: CanonicalSystem,
    ZZ: np.ndarray,
    DD: np.ndarray,
    QQ: np.ndarray,
    EE: np.ndarray,
    ZZ_pseudo: np.ndarray | None = None,
    DD_pseudo: np.ndarray | None = None,
    method: CanonicalSolveMethod = "auto",
) -> System:
    solved = solve_canonical(canonical, method=method)
    transition = solved.transition
    return build_system(
        TTT=transition.TTT,
        RRR=transition.RRR,
        CCC=transition.CCC,
        ZZ=ZZ,
        DD=DD,
        QQ=QQ,
        EE=EE,
        ZZ_pseudo=ZZ_pseudo,
        DD_pseudo=DD_pseudo,
    )


def solve_canonical(
    canonical: CanonicalSystem,
    *,
    method: CanonicalSolveMethod = "auto",
) -> CanonicalSolveResult:
    if method not in {"auto", "direct", "gensys"}:
        msg = f"Unsupported canonical solve method: {method}"
        raise ValueError(msg)
    if method == "direct":
        return solve_canonical_direct(canonical)
    if method == "gensys":
        return gensys(canonical)

    canonical = _as_float64_canonical(canonical)
    if canonical.Pi.size and not np.allclose(canonical.Pi, 0.0):
        return gensys(canonical)
    try:
        return solve_canonical_direct(canonical)
    except NotPortedError:
        return gensys(canonical)


def solve_canonical_direct(canonical: CanonicalSystem) -> CanonicalSolveResult:
    canonical = _as_float64_canonical(canonical)
    canonical.validate()
    n_equations, n_states = canonical.Gamma0.shape
    if n_equations != n_states:
        msg = "Direct canonical solve requires square Gamma0."
        raise ValueError(msg)
    if canonical.Pi.size and not np.allclose(canonical.Pi, 0.0):
        msg = (
            "Direct canonical solve only supports systems without effective "
            "expectational-error terms; full QZ/gensys support is still pending."
        )
        raise NotPortedError(msg)
    try:
        transition = Transition(
            TTT=np.linalg.solve(canonical.Gamma0, canonical.Gamma1),
            RRR=np.linalg.solve(canonical.Gamma0, canonical.Psi),
            CCC=np.linalg.solve(canonical.Gamma0, canonical.C),
        )
    except np.linalg.LinAlgError as err:
        msg = (
            "Direct canonical solve requires nonsingular Gamma0; full QZ/gensys "
            "support is still pending."
        )
        raise NotPortedError(msg) from err
    return CanonicalSolveResult(transition=transition, eu=(1, 1), method="direct")


def gensys(
    canonical: CanonicalSystem,
    *,
    div: float = 0.0,
    eps: float = 1.0e-6,
) -> CanonicalSolveResult:
    canonical = _as_float64_canonical(canonical)
    canonical.validate()
    n_equations, n_states = canonical.Gamma0.shape
    if n_equations != n_states:
        msg = "Gensys requires square Gamma0."
        raise ValueError(msg)

    gamma0 = canonical.Gamma0.astype(np.complex128)
    gamma1 = canonical.Gamma1.astype(np.complex128)
    try:
        a_probe, b_probe, _, _ = linalg.qz(gamma0, gamma1, output="complex")
        if _has_coincident_zeros(a_probe, b_probe, eps=eps):
            return CanonicalSolveResult(
                transition=_empty_transition(),
                eu=(-2, -2),
                method="gensys",
            )
        if div == 0.0:
            div = _gensys_div_from_qz(a_probe, b_probe, eps=eps)
        a, b, _, _, q, z = linalg.ordqz(
            gamma0,
            gamma1,
            sort=lambda alpha, beta: ~(np.abs(beta) > div * np.abs(alpha)),
            output="complex",
        )
    except linalg.LinAlgError:
        return CanonicalSolveResult(
            transition=_empty_transition(),
            eu=(-3, -3),
            method="gensys",
        )

    select = ~(np.abs(np.diag(b)) > div * np.abs(np.diag(a)))
    nunstab = n_states - int(np.count_nonzero(select))
    stable = n_states - nunstab

    qt1 = q[:, :stable]
    qt2 = q[:, stable:]
    pi = canonical.Pi.astype(np.complex128)
    psi = canonical.Psi.astype(np.complex128)
    c = canonical.C.astype(np.complex128)
    neta = pi.shape[1]

    if nunstab == 0:
        ueta = np.zeros((0, 0), dtype=np.complex128)
        deta = np.zeros((0, 0), dtype=np.complex128)
        veta = np.zeros((neta, 0), dtype=np.complex128)
        eta_rank = 0
    else:
        ueta, deta, veta, eta_rank = _selected_svd(qt2.conj().T @ pi, eps=eps)

    eu = [0, 0]
    if eta_rank >= nunstab:
        eu[0] = 1

    if nunstab == n_states:
        ueta1 = np.zeros((0, 0), dtype=np.complex128)
        deta1 = np.zeros((0, 0), dtype=np.complex128)
        veta1 = np.zeros((neta, 0), dtype=np.complex128)
    else:
        ueta1, deta1, veta1, _ = _selected_svd(qt1.conj().T @ pi, eps=eps)

    if veta1.shape[1] == 0:
        unique = True
    else:
        loose = veta1 - (veta @ veta.conj().T) @ veta1
        loose_singular_values = np.linalg.svd(loose, compute_uv=False)
        unique = bool(np.sum(np.abs(loose_singular_values) > eps * n_states) == 0)
    if unique:
        eu[1] = 1

    if nunstab == 0:
        tmat = np.eye(stable, dtype=np.complex128)
    else:
        eta_projection = (
            ueta @ _solve_or_empty(deta, veta.conj().T) @ veta1 @ (deta1 @ ueta1.conj().T)
        )
        tmat = np.hstack(
            [
                np.eye(stable, dtype=np.complex128),
                -eta_projection.conj().T,
            ]
        )

    g0 = np.vstack(
        [
            tmat @ a,
            np.hstack(
                [
                    np.zeros((nunstab, stable), dtype=np.complex128),
                    np.eye(nunstab, dtype=np.complex128),
                ]
            ),
        ]
    )
    g1 = np.vstack(
        [
            tmat @ b,
            np.zeros((nunstab, n_states), dtype=np.complex128),
        ]
    )

    g0i = np.linalg.inv(g0)
    g1 = g0i @ g1
    usix = slice(stable, n_states)
    bottom_c = _solve_or_empty(a[usix, usix] - b[usix, usix], qt2.conj().T @ c)
    c_solution = g0i @ np.concatenate([tmat @ (q.conj().T @ c), bottom_c])
    impact = g0i @ np.vstack(
        [
            tmat @ (q.conj().T @ psi),
            np.zeros((nunstab, psi.shape[1]), dtype=np.complex128),
        ]
    )

    transition = Transition(
        TTT=np.real(z @ (g1 @ z.conj().T)),
        RRR=np.real(z @ impact),
        CCC=np.real(z @ c_solution),
    )
    return CanonicalSolveResult(transition=transition, eu=(eu[0], eu[1]), method="gensys")


def validate_system(system: System) -> None:
    t = system.transition
    m = system.measurement
    n_states = t.TTT.shape[0]
    if t.TTT.shape != (n_states, n_states):
        msg = "TTT must be square."
        raise ValueError(msg)
    if t.RRR.shape[0] != n_states:
        msg = "RRR must have one row per state."
        raise ValueError(msg)
    if t.CCC.shape != (n_states,):
        msg = f"CCC must have shape {(n_states,)}."
        raise ValueError(msg)
    n_shocks = t.RRR.shape[1]
    n_observables = m.ZZ.shape[0]
    if m.ZZ.shape[1] != n_states:
        msg = "ZZ must have one column per state."
        raise ValueError(msg)
    if m.DD.shape != (n_observables,):
        msg = f"DD must have shape {(n_observables,)}."
        raise ValueError(msg)
    if m.QQ.shape != (n_shocks, n_shocks):
        msg = f"QQ must have shape {(n_shocks, n_shocks)}."
        raise ValueError(msg)
    if m.EE.shape != (n_observables, n_observables):
        msg = f"EE must have shape {(n_observables, n_observables)}."
        raise ValueError(msg)
    _validate_finite_array("TTT", t.TTT)
    _validate_finite_array("RRR", t.RRR)
    _validate_finite_array("CCC", t.CCC)
    _validate_finite_array("ZZ", m.ZZ)
    _validate_finite_array("DD", m.DD)
    _validate_finite_array("QQ", m.QQ)
    _validate_finite_array("EE", m.EE)
    if system.pseudo_measurement is not None:
        pseudo = system.pseudo_measurement
        n_pseudo = pseudo.ZZ_pseudo.shape[0]
        if pseudo.ZZ_pseudo.shape[1] != n_states:
            msg = "ZZ_pseudo must have one column per state."
            raise ValueError(msg)
        if pseudo.DD_pseudo.shape != (n_pseudo,):
            msg = f"DD_pseudo must have shape {(n_pseudo,)}."
            raise ValueError(msg)
        _validate_finite_array("ZZ_pseudo", pseudo.ZZ_pseudo)
        _validate_finite_array("DD_pseudo", pseudo.DD_pseudo)


def _validate_finite_array(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)


def _as_float64_canonical(canonical: CanonicalSystem) -> CanonicalSystem:
    return CanonicalSystem(
        Gamma0=np.asarray(canonical.Gamma0, dtype=np.float64),
        Gamma1=np.asarray(canonical.Gamma1, dtype=np.float64),
        C=np.asarray(canonical.C, dtype=np.float64),
        Psi=np.asarray(canonical.Psi, dtype=np.float64),
        Pi=np.asarray(canonical.Pi, dtype=np.float64),
    )


def _empty_transition() -> Transition:
    return Transition(
        TTT=np.empty((0, 0), dtype=np.float64),
        RRR=np.empty((0, 0), dtype=np.float64),
        CCC=np.empty(0, dtype=np.float64),
    )


def _gensys_div_from_qz(a: np.ndarray, b: np.ndarray, *, eps: float) -> float:
    div = 1.01
    for alpha, beta in zip(np.diag(a), np.diag(b), strict=True):
        if abs(alpha) > 0.0:
            divhat = abs(beta) / abs(alpha)
            if 1.0 + eps < divhat <= div:
                div = 0.5 * (1.0 + divhat)
    return float(div)


def _has_coincident_zeros(a: np.ndarray, b: np.ndarray, *, eps: float) -> bool:
    return bool(np.any((np.abs(np.diag(a)) < eps) & (np.abs(np.diag(b)) < eps)))


def _selected_svd(
    matrix: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    selected = np.flatnonzero(singular_values > eps)
    v = vh.conj().T
    return (
        u[:, selected],
        np.diag(singular_values[selected]).astype(np.complex128),
        v[:, selected],
        int(selected.size),
    )


def _solve_or_empty(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0:
        if b.ndim == 1:
            return np.zeros((a.shape[0],), dtype=np.complex128)
        return np.zeros((a.shape[0], b.shape[1]), dtype=np.complex128)
    return np.linalg.solve(a, b)


def compute_system(
    model: DSGEModel,
    *,
    tvis: bool = False,
    verbose: str = "high",
    method: CanonicalSolveMethod = "auto",
) -> System:
    del tvis, verbose
    if model.spec == "m1002" and model.subspec in {"ss10", "ss104"}:
        model1002 = cast(Any, model)
        canonical = model1002.equilibrium_matrices()
        solved = solve_canonical(canonical, method=method)
        if solved.eu != (1, 1):
            msg = f"Model1002 ss10 canonical solve failed existence/uniqueness check: {solved.eu}."
            raise NotPortedError(msg)
        transition = model1002.augment_transition(solved.transition)
        measurement = model1002.measurement_matrices(transition)
        pseudo_measurement = model1002.pseudo_measurement_matrices(transition)
        system = System(
            transition=transition,
            measurement=measurement,
            pseudo_measurement=pseudo_measurement,
        )
        validate_system(system)
        return system
    msg = (
        f"State-space solve for {model.spec} {model.subspec} is not ported yet. "
        "Next required translations: model-specific eqcond, augmentation, measurement, "
        "and pseudo-measurement."
    )
    raise NotPortedError(msg)
