"""Deterministic regime paths and finite-horizon policy solutions.

Regime and scenario numbers are one-based, matching DSGE.jl. Arrays of data
remain periods x observables, as in the existing Python filter. A regime in a
forward information set represents one quarter; its final regime persists.

Source: DSGE.jl c09dfe8, src/solve/gensys2.jl and src/statespace/{util,
statespace_functions}.jl. See docs/ss104_m1.md for contracts and source mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, overload

import numpy as np
import pandas as pd

from nydsge.core import NotPortedError
from nydsge.solve import (
    CanonicalSolveMethod,
    CanonicalSystem,
    Measurement,
    PseudoMeasurement,
    System,
    Transition,
    solve_canonical,
    validate_system,
)

InformationSets = Sequence[Sequence[int]]
MeasurementBuilder = Callable[
    [int, Transition, tuple[Transition, ...], tuple[int, ...]], Measurement
]
PseudoMeasurementBuilder = Callable[
    [int, Transition, tuple[Transition, ...], tuple[int, ...]], PseudoMeasurement
]
TransitionAugmenter = Callable[[int, Transition], Transition]


def _regime_number(value: int, count: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= count:
        raise ValueError(f"Regime number must be an integer between 1 and {count}: {value!r}.")
    return int(value)


def _information_sets(count: int, values: InformationSets | None) -> tuple[tuple[int, ...], ...]:
    if values is None:
        return tuple((reg,) for reg in range(1, count + 1))
    if len(values) != count:
        raise ValueError("Provide one information set per regime.")
    normalized = []
    for reg, info in enumerate(values, start=1):
        ids = tuple(_regime_number(value, count) for value in info)
        if not ids or reg not in ids or ids != tuple(range(ids[0], ids[-1] + 1)):
            raise ValueError(
                "Information sets must be contiguous, ordered, and contain their regime."
            )
        normalized.append(ids)
    return tuple(normalized)


@dataclass(frozen=True)
class RegimeSwitchingSystem:
    """Known regimes, with common state, shock, and observable dimensions."""

    systems: tuple[System, ...]
    information_sets: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        systems = tuple(self.systems)
        if not systems:
            raise ValueError("A regime system must contain at least one regime.")
        object.__setattr__(self, "systems", systems)
        object.__setattr__(
            self, "information_sets", _information_sets(len(systems), self.information_sets)
        )
        shapes = {
            name: systems[0][name].shape for name in ("TTT", "RRR", "CCC", "ZZ", "DD", "QQ", "EE")
        }
        for system in systems:
            validate_system(system)
            if any(system[name].shape != shape for name, shape in shapes.items()):
                raise ValueError(
                    "All regimes must have common state, shock, and observable dimensions."
                )
            first_pseudo = systems[0].pseudo_measurement
            pseudo = system.pseudo_measurement
            if (pseudo is None) != (first_pseudo is None) or (
                pseudo is not None
                and first_pseudo is not None
                and pseudo.ZZ_pseudo.shape != first_pseudo.ZZ_pseudo.shape
            ):
                raise ValueError("All regimes must have common pseudo-observable dimensions.")

    def __len__(self) -> int:
        return len(self.systems)

    @overload
    def __getitem__(self, key: int) -> System: ...

    @overload
    def __getitem__(self, key: tuple[int, str]) -> np.ndarray: ...

    def __getitem__(self, key: int | tuple[int, str]) -> System | np.ndarray:
        if isinstance(key, tuple):
            reg, name = key
            return self.systems[_regime_number(reg, len(self)) - 1][name]
        return self.systems[_regime_number(key, len(self)) - 1]

    def period_systems(
        self, regime_indices: Sequence[int] | np.ndarray | None, periods: int
    ) -> tuple[System, ...]:
        if regime_indices is None:
            if len(self) != 1:
                raise ValueError("regime_indices are required for a multi-regime system.")
            return (self.systems[0],) * periods
        if len(regime_indices) != periods:
            raise ValueError("regime_indices must contain one regime number per period.")
        return tuple(self.systems[_regime_number(reg, len(self)) - 1] for reg in regime_indices)


@dataclass(frozen=True)
class PredictableSystem:
    Gamma0: np.ndarray
    Gamma1: np.ndarray
    Gamma2: np.ndarray
    C: np.ndarray
    Psi: np.ndarray


def gensys_to_predictable_form(canonical: CanonicalSystem) -> PredictableSystem:
    """Convert z_t = E_{t-1}[z_t] + eta_t identities to forward equations.

    As upstream, Pi identifies the expectation rows. Fail explicitly for other
    expectational-error layouts rather than guessing a forward equation.
    """
    canonical.validate()
    if canonical.Gamma0.shape[0] != canonical.Gamma0.shape[1]:
        raise ValueError("gensys2 requires square Gamma0.")
    g0 = np.array(canonical.Gamma0, dtype=np.float64, copy=True)
    g1 = np.array(canonical.Gamma1, dtype=np.float64, copy=True)
    g2 = np.zeros_like(g0)
    for row in np.flatnonzero(np.any(canonical.Pi != 0.0, axis=1)):
        current = np.flatnonzero(g0[row] != 0.0)
        lagged = np.flatnonzero(g1[row] != 0.0)
        if (
            current.size != 1
            or lagged.size != 1
            or g0[row, current[0]] != 1.0
            or g1[row, lagged[0]] != 1.0
            or canonical.C[row] != 0.0
            or np.any(canonical.Psi[row] != 0.0)
        ):
            raise ValueError("gensys2 expectation rows must be unit forecast-error identities.")
        g0[row] = 0.0
        g1[row] = 0.0
        g0[row, lagged[0]] = -1.0
        g2[row, current[0]] = 1.0
    return PredictableSystem(g0, g1, g2, np.asarray(canonical.C), np.asarray(canonical.Psi))


def gensys2(
    canonical: CanonicalSystem | Sequence[CanonicalSystem],
    terminal: Transition,
    *,
    periods: int | None = None,
) -> tuple[Transition, ...]:
    """Solve temporary policies backward, returning temporary periods + terminal.

    For a single canonical, ``periods`` is the number of temporary quarters. For a
    sequence, each canonical is one temporary quarter (the terminal is separate).
    The terminal must be a solved permanent-policy boundary, not a ZLB steady state.
    """
    if isinstance(canonical, CanonicalSystem):
        if isinstance(periods, bool) or not isinstance(periods, Integral) or periods < 0:
            raise ValueError("periods must be a nonnegative integer for a single canonical.")
        canonicals = (canonical,) * periods
    else:
        if periods is not None:
            raise ValueError("periods is only used with a single canonical.")
        canonicals = tuple(canonical)
    n = terminal.TTT.shape[0]
    if (
        terminal.TTT.shape != (n, n)
        or terminal.RRR.ndim != 2
        or terminal.RRR.shape[0] != n
        or terminal.CCC.shape != (n,)
        or not all(np.all(np.isfinite(x)) for x in (terminal.TTT, terminal.RRR, terminal.CCC))
    ):
        raise ValueError("Invalid gensys2 terminal transition.")
    result = [terminal]
    next_transition = terminal
    for system in reversed(canonicals):
        predictable = gensys_to_predictable_form(system)
        if system.Gamma0.shape != (n, n) or system.Psi.shape != terminal.RRR.shape:
            raise ValueError(
                "Temporary and terminal systems must have common state/shock dimensions."
            )
        lhs = predictable.Gamma0 + predictable.Gamma2 @ next_transition.TTT
        next_transition = Transition(
            TTT=np.linalg.solve(lhs, predictable.Gamma1),
            RRR=np.linalg.solve(lhs, predictable.Psi),
            CCC=np.linalg.solve(lhs, predictable.C - predictable.Gamma2 @ next_transition.CCC),
        )
        if not all(
            np.all(np.isfinite(x))
            for x in (next_transition.TTT, next_transition.RRR, next_transition.CCC)
        ):
            raise ValueError("Nonfinite gensys2 solution.")
        result.append(next_transition)
    return tuple(reversed(result))


def solve_regime_switching(
    canonicals: Sequence[CanonicalSystem],
    *,
    temporary_regimes: Sequence[Sequence[int]] = (),
    method: CanonicalSolveMethod = "auto",
) -> tuple[Transition, ...]:
    """gensys per permanent regime, gensys2 per contiguous temporary block.

    Blocks contain only temporary regime numbers. The immediately following regime
    is their permanent terminal boundary. Multiple nonoverlapping blocks are allowed.
    """
    canonicals = tuple(canonicals)
    if not canonicals:
        raise ValueError("Provide at least one canonical regime.")
    blocks = []
    occupied: set[int] = set()
    for block in temporary_regimes:
        ids = tuple(_regime_number(reg, len(canonicals)) for reg in block)
        if not ids or ids != tuple(range(ids[0], ids[-1] + 1)):
            raise ValueError("Temporary blocks must contain contiguous ordered regime numbers.")
        if ids[-1] == len(canonicals):
            raise ValueError("A temporary block requires a following permanent terminal regime.")
        if occupied.intersection(ids):
            raise ValueError("Temporary blocks must not overlap.")
        occupied.update(ids)
        blocks.append(ids)
    if any(ids[-1] + 1 in occupied for ids in blocks):
        raise ValueError("Each temporary block must end before a permanent regime.")
    transitions: dict[int, Transition] = {}
    dimensions = (canonicals[0].Gamma0.shape, canonicals[0].Psi.shape, canonicals[0].Pi.shape)
    for reg, canonical in enumerate(canonicals, start=1):
        canonical.validate()
        if (canonical.Gamma0.shape, canonical.Psi.shape, canonical.Pi.shape) != dimensions:
            raise ValueError(
                "Canonical regimes must have common state, shock, and expectation dimensions."
            )
        if reg not in occupied:
            solved = solve_canonical(canonical, method=method)
            if solved.eu != (1, 1):
                raise NotPortedError(f"Regime {reg} failed existence/uniqueness: {solved.eu}.")
            transitions[reg] = solved.transition
    for ids in blocks:
        path = gensys2([canonicals[reg - 1] for reg in ids], transitions[ids[-1] + 1])
        transitions.update(zip(ids, path[:-1], strict=True))
    return tuple(transitions[reg] for reg in range(1, len(canonicals) + 1))


def _forward_expectations(
    transitions: Sequence[Transition], reg: int, horizon: int, information_set: Sequence[int] | None
) -> list[tuple[np.ndarray, np.ndarray]]:
    reg = _regime_number(reg, len(transitions))
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 0:
        raise ValueError("horizon must be a nonnegative integer.")
    info = (reg,) if information_set is None else tuple(information_set)
    # Use the same validation as the regime container, allowing past regimes in an info set.
    infos: list[tuple[int, ...]] = [(i,) for i in range(1, len(transitions) + 1)]
    infos[reg - 1] = info
    info = _information_sets(len(transitions), infos)[reg - 1]
    n = transitions[reg - 1].TTT.shape[0]
    power = np.eye(n, dtype=np.float64)
    constant = np.zeros(n, dtype=np.float64)
    result = []
    for step in range(1, horizon + 1):
        transition = transitions[min(reg + step, info[-1]) - 1]
        constant = transition.TTT @ constant + transition.CCC
        power = transition.TTT @ power
        result.append((power, constant))
    return result


def k_periods_ahead_expectations(
    transitions: Sequence[Transition],
    horizon: int,
    *,
    reg: int = 1,
    information_set: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine E_t[s_{t+k}], using future regimes through the information horizon."""
    result = _forward_expectations(transitions, reg, horizon, information_set)
    if result:
        return result[-1]
    n = transitions[reg - 1].TTT.shape[0]
    return np.eye(n, dtype=np.float64), np.zeros(n, dtype=np.float64)


def k_periods_ahead_expected_sums(
    transitions: Sequence[Transition],
    horizon: int,
    *,
    reg: int = 1,
    information_set: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine sum of E_t[s_{t+j}] for j = 1, ..., horizon."""
    result = _forward_expectations(transitions, reg, horizon, information_set)
    n = transitions[reg - 1].TTT.shape[0]
    power_sum = np.zeros((n, n), dtype=np.float64)
    constant_sum = np.zeros(n, dtype=np.float64)
    for power, constant in result:
        power_sum += power
        constant_sum += constant
    return power_sum, constant_sum


def compute_regime_system(
    canonicals: Sequence[CanonicalSystem],
    measurement_builder: MeasurementBuilder,
    *,
    temporary_regimes: Sequence[Sequence[int]] = (),
    information_sets: InformationSets | None = None,
    transition_augmenter: TransitionAugmenter | None = None,
    pseudo_measurement_builder: PseudoMeasurementBuilder | None = None,
    method: CanonicalSolveMethod = "auto",
) -> RegimeSwitchingSystem:
    """Build all seven matrices per regime after solving and augmenting the path."""
    transitions = solve_regime_switching(
        canonicals, temporary_regimes=temporary_regimes, method=method
    )
    if transition_augmenter is not None:
        transitions = tuple(
            transition_augmenter(reg, t) for reg, t in enumerate(transitions, start=1)
        )
    infos = _information_sets(len(transitions), information_sets)
    systems = tuple(
        System(
            transition,
            measurement_builder(reg, transition, transitions, infos[reg - 1]),
            None
            if pseudo_measurement_builder is None
            else pseudo_measurement_builder(reg, transition, transitions, infos[reg - 1]),
        )
        for reg, transition in enumerate(transitions, start=1)
    )
    return RegimeSwitchingSystem(systems, infos)


def compute_tvis_system(
    canonical_scenarios: Sequence[Sequence[CanonicalSystem]],
    measurement_builder: MeasurementBuilder,
    *,
    select_system: Sequence[int],
    information_sets: InformationSets,
    temporary_regimes: Sequence[Sequence[int]] = (),
    scenario_temporary_regimes: Sequence[Sequence[Sequence[int]]] | None = None,
    transition_augmenter: TransitionAugmenter | None = None,
    pseudo_measurement_builder: PseudoMeasurementBuilder | None = None,
    method: CanonicalSolveMethod = "auto",
) -> RegimeSwitchingSystem:
    """Select a belief scenario per regime, including its forward measurement path.

    Mirrors compute_tvis_system / compute_system_helper in DSGE.jl: measurements
    use the entire selected scenario, rather than the subsequently spliced path.
    """
    scenarios = tuple(canonical_scenarios)
    if not scenarios:
        raise ValueError("Provide at least one TVIS scenario.")
    count = len(scenarios[0])
    if len(select_system) != count or any(len(scenario) != count for scenario in scenarios):
        raise ValueError("TVIS scenarios and select_system must have one entry per regime.")
    selected = tuple(_regime_number(i, len(scenarios)) for i in select_system)
    if scenario_temporary_regimes is not None and len(scenario_temporary_regimes) != len(scenarios):
        raise ValueError("Provide one set of temporary blocks per TVIS scenario.")
    systems = tuple(
        compute_regime_system(
            scenario,
            measurement_builder,
            temporary_regimes=(
                temporary_regimes
                if scenario_temporary_regimes is None
                else scenario_temporary_regimes[i]
            ),
            information_sets=information_sets,
            transition_augmenter=transition_augmenter,
            pseudo_measurement_builder=pseudo_measurement_builder,
            method=method,
        )
        for i, scenario in enumerate(scenarios)
    )
    return RegimeSwitchingSystem(
        tuple(systems[select - 1].systems[reg] for reg, select in enumerate(selected)),
        _information_sets(count, information_sets),
    )


def compute_model_regime_system(
    models: Sequence[Any],
    *,
    temporary_regimes: Sequence[Sequence[int]] = (),
    information_sets: InformationSets | None = None,
    method: CanonicalSolveMethod = "auto",
) -> RegimeSwitchingSystem:
    """Solve explicit model snapshots per regime without toggling shared parameters."""
    models = tuple(models)
    if not models:
        raise ValueError("Provide at least one model per regime.")
    first_indexes = models[0].indexes
    if any(model.indexes != first_indexes for model in models):
        raise ValueError("Regime models must have identical ordered index maps.")

    def measurement(
        reg: int, t: Transition, ts: tuple[Transition, ...], info: tuple[int, ...]
    ) -> Measurement:
        return models[reg - 1].measurement_matrices(
            t, transitions=ts, reg=reg, information_set=info
        )

    def pseudo(
        reg: int, t: Transition, ts: tuple[Transition, ...], info: tuple[int, ...]
    ) -> PseudoMeasurement:
        return models[reg - 1].pseudo_measurement_matrices(
            t, transitions=ts, reg=reg, information_set=info
        )

    return compute_regime_system(
        [model.equilibrium_matrices() for model in models],
        measurement,
        temporary_regimes=temporary_regimes,
        information_sets=information_sets,
        transition_augmenter=lambda reg, t: models[reg - 1].augment_transition(t),
        pseudo_measurement_builder=pseudo,
        method=method,
    )


def regime_indices_from_dates(regime_dates: Mapping[int, Any], dates: Sequence[Any]) -> np.ndarray:
    """Map observation quarters to one-based regimes; no prehistory extrapolation."""
    count = len(regime_dates)
    if not count or set(regime_dates) != set(range(1, count + 1)):
        raise ValueError("regime_dates must have consecutive one-based regime keys.")

    def quarter(value: Any) -> int:
        parsed = pd.Period(str(value).replace("-Q", "Q"), freq="Q")
        if not isinstance(parsed, pd.Period):
            raise ValueError("Regime and observation dates must be valid quarters.")
        return parsed.ordinal

    starts = np.array([quarter(regime_dates[reg]) for reg in range(1, count + 1)])
    if np.any(np.diff(starts) <= 0):
        raise ValueError("Regime start quarters must be strictly increasing.")
    indices = np.searchsorted(starts, [quarter(date) for date in dates], side="right")
    if np.any(indices == 0):
        raise ValueError("Observation dates precede the first regime.")
    return indices.astype(np.int64)
