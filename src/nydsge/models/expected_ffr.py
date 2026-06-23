from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any


def parse_expected_ffr_horizon_values(raw_values: Any) -> tuple[int, ...]:
    if isinstance(raw_values, Mapping):
        msg = "expected_ffr setting must be a sequence of positive integer horizons."
        raise TypeError(msg)

    if isinstance(raw_values, str):
        msg = "expected_ffr setting must be a sequence of positive integer horizons."
        raise TypeError(msg)

    if not isinstance(raw_values, (list, tuple, set)):
        msg = "expected_ffr setting must be a sequence of positive integer horizons."
        raise TypeError(msg)

    values = tuple(int(horizon) for horizon in raw_values)
    if any(horizon <= 0 for horizon in values):
        msg = "expected_ffr horizons must be positive integers."
        raise ValueError(msg)
    if not isinstance(values, tuple):
        values = tuple(values)
    return values


def parse_expected_ffr_horizons(
    expected_ffr: Any,
    all_ffr_qs: Any,
) -> tuple[int, ...]:
    raw_horizons = expected_ffr
    if not raw_horizons:
        raw_horizons = all_ffr_qs
    if raw_horizons is None:
        return ()

    if isinstance(raw_horizons, Mapping):
        flattened: list[int] = []
        for values in raw_horizons.values():
            flattened.extend(parse_expected_ffr_horizon_values(values))
        horizons = tuple(sorted(set(flattened)))
    else:
        horizons = parse_expected_ffr_horizon_values(raw_horizons)
        horizons = tuple(sorted(set(horizons)))

    if not isinstance(horizons, tuple):
        horizons = tuple(horizons)
    return tuple(horizons)


def parse_expected_ffr_regime_horizons(
    expected_ffr: Any,
    all_ffr_qs: Any,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    raw_horizons = expected_ffr
    if not raw_horizons:
        raw_horizons = all_ffr_qs
    if raw_horizons is None:
        return ()

    if not isinstance(raw_horizons, Mapping):
        horizons = parse_expected_ffr_horizon_values(raw_horizons)
        return (("default", horizons),)

    regime_horizons = OrderedDict[str, tuple[int, ...]]()
    for regime, values in raw_horizons.items():
        horizons = tuple(sorted(set(parse_expected_ffr_horizon_values(values))))
        regime_horizons[str(regime)] = horizons
    return tuple(regime_horizons.items())
