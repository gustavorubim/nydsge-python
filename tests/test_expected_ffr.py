from __future__ import annotations

import pytest

from nydsge.models.expected_ffr import (
    parse_expected_ffr_horizon_values,
    parse_expected_ffr_horizons,
    parse_expected_ffr_regime_horizons,
)


def test_parse_expected_ffr_horizons_flattens_and_sorts_horizons() -> None:
    assert parse_expected_ffr_horizons((4, 1, 4), ()) == (1, 4)


def test_parse_expected_ffr_horizons_uses_all_ffr_qs_when_expected_is_empty() -> None:
    assert parse_expected_ffr_horizons((), (4, 2, 1)) == (1, 2, 4)


def test_parse_expected_ffr_horizons_supports_regime_keyed_collections() -> None:
    assert parse_expected_ffr_horizons({1: (4, 1), 2: [2], 3: {3}}, ()) == (1, 2, 3, 4)


def test_parse_expected_ffr_regime_horizons_preserves_regimes() -> None:
    assert parse_expected_ffr_regime_horizons({"baseline": (4, 1), "shock": {3, 2}}, ()) == (
        ("baseline", (1, 4)),
        ("shock", (2, 3)),
    )


def test_parse_expected_ffr_regime_horizons_preserves_overlaps() -> None:
    assert parse_expected_ffr_regime_horizons({"baseline": (4, 1), "shock": [1, 3]}, ()) == (
        ("baseline", (1, 4)),
        ("shock", (1, 3)),
    )


def test_parse_expected_ffr_horizon_values_requires_positive_integers() -> None:
    with pytest.raises(ValueError, match="expected_ffr horizons must be positive integers."):
        parse_expected_ffr_horizon_values((-1, 2))

    with pytest.raises(TypeError, match="expected_ffr setting must be a sequence"):
        parse_expected_ffr_horizon_values("bad")

    with pytest.raises(TypeError, match="expected_ffr setting must be a sequence"):
        parse_expected_ffr_horizon_values(2)
