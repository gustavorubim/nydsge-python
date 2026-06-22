from __future__ import annotations

import numpy as np
from scipy.stats import norm

from nydsge.financial_frictions import (
    g_fn,
    gamma_fn,
    mu_fn,
    nk_fn,
    omega_fn,
    zeta_bomega_fn,
    zeta_spb_fn,
    zeta_zomega_fn,
)


def test_financial_frictions_default_point_values() -> None:
    z = float(norm.ppf(1.0 - (1.0 - 0.03) ** 0.25))
    sigma = 0.5
    spr = (1.0 + 1.7444 / 100.0) ** 0.25

    assert np.isclose(omega_fn(z, sigma), 0.2620745217846454)
    assert np.isclose(g_fn(z, sigma), 0.0017043640474161438)
    assert np.isclose(gamma_fn(z, sigma), 0.26179081924629516)
    assert np.isclose(mu_fn(z, sigma, spr), 0.13142954165405685)
    assert np.isclose(nk_fn(z, sigma, spr), 0.7372998784691602)


def test_zeta_spb_identity() -> None:
    z = float(norm.ppf(1.0 - (1.0 - 0.03) ** 0.25))
    sigma = 0.5
    spr = (1.0 + 1.7444 / 100.0) ** 0.25

    zeta_ratio = zeta_bomega_fn(z, sigma, spr) / zeta_zomega_fn(z, sigma, spr)
    expected = (
        -zeta_ratio / (1.0 - zeta_ratio) * nk_fn(z, sigma, spr) / (1.0 - nk_fn(z, sigma, spr))
    )

    assert np.isclose(zeta_spb_fn(z, sigma, spr), expected)
    assert np.isclose(zeta_spb_fn(z, sigma, spr), 0.055975462717559314)
