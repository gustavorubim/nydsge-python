from __future__ import annotations

from math import exp

from scipy.stats import norm


def zeta_spb_fn(z: float, sigma: float, spr: float) -> float:
    zeta_ratio = zeta_bomega_fn(z, sigma, spr) / zeta_zomega_fn(z, sigma, spr)
    nk = nk_fn(z, sigma, spr)
    return -zeta_ratio / (1.0 - zeta_ratio) * nk / (1.0 - nk)


def zeta_bomega_fn(z: float, sigma: float, spr: float) -> float:
    nk = nk_fn(z, sigma, spr)
    mu_star = mu_fn(z, sigma, spr)
    omega_star = omega_fn(z, sigma)
    gamma_star = gamma_fn(z, sigma)
    g_star = g_fn(z, sigma)
    dgamma_domega_star = dgamma_domega_fn(z)
    dg_domega_star = dg_domega_fn(z, sigma)
    d2gamma_domega2_star = d2gamma_domega2_fn(z, sigma)
    d2g_domega2_star = d2g_domega2_fn(z, sigma)
    numerator = (
        omega_star
        * mu_star
        * nk
        * (d2gamma_domega2_star * dg_domega_star - d2g_domega2_star * dgamma_domega_star)
    )
    denominator = (
        (dgamma_domega_star - mu_star * dg_domega_star) ** 2
        * spr
        * (
            1.0
            - gamma_star
            + dgamma_domega_star
            * (gamma_star - mu_star * g_star)
            / (dgamma_domega_star - mu_star * dg_domega_star)
        )
    )
    return numerator / denominator


def zeta_zomega_fn(z: float, sigma: float, spr: float) -> float:
    mu_star = mu_fn(z, sigma, spr)
    return (
        omega_fn(z, sigma)
        * (dgamma_domega_fn(z) - mu_star * dg_domega_fn(z, sigma))
        / (gamma_fn(z, sigma) - mu_star * g_fn(z, sigma))
    )


def nk_fn(z: float, sigma: float, spr: float) -> float:
    return 1.0 - (gamma_fn(z, sigma) - mu_fn(z, sigma, spr) * g_fn(z, sigma)) * spr


def mu_fn(z: float, sigma: float, spr: float) -> float:
    denominator = dg_domega_fn(z, sigma) / dgamma_domega_fn(z) * (1.0 - gamma_fn(z, sigma)) + g_fn(
        z, sigma
    )
    return (1.0 - 1.0 / spr) / denominator


def omega_fn(z: float, sigma: float) -> float:
    return exp(sigma * z - sigma**2 / 2.0)


def g_fn(z: float, sigma: float) -> float:
    return float(norm.cdf(z - sigma))


def gamma_fn(z: float, sigma: float) -> float:
    return omega_fn(z, sigma) * (1.0 - float(norm.cdf(z))) + float(norm.cdf(z - sigma))


def dg_domega_fn(z: float, sigma: float) -> float:
    return float(norm.pdf(z)) / sigma


def d2g_domega2_fn(z: float, sigma: float) -> float:
    return -z * float(norm.pdf(z)) / omega_fn(z, sigma) / sigma**2


def dgamma_domega_fn(z: float) -> float:
    return 1.0 - float(norm.cdf(z))


def d2gamma_domega2_fn(z: float, sigma: float) -> float:
    return -float(norm.pdf(z)) / omega_fn(z, sigma) / sigma


def dg_dsigma_fn(z: float, sigma: float) -> float:
    return -z * float(norm.pdf(z - sigma)) / sigma


def d2g_domega_dsigma_fn(z: float, sigma: float) -> float:
    return -float(norm.pdf(z)) * (1.0 - z * (z - sigma)) / sigma**2


def dgamma_dsigma_fn(z: float, sigma: float) -> float:
    return -float(norm.pdf(z - sigma))


def d2gamma_domega_dsigma_fn(z: float, sigma: float) -> float:
    return (z / sigma - 1.0) * float(norm.pdf(z))
