from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from math import exp, log
from typing import Any

from scipy.optimize import root_scalar
from scipy.stats import norm

from nydsge.core import DSGEModel, NotPortedError, Observable, Parameter, PseudoObservable
from nydsge.financial_frictions import (
    d2g_domega_dsigma_fn,
    d2gamma_domega_dsigma_fn,
    dg_domega_fn,
    dg_dsigma_fn,
    dgamma_domega_fn,
    dgamma_dsigma_fn,
    g_fn,
    gamma_fn,
    mu_fn,
    nk_fn,
    omega_fn,
    zeta_bomega_fn,
    zeta_spb_fn,
    zeta_zomega_fn,
)
from nydsge.models.expected_ffr import (
    parse_expected_ffr_horizons,
    parse_expected_ffr_regime_horizons,
)
from nydsge.models.m1002_augment import augment_transition_ss10
from nydsge.models.m1002_eqcond import equilibrium_matrices_ss10
from nydsge.models.m1002_measurement import measurement_matrices_ss10
from nydsge.models.m1002_pseudo_measurement import pseudo_measurement_matrices_ss10
from nydsge.parameters import Prior
from nydsge.runtime import RuntimeConfig
from nydsge.solve import CanonicalSystem, Measurement, PseudoMeasurement, Transition

ParameterSpec = tuple[str, float, bool, tuple[float, float] | None, str]
PseudoObservableSpec = tuple[str, str, str] | tuple[str, str, str, str]

MODEL1002_SS10_STEADY_STATE_NAMES: tuple[str, ...] = (
    "z_star",
    "rstar",
    "Rstarn",
    "r_k_star",
    "wstar",
    "Lstar",
    "kstar",
    "kbarstar",
    "istar",
    "ystar",
    "cstar",
    "wl_c",
    "nstar",
    "vstar",
    "zeta_spsigma_omega",
    "zeta_spmu_e",
    "zeta_nRk",
    "zeta_nR",
    "zeta_nqk",
    "zeta_nn",
    "zeta_nmu_e",
    "zeta_nsigma_omega",
)

MODEL1002_SS10_PARAMETER_SCALINGS: dict[str, str] = {
    "beta": "discount_rate",
    "pi_star": "gross_rate",
    "Fomega": "fomega",
    "spr": "quarterly_spread",
    "gamma": "percent",
}

MODEL1002_SS10_PARAMETER_DESCRIPTIONS: dict[str, str] = {
    "alpha": "Capital share in production.",
    "zeta_p": "Calvo price stickiness.",
    "iota_p": "Price indexation.",
    "delta": "Capital depreciation rate.",
    "Upsilon": "Investment-specific technology growth.",
    "Phi": "Fixed cost parameter.",
    "Spp": "Investment adjustment cost curvature.",
    "h": "External habit persistence.",
    "ppsi": "Capacity utilization adjustment cost curvature.",
    "nu_l": "Inverse Frisch elasticity of labor supply.",
    "zeta_w": "Calvo wage stickiness.",
    "iota_w": "Wage indexation.",
    "lambda_w": "Wage markup.",
    "beta": "Household discount rate parameter.",
    "psi1": "Taylor-rule inflation response.",
    "psi2": "Taylor-rule output-gap response.",
    "psi3": "Taylor-rule output-growth response.",
    "pi_star": "Steady-state inflation target.",
    "sigma_c": "Intertemporal substitution curvature.",
    "rho": "Taylor-rule interest-rate smoothing.",
    "epsilon_p": "Goods-market elasticity of substitution.",
    "epsilon_w": "Labor-market elasticity of substitution.",
    "Fomega": "Financial-frictions default threshold.",
    "spr": "Steady-state external finance spread.",
    "zeta_spb": "Elasticity of spread with respect to leverage.",
    "gamma_star": "Steady-state technology growth.",
    "gamma": "Steady-state growth rate.",
    "Lmean": "Mean log hours.",
    "g_star": "Steady-state government spending share.",
    "me_level": "Measurement-error level parameter.",
    "damp_standard_shocks": "Multiplier damping standard shock standard deviations.",
    "amplify_sigma_r_m": "Multiplier amplifying monetary-policy shock volatility.",
    "amplify_inflation_me": "Multiplier amplifying inflation measurement error.",
    "phi_pi": "Temporary-policy inflation response.",
    "phi_y": "Temporary-policy output response.",
    "rho_smooth": "Temporary-policy smoothing parameter.",
    "kappa_std_bcshocks": "Blue-Chip shock standard-deviation scale.",
    "kappa_covid": "COVID shock standard-deviation scale.",
    "kappa_pce": "PCE inflation shock scale.",
    "eta_gz": "Trend-growth loading for output growth.",
    "eta_lambda_f": "Financial shock loading.",
    "eta_lambda_w": "Wage-markup shock loading.",
    "Iendoalpha": "Endogenous capital-share indicator.",
    "Gamma_gdpdef": "GDP-deflator measurement loading.",
    "delta_gdpdef": "GDP-deflator measurement intercept.",
    "gamma_gdi": "GDI measurement loading.",
    "delta_gdi": "GDI measurement intercept.",
    "meas_pi1": "Inflation measurement-error loading.",
    "rho_exp_rm": "Expected federal funds rate shock persistence.",
}

MODEL1002_SS10_PARAMETER_PRIORS: dict[str, Prior] = {
    "alpha": Prior("normal", mean=0.30, std=0.05),
    "zeta_p": Prior("beta_alt", mean=0.5, std=0.1),
    "iota_p": Prior("beta_alt", mean=0.5, std=0.15),
    "Phi": Prior("normal", mean=1.25, std=0.12),
    "Spp": Prior("normal", mean=4.0, std=1.5),
    "h": Prior("beta_alt", mean=0.7, std=0.1),
    "ppsi": Prior("beta_alt", mean=0.5, std=0.15),
    "nu_l": Prior("normal", mean=2.0, std=0.75),
    "zeta_w": Prior("beta_alt", mean=0.5, std=0.1),
    "iota_w": Prior("beta_alt", mean=0.5, std=0.15),
    "beta": Prior("gamma_alt", mean=0.25, std=0.1),
    "psi1": Prior("normal", mean=1.5, std=0.25),
    "psi2": Prior("normal", mean=0.12, std=0.05),
    "psi3": Prior("normal", mean=0.12, std=0.05),
    "sigma_c": Prior("normal", mean=1.5, std=0.37),
    "rho": Prior("beta_alt", mean=0.75, std=0.10),
    "spr": Prior("gamma_alt", mean=2.0, std=0.1),
    "zeta_spb": Prior("beta_alt", mean=0.05, std=0.005),
    "gamma": Prior("normal", mean=0.4, std=0.1),
    "Lmean": Prior("normal", mean=-45.0, std=5.0),
    "rho_g": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_b": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_mu": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_ztil": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_lambda_f": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_lambda_w": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_rm": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_sigma_w": Prior("beta_alt", mean=0.75, std=0.15),
    "rho_lr": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_z_p": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_tfp": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_gdpdef": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_corepce": Prior("beta_alt", mean=0.5, std=0.2),
    "rho_gdp": Prior("normal", mean=0.0, std=0.2),
    "rho_gdi": Prior("normal", mean=0.0, std=0.2),
    "rho_gdpvar": Prior("normal", mean=0.0, std=0.4),
    "sigma_g": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_b": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_mu": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_ztil": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_lambda_f": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_lambda_w": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_r_m": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_sigma_omega": Prior("root_inverse_gamma", nu=4.0, tau=0.05),
    "sigma_pi_star": Prior("root_inverse_gamma", nu=6.0, tau=0.03),
    "sigma_lr": Prior("root_inverse_gamma", nu=2.0, tau=0.75),
    "sigma_z_p": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_tfp": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_gdpdef": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_corepce": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_gdp": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "sigma_gdi": Prior("root_inverse_gamma", nu=2.0, tau=0.10),
    "eta_gz": Prior("beta_alt", mean=0.5, std=0.2),
    "eta_lambda_f": Prior("beta_alt", mean=0.5, std=0.2),
    "eta_lambda_w": Prior("beta_alt", mean=0.5, std=0.2),
    "Gamma_gdpdef": Prior("normal", mean=1.0, std=2.0),
    "delta_gdpdef": Prior("normal", mean=0.0, std=2.0),
}

MODEL1002_SS10_PSEUDO_OBSERVABLE_SPECS: tuple[PseudoObservableSpec, ...] = (
    ("y_t", "Output Growth", "identity"),
    ("y_f_t", "Flexible Output Growth", "identity"),
    ("NaturalRate", "Real Natural Rate", "quarter_to_annual"),
    ("\u03c0_t", "Inflation", "quarter_to_annual"),
    ("OutputGap", "Output Gap", "identity"),
    ("ExAnteRealRate", "Ex Ante Real Rate", "quarter_to_annual"),
    ("LongRunInflation", "Long Run Inflation", "quarter_to_annual"),
    ("MarginalCost", "Marginal Cost", "identity"),
    ("Wages", "Wages", "identity"),
    ("FlexibleWages", "Flexible Wages", "identity"),
    ("Hours", "Hours", "identity"),
    ("FlexibleHours", "Flexible Hours", "identity"),
    ("z_t", "z_t (Technology Growth minus Steady State Growth)", "identity"),
    ("Expected10YearRateGap", "Expected 10-Year Rate Gap", "quarter_to_annual"),
    ("NominalFFR", "Nominal FFR", "quarter_to_annual"),
    ("Expected10YearRate", "Expected 10-Year Rate", "quarter_to_annual"),
    ("Expected10YearNaturalRate", "Expected 10-Year Natural Rate", "quarter_to_annual"),
    ("ExpectedNominalNaturalRate", "Expected Nominal Natural Rate", "quarter_to_annual"),
    ("NominalRateGap", "Nominal Rate Gap", "quarter_to_annual"),
    ("LaborProductivityGrowth", "Labor Productivity Growth", "quarter_to_annual"),
    ("u_t", "u_t", "identity"),
)

MODEL1002_SS10_OPTIONAL_PSEUDO_OBSERVABLE_SPECS: tuple[PseudoObservableSpec, ...] = (
    ("pgap_t", "pgap", "identity"),
    ("ygap_t", "ygap", "identity"),
)

MODEL1002_SS10_PARAMETER_SPECS: tuple[ParameterSpec, ...] = (
    ("alpha", 0.1596, False, (1e-05, 0.999), "sqrt"),
    ("zeta_p", 0.894, False, (0.0, 1.0), "sqrt"),
    ("iota_p", 0.1865, False, (0.0, 1.0), "sqrt"),
    ("delta", 0.025, True, (0.025, 0.025), "identity"),
    ("Upsilon", 1.0, True, (0.0, 10.0), "exponential"),
    ("Phi", 1.1066, False, (1.0, 10.0), "exponential"),
    ("Spp", 2.7314, False, (-15.0, 15.0), "identity"),
    ("h", 0.5347, False, (0.0, 1.0), "sqrt"),
    ("ppsi", 0.6862, False, (0.0, 1.0), "sqrt"),
    ("nu_l", 2.5975, False, (1e-05, 10.0), "exponential"),
    ("zeta_w", 0.9291, False, (0.0, 1.0), "sqrt"),
    ("iota_w", 0.2992, False, (0.0, 1.0), "sqrt"),
    ("lambda_w", 1.5, True, (1.5, 1.5), "identity"),
    ("beta", 0.1402, False, (1e-05, 10.0), "exponential"),
    ("psi1", 1.3679, False, (1e-05, 10.0), "exponential"),
    ("psi2", 0.0388, False, (-0.5, 0.5), "identity"),
    ("psi3", 0.2464, False, (-0.5, 0.5), "identity"),
    ("pi_star", 0.5, True, (1e-05, 10.0), "exponential"),
    ("sigma_c", 0.8719, False, (1e-05, 10.0), "exponential"),
    ("rho", 0.7126, False, (0.0, 1.0), "sqrt"),
    ("epsilon_p", 10.0, True, (10.0, 10.0), "identity"),
    ("epsilon_w", 10.0, True, (10.0, 10.0), "identity"),
    ("Fomega", 0.03, True, (0.0, 1.0), "sqrt"),
    ("spr", 1.7444, False, (0.0, 100.0), "exponential"),
    ("zeta_spb", 0.0559, False, (0.0, 1.0), "sqrt"),
    ("gamma_star", 0.99, True, (0.0, 1.0), "sqrt"),
    ("gamma", 0.3673, False, (-5.0, 5.0), "identity"),
    ("Lmean", -45.9364, False, (-1000.0, 1000.0), "identity"),
    ("g_star", 0.18, True, (0.18, 0.18), "identity"),
    ("rho_g", 0.9863, False, (0.0, 1.0), "sqrt"),
    ("rho_b", 0.941, False, (0.0, 1.0), "sqrt"),
    ("rho_mu", 0.8735, False, (0.0, 1.0), "sqrt"),
    ("rho_ztil", 0.9446, False, (0.0, 1.0), "sqrt"),
    ("rho_lambda_f", 0.8827, False, (0.0, 1.0), "sqrt"),
    ("rho_lambda_w", 0.3884, False, (0.0, 1.0), "sqrt"),
    ("rho_rm", 0.2135, False, (0.0, 1.0), "sqrt"),
    ("rho_sigma_w", 0.9898, False, (0.0, 1.0), "sqrt"),
    ("rho_mu_e", 0.75, True, (0.0, 1.0), "sqrt"),
    ("rho_gamma", 0.75, True, (0.0, 1.0), "sqrt"),
    ("rho_pi_star", 0.99, True, (0.0, 1.0), "sqrt"),
    ("rho_lr", 0.6936, False, (0.0, 1.0), "sqrt"),
    ("rho_z_p", 0.891, False, (0.0, 1.0), "sqrt"),
    ("rho_tfp", 0.1953, False, (0.0, 1.0), "sqrt"),
    ("rho_gdpdef", 0.5379, False, (0.0, 1.0), "sqrt"),
    ("rho_corepce", 0.232, False, (0.0, 1.0), "sqrt"),
    ("rho_gdp", 0.0, False, (-0.999, 0.999), "sqrt"),
    ("rho_gdi", 0.0, False, (-0.999, 0.999), "sqrt"),
    ("rho_gdpvar", 0.0, False, (-0.999, 0.999), "sqrt"),
    ("me_level", 1.0, True, (1.0, 1.0), "identity"),
    ("rho_meas_pi", 0.232, False, (0.0, 0.999), "sqrt"),
    ("rho_ait_rm", 0.2135, False, (-1e-05, 0.999), "sqrt"),
    ("sigma_g", 2.523, False, (1e-08, 5.0), "exponential"),
    ("sigma_b", 0.0292, False, (1e-08, 5.0), "exponential"),
    ("sigma_mu", 0.4559, False, (1e-08, 5.0), "exponential"),
    ("sigma_ztil", 0.6742, False, (1e-08, 5.0), "exponential"),
    ("sigma_lambda_f", 0.1314, False, (1e-08, 5.0), "exponential"),
    ("sigma_lambda_w", 0.3864, False, (1e-08, 5.0), "exponential"),
    ("sigma_r_m", 0.238, False, (1e-08, 5.0), "exponential"),
    ("sigma_sigma_omega", 0.0428, False, (1e-07, 100.0), "exponential"),
    ("sigma_mu_e", 0.0, True, (0.0, 100.0), "exponential"),
    ("sigma_gamma", 0.0, True, (0.0, 100.0), "exponential"),
    ("sigma_pi_star", 0.0269, False, (1e-08, 5.0), "exponential"),
    ("sigma_lr", 0.1766, False, (1e-08, 10.0), "exponential"),
    ("sigma_z_p", 0.1662, False, (1e-08, 5.0), "exponential"),
    ("sigma_tfp", 0.9391, False, (1e-08, 5.0), "exponential"),
    ("sigma_gdpdef", 0.1575, False, (1e-08, 5.0), "exponential"),
    ("sigma_corepce", 0.0999, False, (1e-08, 5.0), "exponential"),
    ("sigma_gdp", 0.1, False, (1e-08, 5.0), "exponential"),
    ("sigma_gdi", 0.1, False, (1e-08, 5.0), "exponential"),
    ("sigma_meas_pi", 0.0999, False, (0.0, 5.0), "exponential"),
    ("sigma_ait_rm", 0.238, False, (0.0, 5.0), "exponential"),
    ("rho_ziid", 0.0, True, (0.0, 0.999), "identity"),
    ("sigma_ziid", 0.0, False, (0.0, 100.0), "exponential"),
    ("rho_biidc", 0.0, True, (0.0, 0.999), "identity"),
    ("sigma_biidc", 0.0, False, (0.0, 100.0), "exponential"),
    ("rho_phi", 0.0, True, (0.0, 0.999), "identity"),
    ("sigma_phi", 0.0, False, (0.0, 1000.0), "exponential"),
    ("rho_g_covid", 0.9863, False, (0.0, 0.999), "sqrt"),
    ("rho_mu_covid", 0.8735, False, (0.0, 0.999), "sqrt"),
    ("rho_lambda_f_covid", 0.8827, False, (0.0, 0.999), "sqrt"),
    ("rho_sigma_w_covid", 0.9898, False, (0.0, 0.99999), "sqrt"),
    ("rho_lr_covid", 0.6936, False, (0.0, 0.999), "sqrt"),
    ("rho_tfp_covid", 0.1953, False, (0.0, 0.999), "sqrt"),
    ("rho_gdp_covid", 0.0, False, (-0.999, 0.999), "sqrt"),
    ("rho_gdi_covid", 0.0, False, (-0.999, 0.999), "sqrt"),
    ("rho_gdpvar_covid", 0.0, False, (-0.999, 0.999), "sqrt"),
    ("sigma_g_covid", 2.523, False, (0.0, 5.0), "exponential"),
    ("sigma_mu_covid", 0.4559, False, (1e-08, 5.0), "exponential"),
    ("sigma_lambda_f_covid", 0.1314, False, (1e-08, 5.0), "exponential"),
    ("sigma_sigma_omega_covid", 0.0428, False, (1e-07, 100.0), "exponential"),
    ("sigma_lr_covid", 0.1766, False, (1e-08, 10.0), "exponential"),
    ("sigma_tfp_covid", 0.9391, False, (1e-08, 5.0), "exponential"),
    ("sigma_gdp_covid", 0.1, False, (1e-08, 5.0), "exponential"),
    ("sigma_gdi_covid", 0.1, False, (1e-08, 5.0), "exponential"),
    ("rho_z_p_covid", 0.891, False, (0.0, 0.999), "sqrt"),
    ("sigma_z_p_covid", 0.1662, False, (1e-08, 5.0), "exponential"),
    ("damp_standard_shocks", 1.0, False, (0.0, 1000.0), "identity"),
    ("amplify_sigma_r_m", 1.0, False, (0.0, 1000.0), "identity"),
    ("amplify_inflation_me", 1.0, False, (0.0, 1000.0), "identity"),
    ("rho_biidc_sh", 0.75, False, (0.0, 0.999), "identity"),
    ("phi_pi", 4.0, False, (1.25, 15.0), "identity"),
    ("phi_y", 3.0, False, (1.25, 15.0), "identity"),
    ("rho_smooth", 0.9, False, (1e-05, 0.999), "sqrt"),
    ("kappa_std_bcshocks", 1.0, False, (0.0, 1.0), "sqrt"),
    ("kappa_covid", 1.0, False, (0.0, 2.0), "identity"),
    ("kappa_pce", 1.0, True, (0.0, 2.0), "identity"),
    ("sigma_pgap", 0.0, True, (0.0, 100.0), "exponential"),
    ("sigma_ygap", 0.0, True, (0.0, 100.0), "exponential"),
    ("rho_condgdp", 0.0, True, (-0.999, 0.999), "sqrt"),
    ("sigma_condgdp", 0.1, False, (0.0, 5.0), "exponential"),
    ("rho_gdpexp", 0.0, True, (-0.999, 0.999), "sqrt"),
    ("sigma_gdpexp", 0.1, False, (0.0, 5.0), "exponential"),
    ("rho_condcorepce", 0.0, True, (-0.999, 0.999), "sqrt"),
    ("sigma_condcorepce", 0.0999, False, (0.0, 5.0), "exponential"),
    ("eta_gz", 0.84, False, (0.0, 1.0), "sqrt"),
    ("eta_lambda_f", 0.7892, False, (0.0, 1.0), "sqrt"),
    ("eta_lambda_w", 0.4226, False, (0.0, 1.0), "sqrt"),
    ("Iendoalpha", 0.0, True, (0.0, 0.0), "identity"),
    ("Gamma_gdpdef", 1.0354, False, (-10.0, 10.0), "identity"),
    ("delta_gdpdef", 0.0181, False, (-10.0, 10.0), "identity"),
    ("gamma_gdi", 1.0, True, (1.0, 1.0), "identity"),
    ("delta_gdi", 0.0, True, (0.0, 0.0), "identity"),
    ("sigma_lambda_f_iid", 0.0, False, (0.0, 100.0), "exponential"),
    ("rho_lambda_f_iid", 0.0, True, (1e-05, 0.999), "sqrt"),
    ("meas_pi1", 0.0, True, (0.0, 5.0), "sqrt"),
    ("rho_exp_rm", 0.0, True, (-1e-05, 0.999), "sqrt"),
)

MODEL1002_SS10_PARAMETER_SPEC_BY_NAME: dict[str, ParameterSpec] = {
    spec[0]: spec for spec in MODEL1002_SS10_PARAMETER_SPECS
}

MODEL1002_SS10_DEFAULT_PARAMETER_NAMES: tuple[str, ...] = (
    "alpha",
    "zeta_p",
    "iota_p",
    "delta",
    "Upsilon",
    "Phi",
    "Spp",
    "h",
    "ppsi",
    "nu_l",
    "zeta_w",
    "iota_w",
    "lambda_w",
    "beta",
    "psi1",
    "psi2",
    "psi3",
    "pi_star",
    "sigma_c",
    "rho",
    "epsilon_p",
    "epsilon_w",
    "Fomega",
    "spr",
    "zeta_spb",
    "gamma_star",
    "gamma",
    "Lmean",
    "g_star",
    "rho_g",
    "rho_b",
    "rho_mu",
    "rho_ztil",
    "rho_lambda_f",
    "rho_lambda_w",
    "rho_rm",
    "rho_sigma_w",
    "rho_mu_e",
    "rho_gamma",
    "rho_pi_star",
    "rho_lr",
    "rho_z_p",
    "rho_tfp",
    "rho_gdpdef",
    "rho_corepce",
    "rho_gdp",
    "rho_gdi",
    "rho_gdpvar",
    "me_level",
    "sigma_g",
    "sigma_b",
    "sigma_mu",
    "sigma_ztil",
    "sigma_lambda_f",
    "sigma_lambda_w",
    "sigma_r_m",
    "sigma_sigma_omega",
    "sigma_mu_e",
    "sigma_gamma",
    "sigma_pi_star",
    "sigma_lr",
    "sigma_z_p",
    "sigma_tfp",
    "sigma_gdpdef",
    "sigma_corepce",
    "sigma_gdp",
    "sigma_gdi",
)

MODEL1002_SS10_POST_ANTICIPATED_PARAMETER_NAMES: tuple[str, ...] = (
    "eta_gz",
    "eta_lambda_f",
    "eta_lambda_w",
    "Iendoalpha",
    "Gamma_gdpdef",
    "delta_gdpdef",
    "gamma_gdi",
    "delta_gdi",
)


def _solve_sigma_omega_star(zomega_star: float, zeta_spb: float, spr: float) -> float:
    def residual(sigma: float) -> float:
        return zeta_spb_fn(zomega_star, sigma, spr) - zeta_spb

    try:
        result = root_scalar(residual, x0=0.5, x1=0.6, method="secant", maxiter=100)
    except (ValueError, ZeroDivisionError, RuntimeError):
        return 0.5
    if not result.converged:
        return 0.5
    return float(result.root)


def _parameter_description(name: str) -> str:
    if name in MODEL1002_SS10_PARAMETER_DESCRIPTIONS:
        return MODEL1002_SS10_PARAMETER_DESCRIPTIONS[name]
    if name.startswith("rho_"):
        return f"Persistence parameter for {name.removeprefix('rho_')}."
    if name.startswith("sigma_"):
        return f"Standard deviation for {name.removeprefix('sigma_')} shock."
    return name.replace("_", " ").capitalize() + "."


def _parameter_category(name: str) -> str:
    if name.startswith(("rho_cond", "sigma_cond", "rho_gdpexp", "sigma_gdpexp")):
        return "measurement_error"
    if name.startswith("rho_"):
        return "persistence"
    if name.startswith("sigma_"):
        return "shock_std"
    if name in {"psi1", "psi2", "psi3", "rho", "phi_pi", "phi_y", "rho_smooth"}:
        return "policy_rule"
    if name in {"alpha", "Phi", "Spp", "h", "ppsi", "nu_l", "zeta_p", "zeta_w"}:
        return "structural"
    if name in {"beta", "pi_star", "gamma", "gamma_star", "g_star", "Lmean"}:
        return "steady_state"
    if name.startswith(("eta_", "Gamma_", "delta_", "meas_")) or name in {"gamma_gdi"}:
        return "measurement"
    return "other"


def _parameter_regime(name: str) -> str:
    if name.endswith("_covid") or "covid" in name:
        return "covid"
    if "_ait_" in name or name.endswith("_ait_rm"):
        return "average_inflation_targeting"
    if name in {"phi_pi", "phi_y", "rho_smooth", "kappa_std_bcshocks"}:
        return "temporary_policy"
    if name.startswith(("rho_cond", "sigma_cond", "rho_gdpexp", "sigma_gdpexp")):
        return "conditional_forecast"
    if name.endswith("_iid") or name in {"rho_ziid", "sigma_ziid", "rho_biidc", "sigma_biidc"}:
        return "iid_branch"
    return "baseline"


class Model1002(DSGEModel):
    """Python representation of upstream FRBNY DSGE Model1002.

    This class currently ports the model shape needed for construction, settings,
    observable ordering, and index maps. Economic kernels still raise
    `NotPortedError` until the Julia equations are translated.
    """

    def __init__(
        self,
        subspec: str = "ss10",
        *,
        runtime: RuntimeConfig | None = None,
        settings: dict[str, Any] | None = None,
        testing: bool = False,
    ) -> None:
        defaults: dict[str, Any] = {
            "data_vintage": "181115",
            "date_presample_start": "1959-Q3",
            "date_mainsample_start": "1960-Q1",
            "date_zlb_start": "2008-Q4",
            "date_forecast_start": "2018-Q4",
            "data_id": 3,
            "cond_id": 2,
            "dataroot": "save/input_data",
            "n_mon_anticipated_shocks": 6,
            "n_mon_anticipated_shocks_padding": 20,
            "n_z_anticipated_shocks": 0,
            "n_z_anticipated_shocks_padding": 0,
            "use_population_forecast": True,
            "population_mnemonic": "CNP16OV__FRED",
            "hpfilter_population": True,
            "population_hpfilter_lambda": 1600.0,
            "antshocks": {},
            "expected_ffr": (),
            "all_ffr_qs": (),
            "add_initialize_pgap_ygap_pseudoobs": False,
            "add_iid_cond_obs_gdp_meas_err": False,
            "add_iid_anticipated_obs_gdp_meas_err": False,
            "add_iid_cond_obs_corepce_meas_err": False,
        }
        defaults.update(self._subspec_default_settings(subspec))
        defaults.update(settings or {})
        super().__init__(
            spec="m1002",
            subspec=subspec,
            runtime=runtime,
            settings=defaults,
            testing=testing,
        )
        self._init_observable_mappings()
        self._init_pseudo_observable_mappings()
        self._init_parameters()
        self._init_index_maps()
        self._init_steady_state_placeholders()

    def description(self) -> str:
        return f"New York Fed DSGE Model m1002, {self.subspec}."

    def _init_observable_mappings(self) -> None:
        population = str(self.get_setting("population_mnemonic"))
        gdp_observable = Observable(
            "obs_gdp",
            ("GDP__FRED", population, "GDPDEF__FRED"),
            "Real GDP Growth",
            "Real GDP Growth Per Capita",
            reverse_transform="loggrowth_to_pct_annualized_percapita",
            forward_transform="gdp_growth",
        )
        hours_observable = Observable(
            "obs_hours",
            ("AWHNONAG__FRED", "CE16OV__FRED"),
            "Hours Per Capita",
            "Log Hours Per Capita",
            forward_transform="hours_per_capita",
        )
        first_observables = (
            [hours_observable, gdp_observable]
            if self.get_setting("hours_first_observable", False)
            else [gdp_observable, hours_observable]
        )
        base = [
            *first_observables,
            Observable(
                "obs_wages",
                ("COMPNFB__FRED", "GDPDEF__FRED"),
                "Real Wage Growth",
                "Q-to-Q Percent Change of Real Compensation",
                reverse_transform="loggrowth_to_pct_annualized",
                forward_transform="real_wage_growth",
            ),
            Observable(
                "obs_gdpdeflator",
                ("GDPDEF__FRED",),
                "GDP Deflator",
                "Q-to-Q Percent Change of GDP Deflator",
                reverse_transform="loggrowth_to_pct_annualized",
                forward_transform="gdp_deflator_growth",
            ),
            Observable(
                "obs_corepce",
                ("PCEPILFE__FRED",),
                "Core PCE Inflation",
                "Core PCE Inflation",
                reverse_transform="loggrowth_to_pct_annualized",
                forward_transform="core_pce_growth",
            ),
            Observable(
                "obs_nominalrate",
                ("DFF__FRED",),
                "Nominal FFR",
                "Nominal Effective Fed Funds Rate",
                reverse_transform="quarter_to_annual",
                forward_transform="nominal_rate",
            ),
            Observable(
                "obs_consumption",
                ("PCE__FRED", population, "GDPDEF__FRED"),
                "Consumption Growth",
                "Consumption growth adjusted for population filtering",
                reverse_transform="loggrowth_to_pct_annualized_percapita",
                forward_transform="consumption_growth",
            ),
            Observable(
                "obs_investment",
                ("FPI__FRED", population, "GDPDEF__FRED"),
                "Investment Growth",
                "Real investment per capita",
                reverse_transform="loggrowth_to_pct_annualized_percapita",
                forward_transform="investment_growth",
            ),
            Observable(
                "obs_spread",
                ("BAA__FRED", "BAMLC8A0C15PYEY__FRED", "GS10__FRED"),
                "BAA - 10yr Treasury Spread",
                "BAA - 10yr Treasury Spread",
                reverse_transform="quarter_to_annual",
                forward_transform="baa_10y_spread",
            ),
            Observable(
                "obs_longinflation",
                ("ASACX10__DLX",),
                "10-year average inflation expectations",
                "10-year average yr/yr CPI inflation expectations",
                reverse_transform="loggrowth_to_pct_annualized",
                forward_transform="long_inflation_expectations",
            ),
            Observable(
                "obs_longrate",
                ("FYCCZA__DLX",),
                "10-year average interest rate expectations",
                "10T yield",
                reverse_transform="quarter_to_annual",
                forward_transform="long_rate",
            ),
            Observable(
                "obs_tfp",
                ("TFPKQ__DLX", "TFPJQ__DLX"),
                "Total Factor Productivity Growth (Fernald)",
                "Fernald TFP adjusted by alpha",
                reverse_transform="quarter_to_annual",
                forward_transform="fernald_tfp",
            ),
            Observable(
                "obs_gdi",
                ("GDI__FRED", population, "GDPDEF__FRED"),
                "Real GDI Growth",
                "Real GDI Growth Per Capita",
                reverse_transform="loggrowth_to_pct_annualized_percapita",
                forward_transform="gdi_growth",
            ),
        ]
        for observable in base:
            self.add_observable(observable)
        for index in range(1, int(self.get_setting("n_mon_anticipated_shocks")) + 1):
            self.add_observable(
                Observable(
                    f"obs_nominalrate{index}",
                    (f"ant{index}__OIS",),
                    f"Anticipated Shock {index}",
                    f"{index}-period ahead anticipated monetary policy shock",
                    reverse_transform="quarter_to_annual",
                    forward_transform="anticipated_rate",
                )
            )
        for horizon in self._expected_ffr_horizons():
            self.add_observable(
                Observable(
                    f"obs_exp_nominalrate{horizon}",
                    (f"exp_ant{horizon}__SPD",),
                    f"Anticipated FFR {horizon}",
                    f"{horizon}-period ahead anticipated federal funds rate",
                    reverse_transform="quarter_to_annual",
                    forward_transform="expected_ffr_spd",
                )
            )
        if self._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs"):
            self.add_observable(
                Observable(
                    "obs_pgap",
                    ("pgap__INITFLEXAIT",),
                    "Average Inflation Gap",
                    "Average Inflation Gap from Target",
                    forward_transform="flexible_ait_gap",
                )
            )
            self.add_observable(
                Observable(
                    "obs_ygap",
                    ("ygap__INITFLEXAIT",),
                    "Average Output Gap",
                    "Average Output Gap from Target",
                    forward_transform="flexible_ait_gap",
                )
            )
        if self._is_setting_enabled("add_anticipated_obs_gdp"):
            source = str(self.get_setting("filename_anticipated_obs_gdp", "ANTGDP"))
            count = int(self.get_setting("n_anticipated_obs_gdp", 1))
            for index in range(1, count + 1):
                self.add_observable(
                    Observable(
                        f"obs_gdp{index}",
                        (f"antgdp{index}__{source}",),
                        f"Anticipated GDP Growth {index}",
                        f"{index}-period ahead anticipated GDP growth",
                        reverse_transform="loggrowth_to_pct_annualized_percapita",
                        forward_transform="anticipated_gdp_growth",
                    )
                )
        self._move_observable_to_edge("first_observable", first=True)
        self._move_observable_to_edge("last_observable", first=False)
        forward_looking = [
            "obs_longinflation",
            "obs_longrate",
            *[
                f"obs_nominalrate{index}"
                for index in range(1, int(self.get_setting("n_mon_anticipated_shocks")) + 1)
            ],
            *[f"obs_exp_nominalrate{horizon}" for horizon in self._expected_ffr_horizons()],
        ]
        if self._is_setting_enabled("add_anticipated_obs_gdp"):
            forward_looking.extend(
                f"obs_gdp{index}"
                for index in range(1, int(self.get_setting("n_anticipated_obs_gdp", 1)) + 1)
            )
        self.set_setting("forward_looking_observables", forward_looking)

    def _move_observable_to_edge(self, setting_name: str, *, first: bool) -> None:
        observable_name = self.get_setting(setting_name, None)
        if observable_name is None:
            return
        observable_key = str(observable_name)
        if observable_key not in self.observable_mappings:
            msg = f"{setting_name} references unknown observable: {observable_key}"
            raise KeyError(msg)
        observable = self.observable_mappings.pop(observable_key)
        if first:
            self.observable_mappings = OrderedDict(
                [(observable_key, observable), *self.observable_mappings.items()]
            )
        else:
            self.observable_mappings[observable_key] = observable

    def _init_pseudo_observable_mappings(self) -> None:
        for spec in MODEL1002_SS10_PSEUDO_OBSERVABLE_SPECS:
            if len(spec) == 4:
                name, description, reverse_transform, forward_transform = spec
            else:
                name, description, reverse_transform = spec
                forward_transform = "identity"
            self.add_pseudo_observable(
                PseudoObservable(
                    name=name,
                    description=description,
                    reverse_transform=reverse_transform,
                    forward_transform=forward_transform,
                )
            )
        if self._has_pgap_state() or self._has_ygap_state():
            for spec in MODEL1002_SS10_OPTIONAL_PSEUDO_OBSERVABLE_SPECS:
                if len(spec) == 4:
                    name, description, reverse_transform, forward_transform = spec
                else:
                    name, description, reverse_transform = spec
                    forward_transform = "identity"
                self.add_pseudo_observable(
                    PseudoObservable(
                        name=name,
                        description=description,
                        reverse_transform=reverse_transform,
                        forward_transform=forward_transform,
                    )
                )
        self.set_setting(
            "forward_looking_pseudo_observables",
            [
                "Expected10YearRateGap",
                "Expected10YearRate",
                "Expected10YearNaturalRate",
            ],
        )

    def _init_parameters(self) -> None:
        for name in MODEL1002_SS10_DEFAULT_PARAMETER_NAMES:
            self._add_parameter_from_spec(name)
        n_ant = int(self.get_setting("n_mon_anticipated_shocks"))
        n_ant_padding = int(self.get_setting("n_mon_anticipated_shocks_padding"))
        for index in range(1, n_ant_padding + 1):
            is_active = index <= n_ant
            self.add_parameter(
                Parameter(
                    f"sigma_r_m{index}",
                    0.2 if is_active else 0.0,
                    fixed=not is_active,
                    value_bounds=(1.0e-7, 100.0),
                    transform="exponential",
                    prior=Prior("root_inverse_gamma", nu=4.0, tau=0.2),
                    description=(
                        f"{index}-period-ahead anticipated monetary-policy shock "
                        "standard deviation."
                    ),
                    category="shock_std",
                    regime="anticipated_policy",
                )
            )
        for name in MODEL1002_SS10_POST_ANTICIPATED_PARAMETER_NAMES:
            self._add_parameter_from_spec(name)
        horizon_regime_tags = self._expected_ffr_horizon_regimes()
        for horizon in self._expected_ffr_horizons():
            self.add_parameter(
                Parameter(
                    f"sigma_exp_rm{horizon}",
                    0.0375 + 0.00625 * horizon,
                    fixed=True,
                    value_bounds=(0.0, 5.0),
                    transform="exponential",
                    prior=Prior("root_inverse_gamma", nu=4.0, tau=0.2),
                    description=(
                        f"{horizon}-period-ahead expected federal funds rate "
                        "measurement-error standard deviation."
                    ),
                    category="measurement_error",
                    regime=f"expected_ffr_spd{horizon_regime_tags.get(horizon, '')}",
                )
            )
        if self._is_setting_enabled("add_iid_cond_obs_gdp_meas_err"):
            self._add_parameter_from_spec("rho_condgdp")
            self._add_parameter_from_spec("sigma_condgdp")
        if self._is_setting_enabled("add_iid_anticipated_obs_gdp_meas_err"):
            self._add_parameter_from_spec("rho_gdpexp")
            self._add_parameter_from_spec("sigma_gdpexp")
        if self._is_setting_enabled("add_iid_cond_obs_corepce_meas_err"):
            self._add_parameter_from_spec("rho_condcorepce")
            self._add_parameter_from_spec("sigma_condcorepce")
        if self._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs"):
            for name, description in (
                ("sigma_pgap", "Average inflation-gap initialization shock standard deviation."),
                ("sigma_ygap", "Average output-gap initialization shock standard deviation."),
            ):
                self.add_parameter(
                    Parameter(
                        name,
                        0.0,
                        fixed=True,
                        value_bounds=(0.0, 100.0),
                        transform="exponential",
                        prior=Prior("root_inverse_gamma", nu=8000.0, tau=(16.1**0.5)),
                        description=description,
                        category="measurement_error",
                        regime="flexible_ait",
                    )
                )
        if self._is_setting_enabled("add_ait_rm"):
            self._add_parameter_from_spec("rho_ait_rm")

    def _add_parameter_from_spec(self, name: str) -> None:
        try:
            spec = MODEL1002_SS10_PARAMETER_SPEC_BY_NAME[name]
        except KeyError as err:
            msg = f"Unknown Model1002 parameter spec: {name}"
            raise KeyError(msg) from err
        parameter_name, value, fixed, value_bounds, transform = spec
        self.add_parameter(
            Parameter(
                parameter_name,
                value,
                fixed=fixed,
                value_bounds=value_bounds,
                transform=transform,
                scaling=MODEL1002_SS10_PARAMETER_SCALINGS.get(parameter_name, "identity"),
                prior=MODEL1002_SS10_PARAMETER_PRIORS.get(parameter_name),
                description=_parameter_description(parameter_name),
                category=_parameter_category(parameter_name),
                regime=_parameter_regime(parameter_name),
            )
        )

    def _init_index_maps(self) -> None:
        n_ant = int(self.get_setting("n_mon_anticipated_shocks"))
        antshocks = self._anticipated_shock_settings()
        endogenous_states = [
            "y_t",
            "c_t",
            "i_t",
            "qk_t",
            "k_t",
            "kbar_t",
            "u_t",
            "rk_t",
            "Rktil_t",
            "n_t",
            "mc_t",
            "pi_t",
            "mu_omega_t",
            "w_t",
            "L_t",
            "R_t",
            "g_t",
            "b_t",
            "mu_t",
            "z_t",
            "lambda_f_t",
            "lambda_f_t1",
            "lambda_w_t",
            "lambda_w_t1",
            "rm_t",
            "sigma_omega_t",
            "mu_e_t",
            "gamma_t",
            "pi_star_t",
            "Ec_t",
            "Eqk_t",
            "Ei_t",
            "Epi_t",
            "EL_t",
            "Erk_t",
            "Ew_t",
            "ERktil_t",
            "ERktil_f_t",
            "y_f_t",
            "c_f_t",
            "i_f_t",
            "qk_f_t",
            "k_f_t",
            "kbar_f_t",
            "u_f_t",
            "rk_f_t",
            "w_f_t",
            "L_f_t",
            "r_f_t",
            "Ec_f_t",
            "Eqk_f_t",
            "Ei_f_t",
            "EL_f_t",
            "ztil_t",
            "pi_t1",
            "pi_t2",
            "pi_a_t",
            "R_t1",
            "zp_t",
            "Ez_t",
            "Rktil_f_t",
            "n_f_t",
        ] + [f"rm_tl{i}" for i in range(1, n_ant + 1)]
        for key, value in antshocks.items():
            endogenous_states.extend(f"{key}_tl{i}" for i in range(1, value + 1))
        if self._is_setting_enabled("add_rw"):
            endogenous_states.append("rw_t")
            endogenous_states.append("Rref_t")
        if self._has_pgap_state():
            endogenous_states.append("pgap_t")
        if self._has_ygap_state():
            endogenous_states.append("ygap_t")
        if self._is_setting_enabled("add_ait_rm"):
            endogenous_states.append("ait_rm_t")
        exogenous_shocks = [
            "g_sh",
            "b_sh",
            "mu_sh",
            "ztil_sh",
            "lambda_f_sh",
            "lambda_w_sh",
            "rm_sh",
            "sigma_omega_sh",
            "mu_e_sh",
            "gamma_sh",
            "pi_star_sh",
            "zp_sh",
            "lr_sh",
            "tfp_sh",
            "gdpdef_sh",
            "corepce_sh",
            "gdp_sh",
            "gdi_sh",
        ] + [f"rm_shl{i}" for i in range(1, n_ant + 1)]
        if self._is_setting_enabled("add_ait_rm"):
            exogenous_shocks.append("rm_ait_sh")
        for key, value in antshocks.items():
            exogenous_shocks.extend(f"{key}_shl{i}" for i in range(1, value + 1))
        if self._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs"):
            exogenous_shocks.extend(["pgap_sh", "ygap_sh"])
        expected_shocks = [
            "Ec_sh",
            "Eqk_sh",
            "Ei_sh",
            "Epi_sh",
            "EL_sh",
            "Erk_sh",
            "Ew_sh",
            "ERktil_sh",
            "Ec_f_sh",
            "Eqk_f_sh",
            "Ei_f_sh",
            "EL_f_sh",
            "ERktil_f_sh",
        ]
        equilibrium_conditions = [
            "eq_euler",
            "eq_inv",
            "eq_capval",
            "eq_spread",
            "eq_nevol",
            "eq_output",
            "eq_caputl",
            "eq_capsrv",
            "eq_capev",
            "eq_mkupp",
            "eq_phlps",
            "eq_caprnt",
            "eq_msub",
            "eq_wage",
            "eq_mp",
            "eq_res",
            "eq_g",
            "eq_b",
            "eq_mu",
            "eq_z",
            "eq_lambda_f",
            "eq_lambda_w",
            "eq_rm",
            "eq_sigma_omega",
            "eq_mu_e",
            "eq_gamma",
            "eq_lambda_f1",
            "eq_lambda_w1",
            "eq_Ec",
            "eq_Eqk",
            "eq_Ei",
            "eq_Epi",
            "eq_EL",
            "eq_Erk",
            "eq_Ew",
            "eq_ERktil",
            "eq_euler_f",
            "eq_inv_f",
            "eq_capval_f",
            "eq_output_f",
            "eq_caputl_f",
            "eq_capsrv_f",
            "eq_capev_f",
            "eq_mkupp_f",
            "eq_caprnt_f",
            "eq_msub_f",
            "eq_res_f",
            "eq_Ec_f",
            "eq_Eqk_f",
            "eq_Ei_f",
            "eq_EL_f",
            "eq_ztil",
            "eq_pi_star",
            "eq_pi1",
            "eq_pi2",
            "eq_pi_a",
            "eq_Rt1",
            "eq_zp",
            "eq_Ez",
            "eq_spread_f",
            "eq_nevol_f",
            "eq_ERktil_f",
        ] + [f"eq_rml{i}" for i in range(1, n_ant + 1)]
        for key, value in antshocks.items():
            equilibrium_conditions.extend(f"eq_{key}l{i}" for i in range(1, value + 1))
        if self._is_setting_enabled("add_rw"):
            equilibrium_conditions.append("eq_rw")
            equilibrium_conditions.append("eq_Rref")
        if self._is_setting_enabled("add_ait_rm"):
            equilibrium_conditions.append("eq_ait_rm")
        if self._has_pgap_state():
            equilibrium_conditions.append("eq_pgap")
        if self._has_ygap_state():
            equilibrium_conditions.append("eq_ygap")
        endogenous_states_augmented = [
            "y_t1",
            "c_t1",
            "i_t1",
            "w_t1",
            "pi_t1_dup",
            "L_t1",
            "u_t1",
            "Et_pi_t",
            "e_lr_t",
            "e_tfp_t",
            "e_gdpdef_t",
            "e_corepce_t",
            "e_gdp_t",
            "e_gdi_t",
            "e_gdp_t1",
            "e_gdi_t1",
        ]
        for horizon in self._expected_ffr_horizons():
            endogenous_states_augmented.append(f"e_exp_rm{horizon}")
            exogenous_shocks.append(f"exp_rm_sh{horizon}")
        if self._is_setting_enabled("add_iid_cond_obs_gdp_meas_err"):
            endogenous_states_augmented.append("e_condgdp_t")
            exogenous_shocks.append("condgdp_sh")
        if self._is_setting_enabled("add_iid_anticipated_obs_gdp_meas_err"):
            endogenous_states_augmented.append("e_gdpexp_t")
            exogenous_shocks.append("gdpexp_sh")
        if self._is_setting_enabled("add_iid_cond_obs_corepce_meas_err"):
            endogenous_states_augmented.append("e_condcorepce_t")
            exogenous_shocks.append("condcorepce_sh")
        self.indexes.endogenous_states = self.build_one_based_index(endogenous_states)
        self.indexes.exogenous_shocks = self.build_one_based_index(exogenous_shocks)
        self.indexes.expected_shocks = self.build_one_based_index(expected_shocks)
        self.indexes.equilibrium_conditions = self.build_one_based_index(equilibrium_conditions)
        offset_augmented = OrderedDict(
            (name, idx + len(endogenous_states))
            for idx, name in enumerate(endogenous_states_augmented, start=1)
        )
        self.indexes.endogenous_states_augmented = offset_augmented
        self.indexes.observables = self.build_one_based_index(list(self.observable_mappings.keys()))
        self.indexes.pseudo_observables = self.build_one_based_index(
            list(self.pseudo_observable_mappings.keys())
        )

    def _anticipated_shock_settings(self) -> dict[str, int]:
        raw_antshocks = self.get_setting("antshocks", {})
        if raw_antshocks is None:
            return {}
        if not isinstance(raw_antshocks, Mapping):
            msg = "antshocks setting must be a mapping from shock key to count."
            raise TypeError(msg)
        antshocks: dict[str, int] = {}
        for key, value in raw_antshocks.items():
            count = int(value)
            if count < 0:
                msg = f"antshocks count must be nonnegative for {key}."
                raise ValueError(msg)
            antshocks[str(key)] = count
        return antshocks

    @staticmethod
    def _subspec_default_settings(subspec: str) -> dict[str, Any]:
        if subspec == "ss104":
            return {
                "expected_ffr": (1, 2, 3, 4, 5, 6),
                "all_ffr_qs": (1, 2, 3, 4, 5, 6),
                "add_iid_cond_obs_gdp_meas_err": True,
                "add_iid_cond_obs_corepce_meas_err": True,
            }
        return {}

    def _expected_ffr_horizons(self) -> tuple[int, ...]:
        return parse_expected_ffr_horizons(
            self.get_setting("expected_ffr", ()),
            self.get_setting("all_ffr_qs", ()),
        )

    def _expected_ffr_regime_horizons(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        return parse_expected_ffr_regime_horizons(
            self.get_setting("expected_ffr", ()),
            self.get_setting("all_ffr_qs", ()),
        )

    def _expected_ffr_horizon_regimes(self) -> dict[int, str]:
        regime_horizons = self._expected_ffr_regime_horizons()
        if len(regime_horizons) == 1 and regime_horizons[0][0] == "default":
            return {}

        tags: dict[int, list[str]] = {}
        for regime_name, horizons in regime_horizons:
            for horizon in horizons:
                tags.setdefault(horizon, []).append(regime_name)

        return {horizon: f"[{','.join(values)}]" for horizon, values in tags.items() if values}

    @staticmethod
    def _normalize_regime_key(value: Any) -> str | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            try:
                return str(int(normalized))
            except ValueError:
                return normalized
        return None

    def _active_regime_set(self) -> set[str]:
        regime_eqcond_info = self.get_setting("regime_eqcond_info", None)
        if not isinstance(regime_eqcond_info, Mapping) or not bool(regime_eqcond_info):
            return {"1"}

        active_regimes: set[str] = set()
        for regime in regime_eqcond_info:
            normalized = self._normalize_regime_key(regime)
            if normalized is not None:
                active_regimes.add(normalized)
        return active_regimes

    def _is_setting_enabled(self, setting_name: str) -> bool:
        setting_value = self.get_setting(setting_name, False)
        active_regimes = self._active_regime_set()
        if not active_regimes:
            return False
        if isinstance(setting_value, Mapping):
            for regime in active_regimes:
                for key, value in setting_value.items():
                    setting_regime = self._normalize_regime_key(key)
                    if setting_regime is not None and setting_regime == regime:
                        return bool(value)
            return False
        return bool(setting_value)

    def _is_regime_selector_active(self, regime_selector: Any) -> bool:
        active_regimes = self._active_regime_set()
        if not active_regimes:
            return False
        if isinstance(regime_selector, (list, tuple, set)):
            for entry in regime_selector:
                if self._is_regime_selector_active(entry):
                    return True
            return False
        normalized = self._normalize_regime_key(regime_selector)
        return normalized is not None and normalized in active_regimes

    def _is_regime_one_active(self) -> bool:
        return bool({"1", "baseline", "default"} & self._active_regime_set())

    def _has_pgap_state(self) -> bool:
        return bool(
            self._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs")
            or self._is_setting_enabled("add_altpolicy_pgap")
            or self._is_setting_enabled("add_pgap")
        )

    def _has_ygap_state(self) -> bool:
        return bool(
            self._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs")
            or self._is_setting_enabled("add_altpolicy_ygap")
            or self._is_setting_enabled("add_ygap")
        )

    def _init_steady_state_placeholders(self) -> None:
        for name in MODEL1002_SS10_STEADY_STATE_NAMES:
            self.set_steady_state(name, float("nan"))

    def compute_steady_state(self) -> OrderedDict[str, float]:
        if self.subspec not in {"ss10", "ss104"}:
            msg = "Only Model1002 ss10/ss104 steady state is ported."
            raise NotPortedError(msg)

        p = self.numeric_value
        z_star = log(1.0 + p("gamma")) + p("alpha") / (1.0 - p("alpha")) * log(p("Upsilon"))
        rstar = exp(p("sigma_c") * z_star) / p("beta")
        rstarn = 100.0 * (rstar * p("pi_star") - 1.0)
        r_k_star = p("spr") * rstar * p("Upsilon") - (1.0 - p("delta"))
        wstar = (
            p("alpha") ** p("alpha")
            * (1.0 - p("alpha")) ** (1.0 - p("alpha"))
            * r_k_star ** (-p("alpha"))
            / p("Phi")
        ) ** (1.0 / (1.0 - p("alpha")))
        lstar = 1.0
        kstar = (p("alpha") / (1.0 - p("alpha"))) * wstar * lstar / r_k_star
        kbarstar = kstar * (1.0 + p("gamma")) * p("Upsilon") ** (1.0 / (1.0 - p("alpha")))
        istar = kbarstar * (
            1.0
            - (
                (1.0 - p("delta"))
                / ((1.0 + p("gamma")) * p("Upsilon") ** (1.0 / (1.0 - p("alpha"))))
            )
        )
        ystar = kstar ** p("alpha") * lstar ** (1.0 - p("alpha")) / p("Phi")
        cstar = (1.0 - p("g_star")) * ystar - istar
        wl_c = (wstar * lstar) / (cstar * p("lambda_w"))

        zomega_star = float(norm.ppf(p("Fomega")))
        sigma_omega_star = _solve_sigma_omega_star(zomega_star, p("zeta_spb"), p("spr"))
        omega_star = omega_fn(zomega_star, sigma_omega_star)
        gstar = g_fn(zomega_star, sigma_omega_star)
        gamma_star_fn = gamma_fn(zomega_star, sigma_omega_star)
        dgdomega_star = dg_domega_fn(zomega_star, sigma_omega_star)
        dgammadomega_star = dgamma_domega_fn(zomega_star)
        dgdsigma_star = dg_dsigma_fn(zomega_star, sigma_omega_star)
        d2gdomega_dsigma_star = d2g_domega_dsigma_fn(zomega_star, sigma_omega_star)
        dgammadsigma_star = dgamma_dsigma_fn(zomega_star, sigma_omega_star)
        d2gammadomega_dsigma_star = d2gamma_domega_dsigma_fn(zomega_star, sigma_omega_star)

        mu_estar = mu_fn(zomega_star, sigma_omega_star, p("spr"))
        nkstar = nk_fn(zomega_star, sigma_omega_star, p("spr"))
        rhostar = 1.0 / nkstar - 1.0
        betabar_inverse = exp((p("sigma_c") - 1.0) * z_star) / p("beta")
        wekstar = (1.0 - (p("gamma_star") * betabar_inverse)) * nkstar - p(
            "gamma_star"
        ) * betabar_inverse * (p("spr") * (1.0 - mu_estar * gstar) - 1.0)
        vkstar = (nkstar - wekstar) / p("gamma_star")
        nstar = nkstar * kbarstar
        vstar = vkstar * kbarstar

        gamma_mu_g = gamma_star_fn - mu_estar * gstar
        gamma_mu_gprime = dgammadomega_star - mu_estar * dgdomega_star
        zeta_bw = zeta_bomega_fn(zomega_star, sigma_omega_star, p("spr"))
        zeta_zw = zeta_zomega_fn(zomega_star, sigma_omega_star, p("spr"))
        zeta_bw_zw = zeta_bw / zeta_zw
        zeta_bsigma_omega = (
            sigma_omega_star
            * (
                (
                    (1.0 - mu_estar * dgdsigma_star / dgammadsigma_star)
                    / (1.0 - mu_estar * dgdomega_star / dgammadomega_star)
                    - 1.0
                )
                * dgammadsigma_star
                * p("spr")
                + mu_estar
                * nkstar
                * (
                    dgdomega_star * d2gammadomega_dsigma_star
                    - dgammadomega_star * d2gdomega_dsigma_star
                )
                / gamma_mu_gprime**2
            )
            / (
                (1.0 - gamma_star_fn) * p("spr")
                + dgammadomega_star / gamma_mu_gprime * (1.0 - nkstar)
            )
        )
        zeta_zsigma_omega = (
            sigma_omega_star * (dgammadsigma_star - mu_estar * dgdsigma_star) / gamma_mu_g
        )
        zeta_spsigma_omega = (zeta_bw_zw * zeta_zsigma_omega - zeta_bsigma_omega) / (
            1.0 - zeta_bw_zw
        )
        zeta_bmu_e = (
            -mu_estar
            * (
                nkstar * dgammadomega_star * dgdomega_star / gamma_mu_gprime
                + dgammadomega_star * gstar * p("spr")
            )
            / (
                (1.0 - gamma_star_fn) * gamma_mu_gprime * p("spr")
                + dgammadomega_star * (1.0 - nkstar)
            )
        )
        zeta_zmu_e = -mu_estar * gstar / gamma_mu_g
        zeta_spmu_e = (zeta_bw_zw * zeta_zmu_e - zeta_bmu_e) / (1.0 - zeta_bw_zw)

        rkstar = p("spr") * p("pi_star") * rstar
        zeta_gw = dgdomega_star / gstar * omega_star
        zeta_gsigma_omega = dgdsigma_star / gstar * sigma_omega_star
        zeta_nrk = (
            p("gamma_star")
            * rkstar
            / p("pi_star")
            / exp(z_star)
            * (1.0 + rhostar)
            * (1.0 - mu_estar * gstar * (1.0 - zeta_gw / zeta_zw))
        )
        zeta_nr = (
            p("gamma_star")
            * betabar_inverse
            * (1.0 + rhostar)
            * (1.0 - nkstar + mu_estar * gstar * p("spr") * zeta_gw / zeta_zw)
        )
        zeta_nqk = p("gamma_star") * rkstar / p("pi_star") / exp(z_star) * (1.0 + rhostar) * (
            1.0 - mu_estar * gstar * (1.0 + zeta_gw / zeta_zw / rhostar)
        ) - p("gamma_star") * betabar_inverse * (1.0 + rhostar)
        zeta_nn = p("gamma_star") * betabar_inverse + (
            p("gamma_star")
            * rkstar
            / p("pi_star")
            / exp(z_star)
            * (1.0 + rhostar)
            * mu_estar
            * gstar
            * zeta_gw
            / zeta_zw
            / rhostar
        )
        zeta_nmu_e = (
            p("gamma_star")
            * rkstar
            / p("pi_star")
            / exp(z_star)
            * (1.0 + rhostar)
            * mu_estar
            * gstar
            * (1.0 - zeta_gw * zeta_zmu_e / zeta_zw)
        )
        zeta_nsigma_omega = (
            p("gamma_star")
            * rkstar
            / p("pi_star")
            / exp(z_star)
            * (1.0 + rhostar)
            * mu_estar
            * gstar
            * (zeta_gsigma_omega - zeta_gw / zeta_zw * zeta_zsigma_omega)
        )

        values = {
            "z_star": z_star,
            "rstar": rstar,
            "Rstarn": rstarn,
            "r_k_star": r_k_star,
            "wstar": wstar,
            "Lstar": lstar,
            "kstar": kstar,
            "kbarstar": kbarstar,
            "istar": istar,
            "ystar": ystar,
            "cstar": cstar,
            "wl_c": wl_c,
            "nstar": nstar,
            "vstar": vstar,
            "zeta_spsigma_omega": zeta_spsigma_omega,
            "zeta_spmu_e": zeta_spmu_e,
            "zeta_nRk": zeta_nrk,
            "zeta_nR": zeta_nr,
            "zeta_nqk": zeta_nqk,
            "zeta_nn": zeta_nn,
            "zeta_nmu_e": zeta_nmu_e,
            "zeta_nsigma_omega": zeta_nsigma_omega,
        }
        for name, value in values.items():
            self.set_steady_state(name, value)
        return self.steady_state

    def steadystate(self) -> OrderedDict[str, float]:
        return self.compute_steady_state()

    def equilibrium_matrices(self) -> CanonicalSystem:
        if self.subspec not in {"ss10", "ss104"}:
            msg = "Only Model1002 ss10/ss104 equilibrium conditions are ported."
            raise NotPortedError(msg)
        return equilibrium_matrices_ss10(self)

    def measurement_matrices(self, transition: Transition) -> Measurement:
        if self.subspec not in {"ss10", "ss104"}:
            msg = "Only Model1002 ss10/ss104 measurement matrices are ported."
            raise NotPortedError(msg)
        return measurement_matrices_ss10(self, transition)

    def pseudo_measurement_matrices(self, transition: Transition) -> PseudoMeasurement:
        if self.subspec not in {"ss10", "ss104"}:
            msg = "Only Model1002 ss10/ss104 pseudo-measurement matrices are ported."
            raise NotPortedError(msg)
        return pseudo_measurement_matrices_ss10(self, transition)

    def augment_transition(self, transition: Transition) -> Transition:
        if self.subspec not in {"ss10", "ss104"}:
            msg = "Only Model1002 ss10/ss104 transition augmentation is ported."
            raise NotPortedError(msg)
        return augment_transition_ss10(self, transition)
