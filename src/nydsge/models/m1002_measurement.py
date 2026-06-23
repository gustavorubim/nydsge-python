from __future__ import annotations

from collections.abc import Mapping
from math import exp, log
from typing import Any

import numpy as np

from nydsge.models.expected_ffr import parse_expected_ffr_horizons
from nydsge.solve import Measurement, Transition


def measurement_matrices_ss10(model: Any, transition: Transition) -> Measurement:
    """Build default Model1002 ss10 measurement matrices."""
    model.steadystate()
    endo = model.indexes.endogenous_states
    endo_aug = model.indexes.endogenous_states_augmented
    exo = model.indexes.exogenous_shocks
    obs = model.indexes.observables
    n_observables = len(obs)
    n_states = len(endo) + len(endo_aug)
    n_exogenous = len(exo)
    ttt = np.asarray(transition.TTT, dtype=np.float64)
    ccc = np.asarray(transition.CCC, dtype=np.float64)
    if ttt.shape != (n_states, n_states):
        msg = f"TTT must have shape {(n_states, n_states)} for Model1002 measurement."
        raise ValueError(msg)
    if ccc.shape != (n_states,):
        msg = f"CCC must have shape {(n_states,)} for Model1002 measurement."
        raise ValueError(msg)

    zz = np.zeros((n_observables, n_states), dtype=np.float64)
    dd = np.zeros(n_observables, dtype=np.float64)
    ee = np.zeros((n_observables, n_observables), dtype=np.float64)
    qq = np.zeros((n_exogenous, n_exogenous), dtype=np.float64)
    v = model.numeric_value

    def set_zz(
        row_name: str,
        column_map: Mapping[str, int],
        column_name: str,
        value: float,
    ) -> None:
        zz[obs[row_name] - 1, column_map[column_name] - 1] = value

    def set_dd(row_name: str, value: float) -> None:
        dd[obs[row_name] - 1] = value

    def set_qq(shock_name: str, value: float) -> None:
        qq[exo[shock_name] - 1, exo[shock_name] - 1] = value

    trend_growth = 100.0 * (exp(v("z_star")) - 1.0)
    inflation_trend = 100.0 * (v("pi_star") - 1.0)

    set_zz("obs_gdp", endo, "y_t", 1.0)
    set_zz("obs_gdp", endo_aug, "y_t1", -1.0)
    set_zz("obs_gdp", endo, "z_t", 1.0)
    set_zz("obs_gdp", endo_aug, "e_gdp_t", 1.0)
    set_zz("obs_gdp", endo_aug, "e_gdp_t1", -v("me_level"))
    if model._is_setting_enabled("add_iid_cond_obs_gdp_meas_err"):
        set_zz("obs_gdp", endo_aug, "e_condgdp_t", 1.0)
    set_dd("obs_gdp", trend_growth)

    set_zz("obs_gdi", endo, "y_t", v("gamma_gdi"))
    set_zz("obs_gdi", endo_aug, "y_t1", -v("gamma_gdi"))
    set_zz("obs_gdi", endo, "z_t", v("gamma_gdi"))
    set_zz("obs_gdi", endo_aug, "e_gdi_t", 1.0)
    set_zz("obs_gdi", endo_aug, "e_gdi_t1", -v("me_level"))
    set_dd("obs_gdi", trend_growth + v("delta_gdi"))

    set_zz("obs_hours", endo, "L_t", 1.0)
    set_dd("obs_hours", v("Lmean"))

    set_zz("obs_wages", endo, "w_t", 1.0)
    set_zz("obs_wages", endo_aug, "w_t1", -1.0)
    set_zz("obs_wages", endo, "z_t", 1.0)
    set_dd("obs_wages", trend_growth)

    set_zz("obs_gdpdeflator", endo, "pi_t", v("Gamma_gdpdef"))
    set_zz("obs_gdpdeflator", endo_aug, "e_gdpdef_t", 1.0)
    set_dd("obs_gdpdeflator", inflation_trend + v("delta_gdpdef"))

    set_zz("obs_corepce", endo, "pi_t", 1.0)
    set_zz("obs_corepce", endo_aug, "e_corepce_t", 1.0)
    if model._is_setting_enabled("add_iid_cond_obs_corepce_meas_err"):
        set_zz("obs_corepce", endo_aug, "e_condcorepce_t", 1.0)
    set_dd("obs_corepce", inflation_trend)

    set_zz("obs_nominalrate", endo, "R_t", 1.0)
    set_dd("obs_nominalrate", v("Rstarn"))

    set_zz("obs_consumption", endo, "c_t", 1.0)
    set_zz("obs_consumption", endo_aug, "c_t1", -1.0)
    set_zz("obs_consumption", endo, "z_t", 1.0)
    set_dd("obs_consumption", trend_growth)

    set_zz("obs_investment", endo, "i_t", 1.0)
    set_zz("obs_investment", endo_aug, "i_t1", -1.0)
    set_zz("obs_investment", endo, "z_t", 1.0)
    set_dd("obs_investment", trend_growth)

    set_zz("obs_spread", endo, "ERktil_t", 1.0)
    set_zz("obs_spread", endo, "R_t", -1.0)
    set_dd("obs_spread", 100.0 * log(v("spr")))

    ttt10, ccc10 = _k_periods_ahead_expected_sums(ttt, ccc, 40)
    ttt10 = ttt10 / 40.0
    ccc10 = ccc10 / 40.0
    zz[obs["obs_longinflation"] - 1, :] = ttt10[endo["pi_t"] - 1, :]
    set_dd("obs_longinflation", inflation_trend + ccc10[endo["pi_t"] - 1])

    zz[obs["obs_longrate"] - 1, :] = ttt10[endo["R_t"] - 1, :]
    set_zz("obs_longrate", endo_aug, "e_lr_t", 1.0)
    set_dd("obs_longrate", v("Rstarn") + ccc10[endo["R_t"] - 1])

    iendoalpha = v("Iendoalpha")
    tfp_utilization_weight = v("alpha") / ((1.0 - v("alpha")) * (1.0 - iendoalpha) + iendoalpha)
    set_zz("obs_tfp", endo, "z_t", (1.0 - v("alpha")) * iendoalpha + (1.0 - iendoalpha))
    set_zz("obs_tfp", endo_aug, "e_tfp_t", 1.0)
    set_zz("obs_tfp", endo, "u_t", tfp_utilization_weight)
    set_zz("obs_tfp", endo_aug, "u_t1", -tfp_utilization_weight)

    if model._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs"):
        set_zz("obs_pgap", endo, "pgap_t", 1.0)
        set_zz("obs_ygap", endo, "ygap_t", 1.0)

    shock_sigmas = {
        "g_sh": "sigma_g",
        "b_sh": "sigma_b",
        "mu_sh": "sigma_mu",
        "ztil_sh": "sigma_ztil",
        "lambda_f_sh": "sigma_lambda_f",
        "lambda_w_sh": "sigma_lambda_w",
        "rm_sh": "sigma_r_m",
        "sigma_omega_sh": "sigma_sigma_omega",
        "mu_e_sh": "sigma_mu_e",
        "gamma_sh": "sigma_gamma",
        "pi_star_sh": "sigma_pi_star",
        "lr_sh": "sigma_lr",
        "zp_sh": "sigma_z_p",
        "tfp_sh": "sigma_tfp",
        "gdpdef_sh": "sigma_gdpdef",
        "corepce_sh": "sigma_corepce",
        "gdp_sh": "sigma_gdp",
        "gdi_sh": "sigma_gdi",
    }
    for shock_name, sigma_name in shock_sigmas.items():
        set_qq(shock_name, v(sigma_name) ** 2)

    if model._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs"):
        set_qq("pgap_sh", v("sigma_pgap") ** 2)
        set_qq("ygap_sh", v("sigma_ygap") ** 2)
    if model._is_setting_enabled("add_iid_cond_obs_gdp_meas_err"):
        set_qq("condgdp_sh", v("sigma_condgdp") ** 2)
    if model._is_setting_enabled("add_iid_anticipated_obs_gdp_meas_err"):
        set_qq("gdpexp_sh", v("sigma_gdpexp") ** 2)
    if model._is_setting_enabled("add_iid_cond_obs_corepce_meas_err"):
        set_qq("condcorepce_sh", v("sigma_condcorepce") ** 2)

    n_mon_anticipated_shocks = int(model.get_setting("n_mon_anticipated_shocks"))
    anticipated_rate_expectations = _one_to_k_periods_ahead_expectations(
        ttt, ccc, n_mon_anticipated_shocks
    )
    for horizon, (ttt_h, ccc_h) in enumerate(anticipated_rate_expectations, start=1):
        obs_name = f"obs_nominalrate{horizon}"
        zz[obs[obs_name] - 1, :] = ttt_h[endo["R_t"] - 1, :]
        set_dd(obs_name, v("Rstarn") + ccc_h[endo["R_t"] - 1])
        set_qq(f"rm_shl{horizon}", v(f"sigma_r_m{horizon}") ** 2)

    for horizon in _expected_ffr_horizons(model):
        if horizon <= n_mon_anticipated_shocks:
            ttt_h, ccc_h = anticipated_rate_expectations[horizon - 1]
        else:
            ttt_h, ccc_h = _k_periods_ahead_expectations(ttt, ccc, horizon)
        obs_name = f"obs_exp_nominalrate{horizon}"
        zz[obs[obs_name] - 1, :] = ttt_h[endo["R_t"] - 1, :]
        set_zz(obs_name, endo_aug, f"e_exp_rm{horizon}", 1.0)
        set_dd(obs_name, v("Rstarn") + ccc_h[endo["R_t"] - 1])
        set_qq(f"exp_rm_sh{horizon}", v(f"sigma_exp_rm{horizon}") ** 2)

    if model._is_setting_enabled("add_anticipated_obs_gdp"):
        gdp_row = zz[obs["obs_gdp"] - 1, :].copy()
        meas_err = float(model.get_setting("meas_err_anticipated_obs_gdp", 0.0))
        gdp_row[endo_aug["e_gdp_t"] - 1] = meas_err
        gdp_row[endo_aug["e_gdp_t1"] - 1] = -meas_err * v("me_level")
        for horizon in range(1, int(model.get_setting("n_anticipated_obs_gdp", 1)) + 1):
            obs_name = f"obs_gdp{horizon}"
            ttt_h, ccc_h = _k_periods_ahead_expectations(ttt, ccc, horizon)
            zz[obs[obs_name] - 1, :] = gdp_row @ ttt_h
            if model._is_setting_enabled("add_iid_anticipated_obs_gdp_meas_err"):
                set_zz(obs_name, endo_aug, "e_gdpexp_t", 1.0)
            set_dd(obs_name, trend_growth + float(gdp_row @ ccc_h))

    if not np.allclose(ccc, 0.0):
        steady_offset = np.linalg.solve(np.eye(n_states, dtype=np.float64) - ttt, ccc)
        dd += zz @ steady_offset

    return Measurement(ZZ=zz, DD=dd, QQ=qq, EE=ee)


def _one_to_k_periods_ahead_expectations(
    transition_matrix: np.ndarray,
    constant: np.ndarray,
    horizon: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        _k_periods_ahead_expectations(transition_matrix, constant, step)
        for step in range(1, horizon + 1)
    ]


def _expected_ffr_horizons(model: Any) -> tuple[int, ...]:
    return parse_expected_ffr_horizons(
        model.get_setting("expected_ffr", ()),
        model.get_setting("all_ffr_qs", ()),
    )


def _k_periods_ahead_expectations(
    transition_matrix: np.ndarray,
    constant: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_states = transition_matrix.shape[0]
    power = np.eye(n_states, dtype=np.float64)
    constant_accum = np.zeros(n_states, dtype=np.float64)
    for _ in range(horizon):
        constant_accum = transition_matrix @ constant_accum + constant
        power = transition_matrix @ power
    return power, constant_accum


def _k_periods_ahead_expected_sums(
    transition_matrix: np.ndarray,
    constant: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_states = transition_matrix.shape[0]
    power = np.eye(n_states, dtype=np.float64)
    constant_accum = np.zeros(n_states, dtype=np.float64)
    power_sum = np.zeros((n_states, n_states), dtype=np.float64)
    constant_sum = np.zeros(n_states, dtype=np.float64)
    for _ in range(horizon):
        constant_accum = transition_matrix @ constant_accum + constant
        power = transition_matrix @ power
        power_sum += power
        constant_sum += constant_accum
    return power_sum, constant_sum
