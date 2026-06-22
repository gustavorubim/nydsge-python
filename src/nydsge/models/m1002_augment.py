from __future__ import annotations

from typing import Any

import numpy as np

from nydsge.solve import Transition


def augment_transition_ss10(model: Any, transition: Transition) -> Transition:
    """Augment default Model1002 ss10 transition matrices with lag and measurement states."""
    endo = model.indexes.endogenous_states
    endo_aug = model.indexes.endogenous_states_augmented
    exo = model.indexes.exogenous_shocks
    n_endo = len(endo)
    n_exo = len(exo)
    n_aug = len(endo_aug)
    n_states = n_endo + n_aug
    ttt = np.asarray(transition.TTT, dtype=np.float64)
    rrr = np.asarray(transition.RRR, dtype=np.float64)
    ccc = np.asarray(transition.CCC, dtype=np.float64)
    if ttt.shape != (n_endo, n_endo):
        msg = f"TTT must have shape {(n_endo, n_endo)} before augmentation."
        raise ValueError(msg)
    if rrr.shape != (n_endo, n_exo):
        msg = f"RRR must have shape {(n_endo, n_exo)} before augmentation."
        raise ValueError(msg)
    if ccc.shape != (n_endo,):
        msg = f"CCC must have shape {(n_endo,)} before augmentation."
        raise ValueError(msg)

    ttt_aug = np.zeros((n_states, n_states), dtype=np.float64)
    ttt_aug[:n_endo, :n_endo] = ttt
    rrr_aug = np.zeros((n_states, n_exo), dtype=np.float64)
    rrr_aug[:n_endo, :] = rrr
    ccc_aug = np.zeros(n_states, dtype=np.float64)
    ccc_aug[:n_endo] = ccc
    v = model.numeric_value

    def row(name: str) -> int:
        return endo_aug[name] - 1

    def state(name: str) -> int:
        return endo[name] - 1

    def shock(name: str) -> int:
        return exo[name] - 1

    lag_pairs = {
        "y_t1": "y_t",
        "c_t1": "c_t",
        "i_t1": "i_t",
        "w_t1": "w_t",
        "pi_t1_dup": "pi_t",
        "L_t1": "L_t",
        "u_t1": "u_t",
    }
    for lag_name, state_name in lag_pairs.items():
        ttt_aug[row(lag_name), state(state_name)] = 1.0
    ttt_aug[row("e_gdp_t1"), row("e_gdp_t")] = 1.0
    ttt_aug[row("e_gdi_t1"), row("e_gdi_t")] = 1.0

    pi_row = state("pi_t")
    ttt_aug[row("Et_pi_t"), :n_endo] = (ttt @ ttt)[pi_row, :]
    ttt_aug[row("e_lr_t"), row("e_lr_t")] = v("rho_lr")
    ttt_aug[row("e_tfp_t"), row("e_tfp_t")] = v("rho_tfp")
    ttt_aug[row("e_gdpdef_t"), row("e_gdpdef_t")] = v("rho_gdpdef")
    ttt_aug[row("e_corepce_t"), row("e_corepce_t")] = v("rho_corepce")
    ttt_aug[row("e_gdp_t"), row("e_gdp_t")] = v("rho_gdp")
    ttt_aug[row("e_gdi_t"), row("e_gdi_t")] = v("rho_gdi")
    if model.get_setting("add_iid_cond_obs_gdp_meas_err", False):
        ttt_aug[row("e_condgdp_t"), row("e_condgdp_t")] = v("rho_condgdp")
    if model.get_setting("add_iid_anticipated_obs_gdp_meas_err", False):
        ttt_aug[row("e_gdpexp_t"), row("e_gdpexp_t")] = v("rho_gdpexp")
    if model.get_setting("add_iid_cond_obs_corepce_meas_err", False):
        ttt_aug[row("e_condcorepce_t"), row("e_condcorepce_t")] = v("rho_condcorepce")

    rrr_aug[row("Et_pi_t"), :] = (ttt @ rrr)[pi_row, :]
    rrr_aug[row("e_lr_t"), shock("lr_sh")] = 1.0
    rrr_aug[row("e_tfp_t"), shock("tfp_sh")] = 1.0
    rrr_aug[row("e_gdpdef_t"), shock("gdpdef_sh")] = 1.0
    rrr_aug[row("e_corepce_t"), shock("corepce_sh")] = 1.0
    rrr_aug[row("e_gdp_t"), shock("gdp_sh")] = 1.0
    rrr_aug[row("e_gdp_t"), shock("gdi_sh")] = v("rho_gdpvar") * v("sigma_gdp") ** 2
    rrr_aug[row("e_gdi_t"), shock("gdi_sh")] = 1.0
    for horizon in _expected_ffr_horizons(model):
        rrr_aug[row(f"e_exp_rm{horizon}"), shock(f"exp_rm_sh{horizon}")] = 1.0
    if model.get_setting("add_iid_cond_obs_gdp_meas_err", False):
        rrr_aug[row("e_condgdp_t"), shock("condgdp_sh")] = 1.0
    if model.get_setting("add_iid_anticipated_obs_gdp_meas_err", False):
        rrr_aug[row("e_gdpexp_t"), shock("gdpexp_sh")] = 1.0
    if model.get_setting("add_iid_cond_obs_corepce_meas_err", False):
        rrr_aug[row("e_condcorepce_t"), shock("condcorepce_sh")] = 1.0

    ccc_aug[row("Et_pi_t")] = (ccc + ttt @ ccc)[pi_row]

    return Transition(TTT=ttt_aug, RRR=rrr_aug, CCC=ccc_aug)


def _expected_ffr_horizons(model: Any) -> tuple[int, ...]:
    raw_horizons = model.get_setting("expected_ffr", ())
    if raw_horizons is None:
        return ()
    return tuple(sorted({int(horizon) for horizon in raw_horizons}))
