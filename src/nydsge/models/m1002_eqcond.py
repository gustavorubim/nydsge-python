from __future__ import annotations

from collections.abc import Mapping
from math import exp, log
from typing import Any

import numpy as np

from nydsge.solve import CanonicalSystem


def equilibrium_matrices_ss10(model: Any) -> CanonicalSystem:
    """Build default Model1002 ss10 canonical matrices translated from eqcond.jl."""
    model.steadystate()
    endo = model.indexes.endogenous_states
    exo = model.indexes.exogenous_shocks
    expected = model.indexes.expected_shocks
    eq = model.indexes.equilibrium_conditions
    n_states = len(endo)
    n_exogenous = len(exo)
    n_expected = len(expected)
    Gamma0 = np.zeros((n_states, n_states), dtype=np.float64)
    Gamma1 = np.zeros((n_states, n_states), dtype=np.float64)
    C = np.zeros(n_states, dtype=np.float64)
    Psi = np.zeros((n_states, n_exogenous), dtype=np.float64)
    Pi = np.zeros((n_states, n_expected), dtype=np.float64)
    v = model.numeric_value
    noant = 1.0
    nopish = 1.0

    def set_col(
        matrix: np.ndarray,
        row_name: str,
        column_map: Mapping[str, int],
        column_name: str,
        value: float,
    ) -> None:
        matrix[eq[row_name] - 1, column_map[column_name] - 1] = value

    def set_const(row_name: str, value: float) -> None:
        C[eq[row_name] - 1] = value

    set_col(Gamma0, "eq_euler", endo, "c_t", 1.0)  # eqcond.jl:52
    set_col(
        Gamma0,
        "eq_euler",
        endo,
        "R_t",
        (1 - v("h") * exp(-v("z_star"))) / (v("sigma_c") * (1 + v("h") * exp(-v("z_star")))),
    )  # eqcond.jl:53
    set_col(Gamma0, "eq_euler", endo, "b_t", -1.0)  # eqcond.jl:54
    set_col(
        Gamma0,
        "eq_euler",
        endo,
        "Epi_t",
        -(1 - v("h") * exp(-v("z_star"))) / (v("sigma_c") * (1 + v("h") * exp(-v("z_star")))),
    )  # eqcond.jl:55
    set_col(
        Gamma0,
        "eq_euler",
        endo,
        "z_t",
        (v("h") * exp(-v("z_star"))) / (1 + v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:56
    set_col(Gamma0, "eq_euler", endo, "Ec_t", -1 / (1 + v("h") * exp(-v("z_star"))))  # eqcond.jl:57
    set_col(Gamma0, "eq_euler", endo, "Ez_t", -1 / (1 + v("h") * exp(-v("z_star"))))  # eqcond.jl:58
    set_col(
        Gamma0,
        "eq_euler",
        endo,
        "L_t",
        -(v("sigma_c") - 1) * v("wl_c") / (v("sigma_c") * (1 + v("h") * exp(-v("z_star")))),
    )  # eqcond.jl:59
    set_col(
        Gamma0,
        "eq_euler",
        endo,
        "EL_t",
        (v("sigma_c") - 1) * v("wl_c") / (v("sigma_c") * (1 + v("h") * exp(-v("z_star")))),
    )  # eqcond.jl:60
    set_col(
        Gamma1,
        "eq_euler",
        endo,
        "c_t",
        (v("h") * exp(-v("z_star"))) / (1 + v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:61
    set_col(Gamma0, "eq_euler_f", endo, "c_f_t", 1.0)  # eqcond.jl:69
    set_col(
        Gamma0,
        "eq_euler_f",
        endo,
        "r_f_t",
        (1 - v("h") * exp(-v("z_star"))) / (v("sigma_c") * (1 + v("h") * exp(-v("z_star")))),
    )  # eqcond.jl:70
    set_col(Gamma0, "eq_euler_f", endo, "b_t", -1.0)  # eqcond.jl:71
    set_col(
        Gamma0,
        "eq_euler_f",
        endo,
        "z_t",
        (v("h") * exp(-v("z_star"))) / (1 + v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:72
    set_col(
        Gamma0, "eq_euler_f", endo, "Ec_f_t", -1 / (1 + v("h") * exp(-v("z_star")))
    )  # eqcond.jl:73
    set_col(
        Gamma0, "eq_euler_f", endo, "Ez_t", -1 / (1 + v("h") * exp(-v("z_star")))
    )  # eqcond.jl:74
    set_col(
        Gamma0,
        "eq_euler_f",
        endo,
        "L_f_t",
        -(v("sigma_c") - 1) * v("wl_c") / (v("sigma_c") * (1 + v("h") * exp(-v("z_star")))),
    )  # eqcond.jl:75
    set_col(
        Gamma0,
        "eq_euler_f",
        endo,
        "EL_f_t",
        (v("sigma_c") - 1) * v("wl_c") / (v("sigma_c") * (1 + v("h") * exp(-v("z_star")))),
    )  # eqcond.jl:76
    set_col(
        Gamma1,
        "eq_euler_f",
        endo,
        "c_f_t",
        (v("h") * exp(-v("z_star"))) / (1 + v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:77
    set_col(
        Gamma0,
        "eq_inv",
        endo,
        "qk_t",
        -1
        / (
            v("Spp")
            * exp(2.0 * v("z_star"))
            * (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
        ),
    )  # eqcond.jl:87
    set_col(Gamma0, "eq_inv", endo, "i_t", 1.0)  # eqcond.jl:88
    set_col(
        Gamma0, "eq_inv", endo, "z_t", 1 / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
    )  # eqcond.jl:89
    set_col(
        Gamma1, "eq_inv", endo, "i_t", 1 / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
    )  # eqcond.jl:90
    set_col(
        Gamma0,
        "eq_inv",
        endo,
        "Ei_t",
        -v("beta")
        * exp((1 - v("sigma_c")) * v("z_star"))
        / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:91
    set_col(
        Gamma0,
        "eq_inv",
        endo,
        "Ez_t",
        -v("beta")
        * exp((1 - v("sigma_c")) * v("z_star"))
        / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:92
    set_col(Gamma0, "eq_inv", endo, "mu_t", -1.0)  # eqcond.jl:93
    set_col(
        Gamma0,
        "eq_inv_f",
        endo,
        "qk_f_t",
        -1
        / (
            v("Spp")
            * exp(2 * v("z_star"))
            * (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
        ),
    )  # eqcond.jl:96
    set_col(Gamma0, "eq_inv_f", endo, "i_f_t", 1.0)  # eqcond.jl:97
    set_col(
        Gamma0, "eq_inv_f", endo, "z_t", 1 / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
    )  # eqcond.jl:98
    set_col(
        Gamma1,
        "eq_inv_f",
        endo,
        "i_f_t",
        1 / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:99
    set_col(
        Gamma0,
        "eq_inv_f",
        endo,
        "Ei_f_t",
        -v("beta")
        * exp((1 - v("sigma_c")) * v("z_star"))
        / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:100
    set_col(
        Gamma0,
        "eq_inv_f",
        endo,
        "Ez_t",
        -v("beta")
        * exp((1 - v("sigma_c")) * v("z_star"))
        / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:101
    set_col(Gamma0, "eq_inv_f", endo, "mu_t", -1.0)  # eqcond.jl:102
    set_col(Gamma0, "eq_capval", endo, "Rktil_t", 1.0)  # eqcond.jl:108
    set_col(Gamma0, "eq_capval", endo, "pi_t", -1.0)  # eqcond.jl:109
    set_col(
        Gamma0, "eq_capval", endo, "rk_t", -v("r_k_star") / (1 + v("r_k_star") - v("delta"))
    )  # eqcond.jl:110
    set_col(
        Gamma0, "eq_capval", endo, "qk_t", -(1 - v("delta")) / (1 + v("r_k_star") - v("delta"))
    )  # eqcond.jl:111
    set_col(Gamma1, "eq_capval", endo, "qk_t", -1.0)  # eqcond.jl:112
    set_col(Gamma0, "eq_spread", endo, "ERktil_t", 1.0)  # eqcond.jl:116
    set_col(Gamma0, "eq_spread", endo, "R_t", -1.0)  # eqcond.jl:117
    set_col(
        Gamma0,
        "eq_spread",
        endo,
        "b_t",
        (v("sigma_c") * (1 + v("h") * exp(-v("z_star")))) / (1 - v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:118
    set_col(Gamma0, "eq_spread", endo, "qk_t", -v("zeta_spb"))  # eqcond.jl:119
    set_col(Gamma0, "eq_spread", endo, "kbar_t", -v("zeta_spb"))  # eqcond.jl:120
    set_col(Gamma0, "eq_spread", endo, "n_t", v("zeta_spb"))  # eqcond.jl:121
    set_col(Gamma0, "eq_spread", endo, "sigma_omega_t", -1.0)  # eqcond.jl:122
    set_col(Gamma0, "eq_spread", endo, "mu_e_t", -1.0)  # eqcond.jl:123
    set_col(Gamma0, "eq_spread_f", endo, "ERktil_f_t", 1.0)  # eqcond.jl:126
    set_col(Gamma0, "eq_spread_f", endo, "r_f_t", -1.0)  # eqcond.jl:127
    set_col(
        Gamma0,
        "eq_spread_f",
        endo,
        "b_t",
        (v("sigma_c") * (1 + v("h") * exp(-v("z_star")))) / (1 - v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:128
    set_col(Gamma0, "eq_spread_f", endo, "qk_f_t", -v("zeta_spb"))  # eqcond.jl:129
    set_col(Gamma0, "eq_spread_f", endo, "kbar_f_t", -v("zeta_spb"))  # eqcond.jl:130
    set_col(Gamma0, "eq_spread_f", endo, "n_f_t", v("zeta_spb"))  # eqcond.jl:131
    set_col(Gamma0, "eq_spread_f", endo, "sigma_omega_t", -1.0)  # eqcond.jl:132
    set_col(Gamma0, "eq_spread_f", endo, "mu_e_t", -1.0)  # eqcond.jl:133
    set_col(Gamma0, "eq_nevol", endo, "n_t", 1.0)  # eqcond.jl:137
    set_col(Gamma0, "eq_nevol", endo, "gamma_t", -1.0)  # eqcond.jl:138
    set_col(
        Gamma0, "eq_nevol", endo, "z_t", v("gamma_star") * v("vstar") / v("nstar")
    )  # eqcond.jl:139
    set_col(Gamma0, "eq_nevol", endo, "Rktil_t", -v("zeta_nRk"))  # eqcond.jl:140
    set_col(Gamma0, "eq_nevol", endo, "pi_t", (v("zeta_nRk") - v("zeta_nR")))  # eqcond.jl:141
    set_col(
        Gamma1, "eq_nevol", endo, "sigma_omega_t", -v("zeta_nsigma_omega") / v("zeta_spsigma_omega")
    )  # eqcond.jl:142
    set_col(
        Gamma1, "eq_nevol", endo, "mu_e_t", -v("zeta_nmu_e") / v("zeta_spmu_e")
    )  # eqcond.jl:143
    set_col(Gamma1, "eq_nevol", endo, "qk_t", v("zeta_nqk"))  # eqcond.jl:144
    set_col(Gamma1, "eq_nevol", endo, "kbar_t", v("zeta_nqk"))  # eqcond.jl:145
    set_col(Gamma1, "eq_nevol", endo, "n_t", v("zeta_nn"))  # eqcond.jl:146
    set_col(Gamma1, "eq_nevol", endo, "R_t", -v("zeta_nR"))  # eqcond.jl:147
    set_col(
        Gamma1,
        "eq_nevol",
        endo,
        "b_t",
        v("zeta_nR")
        * (
            (v("sigma_c") * (1.0 + v("h") * exp(-v("z_star")))) / (1.0 - v("h") * exp(-v("z_star")))
        ),
    )  # eqcond.jl:148
    set_col(Gamma0, "eq_nevol_f", endo, "n_f_t", 1.0)  # eqcond.jl:151
    set_col(
        Gamma0, "eq_nevol_f", endo, "z_t", v("gamma_star") * v("vstar") / v("nstar")
    )  # eqcond.jl:152
    set_col(Gamma0, "eq_nevol_f", endo, "Rktil_f_t", -v("zeta_nRk"))  # eqcond.jl:153
    set_col(
        Gamma1,
        "eq_nevol_f",
        endo,
        "sigma_omega_t",
        -v("zeta_nsigma_omega") / v("zeta_spsigma_omega"),
    )  # eqcond.jl:154
    set_col(
        Gamma1, "eq_nevol_f", endo, "mu_e_t", -v("zeta_nmu_e") / v("zeta_spmu_e")
    )  # eqcond.jl:155
    set_col(Gamma1, "eq_nevol_f", endo, "qk_f_t", v("zeta_nqk"))  # eqcond.jl:156
    set_col(Gamma1, "eq_nevol_f", endo, "kbar_f_t", v("zeta_nqk"))  # eqcond.jl:157
    set_col(Gamma1, "eq_nevol_f", endo, "n_f_t", v("zeta_nn"))  # eqcond.jl:158
    set_col(Gamma1, "eq_nevol_f", endo, "r_f_t", -v("zeta_nR"))  # eqcond.jl:159
    set_col(
        Gamma1,
        "eq_nevol_f",
        endo,
        "b_t",
        v("zeta_nR")
        * (
            (v("sigma_c") * (1.0 + v("h") * exp(-v("z_star")))) / (1.0 - v("h") * exp(-v("z_star")))
        ),
    )  # eqcond.jl:160
    set_col(Gamma0, "eq_capval_f", endo, "Rktil_f_t", 1.0)  # eqcond.jl:163
    set_col(
        Gamma0, "eq_capval_f", endo, "rk_f_t", -v("r_k_star") / (v("r_k_star") + 1 - v("delta"))
    )  # eqcond.jl:164
    set_col(
        Gamma0, "eq_capval_f", endo, "qk_f_t", -(1 - v("delta")) / (v("r_k_star") + 1 - v("delta"))
    )  # eqcond.jl:165
    set_col(Gamma1, "eq_capval_f", endo, "qk_f_t", -1.0)  # eqcond.jl:166
    set_col(Gamma0, "eq_output", endo, "y_t", 1.0)  # eqcond.jl:171
    set_col(Gamma0, "eq_output", endo, "k_t", -v("Phi") * v("alpha"))  # eqcond.jl:172
    set_col(Gamma0, "eq_output", endo, "L_t", -v("Phi") * (1 - v("alpha")))  # eqcond.jl:173
    set_col(Gamma0, "eq_output_f", endo, "y_f_t", 1.0)  # eqcond.jl:176
    set_col(Gamma0, "eq_output_f", endo, "k_f_t", -v("Phi") * v("alpha"))  # eqcond.jl:177
    set_col(Gamma0, "eq_output_f", endo, "L_f_t", -v("Phi") * (1 - v("alpha")))  # eqcond.jl:178
    set_col(Gamma0, "eq_caputl", endo, "k_t", 1.0)  # eqcond.jl:183
    set_col(Gamma1, "eq_caputl", endo, "kbar_t", 1.0)  # eqcond.jl:184
    set_col(Gamma0, "eq_caputl", endo, "z_t", 1.0)  # eqcond.jl:185
    set_col(Gamma0, "eq_caputl", endo, "u_t", -1.0)  # eqcond.jl:186
    set_col(Gamma0, "eq_caputl_f", endo, "k_f_t", 1.0)  # eqcond.jl:189
    set_col(Gamma1, "eq_caputl_f", endo, "kbar_f_t", 1.0)  # eqcond.jl:190
    set_col(Gamma0, "eq_caputl_f", endo, "z_t", 1.0)  # eqcond.jl:191
    set_col(Gamma0, "eq_caputl_f", endo, "u_f_t", -1.0)  # eqcond.jl:192
    set_col(Gamma0, "eq_capsrv", endo, "u_t", 1.0)  # eqcond.jl:197
    set_col(Gamma0, "eq_capsrv", endo, "rk_t", -(1 - v("ppsi")) / v("ppsi"))  # eqcond.jl:198
    set_col(Gamma0, "eq_capsrv_f", endo, "u_f_t", 1.0)  # eqcond.jl:201
    set_col(Gamma0, "eq_capsrv_f", endo, "rk_f_t", -(1 - v("ppsi")) / v("ppsi"))  # eqcond.jl:202
    set_col(Gamma0, "eq_capev", endo, "kbar_t", 1.0)  # eqcond.jl:207
    set_col(Gamma1, "eq_capev", endo, "kbar_t", 1 - v("istar") / v("kbarstar"))  # eqcond.jl:208
    set_col(Gamma0, "eq_capev", endo, "z_t", 1 - v("istar") / v("kbarstar"))  # eqcond.jl:209
    set_col(Gamma0, "eq_capev", endo, "i_t", -v("istar") / v("kbarstar"))  # eqcond.jl:210
    set_col(
        Gamma0,
        "eq_capev",
        endo,
        "mu_t",
        -v("istar")
        * v("Spp")
        * exp(2 * v("z_star"))
        * (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
        / v("kbarstar"),
    )  # eqcond.jl:211
    set_col(Gamma0, "eq_capev_f", endo, "kbar_f_t", 1.0)  # eqcond.jl:214
    set_col(Gamma1, "eq_capev_f", endo, "kbar_f_t", 1 - v("istar") / v("kbarstar"))  # eqcond.jl:215
    set_col(Gamma0, "eq_capev_f", endo, "z_t", 1 - v("istar") / v("kbarstar"))  # eqcond.jl:216
    set_col(Gamma0, "eq_capev_f", endo, "i_f_t", -v("istar") / v("kbarstar"))  # eqcond.jl:217
    set_col(
        Gamma0,
        "eq_capev_f",
        endo,
        "mu_t",
        -v("istar")
        * v("Spp")
        * exp(2 * v("z_star"))
        * (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
        / v("kbarstar"),
    )  # eqcond.jl:218
    set_col(Gamma0, "eq_mkupp", endo, "mc_t", 1.0)  # eqcond.jl:223
    set_col(Gamma0, "eq_mkupp", endo, "w_t", -1.0)  # eqcond.jl:224
    set_col(Gamma0, "eq_mkupp", endo, "L_t", -v("alpha"))  # eqcond.jl:225
    set_col(Gamma0, "eq_mkupp", endo, "k_t", v("alpha"))  # eqcond.jl:226
    set_col(Gamma0, "eq_mkupp_f", endo, "w_f_t", 1.0)  # eqcond.jl:229
    set_col(Gamma0, "eq_mkupp_f", endo, "L_f_t", v("alpha"))  # eqcond.jl:230
    set_col(Gamma0, "eq_mkupp_f", endo, "k_f_t", -v("alpha"))  # eqcond.jl:231
    set_col(Gamma0, "eq_phlps", endo, "pi_t", 1.0)  # eqcond.jl:236
    set_col(
        Gamma0,
        "eq_phlps",
        endo,
        "mc_t",
        -((1 - v("zeta_p") * v("beta") * exp((1 - v("sigma_c")) * v("z_star"))) * (1 - v("zeta_p")))
        / (v("zeta_p") * ((v("Phi") - 1) * v("epsilon_p") + 1))
        / (1 + v("iota_p") * v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:237
    set_col(
        Gamma1,
        "eq_phlps",
        endo,
        "pi_t",
        v("iota_p") / (1 + v("iota_p") * v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:239
    set_col(
        Gamma0,
        "eq_phlps",
        endo,
        "Epi_t",
        -v("beta")
        * exp((1 - v("sigma_c")) * v("z_star"))
        / (1 + v("iota_p") * v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:240
    set_col(Gamma0, "eq_phlps", endo, "lambda_f_t", -1.0)  # eqcond.jl:244
    set_col(Gamma0, "eq_caprnt", endo, "rk_t", 1.0)  # eqcond.jl:251
    set_col(Gamma0, "eq_caprnt", endo, "k_t", 1.0)  # eqcond.jl:252
    set_col(Gamma0, "eq_caprnt", endo, "L_t", -1.0)  # eqcond.jl:253
    set_col(Gamma0, "eq_caprnt", endo, "w_t", -1.0)  # eqcond.jl:254
    set_col(Gamma0, "eq_caprnt_f", endo, "rk_f_t", 1.0)  # eqcond.jl:257
    set_col(Gamma0, "eq_caprnt_f", endo, "k_f_t", 1.0)  # eqcond.jl:258
    set_col(Gamma0, "eq_caprnt_f", endo, "L_f_t", -1.0)  # eqcond.jl:259
    set_col(Gamma0, "eq_caprnt_f", endo, "w_f_t", -1.0)  # eqcond.jl:260
    set_col(Gamma0, "eq_msub", endo, "mu_omega_t", 1.0)  # eqcond.jl:265
    set_col(Gamma0, "eq_msub", endo, "L_t", v("nu_l"))  # eqcond.jl:266
    set_col(Gamma0, "eq_msub", endo, "c_t", 1 / (1 - v("h") * exp(-v("z_star"))))  # eqcond.jl:267
    set_col(
        Gamma1,
        "eq_msub",
        endo,
        "c_t",
        v("h") * exp(-v("z_star")) / (1 - v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:268
    set_col(
        Gamma0,
        "eq_msub",
        endo,
        "z_t",
        v("h") * exp(-v("z_star")) / (1 - v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:269
    set_col(Gamma0, "eq_msub", endo, "w_t", -1.0)  # eqcond.jl:270
    set_col(Gamma0, "eq_msub_f", endo, "w_f_t", -1.0)  # eqcond.jl:277
    set_col(Gamma0, "eq_msub_f", endo, "L_f_t", v("nu_l"))  # eqcond.jl:278
    set_col(
        Gamma0, "eq_msub_f", endo, "c_f_t", 1 / (1 - v("h") * exp(-v("z_star")))
    )  # eqcond.jl:279
    set_col(
        Gamma1,
        "eq_msub_f",
        endo,
        "c_f_t",
        v("h") * exp(-v("z_star")) / (1 - v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:280
    set_col(
        Gamma0,
        "eq_msub_f",
        endo,
        "z_t",
        v("h") * exp(-v("z_star")) / (1 - v("h") * exp(-v("z_star"))),
    )  # eqcond.jl:281
    set_col(Gamma0, "eq_wage", endo, "w_t", 1)  # eqcond.jl:290
    set_col(
        Gamma0,
        "eq_wage",
        endo,
        "mu_omega_t",
        (1 - v("zeta_w") * v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
        * (1 - v("zeta_w"))
        / (v("zeta_w") * ((v("lambda_w") - 1) * v("epsilon_w") + 1))
        / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:291
    set_col(
        Gamma0,
        "eq_wage",
        endo,
        "pi_t",
        (1 + v("iota_w") * v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
        / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:293
    set_col(
        Gamma1, "eq_wage", endo, "w_t", 1 / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
    )  # eqcond.jl:294
    set_col(
        Gamma0, "eq_wage", endo, "z_t", 1 / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star")))
    )  # eqcond.jl:295
    set_col(
        Gamma1,
        "eq_wage",
        endo,
        "pi_t",
        v("iota_w") / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:296
    set_col(
        Gamma0,
        "eq_wage",
        endo,
        "Ew_t",
        -v("beta")
        * exp((1 - v("sigma_c")) * v("z_star"))
        / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:297
    set_col(
        Gamma0,
        "eq_wage",
        endo,
        "Ez_t",
        -v("beta")
        * exp((1 - v("sigma_c")) * v("z_star"))
        / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:298
    set_col(
        Gamma0,
        "eq_wage",
        endo,
        "Epi_t",
        -v("beta")
        * exp((1 - v("sigma_c")) * v("z_star"))
        / (1 + v("beta") * exp((1 - v("sigma_c")) * v("z_star"))),
    )  # eqcond.jl:299
    set_col(Gamma0, "eq_wage", endo, "lambda_w_t", -1.0)  # eqcond.jl:300
    set_col(Gamma0, "eq_mp", endo, "R_t", 1.0)  # eqcond.jl:306
    set_col(Gamma1, "eq_mp", endo, "R_t", v("rho"))  # eqcond.jl:307
    set_col(Gamma0, "eq_mp", endo, "pi_t", -(1 - v("rho")) * v("psi1"))  # eqcond.jl:308
    set_col(Gamma0, "eq_mp", endo, "pi_star_t", (1 - v("rho")) * v("psi1"))  # eqcond.jl:309
    set_col(Gamma0, "eq_mp", endo, "y_t", -(1 - v("rho")) * v("psi2") - v("psi3"))  # eqcond.jl:310
    set_col(Gamma0, "eq_mp", endo, "y_f_t", (1 - v("rho")) * v("psi2") + v("psi3"))  # eqcond.jl:311
    set_col(Gamma1, "eq_mp", endo, "y_t", -v("psi3"))  # eqcond.jl:312
    set_col(Gamma1, "eq_mp", endo, "y_f_t", v("psi3"))  # eqcond.jl:313
    set_col(Gamma0, "eq_mp", endo, "rm_t", -1.0)  # eqcond.jl:314
    set_col(Gamma0, "eq_res", endo, "y_t", 1.0)  # eqcond.jl:321
    set_col(Gamma0, "eq_res", endo, "g_t", -v("g_star"))  # eqcond.jl:322
    set_col(Gamma0, "eq_res", endo, "c_t", -v("cstar") / v("ystar"))  # eqcond.jl:323
    set_col(Gamma0, "eq_res", endo, "i_t", -v("istar") / v("ystar"))  # eqcond.jl:324
    set_col(
        Gamma0, "eq_res", endo, "u_t", -v("r_k_star") * v("kstar") / v("ystar")
    )  # eqcond.jl:325
    set_col(Gamma0, "eq_res_f", endo, "y_f_t", 1.0)  # eqcond.jl:328
    set_col(Gamma0, "eq_res_f", endo, "g_t", -v("g_star"))  # eqcond.jl:329
    set_col(Gamma0, "eq_res_f", endo, "c_f_t", -v("cstar") / v("ystar"))  # eqcond.jl:330
    set_col(Gamma0, "eq_res_f", endo, "i_f_t", -v("istar") / v("ystar"))  # eqcond.jl:331
    set_col(
        Gamma0, "eq_res_f", endo, "u_f_t", -v("r_k_star") * v("kstar") / v("ystar")
    )  # eqcond.jl:332
    set_col(Gamma0, "eq_pi1", endo, "pi_t1", 1.0)  # eqcond.jl:338
    set_col(Gamma1, "eq_pi1", endo, "pi_t", 1.0)  # eqcond.jl:339
    set_col(Gamma0, "eq_pi2", endo, "pi_t2", 1.0)  # eqcond.jl:342
    set_col(Gamma1, "eq_pi2", endo, "pi_t1", 1.0)  # eqcond.jl:343
    set_col(Gamma0, "eq_pi_a", endo, "pi_a_t", 1.0)  # eqcond.jl:346
    set_col(Gamma0, "eq_pi_a", endo, "pi_t", -1.0)  # eqcond.jl:347
    set_col(Gamma0, "eq_pi_a", endo, "pi_t1", -1.0)  # eqcond.jl:348
    set_col(Gamma0, "eq_pi_a", endo, "pi_t2", -1.0)  # eqcond.jl:349
    set_col(Gamma1, "eq_pi_a", endo, "pi_t2", 1.0)  # eqcond.jl:350
    set_col(Gamma0, "eq_Rt1", endo, "R_t1", 1.0)  # eqcond.jl:353
    set_col(Gamma1, "eq_Rt1", endo, "R_t", 1.0)  # eqcond.jl:354
    set_col(Gamma0, "eq_Ez", endo, "Ez_t", 1.0)  # eqcond.jl:357
    set_col(
        Gamma0, "eq_Ez", endo, "ztil_t", -(v("rho_ztil") - 1) / (1 - v("alpha"))
    )  # eqcond.jl:358
    set_col(Gamma0, "eq_Ez", endo, "zp_t", -v("rho_z_p"))  # eqcond.jl:359
    set_col(Gamma0, "eq_z", endo, "z_t", 1.0)  # eqcond.jl:370
    set_col(Gamma1, "eq_z", endo, "ztil_t", (v("rho_ztil") - 1) / (1 - v("alpha")))  # eqcond.jl:371
    set_col(Gamma0, "eq_z", endo, "zp_t", -1.0)  # eqcond.jl:372
    set_col(Psi, "eq_z", exo, "ztil_sh", 1 / (1 - v("alpha")))  # eqcond.jl:373
    set_col(Gamma0, "eq_ztil", endo, "ztil_t", 1.0)  # eqcond.jl:375
    set_col(Gamma1, "eq_ztil", endo, "ztil_t", v("rho_ztil"))  # eqcond.jl:376
    set_col(Psi, "eq_ztil", exo, "ztil_sh", 1.0)  # eqcond.jl:377
    set_col(Gamma0, "eq_zp", endo, "zp_t", 1.0)  # eqcond.jl:394
    set_col(Gamma1, "eq_zp", endo, "zp_t", v("rho_z_p"))  # eqcond.jl:395
    set_col(Psi, "eq_zp", exo, "zp_sh", 1.0)  # eqcond.jl:396
    set_col(Gamma0, "eq_g", endo, "g_t", 1.0)  # eqcond.jl:399
    set_col(Gamma1, "eq_g", endo, "g_t", v("rho_g"))  # eqcond.jl:400
    set_col(Psi, "eq_g", exo, "g_sh", 1.0)  # eqcond.jl:401
    set_col(Psi, "eq_g", exo, "ztil_sh", v("eta_gz"))  # eqcond.jl:402
    set_col(Gamma0, "eq_b", endo, "b_t", 1.0)  # eqcond.jl:405
    set_col(Gamma1, "eq_b", endo, "b_t", v("rho_b"))  # eqcond.jl:406
    set_col(Psi, "eq_b", exo, "b_sh", 1.0)  # eqcond.jl:407
    set_col(Gamma0, "eq_mu", endo, "mu_t", 1.0)  # eqcond.jl:426
    set_col(Gamma1, "eq_mu", endo, "mu_t", v("rho_mu"))  # eqcond.jl:427
    set_col(Psi, "eq_mu", exo, "mu_sh", 1.0)  # eqcond.jl:428
    set_col(Gamma0, "eq_lambda_f", endo, "lambda_f_t", 1.0)  # eqcond.jl:452
    set_col(Gamma1, "eq_lambda_f", endo, "lambda_f_t", v("rho_lambda_f"))  # eqcond.jl:453
    set_col(Gamma1, "eq_lambda_f", endo, "lambda_f_t1", -v("eta_lambda_f"))  # eqcond.jl:454
    set_col(Psi, "eq_lambda_f", exo, "lambda_f_sh", 1.0)  # eqcond.jl:455
    set_col(Gamma0, "eq_lambda_f1", endo, "lambda_f_t1", 1.0)  # eqcond.jl:458
    set_col(Psi, "eq_lambda_f1", exo, "lambda_f_sh", 1.0)  # eqcond.jl:459
    set_col(Gamma0, "eq_lambda_w", endo, "lambda_w_t", 1.0)  # eqcond.jl:462
    set_col(Gamma1, "eq_lambda_w", endo, "lambda_w_t", v("rho_lambda_w"))  # eqcond.jl:463
    set_col(Gamma1, "eq_lambda_w", endo, "lambda_w_t1", -v("eta_lambda_w"))  # eqcond.jl:464
    set_col(Psi, "eq_lambda_w", exo, "lambda_w_sh", 1.0)  # eqcond.jl:465
    set_col(Gamma0, "eq_lambda_w1", endo, "lambda_w_t1", 1.0)  # eqcond.jl:467
    set_col(Psi, "eq_lambda_w1", exo, "lambda_w_sh", 1.0)  # eqcond.jl:468
    set_col(Gamma0, "eq_rm", endo, "rm_t", 1.0)  # eqcond.jl:474
    set_col(Gamma1, "eq_rm", endo, "rm_t", v("rho_rm"))  # eqcond.jl:475
    set_col(Psi, "eq_rm", exo, "rm_sh", noant)  # eqcond.jl:476
    set_col(Gamma0, "eq_sigma_omega", endo, "sigma_omega_t", 1.0)  # eqcond.jl:524
    set_col(Gamma1, "eq_sigma_omega", endo, "sigma_omega_t", v("rho_sigma_w"))  # eqcond.jl:525
    set_col(Psi, "eq_sigma_omega", exo, "sigma_omega_sh", 1.0)  # eqcond.jl:526
    set_col(Gamma0, "eq_mu_e", endo, "mu_e_t", 1.0)  # eqcond.jl:529
    set_col(Gamma1, "eq_mu_e", endo, "mu_e_t", v("rho_mu_e"))  # eqcond.jl:530
    set_col(Psi, "eq_mu_e", exo, "mu_e_sh", 1.0)  # eqcond.jl:531
    set_col(Gamma0, "eq_gamma", endo, "gamma_t", 1.0)  # eqcond.jl:534
    set_col(Gamma1, "eq_gamma", endo, "gamma_t", v("rho_gamma"))  # eqcond.jl:535
    set_col(Psi, "eq_gamma", exo, "gamma_sh", 1.0)  # eqcond.jl:536
    set_col(Gamma0, "eq_pi_star", endo, "pi_star_t", 1.0)  # eqcond.jl:542
    set_col(Gamma1, "eq_pi_star", endo, "pi_star_t", v("rho_pi_star"))  # eqcond.jl:543
    set_col(Psi, "eq_pi_star", exo, "pi_star_sh", nopish)  # eqcond.jl:544
    set_col(Gamma0, "eq_Ec", endo, "c_t", 1.0)  # eqcond.jl:649
    set_col(Gamma1, "eq_Ec", endo, "Ec_t", 1.0)  # eqcond.jl:650
    set_col(Pi, "eq_Ec", expected, "Ec_sh", 1.0)  # eqcond.jl:651
    set_col(Gamma0, "eq_Ec_f", endo, "c_f_t", 1.0)  # eqcond.jl:654
    set_col(Gamma1, "eq_Ec_f", endo, "Ec_f_t", 1.0)  # eqcond.jl:655
    set_col(Pi, "eq_Ec_f", expected, "Ec_f_sh", 1.0)  # eqcond.jl:656
    set_col(Gamma0, "eq_Eqk", endo, "qk_t", 1.0)  # eqcond.jl:661
    set_col(Gamma1, "eq_Eqk", endo, "Eqk_t", 1.0)  # eqcond.jl:662
    set_col(Pi, "eq_Eqk", expected, "Eqk_sh", 1.0)  # eqcond.jl:663
    set_col(Gamma0, "eq_Eqk_f", endo, "qk_f_t", 1.0)  # eqcond.jl:666
    set_col(Gamma1, "eq_Eqk_f", endo, "Eqk_f_t", 1.0)  # eqcond.jl:667
    set_col(Pi, "eq_Eqk_f", expected, "Eqk_f_sh", 1.0)  # eqcond.jl:668
    set_col(Gamma0, "eq_Ei", endo, "i_t", 1.0)  # eqcond.jl:673
    set_col(Gamma1, "eq_Ei", endo, "Ei_t", 1.0)  # eqcond.jl:674
    set_col(Pi, "eq_Ei", expected, "Ei_sh", 1.0)  # eqcond.jl:675
    set_col(Gamma0, "eq_Ei_f", endo, "i_f_t", 1.0)  # eqcond.jl:678
    set_col(Gamma1, "eq_Ei_f", endo, "Ei_f_t", 1.0)  # eqcond.jl:679
    set_col(Pi, "eq_Ei_f", expected, "Ei_f_sh", 1.0)  # eqcond.jl:680
    set_col(Gamma0, "eq_Epi", endo, "pi_t", 1.0)  # eqcond.jl:685
    set_col(Gamma1, "eq_Epi", endo, "Epi_t", 1.0)  # eqcond.jl:686
    set_col(Pi, "eq_Epi", expected, "Epi_sh", 1.0)  # eqcond.jl:687
    set_col(Gamma0, "eq_EL", endo, "L_t", 1.0)  # eqcond.jl:692
    set_col(Gamma1, "eq_EL", endo, "EL_t", 1.0)  # eqcond.jl:693
    set_col(Pi, "eq_EL", expected, "EL_sh", 1.0)  # eqcond.jl:694
    set_col(Gamma0, "eq_EL_f", endo, "L_f_t", 1.0)  # eqcond.jl:697
    set_col(Gamma1, "eq_EL_f", endo, "EL_f_t", 1.0)  # eqcond.jl:698
    set_col(Pi, "eq_EL_f", expected, "EL_f_sh", 1.0)  # eqcond.jl:699
    set_col(Gamma0, "eq_Erk", endo, "rk_t", 1.0)  # eqcond.jl:704
    set_col(Gamma1, "eq_Erk", endo, "Erk_t", 1.0)  # eqcond.jl:705
    set_col(Pi, "eq_Erk", expected, "Erk_sh", 1.0)  # eqcond.jl:706
    set_col(Gamma0, "eq_ERktil_f", endo, "Rktil_f_t", 1.0)  # eqcond.jl:709
    set_col(Gamma1, "eq_ERktil_f", endo, "ERktil_f_t", 1.0)  # eqcond.jl:710
    set_col(Pi, "eq_ERktil_f", expected, "ERktil_f_sh", 1.0)  # eqcond.jl:711
    set_col(Gamma0, "eq_Ew", endo, "w_t", 1.0)  # eqcond.jl:716
    set_col(Gamma1, "eq_Ew", endo, "Ew_t", 1.0)  # eqcond.jl:717
    set_col(Pi, "eq_Ew", expected, "Ew_sh", 1.0)  # eqcond.jl:718
    set_col(Gamma0, "eq_ERktil", endo, "Rktil_t", 1.0)  # eqcond.jl:723
    set_col(Gamma1, "eq_ERktil", endo, "ERktil_t", 1.0)  # eqcond.jl:724
    set_col(Pi, "eq_ERktil", expected, "ERktil_sh", 1.0)  # eqcond.jl:725

    n_mon_anticipated_shocks = int(model.get_setting("n_mon_anticipated_shocks"))
    if n_mon_anticipated_shocks > 0:
        set_col(Gamma1, "eq_rm", endo, "rm_tl1", noant)
        set_col(Gamma0, "eq_rml1", endo, "rm_tl1", 1.0)
        set_col(Psi, "eq_rml1", exo, "rm_shl1", noant)
        for index in range(2, n_mon_anticipated_shocks + 1):
            set_col(Gamma1, f"eq_rml{index - 1}", endo, f"rm_tl{index}", noant)
            set_col(Gamma0, f"eq_rml{index}", endo, f"rm_tl{index}", 1.0)
            set_col(Psi, f"eq_rml{index}", exo, f"rm_shl{index}", noant)

    if model._is_setting_enabled("add_altpolicy_pgap"):
        set_col(Gamma0, "eq_pgap", endo, "pgap_t", 1.0)
        rho_pgap = None
        pgap_type = model.get_setting("pgap_type", None)
        if isinstance(pgap_type, str):
            pgap_type = pgap_type.lower()
        if model._is_regime_one_active() and pgap_type is not None:
            if pgap_type == "ngdp":
                set_col(Gamma0, "eq_pgap", endo, "pi_t", -1.0)
                set_col(Gamma0, "eq_pgap", endo, "y_t", -1.0)
                set_col(Gamma0, "eq_pgap", endo, "z_t", -1.0)
                set_col(Gamma1, "eq_pgap", endo, "pgap_t", 1.0)
                set_col(Gamma1, "eq_pgap", endo, "y_t", -1.0)
            elif pgap_type == "ait":
                thalf = float(model.get_setting("ait_Thalf", 10))
                rho_pgap = exp(log(0.5) / thalf)
                set_col(Gamma0, "eq_pgap", endo, "pi_t", -1.0)
                set_col(Gamma1, "eq_pgap", endo, "pgap_t", rho_pgap)
            elif pgap_type in [
                "smooth_ait",
                "smooth_ait_gdp",
                "smooth_ait_gdp_alt",
                "flexible_ait",
                "rw",
            ]:
                thalf = float(model.get_setting("ait_Thalf", 10))
                rho_pgap = exp(log(0.5) / thalf)
                set_col(Gamma0, "eq_pgap", endo, "pi_t", -1.0)
                set_col(Gamma1, "eq_pgap", endo, "pgap_t", rho_pgap)

            if rho_pgap is not None:
                set_pgap1 = model.get_setting("set_pgap1", None)
                if isinstance(set_pgap1, (list, tuple)) and len(set_pgap1) == 2:
                    regime_list, pgap_level = set_pgap1
                    applies_to_regime = model._is_regime_selector_active(regime_list)
                    if applies_to_regime:
                        C[eq["eq_pgap"] - 1] = rho_pgap * float(pgap_level)
                        set_col(Gamma1, "eq_pgap", endo, "pgap_t", 0.0)

        if model._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs"):
            set_col(Psi, "eq_pgap", exo, "pgap_sh", 1.0)

    if (
        model._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs")
        and not model._is_setting_enabled("add_altpolicy_pgap")
        and not model._is_setting_enabled("add_pgap")
    ):
        set_col(Gamma0, "eq_pgap", endo, "pgap_t", 1.0)
        set_col(Psi, "eq_pgap", exo, "pgap_sh", 1.0)

    if model._is_setting_enabled("add_altpolicy_ygap"):
        set_col(Gamma0, "eq_ygap", endo, "ygap_t", 1.0)
        ygap_type = model.get_setting("ygap_type", None)
        if isinstance(ygap_type, str):
            ygap_type = ygap_type.lower()
        if model._is_regime_one_active() and ygap_type in [
            "smooth_ait",
            "smooth_ait_gdp",
            "smooth_ait_gdp_alt",
            "flexible_ait",
            "rw",
        ]:
            thalf = float(model.get_setting("gdp_Thalf", 10))
            rho_ygap = exp(log(0.5) / thalf)
            set_col(Gamma1, "eq_ygap", endo, "ygap_t", rho_ygap)
            set_col(Gamma0, "eq_ygap", endo, "y_t", -1.0)
            set_col(Gamma0, "eq_ygap", endo, "z_t", -1.0)
            set_col(Gamma1, "eq_ygap", endo, "y_t", -rho_ygap)

        if model._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs"):
            set_col(Psi, "eq_ygap", exo, "ygap_sh", 1.0)

    if (
        model._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs")
        and not model._is_setting_enabled("add_altpolicy_ygap")
        and not model._is_setting_enabled("add_ygap")
    ):
        set_col(Gamma0, "eq_ygap", endo, "ygap_t", 1.0)
        set_col(Psi, "eq_ygap", exo, "ygap_sh", 1.0)

    if model._is_setting_enabled("add_rw"):
        set_col(Gamma0, "eq_rw", endo, "rw_t", 1.0)
        set_col(Gamma0, "eq_Rref", endo, "Rref_t", 1.0)
        if model._is_regime_one_active() and model.get_setting("Rref_type", None) is not None:
            rho_rw = float(model.get_setting("rho_rw", 0.93))
            set_col(Gamma1, "eq_rw", endo, "rw_t", rho_rw)
            rref_type = model.get_setting("Rref_type", None)
            if isinstance(rref_type, str):
                rref_type = rref_type.lower()
            if rref_type == "ait":
                thalf = float(model.get_setting("ait_Thalf", 10))
                rho_pgap = exp(log(0.5) / thalf)
                phi = float(
                    model.get_setting(
                        "ait_phi",
                        model.get_setting("ait_phi", 0.25),
                    )
                )
                set_col(Gamma0, "eq_Rref", endo, "Rref_t", 1.0)
                set_col(Gamma1, "eq_Rref", endo, "Rref_t", 0.0)
                C[eq["eq_Rref"] - 1] = 0.0
                set_col(Gamma0, "eq_Rref", endo, "rw_t", -1.0)
                if "pgap_t" in endo:
                    set_col(
                        Gamma0,
                        "eq_Rref",
                        endo,
                        "pgap_t",
                        -phi * (1.0 / (1.0 - rho_pgap)),
                    )
            elif rref_type in [
                "smooth_ait",
                "smooth_ait_gdp",
                "smooth_ait_gdp_alt",
                "flexible_ait",
                "rw",
            ]:
                ait_thalf = float(model.get_setting("ait_Thalf", 10))
                gdp_thalf = float(model.get_setting("gdp_Thalf", 10))
                rho_pgap = exp(log(0.5) / ait_thalf)
                rho_ygap = exp(log(0.5) / gdp_thalf)
                rho_smooth = float(
                    model.get_setting(
                        "rw_rho_smooth",
                        model.get_setting("rw_rho_smooth", 0.656),
                    )
                )
                phi_pi = float(
                    model.get_setting(
                        "rw_phi_pi",
                        model.get_setting("rw_phi_pi", 11.13),
                    )
                )
                phi_y = float(
                    model.get_setting(
                        "rw_phi_y",
                        model.get_setting("rw_phi_y", 11.13),
                    )
                )
                set_col(Gamma0, "eq_Rref", endo, "Rref_t", 1.0)
                set_col(Gamma1, "eq_Rref", endo, "Rref_t", rho_smooth)
                C[eq["eq_Rref"] - 1] = 0.0
                if "pgap_t" in endo:
                    set_col(
                        Gamma0,
                        "eq_Rref",
                        endo,
                        "pgap_t",
                        -phi_pi * (1.0 - rho_pgap) * (1.0 - rho_smooth),
                    )
                if "ygap_t" in endo:
                    set_col(
                        Gamma0,
                        "eq_Rref",
                        endo,
                        "ygap_t",
                        -phi_y * (1.0 - rho_ygap) * (1.0 - rho_smooth),
                    )

    if model._is_setting_enabled("add_pgap"):
        pgap_type = model.get_setting("pgap_type", None)
        if isinstance(pgap_type, str):
            pgap_type = pgap_type.lower()
        if pgap_type in [
            "smooth_ait_gdp",
            "smooth_ait",
            "ait",
            "smooth_ait_gdp_alt",
            "flexible_ait",
            "rw",
        ]:
            thalf = float(model.get_setting("ait_Thalf", 10))
            rho_pgap = exp(log(0.5) / thalf)
            set_col(Gamma0, "eq_pgap", endo, "pgap_t", 1.0)
            set_col(Gamma0, "eq_pgap", endo, "pi_t", -1.0)
            set_col(Gamma1, "eq_pgap", endo, "pgap_t", rho_pgap)
            if model._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs"):
                set_col(Psi, "eq_pgap", exo, "pgap_sh", 1.0)

    if model._is_setting_enabled("add_ygap"):
        ygap_type = model.get_setting("ygap_type", None)
        if isinstance(ygap_type, str):
            ygap_type = ygap_type.lower()
        if ygap_type in [
            "smooth_ait",
            "smooth_ait_gdp",
            "smooth_ait_gdp_alt",
            "flexible_ait",
            "rw",
        ]:
            thalf = float(model.get_setting("gdp_Thalf", 10))
            rho_ygap = exp(log(0.5) / thalf)
            set_col(Gamma0, "eq_ygap", endo, "ygap_t", 1.0)
            set_col(Gamma0, "eq_ygap", endo, "y_t", -1.0)
            set_col(Gamma0, "eq_ygap", endo, "z_t", -1.0)
            set_col(Gamma1, "eq_ygap", endo, "ygap_t", rho_ygap)
            set_col(Gamma1, "eq_ygap", endo, "y_t", -rho_ygap)
            if model._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs"):
                set_col(Psi, "eq_ygap", exo, "ygap_sh", 1.0)

    if model._is_setting_enabled("add_ait_rm"):
        set_col(Gamma0, "eq_ait_rm", endo, "ait_rm_t", 1.0)
        set_col(
            Gamma1,
            "eq_ait_rm",
            endo,
            "ait_rm_t",
            float(v("rho_ait_rm")),
        )
        set_col(Psi, "eq_ait_rm", exo, "rm_ait_sh", 1.0)
        if model.get_setting("add_taylor_rm", False):
            set_col(Gamma0, "eq_mp", endo, "ait_rm_t", -1.0)

    canonical = CanonicalSystem(Gamma0=Gamma0, Gamma1=Gamma1, C=C, Psi=Psi, Pi=Pi)
    canonical.validate()
    return canonical
