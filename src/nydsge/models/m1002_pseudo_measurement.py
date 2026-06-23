from __future__ import annotations

from math import exp
from typing import Any

import numpy as np

from nydsge.solve import PseudoMeasurement, Transition


def pseudo_measurement_matrices_ss10(model: Any, transition: Transition) -> PseudoMeasurement:
    """Build default Model1002 ss10 pseudo-measurement matrices."""
    model.steadystate()
    endo = model.indexes.endogenous_states
    endo_aug = model.indexes.endogenous_states_augmented
    pseudo = model.indexes.pseudo_observables
    n_pseudo = len(pseudo)
    n_states = len(endo) + len(endo_aug)
    n_pseudo_states = n_states
    if (
        model._is_setting_enabled("add_initialize_pgap_ygap_pseudoobs")
        and int(model.get_setting("n_mon_anticipated_shocks", 0)) == 0
    ):
        n_pseudo_states += 4
    ttt = np.asarray(transition.TTT, dtype=np.float64)
    ccc = np.asarray(transition.CCC, dtype=np.float64)
    if ttt.shape != (n_states, n_states):
        msg = f"TTT must have shape {(n_states, n_states)} for Model1002 pseudo-measurement."
        raise ValueError(msg)
    if ccc.shape != (n_states,):
        msg = f"CCC must have shape {(n_states,)} for Model1002 pseudo-measurement."
        raise ValueError(msg)

    zz = np.zeros((n_pseudo, n_pseudo_states), dtype=np.float64)
    dd = np.zeros(n_pseudo, dtype=np.float64)
    v = model.numeric_value

    def set_zz(row_name: str, column_name: str, value: float) -> None:
        zz[pseudo[row_name] - 1, endo[column_name] - 1] = value

    def set_zz_aug(row_name: str, column_name: str, value: float) -> None:
        zz[pseudo[row_name] - 1, endo_aug[column_name] - 1] = value

    def set_row(row_name: str, value: np.ndarray) -> None:
        row = np.asarray(value, dtype=np.float64)
        if row.shape != (n_states,):
            msg = f"Pseudo-measurement row {row_name} must have shape {(n_states,)}."
            raise ValueError(msg)
        extended_row = np.zeros(n_pseudo_states, dtype=np.float64)
        extended_row[:n_states] = row
        zz[pseudo[row_name] - 1, :] = extended_row

    def set_dd(row_name: str, value: float) -> None:
        dd[pseudo[row_name] - 1] = value

    trend_growth = 100.0 * (exp(v("z_star")) - 1.0)
    inflation_trend = 100.0 * (v("pi_star") - 1.0)
    ttt10, ccc10 = _k_periods_ahead_expected_sums(ttt, ccc, 40)
    ttt10 = ttt10 / 40.0
    ccc10 = ccc10 / 40.0

    set_zz("y_t", "y_t", 1.0)
    set_zz("y_f_t", "y_f_t", 1.0)

    set_zz("NaturalRate", "r_f_t", 1.0)
    set_dd("NaturalRate", 100.0 * (v("rstar") - 1.0))

    set_zz("\u03c0_t", "pi_t", 1.0)
    set_dd("\u03c0_t", inflation_trend)

    set_zz("OutputGap", "y_t", 1.0)
    set_zz("OutputGap", "y_f_t", -1.0)

    set_zz("ExAnteRealRate", "R_t", 1.0)
    set_zz("ExAnteRealRate", "Epi_t", -1.0)
    set_dd("ExAnteRealRate", v("Rstarn") - inflation_trend)

    set_zz("LongRunInflation", "pi_star_t", 1.0)
    set_dd("LongRunInflation", inflation_trend)

    set_zz("MarginalCost", "mc_t", 1.0)
    set_zz("Wages", "w_t", 1.0)
    set_zz("FlexibleWages", "w_f_t", 1.0)
    set_zz("Hours", "L_t", 1.0)
    set_zz("FlexibleHours", "L_f_t", 1.0)
    set_zz("z_t", "z_t", 1.0)

    r_t = endo["R_t"] - 1
    r_f_t = endo["r_f_t"] - 1
    epi_t = endo["Epi_t"] - 1
    set_row("Expected10YearRateGap", ttt10[r_t, :] - ttt10[r_f_t, :] - ttt10[epi_t, :])
    set_dd("Expected10YearRateGap", ccc10[r_t] - ccc10[r_f_t] - ccc10[epi_t])

    set_zz("NominalFFR", "R_t", 1.0)
    set_dd("NominalFFR", v("Rstarn"))

    set_row("Expected10YearRate", ttt10[r_t, :])
    set_dd("Expected10YearRate", v("Rstarn") + ccc10[r_t])

    set_row("Expected10YearNaturalRate", ttt10[r_f_t, :] + ttt10[epi_t, :])
    set_dd("Expected10YearNaturalRate", v("Rstarn") + ccc10[r_f_t] + ccc10[epi_t])

    set_zz("ExpectedNominalNaturalRate", "r_f_t", 1.0)
    set_zz("ExpectedNominalNaturalRate", "Epi_t", 1.0)
    set_dd("ExpectedNominalNaturalRate", v("Rstarn"))

    set_zz("NominalRateGap", "R_t", 1.0)
    set_zz("NominalRateGap", "r_f_t", -1.0)
    set_zz("NominalRateGap", "Epi_t", -1.0)

    set_zz("LaborProductivityGrowth", "y_t", 1.0)
    set_zz_aug("LaborProductivityGrowth", "y_t1", -1.0)
    set_zz("LaborProductivityGrowth", "z_t", 1.0)
    set_zz_aug("LaborProductivityGrowth", "e_gdp_t", 1.0)
    set_zz_aug("LaborProductivityGrowth", "e_gdp_t1", -v("me_level"))
    set_zz("LaborProductivityGrowth", "L_t", -1.0)
    set_zz_aug("LaborProductivityGrowth", "L_t1", 1.0)
    set_dd("LaborProductivityGrowth", trend_growth)

    set_zz("u_t", "u_t", 1.0)

    if model._has_pgap_state():
        set_zz("pgap_t", "pgap_t", 1.0)
    if model._has_ygap_state():
        set_zz("ygap_t", "ygap_t", 1.0)

    return PseudoMeasurement(ZZ_pseudo=zz, DD_pseudo=dd)


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
