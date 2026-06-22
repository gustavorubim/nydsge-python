from __future__ import annotations

import math

import numpy as np
import pytest

from nydsge.core import NotPortedError
from nydsge.models import Model1002
from nydsge.solve import Transition, compute_system


def test_model1002_constructs_ss10_shape() -> None:
    model = Model1002()
    assert model.spec == "m1002"
    assert model.subspec == "ss10"
    assert model.get_setting("data_vintage") == "181115"
    assert "obs_gdp" in model.observables
    assert "obs_nominalrate6" in model.observables
    assert "OutputGap" in model.pseudo_observables
    assert "Expected10YearRateGap" in model.pseudo_observables
    assert "y_t" in model.indexes.endogenous_states
    assert "rm_shl6" in model.indexes.exogenous_shocks
    assert "alpha" in model.parameters
    assert len(model.parameters) == 95
    assert "sigma_r_m6" in model.parameters
    assert "rho_exp_rm" not in model.parameters


def test_model1002_ss10_default_index_order_matches_upstream_shape() -> None:
    model = Model1002()

    assert len(model.indexes.endogenous_states) == 68
    assert len(model.indexes.exogenous_shocks) == 24
    assert len(model.indexes.expected_shocks) == 13
    assert len(model.indexes.equilibrium_conditions) == 68
    assert len(model.indexes.endogenous_states_augmented) == 16
    assert len(model.indexes.pseudo_observables) == 21

    assert model.indexes.endogenous_states["y_t"] == 1
    assert model.indexes.endogenous_states["n_f_t"] == 62
    assert model.indexes.endogenous_states["rm_tl1"] == 63
    assert model.indexes.endogenous_states["rm_tl6"] == 68
    assert model.indexes.exogenous_shocks["g_sh"] == 1
    assert model.indexes.exogenous_shocks["rm_shl1"] == 19
    assert model.indexes.exogenous_shocks["rm_shl6"] == 24
    assert model.indexes.expected_shocks["Epi_sh"] == 4
    assert model.indexes.equilibrium_conditions["eq_mu"] == 19
    assert model.indexes.equilibrium_conditions["eq_rml1"] == 63
    assert model.indexes.equilibrium_conditions["eq_rml6"] == 68
    assert model.indexes.endogenous_states_augmented["y_t1"] == 69
    assert model.indexes.endogenous_states_augmented["e_gdi_t1"] == 84
    assert model.indexes.pseudo_observables["y_t"] == 1
    assert model.indexes.pseudo_observables["Expected10YearRateGap"] == 14
    assert model.indexes.pseudo_observables["u_t"] == 21


def test_model1002_index_maps_expand_configured_antshocks() -> None:
    model = Model1002(settings={"n_mon_anticipated_shocks": 2, "antshocks": {"foo": 2}})

    assert len(model.indexes.endogenous_states) == 66
    assert len(model.indexes.exogenous_shocks) == 22
    assert len(model.indexes.equilibrium_conditions) == 66
    assert model.indexes.endogenous_states["rm_tl2"] == 64
    assert model.indexes.endogenous_states["foo_tl1"] == 65
    assert model.indexes.endogenous_states["foo_tl2"] == 66
    assert model.indexes.exogenous_shocks["foo_shl2"] == 22
    assert model.indexes.equilibrium_conditions["eq_fool2"] == 66
    assert model.indexes.endogenous_states_augmented["y_t1"] == 67


def test_model1002_rejects_invalid_antshocks_setting() -> None:
    with pytest.raises(TypeError, match="antshocks"):
        Model1002(settings={"antshocks": ["foo"]})

    with pytest.raises(ValueError, match="nonnegative"):
        Model1002(settings={"antshocks": {"foo": -1}})


def test_model1002_observable_order_settings_match_upstream_controls() -> None:
    hours_first = Model1002(
        settings={
            "hours_first_observable": True,
            "n_mon_anticipated_shocks": 0,
        }
    )
    assert list(hours_first.observables)[:2] == ["obs_hours", "obs_gdp"]

    reordered = Model1002(
        settings={
            "first_observable": "obs_tfp",
            "last_observable": "obs_gdp",
            "n_mon_anticipated_shocks": 0,
        }
    )
    assert next(iter(reordered.observables)) == "obs_tfp"
    assert next(reversed(reordered.observables)) == "obs_gdp"

    with pytest.raises(KeyError, match="unknown observable"):
        Model1002(settings={"first_observable": "does_not_exist"})


def test_model1002_anticipated_gdp_observables_and_forward_looking_metadata() -> None:
    model = Model1002(
        settings={
            "add_anticipated_obs_gdp": True,
            "n_anticipated_obs_gdp": 2,
            "filename_anticipated_obs_gdp": "SURVEY",
            "n_mon_anticipated_shocks": 0,
        }
    )

    assert "obs_gdp1" in model.observables
    assert "obs_gdp2" in model.observables
    assert model.observable_mappings["obs_gdp1"].source_names == ("antgdp1__SURVEY",)
    assert model.observable_mappings["obs_gdp1"].forward_transform == "anticipated_gdp_growth"
    assert model.get_setting("forward_looking_observables") == [
        "obs_longinflation",
        "obs_longrate",
        "obs_gdp1",
        "obs_gdp2",
    ]


def test_model1002_expected_ffr_spd_observables_indexes_and_parameters() -> None:
    model = Model1002(
        settings={
            "expected_ffr": [4, 1, 4],
            "n_mon_anticipated_shocks": 2,
        }
    )

    assert "obs_exp_nominalrate1" in model.observables
    assert "obs_exp_nominalrate4" in model.observables
    assert model.observable_mappings["obs_exp_nominalrate4"].source_names == ("exp_ant4__SPD",)
    assert model.observable_mappings["obs_exp_nominalrate4"].forward_transform == (
        "expected_ffr_spd"
    )
    assert model.get_setting("forward_looking_observables") == [
        "obs_longinflation",
        "obs_longrate",
        "obs_nominalrate1",
        "obs_nominalrate2",
        "obs_exp_nominalrate1",
        "obs_exp_nominalrate4",
    ]
    assert model.indexes.endogenous_states_augmented["e_exp_rm1"] == 81
    assert model.indexes.endogenous_states_augmented["e_exp_rm4"] == 82
    assert model.indexes.exogenous_shocks["exp_rm_sh1"] == 21
    assert model.indexes.exogenous_shocks["exp_rm_sh4"] == 22
    assert model.parameters["sigma_exp_rm1"].value == 0.04375
    assert model.parameters["sigma_exp_rm4"].value == 0.0625
    assert model.parameters["sigma_exp_rm4"].category == "measurement_error"
    assert model.parameters["sigma_exp_rm4"].regime == "expected_ffr_spd"


def test_model1002_ss104_enables_expected_ffr_spd_horizons() -> None:
    model = Model1002(subspec="ss104")

    assert model.get_setting("expected_ffr") == (1, 2, 3, 4, 5, 6)
    assert "obs_exp_nominalrate6" in model.observables
    assert "e_exp_rm6" in model.indexes.endogenous_states_augmented
    assert "exp_rm_sh6" in model.indexes.exogenous_shocks


def test_model1002_flexible_ait_initialization_observables_indexes_and_parameters() -> None:
    model = Model1002(
        settings={
            "add_initialize_pgap_ygap_pseudoobs": True,
            "n_mon_anticipated_shocks": 0,
        }
    )

    assert "obs_pgap" in model.observables
    assert "obs_ygap" in model.observables
    assert model.observable_mappings["obs_pgap"].source_names == ("pgap__INITFLEXAIT",)
    assert model.observable_mappings["obs_ygap"].source_names == ("ygap__INITFLEXAIT",)
    assert model.observable_mappings["obs_pgap"].forward_transform == "flexible_ait_gap"
    assert model.indexes.endogenous_states["pgap_t"] == 63
    assert model.indexes.endogenous_states["ygap_t"] == 64
    assert model.indexes.exogenous_shocks["pgap_sh"] == 19
    assert model.indexes.exogenous_shocks["ygap_sh"] == 20
    assert model.indexes.equilibrium_conditions["eq_pgap"] == 63
    assert model.indexes.equilibrium_conditions["eq_ygap"] == 64
    assert model.parameters["sigma_pgap"].value == 0.0
    assert model.parameters["sigma_ygap"].category == "measurement_error"
    assert model.parameters["sigma_ygap"].regime == "flexible_ait"


def test_model1002_conditional_measurement_error_indexes() -> None:
    model = Model1002(
        settings={
            "add_iid_cond_obs_gdp_meas_err": True,
            "add_iid_anticipated_obs_gdp_meas_err": True,
            "add_iid_cond_obs_corepce_meas_err": True,
            "n_mon_anticipated_shocks": 0,
        }
    )

    assert model.indexes.endogenous_states_augmented["e_condgdp_t"] == 79
    assert model.indexes.endogenous_states_augmented["e_gdpexp_t"] == 80
    assert model.indexes.endogenous_states_augmented["e_condcorepce_t"] == 81
    assert model.indexes.exogenous_shocks["condgdp_sh"] == 19
    assert model.indexes.exogenous_shocks["gdpexp_sh"] == 20
    assert model.indexes.exogenous_shocks["condcorepce_sh"] == 21
    assert model.parameters["rho_condgdp"].regime == "conditional_forecast"
    assert model.parameters["sigma_gdpexp"].regime == "conditional_forecast"
    assert model.parameters["sigma_condcorepce"].category == "measurement_error"


def test_model1002_ss10_parameter_metadata() -> None:
    model = Model1002()

    assert model.parameters["Phi"].value == 1.1066
    assert model.parameters["Phi"].fixed is False
    assert model.parameters["Phi"].transform == "exponential"
    assert model.parameters["beta"].scaling == "discount_rate"
    assert model.parameters["pi_star"].scaling == "gross_rate"
    assert model.parameters["Fomega"].scaling == "fomega"
    assert model.parameters["spr"].scaling == "quarterly_spread"
    assert model.parameters["gamma"].scaling == "percent"
    assert model.parameters["gamma_gdi"].value == 1.0
    assert model.parameters["gamma_gdi"].fixed is True
    assert model.parameters["alpha"].prior is not None
    assert model.parameters["alpha"].prior.name == "normal"
    assert model.parameters["zeta_p"].prior is not None
    assert model.parameters["zeta_p"].prior.name == "beta_alt"
    assert model.parameters["sigma_g"].prior is not None
    assert model.parameters["sigma_g"].prior.name == "root_inverse_gamma"
    assert model.parameters["sigma_r_m1"].prior is not None
    assert model.parameters["sigma_r_m1"].prior.name == "root_inverse_gamma"
    assert model.parameters["sigma_r_m7"].fixed is True
    assert model.parameters["sigma_r_m7"].prior is not None
    assert model.parameters["eta_gz"].prior is not None
    assert model.parameters["eta_gz"].prior.name == "beta_alt"
    assert model.parameters["Gamma_gdpdef"].prior is not None
    assert model.parameters["Gamma_gdpdef"].prior.name == "normal"
    assert sum(parameter.prior is not None for parameter in model.parameters.values()) == 77
    assert (
        sum(
            parameter.prior is not None and not parameter.fixed
            for parameter in model.parameters.values()
        )
        == 63
    )
    assert all(parameter.description for parameter in model.parameters.values())
    assert all(parameter.category for parameter in model.parameters.values())
    assert all(parameter.regime for parameter in model.parameters.values())
    assert model.parameters["alpha"].description == "Capital share in production."
    assert model.parameters["alpha"].category == "structural"
    assert model.parameters["sigma_g"].category == "shock_std"
    assert model.parameters["rho_g"].category == "persistence"
    assert model.parameters["sigma_r_m6"].regime == "anticipated_policy"
    assert "rho_ait_rm" not in model.parameters
    assert "sigma_g_covid" not in model.parameters
    assert "rho_condgdp" not in model.parameters
    assert "sigma_ziid" not in model.parameters
    assert "phi_pi" not in model.parameters


def test_model1002_ss10_steady_state_placeholders_and_numeric_values() -> None:
    model = Model1002()

    assert len(model.steady_state) == 22
    assert "z_star" in model.steady_state
    assert "zeta_nsigma_omega" in model.steady_state
    assert math.isnan(model.numeric_value("z_star"))
    assert model.numeric_value("alpha") == model.parameters["alpha"].value
    assert math.isclose(model.numeric_value("beta"), 1.0 / (1.0 + 0.1402 / 100.0))
    assert math.isclose(model.numeric_value("pi_star"), 1.005)

    model.set_steady_state("z_star", 1.25)
    assert model.numeric_value("z_star") == 1.25
    with pytest.raises(KeyError, match="Unknown model parameter"):
        model.numeric_value("does_not_exist")


def test_model1002_computes_ss10_steady_state() -> None:
    model = Model1002()

    steady_state = model.steadystate()

    assert len(steady_state) == 22
    assert math.isclose(steady_state["z_star"], 0.0036662710075261113)
    assert math.isclose(steady_state["rstar"], 1.0046082251693307)
    assert math.isclose(steady_state["Rstarn"], 0.9631266295177188)
    assert math.isclose(steady_state["r_k_star"], 0.03396095086834372)
    assert math.isclose(steady_state["kbarstar"], 5.609571337905602)
    assert math.isclose(steady_state["nstar"], 4.133188141136237)
    assert math.isclose(steady_state["zeta_spsigma_omega"], 0.026921438518770248)
    assert math.isclose(steady_state["zeta_nRk"], 1.3523792935746353)
    assert math.isclose(steady_state["zeta_nsigma_omega"], 0.0023997869687067764)
    assert model.numeric_value("z_star") == steady_state["z_star"]


def test_model1002_builds_ss10_equilibrium_matrices() -> None:
    model = Model1002()

    canonical = model.equilibrium_matrices()
    eq = model.indexes.equilibrium_conditions
    endo = model.indexes.endogenous_states
    exo = model.indexes.exogenous_shocks
    expected = model.indexes.expected_shocks
    value = model.numeric_value

    assert canonical.Gamma0.shape == (68, 68)
    assert canonical.Gamma1.shape == (68, 68)
    assert canonical.C.shape == (68,)
    assert canonical.Psi.shape == (68, 24)
    assert canonical.Pi.shape == (68, 13)
    assert not canonical.C.any()

    expected_rate_coefficient = (1 - value("h") * math.exp(-value("z_star"))) / (
        value("sigma_c") * (1 + value("h") * math.exp(-value("z_star")))
    )
    assert math.isclose(
        canonical.Gamma0[eq["eq_euler"] - 1, endo["R_t"] - 1],
        expected_rate_coefficient,
    )
    assert canonical.Gamma0[eq["eq_euler"] - 1, endo["c_t"] - 1] == 1.0
    assert canonical.Gamma1[eq["eq_rm"] - 1, endo["rm_tl1"] - 1] == 1.0
    assert canonical.Gamma0[eq["eq_rml6"] - 1, endo["rm_tl6"] - 1] == 1.0
    assert canonical.Psi[eq["eq_rml6"] - 1, exo["rm_shl6"] - 1] == 1.0
    assert canonical.Pi[eq["eq_Epi"] - 1, expected["Epi_sh"] - 1] == 1.0
    assert canonical.Psi[eq["eq_g"] - 1, exo["g_sh"] - 1] == 1.0


def test_model1002_equilibrium_supports_flexible_ait_initialization_states() -> None:
    model = Model1002(
        settings={
            "add_initialize_pgap_ygap_pseudoobs": True,
            "n_mon_anticipated_shocks": 0,
        }
    )

    canonical = model.equilibrium_matrices()
    eq = model.indexes.equilibrium_conditions
    endo = model.indexes.endogenous_states
    exo = model.indexes.exogenous_shocks

    assert canonical.Gamma0.shape == (64, 64)
    assert canonical.Psi.shape == (64, 20)
    assert canonical.Gamma0[eq["eq_pgap"] - 1, endo["pgap_t"] - 1] == 1.0
    assert canonical.Gamma0[eq["eq_ygap"] - 1, endo["ygap_t"] - 1] == 1.0
    assert canonical.Psi[eq["eq_pgap"] - 1, exo["pgap_sh"] - 1] == 1.0
    assert canonical.Psi[eq["eq_ygap"] - 1, exo["ygap_sh"] - 1] == 1.0


def test_model1002_non_ss10_equilibrium_matrices_are_unported() -> None:
    model = Model1002(subspec="ss59")
    with pytest.raises(NotPortedError, match="ss10 equilibrium"):
        model.equilibrium_matrices()


def test_model1002_builds_ss10_measurement_matrices() -> None:
    model = Model1002()
    n_states = len(model.indexes.endogenous_states) + len(model.indexes.endogenous_states_augmented)
    transition = Transition(
        TTT=np.eye(n_states),
        RRR=np.zeros((n_states, len(model.indexes.exogenous_shocks))),
        CCC=np.zeros(n_states),
    )

    measurement = model.measurement_matrices(transition)
    obs = model.indexes.observables
    endo = model.indexes.endogenous_states
    endo_aug = model.indexes.endogenous_states_augmented
    exo = model.indexes.exogenous_shocks
    value = model.numeric_value

    assert measurement.ZZ.shape == (19, 84)
    assert measurement.DD.shape == (19,)
    assert measurement.QQ.shape == (24, 24)
    assert measurement.EE.shape == (19, 19)
    assert measurement.ZZ[obs["obs_gdp"] - 1, endo["y_t"] - 1] == 1.0
    assert measurement.ZZ[obs["obs_gdp"] - 1, endo_aug["y_t1"] - 1] == -1.0
    assert measurement.ZZ[obs["obs_gdp"] - 1, endo["z_t"] - 1] == 1.0
    assert measurement.ZZ[obs["obs_gdp"] - 1, endo_aug["e_gdp_t"] - 1] == 1.0
    assert measurement.ZZ[obs["obs_gdp"] - 1, endo_aug["e_gdp_t1"] - 1] == -value("me_level")
    assert math.isclose(
        measurement.DD[obs["obs_gdp"] - 1],
        100.0 * (math.exp(value("z_star")) - 1.0),
    )
    assert measurement.ZZ[obs["obs_spread"] - 1, endo["ERktil_t"] - 1] == 1.0
    assert measurement.ZZ[obs["obs_spread"] - 1, endo["R_t"] - 1] == -1.0
    assert math.isclose(measurement.DD[obs["obs_spread"] - 1], 100.0 * math.log(value("spr")))
    assert measurement.ZZ[obs["obs_longinflation"] - 1, endo["pi_t"] - 1] == 1.0
    assert measurement.ZZ[obs["obs_longrate"] - 1, endo["R_t"] - 1] == 1.0
    assert measurement.ZZ[obs["obs_nominalrate3"] - 1, endo["R_t"] - 1] == 1.0
    assert math.isclose(measurement.DD[obs["obs_nominalrate3"] - 1], value("Rstarn"))
    assert math.isclose(measurement.QQ[exo["g_sh"] - 1, exo["g_sh"] - 1], value("sigma_g") ** 2)
    assert math.isclose(
        measurement.QQ[exo["rm_shl6"] - 1, exo["rm_shl6"] - 1],
        value("sigma_r_m6") ** 2,
    )


def test_model1002_measurement_supports_anticipated_gdp_observables() -> None:
    model = Model1002(
        settings={
            "add_anticipated_obs_gdp": True,
            "n_anticipated_obs_gdp": 1,
            "n_mon_anticipated_shocks": 0,
        }
    )
    n_states = len(model.indexes.endogenous_states) + len(model.indexes.endogenous_states_augmented)
    transition = Transition(
        TTT=np.eye(n_states),
        RRR=np.zeros((n_states, len(model.indexes.exogenous_shocks))),
        CCC=np.zeros(n_states),
    )

    measurement = model.measurement_matrices(transition)
    obs = model.indexes.observables
    endo_aug = model.indexes.endogenous_states_augmented
    expected_gdp_row = measurement.ZZ[obs["obs_gdp"] - 1].copy()
    expected_gdp_row[endo_aug["e_gdp_t"] - 1] = 0.0
    expected_gdp_row[endo_aug["e_gdp_t1"] - 1] = 0.0

    np.testing.assert_allclose(
        measurement.ZZ[obs["obs_gdp1"] - 1],
        expected_gdp_row,
    )
    assert measurement.DD[obs["obs_gdp1"] - 1] == measurement.DD[obs["obs_gdp"] - 1]


def test_model1002_measurement_supports_expected_ffr_spd_observables() -> None:
    model = Model1002(
        settings={
            "expected_ffr": [1, 4],
            "n_mon_anticipated_shocks": 2,
        }
    )
    n_states = len(model.indexes.endogenous_states) + len(model.indexes.endogenous_states_augmented)
    transition = Transition(
        TTT=0.5 * np.eye(n_states),
        RRR=np.zeros((n_states, len(model.indexes.exogenous_shocks))),
        CCC=np.zeros(n_states),
    )

    measurement = model.measurement_matrices(transition)
    obs = model.indexes.observables
    endo = model.indexes.endogenous_states
    endo_aug = model.indexes.endogenous_states_augmented
    exo = model.indexes.exogenous_shocks
    value = model.numeric_value

    assert measurement.ZZ.shape == (17, 82)
    assert measurement.DD.shape == (17,)
    assert measurement.QQ.shape == (22, 22)
    assert measurement.ZZ[obs["obs_exp_nominalrate1"] - 1, endo["R_t"] - 1] == 0.5
    assert measurement.ZZ[obs["obs_exp_nominalrate4"] - 1, endo["R_t"] - 1] == 0.5**4
    assert measurement.ZZ[obs["obs_exp_nominalrate4"] - 1, endo_aug["e_exp_rm4"] - 1] == 1.0
    assert math.isclose(measurement.DD[obs["obs_exp_nominalrate4"] - 1], value("Rstarn"))
    assert math.isclose(
        measurement.QQ[exo["exp_rm_sh4"] - 1, exo["exp_rm_sh4"] - 1],
        value("sigma_exp_rm4") ** 2,
    )


def test_model1002_measurement_supports_flexible_ait_initialization_observables() -> None:
    model = Model1002(
        settings={
            "add_initialize_pgap_ygap_pseudoobs": True,
            "n_mon_anticipated_shocks": 0,
        }
    )
    n_states = len(model.indexes.endogenous_states) + len(model.indexes.endogenous_states_augmented)
    transition = Transition(
        TTT=np.eye(n_states),
        RRR=np.zeros((n_states, len(model.indexes.exogenous_shocks))),
        CCC=np.zeros(n_states),
    )

    measurement = model.measurement_matrices(transition)
    obs = model.indexes.observables
    endo = model.indexes.endogenous_states
    exo = model.indexes.exogenous_shocks

    assert measurement.ZZ.shape == (15, 80)
    assert measurement.QQ.shape == (20, 20)
    assert measurement.ZZ[obs["obs_pgap"] - 1, endo["pgap_t"] - 1] == 1.0
    assert measurement.ZZ[obs["obs_ygap"] - 1, endo["ygap_t"] - 1] == 1.0
    assert (
        measurement.QQ[exo["pgap_sh"] - 1, exo["pgap_sh"] - 1]
        == model.numeric_value("sigma_pgap") ** 2
    )
    assert (
        measurement.QQ[exo["ygap_sh"] - 1, exo["ygap_sh"] - 1]
        == model.numeric_value("sigma_ygap") ** 2
    )


def test_model1002_augmentation_supports_conditional_measurement_error_states() -> None:
    model = Model1002(
        settings={
            "expected_ffr": [1],
            "add_iid_cond_obs_gdp_meas_err": True,
            "add_anticipated_obs_gdp": True,
            "add_iid_anticipated_obs_gdp_meas_err": True,
            "add_iid_cond_obs_corepce_meas_err": True,
            "n_mon_anticipated_shocks": 0,
        }
    )
    n_endo = len(model.indexes.endogenous_states)
    n_exo = len(model.indexes.exogenous_shocks)
    transition = Transition(
        TTT=np.eye(n_endo),
        RRR=np.zeros((n_endo, n_exo)),
        CCC=np.zeros(n_endo),
    )

    augmented = model.augment_transition(transition)
    endo_aug = model.indexes.endogenous_states_augmented
    exo = model.indexes.exogenous_shocks
    value = model.numeric_value

    assert augmented.TTT.shape == (82, 82)
    assert augmented.RRR.shape == (82, 22)
    assert augmented.RRR[endo_aug["e_exp_rm1"] - 1, exo["exp_rm_sh1"] - 1] == 1.0
    assert augmented.TTT[endo_aug["e_condgdp_t"] - 1, endo_aug["e_condgdp_t"] - 1] == value(
        "rho_condgdp"
    )
    assert augmented.RRR[endo_aug["e_condgdp_t"] - 1, exo["condgdp_sh"] - 1] == 1.0
    assert augmented.TTT[endo_aug["e_gdpexp_t"] - 1, endo_aug["e_gdpexp_t"] - 1] == value(
        "rho_gdpexp"
    )
    assert augmented.RRR[endo_aug["e_gdpexp_t"] - 1, exo["gdpexp_sh"] - 1] == 1.0
    assert augmented.TTT[
        endo_aug["e_condcorepce_t"] - 1,
        endo_aug["e_condcorepce_t"] - 1,
    ] == value("rho_condcorepce")
    assert augmented.RRR[endo_aug["e_condcorepce_t"] - 1, exo["condcorepce_sh"] - 1] == 1.0


def test_model1002_measurement_supports_conditional_measurement_error_states() -> None:
    model = Model1002(
        settings={
            "add_iid_cond_obs_gdp_meas_err": True,
            "add_anticipated_obs_gdp": True,
            "add_iid_anticipated_obs_gdp_meas_err": True,
            "add_iid_cond_obs_corepce_meas_err": True,
            "n_mon_anticipated_shocks": 0,
        }
    )
    n_states = len(model.indexes.endogenous_states) + len(model.indexes.endogenous_states_augmented)
    transition = Transition(
        TTT=np.eye(n_states),
        RRR=np.zeros((n_states, len(model.indexes.exogenous_shocks))),
        CCC=np.zeros(n_states),
    )

    measurement = model.measurement_matrices(transition)
    obs = model.indexes.observables
    endo_aug = model.indexes.endogenous_states_augmented
    exo = model.indexes.exogenous_shocks
    value = model.numeric_value

    assert measurement.ZZ.shape == (14, 81)
    assert measurement.QQ.shape == (21, 21)
    assert measurement.ZZ[obs["obs_gdp"] - 1, endo_aug["e_condgdp_t"] - 1] == 1.0
    assert measurement.ZZ[obs["obs_corepce"] - 1, endo_aug["e_condcorepce_t"] - 1] == 1.0
    assert measurement.ZZ[obs["obs_gdp1"] - 1, endo_aug["e_gdpexp_t"] - 1] == 1.0
    assert math.isclose(
        measurement.QQ[exo["condgdp_sh"] - 1, exo["condgdp_sh"] - 1],
        value("sigma_condgdp") ** 2,
    )
    assert math.isclose(
        measurement.QQ[exo["gdpexp_sh"] - 1, exo["gdpexp_sh"] - 1],
        value("sigma_gdpexp") ** 2,
    )
    assert math.isclose(
        measurement.QQ[exo["condcorepce_sh"] - 1, exo["condcorepce_sh"] - 1],
        value("sigma_condcorepce") ** 2,
    )


def test_model1002_builds_ss10_pseudo_measurement_matrices() -> None:
    model = Model1002()
    n_states = len(model.indexes.endogenous_states) + len(model.indexes.endogenous_states_augmented)
    transition = Transition(
        TTT=np.eye(n_states),
        RRR=np.zeros((n_states, len(model.indexes.exogenous_shocks))),
        CCC=np.zeros(n_states),
    )

    pseudo_measurement = model.pseudo_measurement_matrices(transition)
    pseudo = model.indexes.pseudo_observables
    endo = model.indexes.endogenous_states
    endo_aug = model.indexes.endogenous_states_augmented
    value = model.numeric_value
    zz = pseudo_measurement.ZZ_pseudo
    dd = pseudo_measurement.DD_pseudo

    assert zz.shape == (21, 84)
    assert dd.shape == (21,)
    assert zz[pseudo["y_t"] - 1, endo["y_t"] - 1] == 1.0
    assert zz[pseudo["OutputGap"] - 1, endo["y_t"] - 1] == 1.0
    assert zz[pseudo["OutputGap"] - 1, endo["y_f_t"] - 1] == -1.0
    assert zz[pseudo["NaturalRate"] - 1, endo["r_f_t"] - 1] == 1.0
    assert math.isclose(dd[pseudo["NaturalRate"] - 1], 100.0 * (value("rstar") - 1.0))
    assert zz[pseudo["\u03c0_t"] - 1, endo["pi_t"] - 1] == 1.0
    assert math.isclose(dd[pseudo["\u03c0_t"] - 1], 100.0 * (value("pi_star") - 1.0))
    assert zz[pseudo["Expected10YearRateGap"] - 1, endo["R_t"] - 1] == 1.0
    assert zz[pseudo["Expected10YearRateGap"] - 1, endo["r_f_t"] - 1] == -1.0
    assert zz[pseudo["Expected10YearRateGap"] - 1, endo["Epi_t"] - 1] == -1.0
    assert zz[pseudo["Expected10YearRate"] - 1, endo["R_t"] - 1] == 1.0
    assert math.isclose(dd[pseudo["Expected10YearRate"] - 1], value("Rstarn"))
    assert zz[pseudo["LaborProductivityGrowth"] - 1, endo["y_t"] - 1] == 1.0
    assert zz[pseudo["LaborProductivityGrowth"] - 1, endo_aug["y_t1"] - 1] == -1.0
    assert zz[pseudo["LaborProductivityGrowth"] - 1, endo["L_t"] - 1] == -1.0
    assert zz[pseudo["LaborProductivityGrowth"] - 1, endo_aug["L_t1"] - 1] == 1.0
    assert math.isclose(
        dd[pseudo["LaborProductivityGrowth"] - 1],
        100.0 * (math.exp(value("z_star")) - 1.0),
    )


def test_model1002_non_ss10_measurement_matrices_are_unported() -> None:
    model = Model1002(subspec="ss59")
    transition = Transition(TTT=np.eye(1), RRR=np.zeros((1, 1)), CCC=np.zeros(1))
    with pytest.raises(NotPortedError, match="ss10 measurement"):
        model.measurement_matrices(transition)


def test_model1002_augments_ss10_transition_matrices() -> None:
    model = Model1002()
    endo = model.indexes.endogenous_states
    endo_aug = model.indexes.endogenous_states_augmented
    exo = model.indexes.exogenous_shocks
    n_endo = len(endo)
    n_exo = len(exo)
    ttt = np.zeros((n_endo, n_endo))
    rrr = np.zeros((n_endo, n_exo))
    ccc = np.zeros(n_endo)
    pi_row = endo["pi_t"] - 1
    ttt[pi_row, pi_row] = 0.5
    rrr[pi_row, exo["g_sh"] - 1] = 2.0
    ccc[pi_row] = 3.0

    augmented = model.augment_transition(Transition(TTT=ttt, RRR=rrr, CCC=ccc))

    assert augmented.TTT.shape == (84, 84)
    assert augmented.RRR.shape == (84, 24)
    assert augmented.CCC.shape == (84,)
    assert augmented.TTT[endo_aug["y_t1"] - 1, endo["y_t"] - 1] == 1.0
    assert augmented.TTT[endo_aug["pi_t1_dup"] - 1, endo["pi_t"] - 1] == 1.0
    assert augmented.TTT[endo_aug["e_gdp_t1"] - 1, endo_aug["e_gdp_t"] - 1] == 1.0
    assert augmented.TTT[endo_aug["Et_pi_t"] - 1, endo["pi_t"] - 1] == 0.25
    assert augmented.RRR[endo_aug["Et_pi_t"] - 1, exo["g_sh"] - 1] == 1.0
    assert augmented.RRR[endo_aug["e_lr_t"] - 1, exo["lr_sh"] - 1] == 1.0
    assert augmented.RRR[endo_aug["e_gdp_t"] - 1, exo["gdi_sh"] - 1] == 0.0
    assert augmented.CCC[endo_aug["Et_pi_t"] - 1] == 4.5


def test_compute_system_builds_model1002_ss10_system() -> None:
    model = Model1002()

    system = compute_system(model)

    endo = model.indexes.endogenous_states
    endo_aug = model.indexes.endogenous_states_augmented
    exo = model.indexes.exogenous_shocks
    obs = model.indexes.observables
    assert system.transition.TTT.shape == (84, 84)
    assert system.transition.RRR.shape == (84, 24)
    assert system.transition.CCC.shape == (84,)
    assert system.measurement.ZZ.shape == (19, 84)
    assert system.measurement.DD.shape == (19,)
    assert system.measurement.QQ.shape == (24, 24)
    assert system.pseudo_measurement is not None
    assert system.pseudo_measurement.ZZ_pseudo.shape == (21, 84)
    assert system.pseudo_measurement.DD_pseudo.shape == (21,)
    assert system.transition.TTT[endo_aug["y_t1"] - 1, endo["y_t"] - 1] == 1.0
    assert system.transition.RRR[endo_aug["e_lr_t"] - 1, exo["lr_sh"] - 1] == 1.0
    assert system.measurement.ZZ[obs["obs_gdp"] - 1, endo["y_t"] - 1] == 1.0
    assert (
        system["ZZ_pseudo"][model.indexes.pseudo_observables["NominalFFR"] - 1, endo["R_t"] - 1]
        == 1.0
    )
