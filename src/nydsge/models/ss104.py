"""Dated ss104 parameter snapshots (DSGE.jl c09dfe8, ss64/ss104)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pandas as pd

from nydsge.core import Parameter
from nydsge.parameters import Prior


def initialize_covid(model) -> None:
    """Install explicit parameter regimes; no hidden mutation during solving."""
    dates = {1: model.get_setting("date_presample_start")}
    dates.update(
        {i + 2: str(q) for i, q in enumerate(pd.period_range("2020Q1", "2026Q4", freq="Q"))}
    )
    model.set_setting("regime_dates", dates)
    model.set_setting("n_regimes", len(dates))
    model.set_setting("regime_switching", True)
    values, mapping = {}, {}

    def install(name, regimes, path, fixed=None, priors=None):
        values[name] = {
            r: replace(
                model.parameters[name],
                value=value,
                fixed=(fixed or {}).get(r, model.parameters[name].fixed),
                prior=(priors or {}).get(r, model.parameters[name].prior),
            )
            for r, value in regimes.items()
        }
        mapping[name] = {r: path(r) for r in dates}

    for name in (
        "sigma_g",
        "sigma_b",
        "sigma_mu",
        "sigma_ztil",
        "sigma_lambda_f",
        "sigma_lambda_w",
        "sigma_sigma_omega",
        "sigma_mu_e",
        "sigma_gamma",
        "sigma_pi_star",
        "sigma_lr",
        "sigma_tfp",
        "sigma_gdp",
        "sigma_gdi",
        "sigma_z_p",
    ):
        p = model.parameters[name]
        mode = float(model.get_setting("standard_shocks_mode_adjust", 1.0))
        spread = float(model.get_setting("standard_shocks_spread_adjust", 1.0))
        prior = p.prior
        adjusted = replace(prior, nu=prior.nu / spread, tau=prior.tau * mode) if prior else None
        install(
            name,
            {1: p.value, 2: 0.0 if name == "sigma_z_p" else mode * p.value},
            lambda r: 2 if r in (2, 3) else 1,
            fixed={2: True} if name == "sigma_z_p" else {},
            priors={2: adjusted},
        )
    for name in ("sigma_r_m", "sigma_gdpdef", "sigma_corepce"):
        p = model.parameters[name]
        install(
            name,
            {1: p.value, 2: 10.0 * p.value},
            lambda r, name=name: 2 if r in (2, 3) or (r == 4 and name != "sigma_r_m") else 1,
            priors={2: replace(p.prior, tau=10.0 * p.prior.tau)},
        )
    for key, large, small, prior, late_prior in (
        (
            "phi",
            400.0,
            4.0,
            Prior("root_inverse_gamma", nu=80.0, tau=400.0012499980469),
            Prior("root_inverse_gamma", nu=80.0, tau=4.0),
        ),
        (
            "ziid",
            5.0,
            0.05,
            Prior("root_inverse_gamma", nu=10.0, tau=25.1**0.5),
            Prior("root_inverse_gamma", nu=10.0, tau=0.0501),
        ),
        (
            "biidc",
            4.0,
            0.04,
            Prior("root_inverse_gamma", nu=8.0, tau=16.1**0.5),
            Prior("root_inverse_gamma", nu=8.0, tau=0.0401248),
        ),
    ):
        for name in (f"rho_{key}", f"sigma_{key}"):
            model._add_parameter_from_spec(name)

        model.add_parameter(replace(model.parameters[f"sigma_{key}"], prior=prior, fixed=True))
        for suffix in ("1", "_prop"):
            model.add_parameter(
                Parameter(
                    f"sigma_{key}{suffix}",
                    0.0,
                    fixed=True,
                    value_bounds=(0.0, 100.0),
                    transform="exponential",
                    prior=prior,
                )
            )
        install(
            f"sigma_{key}",
            {1: 0.0, 2: large, 3: small},
            lambda r: 2 if 2 <= r <= 4 else 3 if r == 5 else 1,
            fixed={1: True, 2: False, 3: False},
            priors={3: late_prior},
        )
        install(
            f"sigma_{key}_prop",
            {1: 0.0, 2: 2.0},
            lambda r: 2 if r == 2 else 1,
            fixed={1: True, 2: True},
        )
    install(
        "sigma_biidc1",
        {1: 0.0, 2: 4.0},
        lambda r: 2 if r in (4, 5) else 1,
        fixed={1: True, 2: False},
    )
    model.set_setting("parameter_regimes", values)
    model.set_setting("model2para_regime", mapping)


def regime_snapshot(model, reg: int):
    dates = model.get_setting("regime_dates", {})
    if reg not in dates:
        raise ValueError(f"Unknown ss104 model regime: {reg}")
    snapshot = deepcopy(model)
    snapshot.set_setting("model_regime", reg)
    snapshot.set_setting("regime_switching", False)
    for name, path in model.get_setting("model2para_regime", {}).items():
        snapshot.parameters[name] = model.get_setting("parameter_regimes")[name][path[reg]]
    if model.get_setting("ss104_policy"):
        snapshot.set_setting("policy", "zlb" if 5 <= reg <= 9 else "ait" if reg >= 10 else "taylor")
    return snapshot


def initialize_inflation_error(model) -> None:
    """ss87 -> ss93 -> ss104: differenced AR(1) error, iid PCE off in regimes 2-10."""
    values = model.get_setting("parameter_regimes")
    mapping = model.get_setting("model2para_regime")
    dates = model.get_setting("regime_dates")
    for name, baseline, covid, path in (
        ("rho_meas_pi", 0.0, 0.2320, lambda r: 1 if r == 1 else 2),
        ("sigma_meas_pi", 0.0, 0.0999, lambda r: 2 if 2 <= r <= 10 else 1),
        ("rho_corepce", 0.2320, 0.0, lambda r: 1),
        ("sigma_corepce", 0.0999, 0.0, lambda r: 2 if 2 <= r <= 10 else 1),
    ):
        if name not in model.parameters:
            model._add_parameter_from_spec(name)
        parameter = model.parameters[name]
        values[name] = {
            1: replace(parameter, value=baseline, fixed=name in ("rho_meas_pi", "sigma_meas_pi")),
            2: replace(parameter, value=covid, fixed=name in ("rho_corepce", "sigma_corepce")),
        }
        if name == "sigma_meas_pi":
            values[name][2] = replace(
                parameter,
                value=covid,
                fixed=False,
                prior=Prior("root_inverse_gamma", nu=2.0, tau=0.4),
            )
        mapping[name] = {r: path(r) for r in dates}
        model.parameters[name] = values[name][1]


def initialize_policy(model) -> None:
    from nydsge.regimes import regime_indices_from_dates

    dates = model.get_setting("regime_dates")
    current = int(regime_indices_from_dates(dates, [model.get_setting("date_forecast_start")])[0])
    model.set_setting("cur_model_regime", current)
    model.set_setting("pgap_ygap_init_date", "2020-Q2")
    model.set_setting("credibility", {r: (r - 5) / 24 for r in range(5, 30)})
    model.set_setting(
        "tvis_information_set", tuple(tuple(range(r, 11)) if 5 <= r <= 10 else (r,) for r in dates)
    )
    values, mapping = model.get_setting("parameter_regimes"), model.get_setting("model2para_regime")
    for name in ("sigma_pgap", "sigma_ygap"):
        p = model.parameters[name]
        values[name] = {1: replace(p, value=0.0), 2: replace(p, value=20.0)}
        mapping[name] = {r: 2 if r == 3 else 1 for r in dates}
    for name in (
        "sigma_r_m",
        *(f"sigma_r_m{i}" for i in range(1, 7)),
        "sigma_ait_rm",
        *(f"sigma_ait_r_m{i}" for i in range(1, 7)),
    ):
        if name == "sigma_ait_rm":
            model._add_parameter_from_spec(name)
        elif name not in model.parameters:
            model.add_parameter(
                Parameter(
                    name,
                    0.2,
                    fixed=False,
                    value_bounds=(0.0, 100.0),
                    transform="exponential",
                    prior=Prior("root_inverse_gamma", nu=4.0, tau=0.2),
                )
            )
        p = model.parameters[name]
        is_ait = "ait" in name
        values[name] = {
            1: replace(p, value=0.0 if is_ait else p.value, fixed=is_ait),
            2: replace(p, value=p.value if is_ait else 0.0, fixed=not is_ait),
        }
        mapping[name] = {
            r: (
                2
                if (9 if name.startswith("sigma_ait_r_m") else 10) <= r
                and (r <= current or not name.startswith("sigma_ait_r_m"))
                else 1
            )
            for r in dates
        }


def policy_canonical(model):
    """Replace ss104 policy and gap rows exactly as flexible_ait/zlb_rule.jl."""
    import numpy as np

    # Called after building the common equations; this function works on their arrays.
    from nydsge.models.m1002_eqcond import equilibrium_matrices_ss10

    model.steadystate()
    model = deepcopy(model)
    model.set_setting("apply_ss104_policy", False)
    c = equilibrium_matrices_ss10(model)
    reg = model.get_setting("model_regime", 1)
    endo, eq, exo = (
        model.indexes.endogenous_states,
        model.indexes.equilibrium_conditions,
        model.indexes.exogenous_shocks,
    )

    def row(key):
        return eq[key] - 1

    def col(key):
        return endo[key] - 1

    for gap in ("pgap", "ygap"):
        i = row(f"eq_{gap}")
        c.Gamma0[i] = 0
        c.Gamma1[i] = 0
        c.Gamma0[i, col(f"{gap}_t")] = 1
        if reg >= 5:
            rho = np.exp(
                np.log(0.5) / model.get_setting("ait_Thalf" if gap == "pgap" else "gdp_Thalf")
            )
            c.Gamma1[i, col(f"{gap}_t")] = rho
            if gap == "pgap":
                c.Gamma0[i, col("pi_t")] = -1
            else:
                c.Gamma0[i, col("y_t")] = c.Gamma0[i, col("z_t")] = -1
                c.Gamma1[i, col("y_t")] = -1
    if reg >= model.get_setting("remove_rm_t_shocks", 8):
        c.Psi[row("eq_rm"), exo["rm_sh"] - 1] = 0
    if reg >= model.get_setting("remove_pistar_shocks", 5):
        c.Psi[row("eq_pi_star"), exo["pi_star_sh"] - 1] = 0
    for i in model.get_setting("mon_anticipated_ait_shocks", ()):
        c.Gamma0[row(f"eq_ait_rml{i}"), col(f"rm_ait_tl{i}")] = 1
        c.Gamma1[row("eq_ait_rm" if i == 1 else f"eq_ait_rml{i - 1}"), col(f"rm_ait_tl{i}")] = 1
        c.Psi[row(f"eq_ait_rml{i}"), exo[f"rm_ait_shl{i}"] - 1] = 1
    policy = model.get_setting("policy", "taylor")
    if policy != "taylor":
        i = row("eq_mp")
        c.Gamma0[i] = c.Gamma1[i] = c.Psi[i] = 0
        c.Gamma0[i, col("R_t")] = 1
        if policy == "ait":
            smooth = model.get_setting("flexible_ait_rho_smooth")
            c.Gamma1[i, col("R_t")] = smooth
            for gap, phi in (("pgap", "pi"), ("ygap", "y")):
                rho = c.Gamma1[row(f"eq_{gap}"), col(f"{gap}_t")]
                c.Gamma0[i, col(f"{gap}_t")] = (
                    -model.get_setting(f"flexible_ait_phi_{phi}") * (1 - rho) * (1 - smooth)
                )
            c.Gamma0[i, col("ait_rm_t")] = -1
        elif policy == "zlb":
            c.C[i] = 0.1 / 4 - model.numeric_value("Rstarn")
            for h in range(1, model.get_setting("n_mon_anticipated_shocks") + 1):
                c.Gamma1[row("eq_rm" if h == 1 else f"eq_rml{h - 1}"), col(f"rm_tl{h}")] = 0
                c.Psi[row(f"eq_rml{h}"), exo[f"rm_shl{h}"] - 1] = 0
        else:
            raise ValueError(f"Unknown ss104 policy: {policy}")
    return c


def compute_ss104_system(model, *, method="auto"):
    """Perfect-policy paths plus imperfect-awareness equations and measurement mixtures."""
    import numpy as np

    from nydsge.regimes import (
        RegimeSwitchingSystem,
        gensys2,
        gensys_to_predictable_form,
    )
    from nydsge.solve import Measurement, PseudoMeasurement, System, Transition, solve_canonical

    model.steadystate()
    models = tuple(model.regime_snapshot(r) for r in range(1, 30))
    # Actual policy, Taylor after the same ZLB, and survey-belief AIT/Taylor pairs.
    scenarios = (
        (9, "ait"),
        (9, "taylor"),
        (10, "ait"),
        (10, "taylor"),
        (14, "ait"),
        (14, "taylor"),
        (15, "ait"),
        (15, "taylor"),
        (18, "ait"),
        (18, "taylor"),
    )
    perfect = []
    solved_cache = {}
    for last_zlb, terminal_policy in scenarios:
        snapshots = deepcopy(models)
        canonicals = []
        for r, m in enumerate(snapshots, 1):
            m.set_setting(
                "policy", "taylor" if r < 5 else "zlb" if r <= last_zlb else terminal_policy
            )
            canonicals.append(m.equilibrium_matrices())
        path = [None] * 29
        for r, c in enumerate(canonicals, 1):
            if 5 <= r <= last_zlb:
                continue
            key = tuple(x.tobytes() for x in (c.Gamma0, c.Gamma1, c.C, c.Psi, c.Pi))
            if key not in solved_cache:
                result = solve_canonical(c, method=method)
                if result.eu != (1, 1):
                    raise ValueError(f"ss104 permanent policy failed in regime {r}: {result.eu}")
                solved_cache[key] = result.transition
            path[r - 1] = solved_cache[key]
        path[4:last_zlb] = gensys2(canonicals[4:last_zlb], path[last_zlb])[:-1]
        path = tuple(path)
        augmented = tuple(m.augment_transition(t) for m, t in zip(snapshots, path, strict=True))
        # The actual path uses ss104's shortened TVIS; alternative paths see their entire future.
        infos = (
            model.get_setting("tvis_information_set")
            if not perfect
            else tuple(tuple(range(r, 30)) if r >= 5 else (r,) for r in range(1, 30))
        )
        systems = tuple(
            System(
                t,
                m.measurement_matrices(
                    t, transitions=augmented, reg=r, information_set=infos[r - 1]
                ),
                m.pseudo_measurement_matrices(
                    t, transitions=augmented, reg=r, information_set=infos[r - 1]
                ),
            )
            for r, (m, t) in enumerate(zip(snapshots, augmented, strict=True), 1)
        )
        perfect.append((path, systems))
    realized = []
    for r, m in enumerate(models, 1):
        if r < 5:
            realized.append(perfect[0][1][r - 1].transition)
            continue
        pair = 8 if r == 5 else 6 if r <= 7 else 4 if r == 8 else 2 if r == 9 else 0
        weight = model.get_setting("credibility")[r]
        future = r if r >= 10 else r + 1
        a, b = perfect[pair][0][future - 1], perfect[pair + 1][0][future - 1]
        p = gensys_to_predictable_form(m.equilibrium_matrices())
        lhs = p.Gamma0 + p.Gamma2 @ (weight * a.TTT + (1 - weight) * b.TTT)
        transition = m.augment_transition(
            Transition(
                np.linalg.solve(lhs, p.Gamma1),
                np.linalg.solve(lhs, p.Psi),
                np.linalg.solve(lhs, p.C - p.Gamma2 @ (weight * a.CCC + (1 - weight) * b.CCC)),
            )
        )
        realized.append(transition)
    systems = []
    for r, (m, transition) in enumerate(zip(models, realized, strict=True), 1):
        if r < 5:
            systems.append(perfect[0][1][r - 1])
            continue
        pair = 8 if r == 5 else 6 if r <= 7 else 4 if r == 8 else 2 if r == 9 else 0
        weight = model.get_setting("credibility")[r]
        meas = m.measurement_matrices(
            transition,
            transitions=realized,
            reg=r,
            information_set=model.get_setting("tvis_information_set")[r - 1],
        )
        zz, dd = meas.ZZ.copy(), meas.DD.copy()
        for name in model.get_setting("forward_looking_observables"):
            i = model.observables[name] - 1
            zz[i] = (
                weight * perfect[pair][1][r - 1]["ZZ"][i]
                + (1 - weight) * perfect[pair + 1][1][r - 1]["ZZ"][i]
            )
            dd[i] = (
                weight * perfect[pair][1][r - 1]["DD"][i]
                + (1 - weight) * perfect[pair + 1][1][r - 1]["DD"][i]
            )
        pa, pb = (
            perfect[pair][1][r - 1].pseudo_measurement,
            perfect[pair + 1][1][r - 1].pseudo_measurement,
        )
        pseudo = PseudoMeasurement(
            weight * pa.ZZ_pseudo + (1 - weight) * pb.ZZ_pseudo,
            weight * pa.DD_pseudo + (1 - weight) * pb.DD_pseudo,
        )
        systems.append(System(transition, Measurement(zz, dd, meas.QQ, meas.EE), pseudo))
    return RegimeSwitchingSystem(
        tuple(systems), information_sets=model.get_setting("tvis_information_set")
    )


def initialize_gap_observations(model, observations):
    """Insert the one-quarter negative gap initialization used by ss104."""
    result = observations.copy()
    date = pd.Period(model.get_setting("pgap_ygap_init_date"), freq="Q")
    for gap in ("pgap", "ygap"):
        result[f"obs_{gap}"] = float("nan")
        if date in result.index:
            result.loc[date, f"obs_{gap}"] = -float(model.get_setting(f"{gap}_value"))
    return result


def initialize_expected_rates(model) -> None:
    model._add_parameter_from_spec("rho_exp_rm")
    values, mapping = model.get_setting("parameter_regimes"), model.get_setting("model2para_regime")
    for horizon in range(1, 7):
        name = f"sigma_exp_rm{horizon}"
        if name not in model.parameters:
            continue
        p = model.parameters[name]
        values[name] = {1: replace(p, value=0.0, fixed=True), 2: replace(p, fixed=False)}
        mapping[name] = {r: 1 if r < 10 else 2 for r in model.get_setting("regime_dates")}
