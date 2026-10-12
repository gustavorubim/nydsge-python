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
    return snapshot
