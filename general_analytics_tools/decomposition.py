"""Diagnostic multi-regime disturbance attribution using the native filter/smoother.

This is not a replacement for nydsge's single-regime historical_decomposition API.
Every contribution, initial condition, deterministic term and remainder is retained.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nydsge.decomposition import regime_disturbance_attribution as disturbance_attribution

__all__ = ["disturbance_attribution", "export_decomposition", "shock_groups"]

HD_TOLERANCE = 1e-8  # Annualized log percentage points, never loosened by configuration.


def shock_groups(shocks: list[str]) -> dict[str, list[int]]:
    groups = {
        "Policy residual": [s for s in shocks if s.startswith(("rm_sh", "rm_ait_sh"))],
        "Price markups": ["lambda_f_sh"],
        "Wage markups": ["lambda_w_sh"],
        "Inflation target": ["pi_star_sh"],
        "Private demand": ["b_sh"],
        "Productivity": ["ztil_sh", "zp_sh"],
        "Investment / financial": ["mu_sh", "sigma_omega_sh", "mu_e_sh", "gamma_sh"],
        "Government spending": ["g_sh"],
        "Measurement / gap initialization": [
            "lr_sh",
            "tfp_sh",
            "gdpdef_sh",
            "corepce_sh",
            "gdp_sh",
            "gdi_sh",
            "meas_pi_sh",
            "pgap_sh",
            "ygap_sh",
            "condgdp_sh",
            "condcorepce_sh",
            *[f"exp_rm_sh{i}" for i in range(1, 7)],
        ],
    }
    indices = {
        label: [i for i, name in enumerate(shocks) if name in names]
        for label, names in groups.items()
    }
    used = {i for ids in indices.values() for i in ids}
    indices["COVID / other innovations"] = [i for i in range(len(shocks)) if i not in used]
    return indices


def export_decomposition(
    arrays: dict[str, np.ndarray], dates: pd.PeriodIndex, names: list[str], shocks: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation = {}
    for name in ("obs_corepce", "obs_gdp", "obs_nominalrate"):
        if name not in names:
            continue
        i = names.index(name)
        components = {
            label: 4 * arrays["contributions"][:, i, ids].sum(axis=1)
            for label, ids in shock_groups(shocks).items()
        }
        components.update(
            {
                "Initial state": 4 * arrays["initial"][:, i],
                "Deterministic terms": 4 * arrays["deterministic"][:, i],
                "Numerical remainder": 4 * arrays["remainder"][:, i],
            }
        )
        error = float(np.max(np.abs(components["Numerical remainder"])))
        validation[name] = {
            "max_reconstruction_error_plotted_units": error,
            "units": "annual rate pp" if name == "obs_nominalrate" else "annualized log pp",
            "max_reconstruction_error_annual_log_pp": error,
            "tolerance": HD_TOLERANCE,
            "status": "reconstruction_pass" if error <= HD_TOLERANCE else "diagnostic",
            "worst_reconstruction_quarter": str(
                dates[int(np.argmax(np.abs(components["Numerical remainder"])))]
            ),
            **(
                {"max_innovation_condition_number": float(np.max(arrays["innovation_condition"]))}
                if "innovation_condition" in arrays
                else {}
            ),
            **(
                {"max_state_reconstruction_error": float(np.max(np.abs(arrays["state_remainder"])))}
                if "state_remainder" in arrays
                else {}
            ),
        }
        for label, values in components.items():
            rows.extend(
                {
                    "quarter": str(date),
                    "variable": name,
                    "component": label,
                    "contribution": float(value),
                }
                for date, value in zip(dates, values, strict=True)
            )
    return pd.DataFrame(rows), validation
