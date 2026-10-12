"""Compare fixed-default Python ss104 to the source Julia export (no Julia runtime)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from nydsge.forecast import forecast_linear_system
from nydsge.kalman import kalman_log_likelihood, observables_from_states, smooth_kalman_result
from nydsge.models import Model1002
from nydsge.regimes import regime_indices_from_dates
from nydsge.solve import compute_system
from nydsge.ss104_workflow import forecast_ss104

TRANSLATION = {
    "α": "alpha",
    "β": "beta",
    "ρ": "rho",
    "σ": "sigma",
    "π": "pi",
    "ζ": "zeta",
    "ι": "iota",
    "μ": "mu",
    "ω": "omega",
    "γ": "gamma",
    "λ": "lambda",
    "δ": "delta",
    "Φ": "Phi",
    "Γ": "Gamma",
    "ν": "nu",
    "ϵ": "epsilon",
    "φ": "phi",
    "S′′": "Spp",
    "ψ": "psi",
    "η": "eta",
}


def translate(name):
    for before, after in TRANSLATION.items():
        name = name.replace(before, after)
    return name


def compare(path, *, model=None, system=None):
    model = model or Model1002("ss104")
    system = compute_system(model) if system is None else system
    report = {"max_abs": {}, "regimes": {}}
    with h5py.File(path) as f:

        def names(key):
            return [translate(n) for n in f[f"names/{key}"].asstr()[...]]

        def permutation(reference, python):
            if set(reference) != set(python):
                raise ValueError(
                    f"Name mismatch: Julia-only {set(reference) - set(python)}; "
                    f"Python-only {set(python) - set(reference)}"
                )
            return [reference.index(k) for k in python]

        states = permutation(
            names("endogenous_states") + names("endogenous_states_augmented"),
            list(model.indexes.endogenous_states) + list(model.indexes.endogenous_states_augmented),
        )
        shocks = permutation(names("exogenous_shocks"), list(model.indexes.exogenous_shocks))
        obs = permutation(names("observables"), list(model.observables))
        eq = permutation(
            names("equilibrium_conditions"), list(model.indexes.equilibrium_conditions)
        )
        expected = permutation(names("expected_shocks"), list(model.indexes.expected_shocks))
        endo = permutation(names("endogenous_states"), list(model.indexes.endogenous_states))
        pseudo_aliases = {"pgap_t": "pgap", "ygap_t": "ygap"}
        pseudo = permutation(
            names("pseudo_observables"),
            [pseudo_aliases.get(translate(k), translate(k)) for k in model.pseudo_observables],
        )
        layouts = {
            "TTT": (states, states),
            "RRR": (states, shocks),
            "CCC": (states,),
            "ZZ": (obs, states),
            "DD": (obs,),
            "QQ": (shocks, shocks),
            "EE": (obs, obs),
            "Gamma0": (eq, endo),
            "Gamma1": (eq, endo),
            "C": (eq,),
            "Psi": (eq, shocks),
            "Pi": (eq, expected),
            "ZZ_pseudo": (pseudo, states),
            "DD_pseudo": (pseudo,),
        }
        report["metadata"] = json.loads(f["metadata/json"].asstr()[()])
        for r in range(1, 30):
            snapshot = model.regime_snapshot(r)
            snapshot.set_setting("policy", "taylor")
            canonical = snapshot.equilibrium_matrices()
            errors = {}
            for name, layout in layouts.items():
                namespace = (
                    "canonical" if name in {"Gamma0", "Gamma1", "C", "Psi", "Pi"} else "system"
                )
                actual = getattr(canonical, name) if namespace == "canonical" else system[r, name]
                reference = f[f"{namespace}/{r}/{name}"][...].T
                reference = reference[np.ix_(*layout)] if len(layout) == 2 else reference[layout[0]]
                error = float(np.max(np.abs(reference - actual)))
                errors[name] = error
                report["max_abs"][name] = max(report["max_abs"].get(name, 0), error)
                if error > 1e-8:
                    index = np.unravel_index(np.argmax(np.abs(reference - actual)), actual.shape)
                    print(r, name, error, index, "py", actual[index], "jl", reference[index])
            parameters = dict(
                zip(
                    map(translate, f[f"parameters/{r}/names"].asstr()[...]),
                    f[f"parameters/{r}/values"][...],
                    strict=True,
                )
            )
            if set(parameters) != set(snapshot.parameters):
                raise ValueError("Oracle and candidate parameter names differ.")
            parameter_error = max(
                abs(p.value - parameters[name]) for name, p in snapshot.parameters.items()
            )
            report["max_abs"]["parameters"] = max(
                report["max_abs"].get("parameters", 0), parameter_error
            )
            for name, parameter in snapshot.parameters.items():
                if name in parameters and abs(parameter.value - parameters[name]) > 1e-12:
                    print("parameter", r, name, parameter.value, parameters[name])
            report["regimes"][str(r)] = errors
        if "workflow/data" in f:
            data = f["workflow/data"][...][:, obs]
            report["workflow"] = {}
            for label, count in (("history", 27), ("conditional", 28)):
                result = kalman_log_likelihood(
                    system, data[:count], regime_indices=np.arange(1, count + 1)
                )
                smooth = smooth_kalman_result(system, result, data=data[:count])
                reconstructed = observables_from_states(
                    system, smooth.smoothed_states, regime_indices=np.arange(1, count + 1)
                )
                report["workflow"][f"{label}_smoother_observation_residual"] = float(
                    np.nanmax(np.abs(reconstructed - data[:count]))
                )
                prefix = f"workflow/{label}"
                reference_likelihood = float(f[f"{prefix}/loglik"][...].sum())
                report["workflow"][f"{label}_loglik"] = {
                    "python": result.log_likelihood,
                    "julia": reference_likelihood,
                    "error": abs(result.log_likelihood - reference_likelihood),
                }
                for name in ("predicted_states", "filtered_states", "smoothed_states"):
                    reference = f[f"{prefix}/{name}"][...][:, states]
                    report["workflow"][f"{label}_{name}_error"] = float(
                        np.max(np.abs(reference - getattr(smooth, name)))
                    )
                for name in ("predicted_covariances", "filtered_covariances"):
                    reference = f[f"{prefix}/{name}"][...][:, states][:, :, states]
                    report["workflow"][f"{label}_{name}_error"] = float(
                        np.max(np.abs(reference - getattr(result, name)))
                    )
                if label == "history":
                    path = regime_indices_from_dates(
                        model.get_setting("regime_dates"),
                        pd.period_range("2026Q3", periods=40, freq="Q"),
                    )
                    prediction = forecast_linear_system(
                        system,
                        result.final_filtered_state,
                        horizon=40,
                        regime_indices=path,
                        include_pseudo=True,
                    )
                else:
                    frame = pd.DataFrame(
                        data,
                        index=pd.period_range("2019Q4", periods=28, freq="Q"),
                        columns=list(model.observables),
                    )
                    prediction = forecast_ss104(
                        model, system=system, data=frame, horizon=40, cond_type="full"
                    )
                forecast_label = "forecast" if label == "history" else "conditional_forecast"
                for name, layout in (
                    ("states", states),
                    ("observables", obs),
                    ("pseudo_observables", pseudo),
                ):
                    dataset_name = "pseudo" if name == "pseudo_observables" else name
                    reference = f[f"workflow/{forecast_label}/{dataset_name}"][...][:, layout]
                    report["workflow"][f"{forecast_label}_{name}_error"] = float(
                        np.max(np.abs(reference - getattr(prediction, name)))
                    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/ss104-parity.json"))
    args = parser.parse_args()
    report = compare(args.oracle)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["max_abs"], indent=2))
    print(json.dumps(report.get("workflow", {}), indent=2))
