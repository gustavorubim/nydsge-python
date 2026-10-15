"""Verified deterministic policy paths using native regime-aware forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nydsge.forecast import forecast_linear_system
from nydsge.regimes import RegimeSwitchingSystem


def policy_paths(
    system: RegimeSwitchingSystem,
    state: np.ndarray,
    regimes: np.ndarray,
    dates: pd.PeriodIndex,
    names: list[str],
    shocks: list[str],
    specifications: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if len(specifications) > 8:
        raise ValueError("At most eight policy paths.")
    n, rate = len(dates), names.index("obs_nominalrate")
    baseline = forecast_linear_system(system, state, horizon=n, regime_indices=regimes).observables
    active = system.period_systems(regimes, n)
    outputs, records = {}, []
    for spec in specifications:
        label = spec["label"]
        if label in outputs or label == "Baseline":
            raise ValueError("Policy path labels must be unique.")
        start = pd.Period(spec["start_quarter"], freq="Q")
        if start not in dates:
            raise ValueError("Policy target start outside forecast.")
        first = dates.get_loc(start)
        absolute = "rates_annual_percent" in spec
        if absolute == ("deviations_bp" in spec):
            raise ValueError("Specify exactly one absolute rate or baseline-deviation path.")
        values = spec["rates_annual_percent"] if absolute else spec["deviations_bp"]
        if not values or first + len(values) > n:
            raise ValueError("Policy targets outside forecast horizon.")
        indices = [first + k for k, value in enumerate(values) if value is not None]
        if not indices:
            raise ValueError("Policy path needs a finite target.")
        targets = np.asarray([values[k - first] for k in indices], dtype=float)
        if not np.isfinite(targets).all():
            raise ValueError("Policy path targets must be finite.")
        if not absolute:
            targets = 4 * baseline[indices, rate] + targets / 100
        innovations = np.zeros((n, len(shocks)))
        support = []
        if spec["kind"] == "held_rate":
            if spec.get("information") != "sequential_surprises":
                raise ValueError("Held-rate paths require sequential_surprises information.")
            current = state.copy()
            target_map = dict(zip(indices, targets, strict=True))
            for t in range(n):
                step = forecast_linear_system(active[t], current, horizon=1)
                if t in target_map:
                    candidates = [s for s in ("rm_ait_sh", "rm_sh") if s in shocks]
                    selected = None
                    for shock in candidates:
                        j = shocks.index(shock)
                        loading = float(
                            4 * active[t].measurement.ZZ[rate] @ active[t].transition.RRR[:, j]
                        )
                        if abs(loading) > 1e-12 and active[t].measurement.QQ[j, j] > 0:
                            selected = j
                            break
                    if selected is None:
                        raise ValueError("No active monetary surprise for held-rate target.")
                    innovations[t, selected] = (
                        target_map[t] - 4 * step.observables[0, rate]
                    ) / loading
                    step = forecast_linear_system(
                        active[t], current, horizon=1, shocks=innovations[t : t + 1]
                    )
                current = step.states[-1]
            assumption = "Each innovation is unexpected when it arrives; no advance announcement."
        elif spec["kind"] == "forward_guidance":
            if spec.get("information") != "announced":
                raise ValueError("Forward guidance requires announced information.")
            announcement = pd.Period(spec["announcement_quarter"], freq="Q")
            if announcement not in dates:
                raise ValueError("Announcement outside forecast.")
            a = dates.get_loc(announcement)
            horizons = [t - a for t in indices]
            if any(h < 1 or h > 6 for h in horizons):
                raise ValueError("Native forward guidance supports target horizons 1 through 6.")
            columns = []
            shock_indices = []
            for h in horizons:
                shock = f"rm_ait_shl{h}"
                if shock not in shocks:
                    raise ValueError("Unsupported anticipated monetary shock.")
                j = shocks.index(shock)
                unit = np.zeros_like(innovations)
                unit[a, j] = 1
                trial = forecast_linear_system(
                    system, state, horizon=n, regime_indices=regimes, shocks=unit
                )
                columns.append(4 * (trial.observables[indices, rate] - baseline[indices, rate]))
                shock_indices.append(j)
                support.append(
                    {
                        "shock": shock,
                        "positive_prior_variance": bool(active[a].measurement.QQ[j, j] > 0),
                    }
                )
            design = np.column_stack(columns)
            solution, _, rank, _ = np.linalg.lstsq(
                design, targets - 4 * baseline[indices, rate], rcond=None
            )
            if rank != len(indices):
                raise ValueError("Forward-guidance targets are not independently reachable.")
            innovations[a, shock_indices] = solution
            assumption = (
                "Future primitive monetary news becomes known at announcement; "
                "no earlier information. Zero-variance news is an "
                "off-prior-support hypothetical."
            )
        else:
            raise ValueError("Unsupported policy path kind.")
        result = forecast_linear_system(
            system, state, horizon=n, regime_indices=regimes, shocks=innovations
        )
        actual = 4 * result.observables[indices, rate]
        error = float(np.max(np.abs(actual - targets)))
        if not np.isfinite(result.observables).all() or error > 1e-8:
            raise ValueError("Policy path failed actual annual-rate target verification.")
        outputs[label] = result.observables
        records.append(
            {
                "label": label,
                "kind": spec["kind"],
                "information": spec["information"],
                "assumption": assumption,
                "specification": spec,
                "max_target_error_annual_pp": error,
                "targets": [
                    {
                        "quarter": str(dates[t]),
                        "target_annual_percent": float(v),
                        "actual_annual_percent": float(x),
                    }
                    for t, v, x in zip(indices, targets, actual, strict=True)
                ],
                "innovations": innovations.tolist(),
                "anticipated_shock_support": support,
            }
        )
    return outputs, records
