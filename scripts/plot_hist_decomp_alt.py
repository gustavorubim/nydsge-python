#!/usr/bin/env python3
"""Alternative historical-decomposition category groupings for desk review.

Reuses Model1002 + desk_fomc vintage (refreshed shock mode + observables).
Runs historical_decomposition once, exports per-shock contribs, plots presets.

Usage:
  uv run python scripts/plot_hist_decomp_alt.py \\
    --vintage outputs/desk_fomc_20260916 \\
    --out-dir /workspace/desk-plots/2026-09-16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nydsge.data import reverse_transform_observables
from nydsge.estimate import load_estimation_mode
from nydsge.forecast import historical_decomposition
from nydsge.models import Model1002
from nydsge.parameters import update_parameter_value

# ---------------------------------------------------------------------------
# Desk taxonomy (reference) — same as economy.SHOCK_GROUPS
# ---------------------------------------------------------------------------
DESK_GROUPS: dict[str, tuple[str, ...]] = {
    "Government spending": ("g_sh",),
    "Private demand": ("b_sh",),
    "Productivity and supply": ("ztil_sh", "zp_sh"),
    "Price markup": ("lambda_f_sh",),
    "Wage markup": ("lambda_w_sh",),
    "Monetary policy": (
        "rm_sh",
        "rm_shl1",
        "rm_shl2",
        "rm_shl3",
        "rm_shl4",
        "rm_shl5",
        "rm_shl6",
    ),
    "Inflation target": ("pi_star_sh",),
    "Investment and finance": (
        "mu_sh",
        "sigma_omega_sh",
        "mu_e_sh",
        "gamma_sh",
    ),
    "Measurement innovations": (
        "lr_sh",
        "tfp_sh",
        "gdpdef_sh",
        "corepce_sh",
        "gdp_sh",
        "gdi_sh",
    ),
}

RM_ANTICIPATED = ("rm_shl1", "rm_shl2", "rm_shl3", "rm_shl4", "rm_shl5", "rm_shl6")
RM_ALL = ("rm_sh",) + RM_ANTICIPATED

PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "desk": {
        "Government": DESK_GROUPS["Government spending"],
        "Private demand": DESK_GROUPS["Private demand"],
        "Productivity and supply": DESK_GROUPS["Productivity and supply"],
        "Price markup": DESK_GROUPS["Price markup"],
        "Wage markup": DESK_GROUPS["Wage markup"],
        "Monetary policy": DESK_GROUPS["Monetary policy"],
        "Inflation target": DESK_GROUPS["Inflation target"],
        "Investment and finance": DESK_GROUPS["Investment and finance"],
        "Measurement innovations": DESK_GROUPS["Measurement innovations"],
    },
    "nk": {
        "Cost-push": ("lambda_f_sh", "lambda_w_sh"),
        "Demand/IS": ("b_sh", "g_sh"),
        "Technology": ("ztil_sh", "zp_sh"),
        "Policy": RM_ALL,
        "Inflation target": ("pi_star_sh",),
        "Financial": ("mu_sh", "sigma_omega_sh", "mu_e_sh", "gamma_sh"),
        "Measurement": (
            "lr_sh",
            "tfp_sh",
            "gdpdef_sh",
            "corepce_sh",
            "gdp_sh",
            "gdi_sh",
        ),
    },
    "mandate": {
        "Inflation-side": (
            "lambda_f_sh",
            "lambda_w_sh",
            "pi_star_sh",
            "corepce_sh",
        ),
        "Real activity": ("b_sh", "g_sh", "ztil_sh", "zp_sh"),
        "Financial": ("mu_sh", "sigma_omega_sh", "mu_e_sh", "gamma_sh"),
        "Monetary": RM_ALL,
        "Other measurement": (
            "lr_sh",
            "tfp_sh",
            "gdpdef_sh",
            "gdp_sh",
            "gdi_sh",
        ),
    },
    "policy_split": {
        "Unexpected policy": ("rm_sh",),
        "Anticipated policy": RM_ANTICIPATED,
        "Markups": ("lambda_f_sh", "lambda_w_sh"),
        "Demand": ("b_sh",),
        "Government": ("g_sh",),
        "Supply": ("ztil_sh", "zp_sh"),
        "Inflation target": ("pi_star_sh",),
        "Financial": ("mu_sh", "sigma_omega_sh", "mu_e_sh", "gamma_sh"),
        "Measurement": (
            "lr_sh",
            "tfp_sh",
            "gdpdef_sh",
            "corepce_sh",
            "gdp_sh",
            "gdi_sh",
        ),
    },
    "collapsed": {
        "Cost-push": ("lambda_f_sh", "lambda_w_sh"),
        "Demand": ("b_sh", "g_sh"),
        "Policy": RM_ALL,
        "Target": ("pi_star_sh",),
        "Other": (
            "mu_sh",
            "sigma_omega_sh",
            "mu_e_sh",
            "gamma_sh",
            "lr_sh",
            "tfp_sh",
            "gdpdef_sh",
            "corepce_sh",
            "gdp_sh",
            "gdi_sh",
            "ztil_sh",
            "zp_sh",
        ),
    },
}

# Panel-B exclusions: IC+trend always; target-like buckets when present.
LEVEL_ANCHOR_NAMES = {
    "desk": ("IC+trend", "Inflation target"),
    "nk": ("IC+trend", "Inflation target"),
    "mandate": ("IC+trend",),  # pi_star inside Inflation-side; exclude IC only
    "policy_split": ("IC+trend", "Inflation target"),
    "collapsed": ("IC+trend", "Target"),
    "topn": ("IC+trend",),
}

COLORS: dict[str, str] = {
    "IC+trend": "#C8CDD3",
    "Initial conditions and trend": "#C8CDD3",
    "Inflation target": "#8FB8D8",
    "Target": "#8FB8D8",
    "Price and wage markups": "#A65A52",
    "Price markup": "#A65A52",
    "Wage markup": "#D4A5A0",
    "Markups": "#A65A52",
    "Cost-push": "#A65A52",
    "Private demand": "#D9822B",
    "Demand": "#D9822B",
    "Demand/IS": "#D9822B",
    "Monetary policy": "#34699A",
    "Policy": "#34699A",
    "Monetary": "#34699A",
    "Unexpected policy": "#1B4F72",
    "Anticipated policy": "#5DADE2",
    "Productivity and supply": "#6E9E45",
    "Technology": "#6E9E45",
    "Supply": "#6E9E45",
    "Government": "#E7B64A",
    "Government spending": "#E7B64A",
    "Investment and finance": "#A56CC1",
    "Financial": "#A56CC1",
    "Measurement innovations": "#7A8088",
    "Measurement": "#7A8088",
    "Other measurement": "#9AA0A6",
    "Other": "#7A8088",
    "Other shocks": "#7A8088",
    "Inflation-side": "#C0392B",
    "Real activity": "#27AE60",
    "Observed-model residual": "#3F454C",
}

# Extra palette for individual shock names in topn
_TOPN_PALETTE = [
    "#A65A52",
    "#D9822B",
    "#34699A",
    "#6E9E45",
    "#E7B64A",
    "#A56CC1",
    "#16A085",
    "#E74C3C",
]


def _apply_refreshed_mode(model: Model1002, mode_path: Path) -> list[str]:
    mode = load_estimation_mode(mode_path)
    for name, value in zip(mode.parameter_names, mode.estimation_values, strict=True):
        model.parameters[name] = update_parameter_value(model.parameters[name], float(value))
    return list(mode.parameter_names)


def _report_scale(model, smoothed_model: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match economy._group_decomposition_rows scale for report-unit mapping."""
    smoothed_report = reverse_transform_observables(model, smoothed_model)
    epsilon = 1.0e-7
    derivative = (
        reverse_transform_observables(model, np.full_like(smoothed_model, epsilon)) / epsilon
    )
    scale = np.divide(
        smoothed_report,
        smoothed_model,
        out=derivative,
        where=np.abs(smoothed_model) > 1.0e-10,
    )
    return scale, smoothed_report


def export_per_shock(
    *,
    vintage: Path,
    out_csv: Path,
    variable: str = "obs_corepce",
    also_gdp: bool = False,
) -> pd.DataFrame:
    """Run historical_decomposition and write per-shock report-unit CSV."""
    observables = pd.read_csv(vintage / "observables.csv")
    mode_path = vintage / "updated_shock_mode.npz"
    meta = json.loads((vintage / "run_metadata.json").read_text())
    model_end = meta["configuration"]["model_end_date"]
    forecast_start = _next_quarter(model_end)

    model = Model1002(
        subspec="ss10",
        settings={
            "data_vintage": model_end.replace("-Q", "q"),
            "date_forecast_start": forecast_start,
        },
    )
    applied = _apply_refreshed_mode(model, mode_path)
    print(f"Applied refreshed mode params: {applied}")

    decomp = historical_decomposition(
        model,
        data=observables,
        check_empty_columns=False,
    )
    shock_names = list(model.indexes.exogenous_shocks)
    dates = list(observables["date"].astype(str))[-decomp.observed.shape[0] :]
    var_names = list(model.observables)
    if variable not in var_names:
        raise SystemExit(f"{variable} not in observables: {var_names}")

    smoothed_model = decomp.smoothed_observables
    scale, smoothed_report = _report_scale(model, smoothed_model)
    # contributions: (T, n_obs, n_shocks)
    contrib_report = decomp.observable_contributions * scale[:, :, np.newaxis]
    baseline_report = decomp.observable_baseline * scale

    rows: list[dict] = []
    targets = [variable]
    if also_gdp and "obs_gdp" in var_names and "obs_gdp" != variable:
        targets.append("obs_gdp")

    for varname in targets:
        vi = var_names.index(varname)
        for t, date in enumerate(dates):
            sm = float(smoothed_report[t, vi])
            for j, shock in enumerate(shock_names):
                rows.append(
                    {
                        "date": date,
                        "variable": varname,
                        "shock": shock,
                        "contribution": float(contrib_report[t, vi, j]),
                        "smoothed": sm,
                    }
                )
            rows.append(
                {
                    "date": date,
                    "variable": varname,
                    "shock": "IC+trend",
                    "contribution": float(baseline_report[t, vi]),
                    "smoothed": sm,
                }
            )

    frame = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Task schema: date, shock, contribution, smoothed — filter to corepce for primary CSV
    core = frame[frame["variable"] == variable][["date", "shock", "contribution", "smoothed"]]
    core.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(core)} rows)")

    # Sanity: reconcile vs existing grouped for overlap quarters
    grouped_path = vintage / "historical_decomposition_grouped.csv"
    if grouped_path.exists():
        _validate_against_grouped(core, grouped_path, variable, shock_names)

    # Caveats on anticipated rm
    caveats = []
    for sh in RM_ANTICIPATED:
        sub = core[core["shock"] == sh]
        if sub.empty or (sub["contribution"].abs().max() < 1e-14):
            caveats.append(f"{sh} contributions are ~0 / empty in sample")
    if caveats:
        print("CAVEATS:")
        for c in caveats:
            print(f"  - {c}")

    return frame


def _validate_against_grouped(
    per_shock: pd.DataFrame,
    grouped_path: Path,
    variable: str,
    shock_names: list[str],
) -> None:
    g = pd.read_csv(grouped_path)
    g = g[(g["variable"] == variable) & (g["variable_kind"] == "observable")]
    if g.empty:
        return
    # Rebuild desk groups from per-shock and compare latest shared date
    shared_dates = sorted(set(per_shock["date"]) & set(g["date"]))
    if not shared_dates:
        return
    d = shared_dates[-1]
    ps = per_shock[per_shock["date"] == d].set_index("shock")["contribution"]
    gg = g[g["date"] == d].set_index("driver")["contribution"]
    max_err = 0.0
    for group, shocks in DESK_GROUPS.items():
        if group not in gg.index:
            continue
        rebuilt = float(sum(float(ps.get(s, 0.0)) for s in shocks))
        err = abs(rebuilt - float(gg[group]))
        max_err = max(max_err, err)
    if "IC+trend" in ps.index and "Initial conditions and trend" in gg.index:
        max_err = max(
            max_err,
            abs(float(ps["IC+trend"]) - float(gg["Initial conditions and trend"])),
        )
    print(f"Validation vs grouped CSV @ {d}: max abs group err = {max_err:.3e}")


def _next_quarter(label: str) -> str:
    year, q = label.split("-Q")
    year_i, q_i = int(year), int(q)
    if q_i == 4:
        return f"{year_i + 1}-Q1"
    return f"{year_i}-Q{q_i + 1}"


def _stack_bars(ax, wide: pd.DataFrame, drivers: list[str], colors: dict[str, str]) -> None:
    bottom_pos = np.zeros(len(wide))
    bottom_neg = np.zeros(len(wide))
    x = np.arange(len(wide))
    for d in drivers:
        if d not in wide.columns:
            continue
        v = wide[d].fillna(0).to_numpy()
        pos = np.where(v >= 0, v, 0)
        neg = np.where(v < 0, v, 0)
        color = colors.get(d, "#888888")
        ax.bar(x, pos, bottom=bottom_pos, color=color, width=0.9, label=d)
        ax.bar(x, neg, bottom=bottom_neg, color=color, width=0.9)
        bottom_pos = bottom_pos + pos
        bottom_neg = bottom_neg + neg


def _aggregate_preset(
    per_shock: pd.DataFrame,
    *,
    groups: dict[str, tuple[str, ...]],
    dates: list[str],
) -> pd.DataFrame:
    """Return wide date x driver contributions including IC+trend."""
    sub = per_shock[per_shock["date"].isin(dates)].copy()
    pieces = []
    for group, shocks in groups.items():
        part = sub[sub["shock"].isin(shocks)].groupby("date", as_index=False)["contribution"].sum()
        part["driver"] = group
        pieces.append(part)
    ic = sub[sub["shock"] == "IC+trend"][["date", "contribution"]].copy()
    ic["driver"] = "IC+trend"
    pieces.append(ic)
    long = pd.concat(pieces, ignore_index=True)
    order = ["IC+trend", *groups.keys()]
    wide = long.pivot(index="date", columns="driver", values="contribution").reindex(
        index=dates, columns=order
    )
    return wide


def _topn_groups(
    per_shock: pd.DataFrame,
    *,
    dates: list[str],
    n: int = 6,
) -> dict[str, tuple[str, ...]]:
    sub = per_shock[per_shock["date"].isin(dates) & (per_shock["shock"] != "IC+trend")]
    ranks = (
        sub.groupby("shock")["contribution"]
        .apply(lambda s: float(np.mean(np.abs(s))))
        .sort_values(ascending=False)
    )
    top = list(ranks.head(n).index)
    rest = [s for s in ranks.index if s not in top]
    groups: dict[str, tuple[str, ...]] = {sh: (sh,) for sh in top}
    if rest:
        groups["Other shocks"] = tuple(rest)
    return groups


def plot_preset(
    per_shock: pd.DataFrame,
    *,
    preset: str,
    out: Path,
    n_quarters: int = 40,
    variable_label: str = "obs_corepce",
) -> Path:
    dates_all = sorted(per_shock["date"].unique())
    dates = dates_all[-n_quarters:]
    sm = per_shock[per_shock["date"].isin(dates)].groupby("date")["smoothed"].first().reindex(dates)

    if preset == "topn":
        groups = _topn_groups(per_shock, dates=dates, n=6)
        # Assign colors for individual shocks
        colors = dict(COLORS)
        for i, name in enumerate(groups):
            if name not in colors:
                colors[name] = _TOPN_PALETTE[i % len(_TOPN_PALETTE)]
    else:
        groups = PRESETS[preset]
        colors = COLORS

    wide = _aggregate_preset(per_shock, groups=groups, dates=dates)
    order = list(wide.columns)
    anchors = LEVEL_ANCHOR_NAMES[preset]
    cyc = [d for d in order if d not in anchors]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = np.arange(len(wide))

    ax = axes[0]
    _stack_bars(ax, wide, order, colors)
    ax.plot(x, sm.to_numpy(), "k-", lw=1.5, label=f"Smoothed {variable_label}")
    ax.set_ylabel("% SAAR")
    ax.set_title(f"A. Full hist decomp (levels) — preset `{preset}` — Model1002 {variable_label}")
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    ax.axhline(0, color="k", lw=0.5)

    ax = axes[1]
    _stack_bars(ax, wide, cyc, colors)
    cyc_sum = wide[[c for c in cyc if c in wide.columns]].sum(axis=1)
    ax.plot(x, cyc_sum.to_numpy(), "k-", lw=1.5, label="Sum of cyclical drivers")
    ax.set_ylabel("pp")
    excl = " + ".join(anchors)
    ax.set_title(f"B. Cyclical shocks only (ex {excl}) — preset `{preset}`")
    ax.legend(loc="upper left", fontsize=6.5, ncol=2)
    ax.axhline(0, color="k", lw=0.5)

    step = max(1, len(x) // 10)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45, ha="right")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_contact_sheet(
    per_shock: pd.DataFrame,
    *,
    out: Path,
    n_quarters: int = 40,
    presets: list[str] | None = None,
) -> Path:
    presets = presets or ["desk", "nk", "mandate", "policy_split", "collapsed", "topn"]
    dates_all = sorted(per_shock["date"].unique())
    dates = dates_all[-n_quarters:]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, preset in zip(axes.ravel(), presets, strict=True):
        if preset == "topn":
            groups = _topn_groups(per_shock, dates=dates, n=6)
            colors = dict(COLORS)
            for i, name in enumerate(groups):
                if name not in colors:
                    colors[name] = _TOPN_PALETTE[i % len(_TOPN_PALETTE)]
        else:
            groups = PRESETS[preset]
            colors = COLORS
        wide = _aggregate_preset(per_shock, groups=groups, dates=dates)
        anchors = LEVEL_ANCHOR_NAMES[preset]
        cyc = [d for d in wide.columns if d not in anchors]
        _stack_bars(ax, wide, cyc, colors)
        cyc_sum = wide[[c for c in cyc if c in wide.columns]].sum(axis=1)
        x = np.arange(len(wide))
        ax.plot(x, cyc_sum.to_numpy(), "k-", lw=1.0)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_title(f"`{preset}` (panel B)", fontsize=10)
        step = max(1, len(x) // 5)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(
            [dates[i] for i in range(0, len(dates), step)], rotation=45, ha="right", fontsize=7
        )
    fig.suptitle(
        "Hist-decomp alt presets — cyclical panel (ex IC+trend / target anchors)",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def contribution_table_2026q2(
    per_shock: pd.DataFrame,
    *,
    date: str = "2026-Q2",
) -> dict[str, dict[str, float]]:
    tables: dict[str, dict[str, float]] = {}
    dates_all = sorted(per_shock["date"].unique())
    window = dates_all[-40:]
    for preset in ["desk", "nk", "mandate", "policy_split", "collapsed", "topn"]:
        if preset == "topn":
            groups = _topn_groups(per_shock, dates=window, n=6)
        else:
            groups = PRESETS[preset]
        wide = _aggregate_preset(
            per_shock, groups=groups, dates=[date] if date in dates_all else window[-1:]
        )
        use_date = date if date in wide.index else wide.index[-1]
        tables[preset] = {
            col: float(wide.loc[use_date, col])
            for col in wide.columns
            if pd.notna(wide.loc[use_date, col])
        }
    return tables


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vintage",
        type=Path,
        default=Path("outputs/desk_fomc_20260916"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/desk-plots/2026-09-16"),
    )
    p.add_argument("--n-quarters", type=int, default=40)
    p.add_argument(
        "--per-shock-csv",
        type=Path,
        default=None,
        help="Default: <out-dir>/hist_decomp_per_shock_corepce.csv",
    )
    p.add_argument("--skip-export", action="store_true")
    args = p.parse_args()

    vintage = args.vintage.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.per_shock_csv or (out_dir / "hist_decomp_per_shock_corepce.csv")

    if args.skip_export and csv_path.exists():
        per = pd.read_csv(csv_path)
        # reconstruct variable column if absent
        if "variable" not in per.columns:
            per["variable"] = "obs_corepce"
        print(f"Reusing existing {csv_path}")
    else:
        full = export_per_shock(
            vintage=vintage,
            out_csv=csv_path,
            variable="obs_corepce",
            also_gdp=False,
        )
        per = full[full["variable"] == "obs_corepce"].copy()

    # Ensure IC+trend present
    if "IC+trend" not in set(per["shock"]):
        raise SystemExit("per-shock CSV missing IC+trend rows")

    pngs = []
    for preset in ["desk", "nk", "mandate", "policy_split", "collapsed", "topn"]:
        path = plot_preset(
            per,
            preset=preset,
            out=out_dir / f"hist_decomp_alt_{preset}.png",
            n_quarters=args.n_quarters,
        )
        pngs.append(path)
        print(f"Wrote {path}")

    contact = plot_contact_sheet(
        per,
        out=out_dir / "hist_decomp_alt_contact.png",
        n_quarters=args.n_quarters,
    )
    pngs.append(contact)
    print(f"Wrote {contact}")

    tables = contribution_table_2026q2(per)
    summary_path = out_dir / "hist_decomp_alt_2026Q2_table.json"
    summary_path.write_text(json.dumps(tables, indent=2))
    print(f"Wrote {summary_path}")
    print("\n=== 2026-Q2 contributions by preset ===")
    for preset, buckets in tables.items():
        print(f"\n[{preset}]")
        for k, v in buckets.items():
            print(f"  {k:28s} {v:+8.3f}")

    print("\nPNG paths:")
    for path in pngs:
        print(path)


if __name__ == "__main__":
    main()
