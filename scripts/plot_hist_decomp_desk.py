#!/usr/bin/env python3
"""Desk-standard Model1002 historical decomposition: levels + cyclical panels.

Standing preference (2026-09-17):
  - Panel A = full levels (incl. IC+trend and Inflation target / pi_star_sh)
  - Panel B = cyclical shocks only (ex those two)
  - Desk groups with Price markup (λ_f) and Wage markup (λ_w) **separate**
    (not combined). Narrative focuses on B; A is accounting audit.

Usage (prefer per-shock CSV so λ_f / λ_w stay split):
  uv run python scripts/plot_hist_decomp_desk.py \
    --per-shock-csv /workspace/desk-plots/YYYY-MM-DD/hist_decomp_per_shock_corepce.csv \
    --out /workspace/desk-plots/YYYY-MM-DD/corepce_hist_decomp_levels_vs_cyclical.png

  # or grouped CSV from `nydsge study economy` after SHOCK_GROUPS split:
  uv run python scripts/plot_hist_decomp_desk.py \
    --csv outputs/.../historical_decomposition_grouped.csv \
    --variable obs_corepce \
    --out .../corepce_hist_decomp_levels_vs_cyclical.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = [
    "Initial conditions and trend",
    "Inflation target",
    "Price markup",
    "Wage markup",
    "Private demand",
    "Monetary policy",
    "Productivity and supply",
    "Government spending",
    "Investment and finance",
    "COVID shocks",
    "Measurement innovations",
    "Observed-model residual",
]

COLORS = {
    "Initial conditions and trend": "#C8CDD3",
    "Inflation target": "#8FB8D8",
    "Price markup": "#A65A52",
    "Wage markup": "#D4A5A0",
    "Price and wage markups": "#A65A52",  # legacy combined
    "Private demand": "#D9822B",
    "Monetary policy": "#34699A",
    "Productivity and supply": "#6E9E45",
    "Government spending": "#E7B64A",
    "Investment and finance": "#A56CC1",
    "COVID shocks": "#2A9D8F",  # ss104 only
    "Measurement innovations": "#7A8088",
    "Observed-model residual": "#3F454C",
}

LEVEL_ANCHORS = ("Initial conditions and trend", "Inflation target")

SHOCK_TO_DRIVER = {
    "g_sh": "Government spending",
    "b_sh": "Private demand",
    "ztil_sh": "Productivity and supply",
    "zp_sh": "Productivity and supply",
    "lambda_f_sh": "Price markup",
    "lambda_w_sh": "Wage markup",
    "rm_sh": "Monetary policy",
    **{f"rm_shl{i}": "Monetary policy" for i in range(1, 7)},
    "pi_star_sh": "Inflation target",
    "mu_sh": "Investment and finance",
    "sigma_omega_sh": "Investment and finance",
    "mu_e_sh": "Investment and finance",
    "gamma_sh": "Investment and finance",
    "lr_sh": "Measurement innovations",
    "tfp_sh": "Measurement innovations",
    "gdpdef_sh": "Measurement innovations",
    "corepce_sh": "Measurement innovations",
    "gdp_sh": "Measurement innovations",
    "gdi_sh": "Measurement innovations",
    # ss104 additions (AIT policy, survey-expectation, COVID and conditioning shocks)
    **{f"rm_ait_shl{i}": "Monetary policy" for i in range(1, 7)},
    "rm_ait_sh": "Monetary policy",
    **{f"exp_rm_sh{i}": "Monetary policy" for i in range(1, 7)},
    "pgap_sh": "Monetary policy",
    "ygap_sh": "Monetary policy",
    **{
        s: "COVID shocks"
        for s in ("biidc_sh", "phi_sh", "ziid_sh", "biidc_shl1", "phi_shl1", "ziid_shl1")
    },
    "meas_pi_sh": "Measurement innovations",
    "condgdp_sh": "Measurement innovations",
    "condcorepce_sh": "Measurement innovations",
    "Observed-model residual": "Observed-model residual",
    "IC+trend": "Initial conditions and trend",
    "Initial conditions and trend": "Initial conditions and trend",
}


def _stack_bars(ax, wide: pd.DataFrame, drivers: list[str]) -> None:
    bottom_pos = np.zeros(len(wide))
    bottom_neg = np.zeros(len(wide))
    x = np.arange(len(wide))
    for d in drivers:
        if d not in wide.columns:
            continue
        v = wide[d].fillna(0).to_numpy()
        pos = np.where(v >= 0, v, 0)
        neg = np.where(v < 0, v, 0)
        ax.bar(x, pos, bottom=bottom_pos, color=COLORS.get(d, "#888"), width=0.9, label=d)
        ax.bar(x, neg, bottom=bottom_neg, color=COLORS.get(d, "#888"), width=0.9)
        bottom_pos = bottom_pos + pos
        bottom_neg = bottom_neg + neg


def _wide_from_per_shock(
    csv_path: Path, n_quarters: int
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(csv_path)
    if "shock" not in df.columns:
        raise SystemExit(f"Expected shock column in {csv_path}")
    df = df.copy()
    df["driver"] = df["shock"].map(SHOCK_TO_DRIVER)
    missing = df.loc[df["driver"].isna(), "shock"].unique()
    if len(missing):
        raise SystemExit(f"Unmapped shocks: {sorted(missing)}")
    agg = df.groupby(["date", "driver"], as_index=False).agg(
        contribution=("contribution", "sum"),
        smoothed=("smoothed", "first"),
    )
    dates = sorted(agg["date"].unique())[-n_quarters:]
    agg = agg[agg["date"].isin(dates)]
    order = [d for d in ORDER if d in set(agg["driver"])]
    wide = agg.pivot(index="date", columns="driver", values="contribution").reindex(columns=order)
    sm = agg.groupby("date")["smoothed"].first().reindex(wide.index)
    return wide, sm, dates


def _wide_from_grouped(
    csv_path: Path, variable: str, n_quarters: int
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    h = pd.read_csv(csv_path)
    sub = h[h["variable"] == variable].copy()
    if sub.empty:
        raise SystemExit(f"No rows for variable={variable!r} in {csv_path}")
    # normalize legacy combined label if somehow still present alone
    dates = sorted(sub["date"].unique())[-n_quarters:]
    sub = sub[sub["date"].isin(dates)]
    drivers = list(sub["driver"].unique())
    order = [d for d in ORDER if d in drivers]
    # keep legacy combined column if new split absent
    if "Price and wage markups" in drivers and "Price markup" not in drivers:
        order = [
            "Initial conditions and trend",
            "Inflation target",
            "Price and wage markups",
            "Private demand",
            "Monetary policy",
            "Productivity and supply",
            "Government spending",
            "Investment and finance",
            "Measurement innovations",
            "Observed-model residual",
        ]
        order = [d for d in order if d in drivers]
    wide = sub.pivot(index="date", columns="driver", values="contribution").reindex(columns=order)
    sm = sub.groupby("date")["smoothed"].first().reindex(wide.index)
    return wide, sm, dates


def plot_hist_decomp_desk(
    *,
    out: Path,
    csv_path: Path | None = None,
    per_shock_csv: Path | None = None,
    variable: str = "obs_corepce",
    n_quarters: int = 40,
    title_var: str | None = None,
) -> Path:
    if per_shock_csv is not None:
        wide, sm, dates = _wide_from_per_shock(per_shock_csv, n_quarters)
    elif csv_path is not None:
        wide, sm, dates = _wide_from_grouped(csv_path, variable, n_quarters)
    else:
        raise SystemExit("Pass --per-shock-csv or --csv")

    label = title_var or variable
    drivers = list(wide.columns)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = np.arange(len(wide))

    ax = axes[0]
    _stack_bars(ax, wide, drivers)
    ax.plot(x, sm.to_numpy(), "k-", lw=1.5, label=f"Smoothed {label}")
    ax.set_ylabel("% SAAR" if "pce" in variable.lower() or "gdp" in variable.lower() else "")
    ax.set_title(f"A. Full hist decomp (levels) — Model1002 {label}")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.axhline(0, color="k", lw=0.5)

    cyc = [d for d in drivers if d not in LEVEL_ANCHORS]
    ax = axes[1]
    _stack_bars(ax, wide, cyc)
    cyc_sum = wide[cyc].sum(axis=1)
    ax.plot(x, cyc_sum.to_numpy(), "k-", lw=1.5, label="Sum of cyclical drivers")
    ax.set_ylabel("pp")
    ax.set_title("B. Cyclical / structural shocks only (ex IC+trend and π*_shocks)")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--per-shock-csv", type=Path, default=None)
    p.add_argument("--variable", default="obs_corepce")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-quarters", type=int, default=40)
    p.add_argument("--title-var", default=None)
    args = p.parse_args()
    path = plot_hist_decomp_desk(
        csv_path=args.csv,
        per_shock_csv=args.per_shock_csv,
        variable=args.variable,
        out=args.out,
        n_quarters=args.n_quarters,
        title_var=args.title_var,
    )
    print(path)


if __name__ == "__main__":
    main()
