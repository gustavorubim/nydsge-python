"""Cheap, offline rendering from completed archives; never solves a model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from general_analytics_tools.compute import VARIABLES
from general_analytics_tools.inputs import quarter, sha256

PALETTE = (
    "#4169a6",
    "#e09633",
    "#9c6ba5",
    "#168878",
    "#729b4b",
    "#a67748",
    "#6e9fbc",
    "#b97a92",
    "#d5a7a7",
    "#c5c5c5",
    "#525252",
    "#df2525",
)


def render(archive: Path, destination: Path | None = None) -> list[Path]:
    if json.loads((archive / "status.json").read_text())["status"] != "completed":
        raise ValueError("Render only completed computation archives.")
    meta = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in meta["output_sha256"].items():
        if sha256(archive / name) != digest:
            raise ValueError(f"Archive hash mismatch: {name}")
    out = destination or archive / "figures"
    out.mkdir(parents=True, exist_ok=False)
    history = pd.read_csv(archive / "history.csv")
    forecasts = pd.read_csv(archive / "forecasts.csv")
    hd = pd.read_csv(archive / "decomposition.csv")
    start = quarter(meta["history_plot_start"])
    history["period"] = history.quarter.map(quarter)
    history = history.loc[history.period >= start]
    forecasts["period"] = forecasts.quarter.map(quarter)
    observed_end = quarter(meta["sample"]["history_end"])
    cond_end = quarter(meta["sample"]["conditioning_end"])
    origin = quarter(meta["forecast_start"])
    source = meta["source"]
    footer = (
        f"Input vintage: {source['vintage']}; retrieved {source['retrieved_at_utc']}. "
        f"Forecasts {meta['forecast_start']}–{meta['forecast_end']}; CPI not substituted. No bands."
    )
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (name, title, _, unit) in zip(axes.flat, VARIABLES, strict=True):
        h = history.loc[history.variable == name]
        b = forecasts.loc[(forecasts.variable == name) & (forecasts.scenario == "Baseline")]
        ax.plot(
            pd.PeriodIndex(h.period).to_timestamp(),
            h.actual,
            "o-",
            color="#252525",
            lw=1.5,
            ms=3,
            label="Actual compatible observation",
        )
        ax.plot(
            pd.PeriodIndex(h.period).to_timestamp(),
            h.fitted,
            "--",
            color="#9c9c9c",
            lw=1.2,
            label="Model smoothed / latent",
        )
        conditional = h.loc[(h.period > observed_end) & (h.period <= cond_end)]
        ax.scatter(
            pd.PeriodIndex(conditional.period).to_timestamp(),
            conditional.fitted,
            marker="D",
            s=32,
            color="#cc9a28",
            zorder=5,
            label="Partial-information nowcast",
        )
        ax.plot(
            pd.PeriodIndex(b.period).to_timestamp(),
            b.value,
            color="#006b6b",
            lw=2.3,
            label="Fixed-default forecast",
        )
        ax.axvline(origin.start_time, color="#777777", ls=":", lw=1)
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=35)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "ss104 public defaults | observed history, conditional nowcast and forecasts", fontsize=16
    )
    fig.text(0.02, 0.02, footer, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    paths = [out / "01_baseline_dashboard.png"]
    fig.savefig(paths[-1], dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = [
        "#386cb0",
        "#d36b32",
        "#17846b",
        "#9467bd",
        "#a67748",
        "#d85f97",
        "#6b8e23",
        "#585858",
    ]
    for ax, (name, title, _, _) in zip(axes.flat, VARIABLES, strict=True):
        for record, color in zip(meta["policy_scenarios"], colors, strict=False):
            data = forecasts.loc[
                (forecasts.variable == name) & (forecasts.scenario == record["label"])
            ]
            ax.plot(
                data.quarter, data.difference, "o-", color=color, lw=2, ms=4, label=record["label"]
            )
        ax.axhline(0, color="#777777", ls="--", label="Baseline (zero difference)")
        ax.set_title(title)
        unit = (
            "annual rate pp"
            if name == "obs_nominalrate"
            else "log points"
            if name == "obs_hours"
            else "annualized log pp"
        )
        ax.set_ylabel(f"Difference: {unit}")
        ax.grid(alpha=0.2)
        ax.tick_params(axis="x", rotation=35)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Unanticipated monetary innovations at {meta['forecast_start']} "
        "| actual FFR impact normalization",
        fontsize=15,
    )
    fig.text(
        0.02,
        0.02,
        "One innovation at impact; zero later innovations. Subsequent rates endogenous: "
        "not held paths or intraday FOMC estimates. " + footer,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    paths.append(out / "02_policy_surprises.png")
    fig.savefig(paths[-1], dpi=170)
    plt.close(fig)
    arrays = np.load(archive / "decomposition_arrays.npz", allow_pickle=False)
    full_history = pd.read_csv(archive / "history.csv")
    for name, title, filename in (
        ("obs_corepce", "Core PCE inflation", "03_hd_corepce.png"),
        ("obs_gdp", "Real GDP per capita growth", "04_hd_gdp_percapita.png"),
    ):
        frame = full_history.loc[full_history.variable == name].copy()
        periods = pd.PeriodIndex(frame.quarter.map(quarter), freq="Q")
        keep = periods >= start
        frame = frame.loc[keep]
        selected_periods = periods[keep]
        decomposition = hd.loc[(hd.variable == name) & (hd.quarter.map(quarter) >= start)]
        components = decomposition.pivot(
            index="quarter", columns="component", values="contribution"
        )
        order = list(dict.fromkeys(hd.component))
        components = components.reindex(index=frame.quarter, columns=order)
        if not np.isfinite(components.to_numpy()).all():
            raise ValueError("Incomplete decomposition components.")
        i = meta["observable_names"].index(name)
        error = meta["HD"][name]["max_reconstruction_error_annual_log_pp"]
        diagnostic = meta["HD"][name]["status"] == "diagnostic"
        fig, (ax, residual_ax) = plt.subplots(
            2, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
        )
        x = np.arange(len(frame))
        positive, negative = np.zeros(len(frame)), np.zeros(len(frame))
        for label, color in zip(order, PALETTE, strict=True):
            values = components[label].to_numpy()
            ax.bar(
                x,
                values,
                bottom=np.where(values >= 0, positive, negative),
                width=0.78,
                color=color,
                label=label,
            )
            positive += np.maximum(values, 0)
            negative += np.minimum(values, 0)
        ax.plot(x, frame.fitted, color="black", lw=2, label="Model smoothed total")
        ax.plot(x, frame.actual, "o", color="black", ms=3, label="Actual observed")
        suffix = (
            "DIAGNOSTIC: reconstruction tolerance failed"
            if diagnostic
            else "reconstruction verified"
        )
        ax.set_title(f"{title} | {suffix}")
        ax.set_ylabel("Annualized quarterly log rate / contributions (pp)")
        ax.axhline(0, color="grey", lw=0.8)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        residual_ax.plot(
            x,
            4 * arrays["remainder"][keep, i],
            color="#d62728",
            marker=".",
            label="Numerical reconstruction remainder",
        )
        residual_ax.plot(
            x,
            4 * arrays["measurement_gap"][keep, i],
            color="#777777",
            ls=":",
            label="Observed − fitted measurement gap",
        )
        residual_ax.set_ylabel("Residual pp")
        residual_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        residual_ax.grid(alpha=0.2)
        residual_ax.legend(fontsize=8)
        residual_ax.set_xticks(x[::2], [str(d) for d in selected_periods[::2]], rotation=35)
        fig.suptitle(
            "ss104 historical decomposition | conditional structural interpretation", fontsize=15
        )
        fig.text(
            0.02,
            0.015,
            f"Max full-sample reconstruction error {error:.3g} annualized log pp; threshold 1e-8. "
            "Partial quarters are latent except declared observations. " + footer,
            fontsize=8,
        )
        fig.subplots_adjust(top=0.88, bottom=0.13, hspace=0.72)
        paths.append(out / filename)
        fig.savefig(paths[-1], dpi=170)
        plt.close(fig)
    (out / "figures.json").write_text(
        json.dumps(
            {
                "archive_manifest_sha256": sha256(archive / "manifest.json"),
                "figures": {p.name: sha256(p) for p in paths},
                "captions": {
                    paths[0].name: (
                        "Actual compatible history and explicitly latent conditional nowcasts, "
                        "then fixed-default forecasts; no uncertainty bands."
                    ),
                    paths[1].name: (
                        "Policy surprises normalized to actual annualized Fed funds impact; "
                        "later rates follow model dynamics."
                    ),
                    paths[2].name: (
                        "Core PCE contributions include initial state, deterministic terms and "
                        "visible numerical remainder; attribution is conditional on the model."
                    ),
                    paths[3].name: (
                        "GDP per capita contributions and explicit residual; diagnostic whenever "
                        "the strict reconstruction gate fails."
                    ),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths
