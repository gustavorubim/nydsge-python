"""Plot saved ss104 and C1/C2 ss10 estimates offline; never re-estimate.

Run: .venv/Scripts/python.exe scripts/plot_ss104_enhancements.py
Requires the optional matplotlib dependency. Generated artifacts stay in outputs/plots.
M1-M5 have no independent ablation API; contribution counterfactuals are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from c1c2_common import FRED_SNAPSHOT, ROOT, build_model, build_observables  # noqa: E402
from c1c2_compare import COND_OBS  # noqa: E402
from ss104_local import build_local_data, build_local_model, input_provenance  # noqa: E402

from nydsge.data import prepare_population_data  # noqa: E402
from nydsge.economy import _restrict_conditioning_quarters  # noqa: E402
from nydsge.forecast import forecast_one  # noqa: E402
from nydsge.kalman import (  # noqa: E402
    kalman_log_likelihood,
    model_process_covariances,
    smooth_kalman_result,
)
from nydsge.regimes import regime_indices_from_dates  # noqa: E402
from nydsge.solve import compute_system  # noqa: E402
from nydsge.ss104_estimation import ss104_parameter_catalog, update_ss104_parameters  # noqa: E402

SS10_MODE = ROOT / "data/estimates/m1002_ss10_c1c2_mode_maskcovid.json"
SS104_MODE = ROOT / "data/estimates/m1002_ss104_mode.json"
ANNUAL = ROOT / "data/estimates/ss104_forecast_annual.csv"
NYFED_PARAMS_ANNUAL = ROOT / "data/estimates/ss104_nyfed_params_forecast_annual.csv"
NYFED_PARAMS_META = ROOT / "data/estimates/ss104_nyfed_params_forecast_metadata.json"
QUARTERLY = ROOT / "data/estimates/ss104_forecast_quarterly.csv"
NYFED = ROOT / "data/public/nyfed_dsge_20260918/nyfed_dsgeTableData.csv"
COLORS = ("#737b87", "#1768ac", "#d47921")
NOTE = "ss104 best saved estimate; convergence uncertified."


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, out, filename, caption):
    """Reserve a bottom margin for an actual single-line fig.text caption."""
    fig.text(0.5, 0.025, caption, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.01, 0.075, 0.99, 0.94))
    fig.savefig(out / filename, dpi=150, facecolor="white")
    plt.close(fig)


def style(ax):
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def ss10_replay(mode):
    """Use exactly the model/data/masking/conditioning recipe in c1c2_compare.after."""
    model = build_model(mode["n_mon_anticipated_shocks"], date_forecast_start="2026-Q4")
    for key in mode["estimated_parameters"]:
        model.parameters[key] = replace(
            model.parameters[key], value=float(mode["parameter_values"][key])
        )
    history = build_observables(model, end="2026-Q3")
    cols = [c for c in history if c.startswith("obs_")]
    history.loc[history.date.isin(mode["mask_covid_quarters"]), cols] = np.nan
    history = _restrict_conditioning_quarters(history, model_end_date="2026-Q2", keep=COND_OBS)
    forecast = forecast_one(
        model,
        input_type="mode",
        cond_type="none",
        output_vars=["histstates", "forecastpseudo"],
        check_empty_columns=False,
        horizon=13,
        data=history,
        history_method="smoothed",
    )
    system = compute_system(model)
    hist_pseudo = (
        forecast.history_states @ system.pseudo_measurement.ZZ_pseudo.T
        + system.pseudo_measurement.DD_pseudo
    )
    dates = pd.period_range("2026Q3", "2029Q4", freq="Q")
    pseudo = pd.DataFrame(
        np.vstack((hist_pseudo[-1:], forecast.pseudo_observables)),
        index=dates,
        columns=list(model.pseudo_observables),
    )
    hist_obs = forecast.history_states @ system.measurement.ZZ.T + system.measurement.DD
    obs = pd.DataFrame(
        np.vstack((hist_obs[-1:], forecast.observables)),
        index=dates,
        columns=list(model.observables),
    )
    # Same terminal HP-trend population approximation as forecast_ss104_mode.py.
    levels = pd.read_csv(FRED_SNAPSHOT)
    quarters = pd.PeriodIndex(levels.date, freq="Q")
    levels = levels.loc[
        levels.CNP16OV.notna()
        & (quarters >= pd.Period("1959Q2"))
        & (quarters <= pd.Period("2026Q2"))
    ].reset_index(drop=True)
    pop = float(prepare_population_data(model, levels).dlfiltered_population_recorded.iloc[-1])
    # Use smoothed Q3 for the conditioning quarter, actual Q1-Q2 for annual 2026.
    hist = history.copy()
    hist.index = pd.PeriodIndex(hist.pop("date"), freq="Q")
    annual = pd.DataFrame(index=range(2026, 2030))
    for name, observable in (
        ("gdp", "obs_gdp"),
        ("corepce", "obs_corepce"),
    ):
        series = pd.concat((hist.loc[:"2026Q2", observable], obs[observable]))
        if name == "gdp":
            series += pop
        for year in annual.index:
            values = series.reindex(pd.period_range(f"{year}Q1", f"{year}Q4", freq="Q"))
            if not np.isfinite(values).all():
                raise ValueError(f"Incomplete ss10 {name} data in {year}")
            annual.loc[year, name] = 100 * np.expm1(values.sum() / 100)
    # C1/C2 reporting uses 400*expm1(quarterly log growth/100), rather than
    # exact annual compounding. This explains small differences from doc rounding.
    return (
        model,
        annual,
        pd.DataFrame({"ffr_ss10": 4 * obs.obs_nominalrate, "output_gap_ss10": pseudo.OutputGap}),
        pop,
    )


def covid_shocks(model):
    """Recover conditional structural innovations with a covariance-weighted DK recursion.

    pinv(RRR) would incorrectly split proportional COVID innovations across
    zero-variance shocks. Instead eps_hat[t] = QQ[t] RRR[t]' r[t], using the
    same backward information recursion as the package disturbance smoother.
    """
    system = compute_system(model)
    history = build_local_data(model, end="2026-Q2")
    data = history.to_numpy(dtype=float)
    indices = regime_indices_from_dates(model.get_setting("regime_dates"), history.index)
    filtered = kalman_log_likelihood(
        system,
        data,
        regime_indices=indices,
        log_likelihood_start=2,
        process_covariances=model_process_covariances(
            model, system, len(history), start_date=history.index[0]
        ),
    )
    path = system.period_systems(indices, len(history))
    r = np.zeros(filtered.filtered_states.shape[1])
    names = list(model.indexes.exogenous_shocks)
    shocks = np.zeros((len(history), len(names)))
    for t in range(len(history) - 1, -1, -1):
        current = path[t]
        transition = path[min(t + 1, len(history) - 1)].transition.TTT
        mask = np.isfinite(data[t])
        z = current.measurement.ZZ[mask]
        covariance = filtered.predicted_covariances[t]
        if mask.any():
            innovation = data[t, mask] - z @ filtered.predicted_states[t]
            innovation -= current.measurement.DD[mask]
            f = z @ covariance @ z.T + current.measurement.EE[np.ix_(mask, mask)]
            inverse_z = np.linalg.solve(f, z)
            gain = transition @ covariance @ inverse_z.T
            r = z.T @ np.linalg.solve(f, innovation) + (transition - gain @ z).T @ r
        else:
            r = transition.T @ r
        shocks[t] = current.measurement.QQ @ current.transition.RRR.T @ r
    # Verify structural shock means reconstruct smoothed state innovations in COVID quarters.
    smoothed = smooth_kalman_result(system, filtered, data=data).smoothed_states
    errors = []
    for t in np.flatnonzero((history.index >= "2020Q1") & (history.index <= "2021Q4")):
        current = path[t].transition
        residual = smoothed[t] - current.TTT @ smoothed[t - 1] - current.CCC
        errors.append(np.max(np.abs(residual - current.RRR @ shocks[t])))
    error = float(max(errors))
    if error > 1e-5:
        raise ValueError(f"COVID shock transition reconstruction error {error:g}")
    frame = pd.DataFrame(shocks, index=history.index, columns=names).loc["2020Q1":"2021Q4"]
    return frame[[n for n in names if n.startswith(("phi_sh", "ziid_sh", "biidc_sh"))]], {
        "history_end": "2026Q2",
        "log_likelihood": filtered.log_likelihood,
        "covid_transition_reconstruction_max_abs_error": error,
    }


def annual_plot(out, old, new, defaults, ny):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, name, row, title in zip(
        axes,
        ("gdp", "corepce"),
        ("GDP growth (Q4/Q4)", "Core PCE inflation (Q4/Q4)"),
        ("Real GDP growth", "Core PCE inflation"),
        strict=True,
    ):
        for values, label, color in (
            (old[name], "ss10 C1/C2 (COVID-masked)", COLORS[0]),
            (new[f"{name}_ss104"], "ss104 saved estimate", COLORS[1]),
            (defaults[f"{name}_ss104"], "ss104 (NY Fed/default params)", "#28804b"),
            (
                ny.loc[row, [str(y) for y in old.index]].to_numpy(float),
                "NY Fed Sep 2026",
                COLORS[2],
            ),
        ):
            ax.plot(old.index, values, "o-", color=color, label=label, linewidth=2)
        ax.set(title=title, xlabel="Calendar year", ylabel="Q4/Q4 change (%)", xticks=old.index)
        ax.legend(fontsize=9)
        style(ax)
    fig.suptitle("Annual forecasts at fixed parameters (2026–2029)", fontsize=15)
    caption = (
        f"2027 GDP/PCE: ss104 defaults {defaults.loc[2027, 'gdp_ss104']:.2f}/"
        f"{defaults.loc[2027, 'corepce_ss104']:.2f}%; saved estimate "
        f"{new.loc[2027, 'gdp_ss104']:.2f}/{new.loc[2027, 'corepce_ss104']:.2f}%; "
        "NY Fed 0.10/2.10%. Defaults are code calibration; estimate convergence uncertified."
    )
    save(fig, out, "forecast_gdp_pce_annual.png", caption)
    return caption


def quarterly_plot(out, data):
    data = data.iloc[:12]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    dates = data.index.to_timestamp()
    for ax, name, title, unit in zip(
        axes,
        ("ffr", "output_gap"),
        ("Federal funds rate", "Output gap (output − flexible output)"),
        ("Annual rate (%)", "Log percentage points"),
        strict=True,
    ):
        for spec, color, label in (
            ("ss10", COLORS[0], "ss10 C1/C2 (COVID-masked)"),
            ("ss104", COLORS[1], "ss104 saved estimate"),
        ):
            ax.plot(dates, data[f"{name}_{spec}"], "o-", label=label, color=color, linewidth=2)
        ax.set(title=title, xlabel="Quarter", ylabel=unit)
        ax.set_xticks(dates[::2], [str(q) for q in data.index[::2]], rotation=35)
        ax.legend(fontsize=9)
        style(ax)
    fig.suptitle("Quarterly policy and output-gap paths (12 quarters)", fontsize=15)
    caption = (
        f"2027Q4 FFR: ss10 {data.loc['2027Q4', 'ffr_ss10']:.2f}%, "
        f"ss104 {data.loc['2027Q4', 'ffr_ss104']:.2f}%; gaps "
        f"{data.loc['2027Q4', 'output_gap_ss10']:.2f}/"
        f"{data.loc['2027Q4', 'output_gap_ss104']:.2f} log pp. {NOTE}"
    )
    save(fig, out, "forecast_ffr_outputgap.png", caption)
    return caption


def regime_plot(out, model):
    dates = model.get_setting("regime_dates")
    starts = pd.PeriodIndex(list(dates.values()), freq="Q").to_timestamp()
    end = pd.Period("2027Q1").to_timestamp()
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), height_ratios=(1, 1.5, 2))
    axes[0].step([*starts, end], [*dates, 29], where="post", color=COLORS[1], label="Model regime")
    axes[0].text(pd.Timestamp("1988-01-01"), 5, "Regime 1: 1959Q3–2019Q4", ha="center")
    axes[0].set(title="Full dated schedule (29 regimes)", ylabel="Regime ID (index)", ylim=(0, 31))
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].xaxis.set_major_locator(mdates.YearLocator(10))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for i in range(1, 29):
        finish = starts[i + 1] if i < 28 else end
        axes[1].axvspan(starts[i], finish, color=COLORS[1], alpha=0.08 if i % 2 else 0.17)
        axes[1].text(
            starts[i] + (finish - starts[i]) / 2, 0.5, str(i + 1), ha="center", va="center"
        )
    axes[1].set(title="Quarterly detail: regimes 2–29", ylabel="Regime ID (labels)", ylim=(0, 1))
    axes[1].set_yticks([])
    axes[1].set_xlim(starts[1], end)
    beliefs = model.get_setting("credibility")
    cred_dates = pd.PeriodIndex([dates[r] for r in beliefs], freq="Q")
    axes[2].plot(
        cred_dates.to_timestamp(),
        list(beliefs.values()),
        "o-",
        color=COLORS[1],
        label="AIT credibility (fixed calibration)",
    )
    axes[2].set(
        title="AIT credibility: 2020Q4–2026Q4",
        ylabel="Credibility weight (0–1)",
        ylim=(-0.04, 1.06),
        xlim=(starts[4], end),
    )
    for ax in axes:
        ax.axvspan(starts[4], starts[9], color="#d47921", alpha=0.24)
        ax.axvspan(starts[9], starts[18], facecolor="#d47921", alpha=0.10, hatch="//")
        ax.set_xlabel("Date (calendar year)")
        style(ax)
    for ax in axes[1:]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles, _ = axes[2].get_legend_handles_labels()
    axes[2].legend(
        handles=[
            *handles,
            Patch(facecolor="#d47921", alpha=0.24, label="Actual temporary ZLB: 2020Q4–2021Q4"),
            Patch(
                facecolor="#d47921",
                alpha=0.10,
                hatch="//",
                label="Latest belief's ZLB extension: 2022Q1–2024Q1",
            ),
        ],
        fontsize=9,
    )
    fig.suptitle("ss104 regime schedule and policy credibility", fontsize=15)
    caption = (
        "29 regimes; 5 actual ZLB quarters; AIT credibility rises from 0.000 (2020Q4) "
        "to 1.000 (2026Q4), fixed rather than estimated."
    )
    save(fig, out, "regime_timeline.png", caption)
    return caption


def covid_plot(out, shocks):
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    for ax, name, title in zip(
        axes,
        ("phi", "ziid", "biidc"),
        ("Labor-supply shock (phi)", "Productivity shock (ziid)", "Discount-factor shock (biidc)"),
        strict=True,
    ):
        for key, color, label in (
            (f"{name}_sh", COLORS[1], "Contemporaneous innovation"),
            (f"{name}_shl1", COLORS[2], "One-quarter anticipated innovation"),
        ):
            ax.plot(range(8), shocks[key], "o-", label=label, color=color, linewidth=2)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set(title=title, ylabel="Innovation (model units)")
        ax.legend(fontsize=9, loc="best")
        style(ax)
    axes[-1].set(xticks=range(8), xticklabels=shocks.index.astype(str), xlabel="Quarter")
    fig.suptitle("Smoothed COVID innovations at the saved ss104 estimate (2020–2021)", fontsize=15)
    caption = (
        "2020Q2 innovations: "
        + ", ".join(
            f"{name} {shocks.loc['2020Q2', name + '_sh']:.2f}" for name in ("phi", "ziid", "biidc")
        )
        + f" model units. {NOTE}"
    )
    save(fig, out, "covid_shocks.png", caption)
    return caption


def key_params(out, new, old, model):
    defaults = build_local_model()
    lines = [
        "# Saved parameter comparison",
        "",
        f"ss104: `{SS104_MODE.relative_to(ROOT).as_posix()}`; status `{new['status']}`; "
        "best finite estimate, not a certified converged mode.",
        "",
        f"ss10: `{SS10_MODE.relative_to(ROOT).as_posix()}`; status `{old['status']}`; "
        "recommended COVID-masked C1/C2 estimate used in both forecast plots.",
        "",
        "ss104 NY Fed/default line uses DSGE.jl c09dfe8 ss104 defaults, the published "
        "code's calibration rather than the FRBNY production posterior.",
        "",
        "| Parameter | ss104 saved estimate | ss104 code defaults | ss10 C1/C2 | Units / status |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label in (
        ("h", "Habit persistence"),
        ("zeta_p", "Price rigidity"),
        ("zeta_w", "Wage rigidity"),
        ("psi1", "Taylor inflation response"),
        ("psi2", "Taylor output response"),
        ("rho", "Taylor policy smoothing"),
    ):
        lines.append(
            f"| {key} ({label}) | {new['parameter_values'][key]:.6f} | "
            f"{defaults.numeric_value(key):.6f} | "
            f"{old['parameter_values'][key]:.6f} | Dimensionless; estimated |"
        )
    for key in new["estimated_parameters"]:
        if "ait" in key or "credib" in key:
            name, regime = key.rsplit("@", 1)
            value = new["parameter_regime_values"][name][regime]
            lines.append(
                f"| {key} | {value:.6f} | "
                f"{defaults.get_setting('parameter_regimes')[name][int(regime)].value:.6f} | N/A | "
                "Quarterly model-unit shock SD; "
                f"estimated parameter regime {regime} |"
            )
    lines += [
        "",
        "AIT policy coefficients and credibility are calibrated, not estimated. "
        "The estimated `rho` above is the historical Taylor-rule coefficient; "
        "the flexible AIT rule uses its own smoothing setting.",
        "",
        "| Fixed ss104 policy setting | Value | Units |",
        "|---|---:|---|",
    ]
    for key, unit in (
        ("flexible_ait_phi_pi", "Dimensionless"),
        ("flexible_ait_phi_y", "Dimensionless"),
        ("flexible_ait_rho_smooth", "Dimensionless"),
        ("ait_Thalf", "Quarters"),
        ("gdp_Thalf", "Quarters"),
    ):
        lines.append(f"| {key} | {model.get_setting(key):.6f} | {unit} |")
    lines += [
        "| AIT credibility | 0 to 1, 2020Q4–2026Q4 | Weight; 1/24 increase per quarter |",
        "",
        "`rho_ait_rm` = 0.2135 and `sigma_ait_rm@2` = 0.238 remain at defaults "
        "because they have no defined prior; they were not estimated. ss10 has no AIT block.",
        "",
    ]
    (out / "key_params.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/plots")
    args = parser.parse_args()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "ss104-port":
        raise RuntimeError("Plot generation is restricted to ss104-port.")
    started = time.monotonic()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT / "outputs/plots"):
        raise ValueError("Output must stay within outputs/plots.")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    new_mode, old_mode = load_json(SS104_MODE), load_json(SS10_MODE)
    if new_mode["status"] == "running" or new_mode["input_sha256"] != input_provenance():
        raise ValueError("Require a finished ss104 estimate and matching local inputs.")
    model = build_local_model()
    catalog = ss104_parameter_catalog(model)
    if list(catalog) != new_mode["estimated_parameters"]:
        raise ValueError("Saved ss104 parameter coordinates differ from the current model.")
    update_ss104_parameters(model, catalog, new_mode["estimation_values"])
    print("Replaying ss10 forecast at saved parameters...", flush=True)
    _, old_annual, old_quarterly, pop = ss10_replay(old_mode)
    new_annual = pd.read_csv(ANNUAL, index_col="year").loc[2026:2029]
    defaults_meta = load_json(NYFED_PARAMS_META)
    if defaults_meta["input_sha256"] != input_provenance():
        raise ValueError("Default forecast inputs differ from the local snapshot.")
    defaults_annual = pd.read_csv(NYFED_PARAMS_ANNUAL, index_col="year").loc[2026:2029]
    ny = pd.read_csv(NYFED, index_col=0)
    new_quarterly = pd.read_csv(QUARTERLY, index_col="quarter")
    new_quarterly.index = pd.PeriodIndex(new_quarterly.index, freq="Q")
    combined = old_quarterly.join(new_quarterly[["ffr_ss104", "output_gap_ss104"]])
    if not np.isfinite(combined.to_numpy()).all():
        raise ValueError("Incomplete quarterly forecast comparison.")
    annual_comparison = old_annual.add_suffix("_ss10").join(new_annual)
    annual_comparison = annual_comparison.join(
        defaults_annual[["gdp_ss104", "corepce_ss104"]].rename(
            columns=lambda k: k.replace("_ss104", "_ss104_defaults")
        )
    )
    for name, row in (("gdp", "GDP growth (Q4/Q4)"), ("corepce", "Core PCE inflation (Q4/Q4)")):
        np.testing.assert_allclose(new_annual[f"{name}_nyfed"], ny.loc[row].to_numpy(float))
    annual_comparison.to_csv(out / "forecast_annual_comparison.csv", index_label="year")
    combined.iloc[:12].to_csv(out / "forecast_quarterly_comparison.csv", index_label="quarter")
    captions = {
        "forecast_gdp_pce_annual.png": annual_plot(
            out, old_annual, new_annual, defaults_annual, ny
        ),
        "forecast_ffr_outputgap.png": quarterly_plot(out, combined),
        "regime_timeline.png": regime_plot(out, model),
    }
    print("Smoothing ss104 COVID innovations at saved parameters...", flush=True)
    shocks, diagnostics = covid_shocks(model)
    np.testing.assert_allclose(
        diagnostics["log_likelihood"], new_mode["log_likelihood"], atol=1e-5, rtol=0
    )
    shocks.to_csv(out / "covid_smoothed_shocks.csv", index_label="quarter")
    captions["covid_shocks.png"] = covid_plot(out, shocks)
    key_params(out, new_mode, old_mode, model)
    lines = ["# Plot takeaways", ""] + [
        f"- **{name}**: {caption}" for name, caption in captions.items()
    ]
    lines += [
        "",
        "**enhancement_contributions.png — skipped:** M1–M5 are integrated infrastructure "
        "and model blocks with no supported independent toggle API. Defining and validating "
        "five comparable counterfactual models is outside a cheap plotting replay; no ablation "
        "or re-estimation was run.",
        "",
        "## Numbers plotted (Q4/Q4, %)",
        "",
        "| Year | GDP ss10 | GDP ss104 estimate | GDP ss104 defaults | GDP NY Fed | "
        "PCE ss10 | PCE ss104 estimate | PCE ss104 defaults | PCE NY Fed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year, row in annual_comparison.iterrows():
        values = [
            row[k]
            for k in (
                "gdp_ss10",
                "gdp_ss104",
                "gdp_ss104_defaults",
                "gdp_nyfed",
                "corepce_ss10",
                "corepce_ss104",
                "corepce_ss104_defaults",
                "corepce_nyfed",
            )
        ]
        lines.append(f"| {year} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += [
        "",
        "## Sources and interpretation",
        "",
        "- ss10 baseline: recommended COVID-masked C1/C2 saved estimate, replayed with "
        "the existing `c1c2_compare.after` recipe; not the earlier desk/default run. "
        "No optimizer or sampler is called.",
        "- ss104 annual/quarterly paths: committed `data/estimates/ss104_forecast_*.csv`; "
        "NY Fed annual figures: cached September 18, 2026 `nyfed_dsgeTableData.csv`.",
        "- ss104 (NY Fed/default params): `ss104_nyfed_params_forecast_annual.csv`; "
        + defaults_meta["parameter_source"]
        + ". Same local data, semi conditioning and growth conventions as the saved estimate.",
        "- GDP adds constant terminal HP population log growth; annual GDP and PCE use "
        "100 × expm1(sum of four quarterly log-percent growth rates / 100). "
        "Existing C1/C2 docs compound their approximate annualized reporting rates, "
        "so their rounded annual numbers differ slightly.",
        "- 2026 uses actual Q1–Q2 and model Q3–Q4. ss10 conditions Q3 on rates, "
        "spread and long-run inflation; ss104 also uses available SME rate expectations. "
        "The specification, parameters and conditioning all differ; this is not a causal ablation.",
        "- FFR = 4 × quarterly NominalFFR; output gap = y − flexible output in log percentage "
        "points, not cumulative AIT ygap. Quarterly panel: 2026Q3–2029Q2 (12 quarters).",
        "- Regimes/credibility come from the current ss104 settings. Actual temporary ZLB: "
        "regimes 5–9. Hatched extension represents the latest alternative survey belief "
        "(ZLB through regime 18), not the realized rate path; it is separate from credibility.",
        "- COVID shocks: full-history smoothing through 2026Q2 at the saved ss104 "
        "estimate, preserving missing data and dated covariance regimes. Covariance-weighted "
        "Durbin–Koopman structural innovations are in native model units; one-quarter anticipated "
        "innovations are dated when the news arrives. Zero estimates follow zero shock variance, "
        "not missing-data imputation.",
        f"- Historical likelihood replay: {diagnostics['log_likelihood']:.6f}; COVID transition "
        "reconstruction max error: "
        f"{diagnostics['covid_transition_reconstruction_max_abs_error']:.3g}.",
        "- ss104 optimizer status is time_limit; saved projected gradient norm "
        f"{new_mode['diagnostics']['saved_point_gradient_inf_norm']:.4f}; convergence uncertified. "
        "These are point paths with no uncertainty bands.",
        "",
    ]
    (out / "captions.md").write_text("\n".join(lines), encoding="utf-8")
    sources = [
        SS10_MODE,
        SS104_MODE,
        ANNUAL,
        QUARTERLY,
        NYFED,
        NYFED_PARAMS_ANNUAL,
        NYFED_PARAMS_META,
    ]
    metadata = {
        "branch": branch,
        "dpi": 150,
        "re_estimation": False,
        "ss10_population_log_growth_annual_pp": 4 * pop,
        "smoothing": diagnostics,
        "source_sha256": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "input_sha256": input_provenance(),
        "elapsed_seconds": time.monotonic() - started,
        "enhancement_contributions": "skipped: no independent M1-M5 toggle API",
    }
    (out / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("\n".join(captions.values()).encode("ascii", errors="replace").decode("ascii"))
    print(
        f"Wrote 4 PNGs, captions and parameter table to {out} in {metadata['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
