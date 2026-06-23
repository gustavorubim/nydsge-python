"""User-facing forecast reporting: numeric exports and figures.

This module productizes the forecast analysis artifacts described in
PORTING_PLAN.md section 12. It computes forecast paths, impulse responses, and a
true historical shock decomposition, exports the numeric arrays with full axis
labels, and (optionally) renders deterministic figures.

Plotting depends on ``matplotlib``, declared as the optional ``plot`` extra
(``pip install nydsge[plot]``). The numeric exports do not require matplotlib, so
report arrays can be produced in a headless/minimal environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nydsge.core import DSGEModel
from nydsge.data import (
    date_labels_for_sample,
    load_data,
    quarter_labels_from_start,
)
from nydsge.forecast import (
    ForecastOutput,
    HistoricalDecomposition,
    ImpulseResponse,
    forecast_one,
    historical_decomposition,
    observable_irf,
)
from nydsge.solve import System, compute_system

# Deterministic figure DPI keeps rendered output stable across runs.
_FIGURE_DPI = 150


class ReportError(ValueError):
    """Raised when report inputs are inconsistent (missing labels/arrays/dims)."""


@dataclass(frozen=True)
class ReportArtifacts:
    """Files and metadata produced by a report run."""

    output_dir: Path
    figures: list[Path]
    arrays: list[Path]
    summary: Path
    manifest: dict[str, Any]


def _require_matplotlib() -> Any:
    try:
        import matplotlib
    except ModuleNotFoundError as err:  # pragma: no cover - exercised via CLI
        msg = (
            "Forecast plotting requires matplotlib. Install the optional plotting "
            "extra, e.g. `uv pip install nydsge[plot]` or `pip install matplotlib`."
        )
        raise ReportError(msg) from err
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _observable_labels(model: DSGEModel) -> list[str]:
    return list(model.observables)


def _shock_labels(model: DSGEModel) -> list[str]:
    return list(model.indexes.exogenous_shocks)


def _state_labels(model: DSGEModel) -> list[str]:
    return list(model.indexes.endogenous_states) + list(model.indexes.endogenous_states_augmented)


def _forecast_dates(model: DSGEModel, horizon: int) -> list[str]:
    return quarter_labels_from_start(model.get_setting("date_forecast_start"), horizon)


def _history_dates(model: DSGEModel, data: Any) -> list[str]:
    if not hasattr(data, "columns"):
        return []
    return date_labels_for_sample(model, data)


def _check_axis(name: str, labels: list[str], expected: int) -> None:
    if len(labels) != expected:
        msg = (
            f"Report axis '{name}' has {len(labels)} labels for {expected} entries; "
            "labels and array dimensions are inconsistent."
        )
        raise ReportError(msg)


# --------------------------------------------------------------------------- #
# Impulse responses
# --------------------------------------------------------------------------- #


def generate_irf_report(
    model: DSGEModel,
    *,
    output_dir: Path,
    horizon: int = 40,
    normalization: str = "one_sd",
    make_plots: bool = True,
    combined: bool = True,
    system: System | None = None,
) -> ReportArtifacts:
    """Export numeric impulse responses and one figure per observable.

    Shock normalization is documented on the array summary: ``"unit"`` applies a
    1.0 impulse, ``"one_sd"`` a one-standard-deviation impulse (``sqrt(diag(QQ))``).
    """

    output_dir = Path(output_dir)
    figure_dir = output_dir / "impulse_responses"
    solved = system if system is not None else compute_system(model)
    irf = observable_irf(solved, horizon=horizon, normalization=normalization)
    obs_labels = _observable_labels(model)
    shock_labels = _shock_labels(model)
    horizon_labels = [f"h{step}" for step in range(horizon)]
    _check_axis("observables", obs_labels, irf.observables.shape[1])
    _check_axis("shocks", shock_labels, irf.observables.shape[2])

    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / "impulse_responses.npz"
    np.savez(
        array_path,
        observables=irf.observables,
        states=irf.states,
        shock_scales=irf.shock_scales,
    )
    arrays = [array_path]
    figures: list[Path] = []
    if make_plots:
        figures = _plot_irf(
            irf,
            figure_dir,
            obs_labels=obs_labels,
            shock_labels=shock_labels,
            normalization=normalization,
            combined=combined,
        )
    manifest = {
        "kind": "irf_report",
        "model": model.spec,
        "subspec": model.subspec,
        "horizon": horizon,
        "normalization": normalization,
        "labels": {
            "impulse_responses/observables": {
                "axis0": horizon_labels,
                "axis1": obs_labels,
                "axis2": shock_labels,
            },
            "impulse_responses/states": {
                "axis0": horizon_labels,
                "axis1": _state_labels(model),
                "axis2": shock_labels,
            },
            "impulse_responses/shock_scales": {"axis0": shock_labels},
        },
        "arrays": [str(path) for path in arrays],
        "figures": [str(path) for path in figures],
    }
    summary = _write_summary(output_dir, "irf_report.json", manifest)
    return ReportArtifacts(output_dir, figures, arrays, summary, manifest)


def _plot_irf(
    irf: ImpulseResponse,
    figure_dir: Path,
    *,
    obs_labels: list[str],
    shock_labels: list[str],
    normalization: str,
    combined: bool,
) -> list[Path]:
    plt = _require_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    horizon = irf.observables.shape[0]
    steps = np.arange(horizon)
    # Rank shocks by total absolute response so the legend highlights movers.
    scores = np.abs(irf.observables).sum(axis=(0, 1))
    order = np.argsort(scores)[::-1]
    top = order[: min(6, order.size)]
    figures: list[Path] = []
    for obs_index, obs_name in enumerate(obs_labels):
        fig, ax = plt.subplots(figsize=(9, 5))
        for shock_index in top:
            ax.plot(
                steps,
                irf.observables[:, obs_index, shock_index],
                linewidth=1.6,
                label=shock_labels[shock_index],
            )
        ax.set_title(f"IRF: {obs_name} ({normalization} shock)")
        ax.set_xlabel("Quarters after shock")
        ax.set_ylabel("Response")
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color="#444", linewidth=0.8)
        ax.legend(ncol=2, fontsize=8)
        path = figure_dir / f"irf_{obs_name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=_FIGURE_DPI)
        plt.close(fig)
        figures.append(path)
    if combined:
        figures.append(
            _plot_panel(
                plt,
                figure_dir / "irf_all_observables.png",
                obs_labels=obs_labels,
                series=[irf.observables[:, i, top].sum(axis=1) for i in range(len(obs_labels))],
                x=steps,
                suptitle=f"Impulse responses ({normalization} shocks, top movers summed)",
                xlabel="Quarters after shock",
            )
        )
    return figures


# --------------------------------------------------------------------------- #
# Historical decomposition
# --------------------------------------------------------------------------- #


def generate_historical_decomposition_report(
    model: DSGEModel,
    *,
    output_dir: Path,
    data: Any | None = None,
    check_empty_columns: bool = True,
    reconciliation_tolerance: float = 1.0e-5,
    make_plots: bool = True,
) -> ReportArtifacts:
    """Export a true historical shock decomposition and per-observable charts.

    The decomposition is built from the RTS-smoothed state path and the recovered
    smoothed structural shocks; contributions reconcile to the model-implied
    observable path within ``reconciliation_tolerance``. The baseline series
    carries the initial-condition contribution and ``residual`` is the
    measurement-error gap to the observed data.
    """

    output_dir = Path(output_dir)
    if data is None:
        data = load_data(model, check_empty_columns=check_empty_columns)
    decomposition = historical_decomposition(
        model,
        data=data,
        check_empty_columns=check_empty_columns,
    )
    if decomposition.reconciliation_max_abs_error > reconciliation_tolerance:
        msg = (
            "Historical decomposition failed to reconcile: max abs error "
            f"{decomposition.reconciliation_max_abs_error:.3e} exceeds tolerance "
            f"{reconciliation_tolerance:.3e}."
        )
        raise ReportError(msg)
    obs_labels = _observable_labels(model)
    shock_labels = _shock_labels(model)
    history_dates = _history_dates(model, data)
    periods = decomposition.observable_contributions.shape[0]
    if not history_dates:
        history_dates = [f"t{index}" for index in range(periods)]
    _check_axis("dates", history_dates, periods)
    _check_axis("observables", obs_labels, decomposition.observable_contributions.shape[1])
    _check_axis("shocks", shock_labels, decomposition.observable_contributions.shape[2])

    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / "historical_decomposition.npz"
    np.savez(
        array_path,
        observable_contributions=decomposition.observable_contributions,
        observable_baseline=decomposition.observable_baseline,
        smoothed_observables=decomposition.smoothed_observables,
        smoothed_shocks=decomposition.smoothed_shocks,
        observed=decomposition.observed,
        residual=decomposition.residual,
    )
    arrays = [array_path]
    figures: list[Path] = []
    if make_plots:
        figures = _plot_historical_decomposition(
            decomposition,
            output_dir / "historical_decomposition",
            obs_labels=obs_labels,
            shock_labels=shock_labels,
            dates=history_dates,
        )
    manifest = {
        "kind": "historical_decomposition_report",
        "model": model.spec,
        "subspec": model.subspec,
        "periods": periods,
        "reconciliation_max_abs_error": decomposition.reconciliation_max_abs_error,
        "reconciliation_tolerance": reconciliation_tolerance,
        "method": "rts_smoothed_shock_contributions",
        "labels": {
            "historical_decomposition/observable_contributions": {
                "axis0": history_dates,
                "axis1": obs_labels,
                "axis2": shock_labels,
            },
            "historical_decomposition/observable_baseline": {
                "axis0": history_dates,
                "axis1": obs_labels,
            },
            "historical_decomposition/smoothed_shocks": {
                "axis0": history_dates,
                "axis1": shock_labels,
            },
        },
        "arrays": [str(path) for path in arrays],
        "figures": [str(path) for path in figures],
    }
    summary = _write_summary(output_dir, "historical_decomposition_report.json", manifest)
    return ReportArtifacts(output_dir, figures, arrays, summary, manifest)


def _plot_historical_decomposition(
    decomposition: HistoricalDecomposition,
    figure_dir: Path,
    *,
    obs_labels: list[str],
    shock_labels: list[str],
    dates: list[str],
    top_k: int = 6,
    tail_periods: int = 60,
) -> list[Path]:
    plt = _require_matplotlib()
    figure_dir.mkdir(parents=True, exist_ok=True)
    contributions = decomposition.observable_contributions
    baseline = decomposition.observable_baseline
    smoothed = decomposition.smoothed_observables
    periods = contributions.shape[0]
    window = slice(max(0, periods - tail_periods), periods)
    x = np.arange(periods)[window]
    tick_dates = dates[window]
    figures: list[Path] = []
    for obs_index, obs_name in enumerate(obs_labels):
        obs_contrib = contributions[window, obs_index, :]
        scores = np.abs(obs_contrib).sum(axis=0)
        order = np.argsort(scores)[::-1]
        top = order[: min(top_k, order.size)]
        other = order[min(top_k, order.size) :]
        fig, ax = plt.subplots(figsize=(12, 5))
        pos_base = np.zeros(x.size)
        neg_base = np.zeros(x.size)
        # Baseline (initial condition + deterministic) as the first stacked block.
        deviation_base = baseline[window, obs_index] - smoothed[window, obs_index].mean()
        _stack_bar(ax, x, deviation_base, pos_base, neg_base, label="initial/baseline")
        for shock_index in top:
            _stack_bar(
                ax,
                x,
                obs_contrib[:, shock_index],
                pos_base,
                neg_base,
                label=shock_labels[shock_index],
            )
        if other.size:
            _stack_bar(
                ax,
                x,
                obs_contrib[:, other].sum(axis=1),
                pos_base,
                neg_base,
                label="other shocks",
            )
        ax.plot(
            x,
            smoothed[window, obs_index] - smoothed[window, obs_index].mean(),
            color="#111",
            linewidth=1.6,
            label="smoothed (demeaned)",
        )
        ax.set_title(f"Historical shock decomposition: {obs_name}")
        ax.set_ylabel("Contribution")
        ax.set_xlabel("Quarter")
        step = max(1, x.size // 12)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(tick_dates[::step], rotation=45, ha="right")
        ax.axhline(0.0, color="#444", linewidth=0.8)
        ax.legend(ncol=3, fontsize=7, loc="upper left")
        ax.grid(axis="y", alpha=0.25)
        path = figure_dir / f"historical_decomposition_{obs_name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=_FIGURE_DPI)
        plt.close(fig)
        figures.append(path)
    return figures


def _stack_bar(
    ax: Any,
    x: np.ndarray,
    values: np.ndarray,
    pos_base: np.ndarray,
    neg_base: np.ndarray,
    *,
    label: str,
) -> None:
    positive = np.where(values >= 0.0, values, 0.0)
    negative = np.where(values < 0.0, values, 0.0)
    ax.bar(x, positive, bottom=pos_base, width=0.9, label=label)
    ax.bar(x, negative, bottom=neg_base, width=0.9)
    pos_base += positive
    neg_base += negative


# --------------------------------------------------------------------------- #
# Combined forecast report
# --------------------------------------------------------------------------- #


def generate_forecast_report(
    model: DSGEModel,
    *,
    output_dir: Path,
    horizon: int = 40,
    data: Any | None = None,
    history_method: str = "smoothed",
    irf_normalization: str = "one_sd",
    check_empty_columns: bool = True,
    make_plots: bool = True,
) -> ReportArtifacts:
    """Produce the full forecast analysis bundle.

    Generates the all-observable macro forecast panel, per-observable impulse
    responses, and a true per-observable historical shock decomposition, plus the
    numeric arrays and a JSON summary tying them together.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    system = compute_system(model)
    if data is None:
        data = load_data(model, check_empty_columns=check_empty_columns)
    forecast = forecast_one(
        model,
        input_type="mode",
        cond_type="none",
        output_vars=["forecastobs", "forecaststates", "histobs", "histstates"],
        horizon=horizon,
        data=data,
        history_method=history_method,
        check_empty_columns=check_empty_columns,
    )
    obs_labels = _observable_labels(model)
    forecast_dates = _forecast_dates(model, forecast.observables.shape[0])
    history_dates = _history_dates(model, data)
    _check_axis("forecast_dates", forecast_dates, forecast.observables.shape[0])
    _check_axis("observables", obs_labels, forecast.observables.shape[1])

    forecast_array = output_dir / "forecast.npz"
    np.savez(
        forecast_array,
        observables=forecast.observables,
        states=forecast.states,
        history_observables=(
            forecast.history_observables
            if forecast.history_observables is not None
            else np.empty((0, len(obs_labels)))
        ),
    )
    arrays = [forecast_array]
    figures: list[Path] = []
    if make_plots:
        figures.append(
            _plot_macro_forecast(
                forecast,
                output_dir / "figures" / "macro_forecasts_all_observables.png",
                obs_labels=obs_labels,
                forecast_dates=forecast_dates,
                history_dates=history_dates,
                horizon=horizon,
            )
        )

    irf_artifacts = generate_irf_report(
        model,
        output_dir=output_dir,
        horizon=horizon,
        normalization=irf_normalization,
        make_plots=make_plots,
        system=system,
    )
    decomp_artifacts = generate_historical_decomposition_report(
        model,
        output_dir=output_dir,
        data=data,
        check_empty_columns=check_empty_columns,
        make_plots=make_plots,
    )
    arrays.extend(irf_artifacts.arrays)
    arrays.extend(decomp_artifacts.arrays)
    figures.extend(irf_artifacts.figures)
    figures.extend(decomp_artifacts.figures)

    manifest = {
        "kind": "forecast_report",
        "model": model.spec,
        "subspec": model.subspec,
        "data_vintage": str(model.get_setting("data_vintage", "")),
        "forecast_start": str(model.get_setting("date_forecast_start", "")),
        "horizon": horizon,
        "history_method": history_method,
        "irf_normalization": irf_normalization,
        "labels": {
            "forecast/observables": {"axis0": forecast_dates, "axis1": obs_labels},
            "forecast/states": {"axis0": forecast_dates, "axis1": _state_labels(model)},
        },
        "components": {
            "irf": irf_artifacts.manifest,
            "historical_decomposition": decomp_artifacts.manifest,
        },
        "arrays": [str(path) for path in arrays],
        "figures": [str(path) for path in figures],
    }
    summary = _write_summary(output_dir, "forecast_report.json", manifest)
    return ReportArtifacts(output_dir, figures, arrays, summary, manifest)


def _plot_macro_forecast(
    forecast: ForecastOutput,
    path: Path,
    *,
    obs_labels: list[str],
    forecast_dates: list[str],
    history_dates: list[str],
    horizon: int,
) -> Path:
    plt = _require_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    history = forecast.history_observables
    n_obs = len(obs_labels)
    cols = 3
    rows = (n_obs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.2 * rows))
    axes = np.array(axes).reshape(-1)
    hist_len = 0 if history is None else history.shape[0]
    x_hist = np.arange(hist_len)
    x_forecast = np.arange(hist_len, hist_len + horizon)
    for index, ax in enumerate(axes):
        if index >= n_obs:
            ax.axis("off")
            continue
        if history is not None and hist_len:
            ax.plot(x_hist, history[:, index], linewidth=1.4, color="#1f77b4", label="history")
        ax.plot(
            x_forecast,
            forecast.observables[:, index],
            linewidth=2.0,
            color="#ff7f0e",
            label="forecast",
        )
        if hist_len:
            ax.axvline(hist_len - 0.5, color="#777", linestyle="--", linewidth=0.8)
        ax.set_title(obs_labels[index], fontsize=10)
        ax.grid(alpha=0.3)
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", linewidth=1.4, label="history"),
        plt.Line2D([0], [0], color="#ff7f0e", linewidth=2.0, label="forecast"),
    ]
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle(
        f"Model1002 all macro observables: history + {horizon}-quarter forecast",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(path, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_panel(
    plt: Any,
    path: Path,
    *,
    obs_labels: list[str],
    series: list[np.ndarray],
    x: np.ndarray,
    suptitle: str,
    xlabel: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(obs_labels)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 3.0 * rows))
    axes = np.array(axes).reshape(-1)
    for index, ax in enumerate(axes):
        if index >= n:
            ax.axis("off")
            continue
        ax.plot(x, series[index], linewidth=1.5, color="#2ca02c")
        ax.axhline(0.0, color="#444", linewidth=0.7)
        ax.set_title(obs_labels[index], fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle(suptitle, fontsize=14)
    fig.text(0.5, 0.01, xlabel, ha="center")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    fig.savefig(path, dpi=_FIGURE_DPI)
    plt.close(fig)
    return path


def _write_summary(output_dir: Path, filename: str, manifest: dict[str, Any]) -> Path:
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return path
