from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nydsge.models import Model1002
from nydsge.solve import compute_system
from nydsge.forecast import _observable_shock_design

run_dir = Path(__file__).resolve().parent
forecast_path = run_dir / "forecast_period.npz"
manifest_path = run_dir / "manifest.json"
figure_dir = run_dir / "figures"
irf_dir = figure_dir / "impulse_responses"
hd_dir = figure_dir / "historical_decomposition"
figure_dir.mkdir(parents=True, exist_ok=True)
irf_dir.mkdir(parents=True, exist_ok=True)
hd_dir.mkdir(parents=True, exist_ok=True)

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
labels = manifest["labels"]
obs_labels = labels["forecast_period/observables"]["axis1"]
obs_dates = labels["forecast_period/observables"]["axis0"]
forecast_data = np.load(forecast_path, allow_pickle=True)["observables"]
history_obs = np.load(forecast_path, allow_pickle=True)["history_observables"]
hist_dates = labels["forecast_period/history_observables"]["axis0"]

n_obs = forecast_data.shape[1]
cols = 3
rows = (n_obs + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(18, 3.2 * rows), sharex=False)
axes = np.array(axes).reshape(-1)
x_hist = np.arange(len(hist_dates))
x_f = np.arange(len(hist_dates), len(hist_dates) + len(obs_dates))

for i, ax in enumerate(axes):
    if i < n_obs:
        ax.plot(x_hist, history_obs[:, i], linewidth=1.5, color="#1f77b4", label="history")
        ax.plot(x_f, forecast_data[:, i], linewidth=2, color="#ff7f0e", label="forecast")
        ax.axvline(x_f[0] - 0.5, color="#777", linestyle="--", linewidth=0.8)
        ax.set_title(obs_labels[i], fontsize=10)
        ax.grid(alpha=0.3)
    else:
        ax.axis("off")

handles = [
    plt.Line2D([0], [0], color="#1f77b4", linewidth=1.5, label="history"),
    plt.Line2D([0], [0], color="#ff7f0e", linewidth=2, label="forecast"),
]
fig.legend(handles=handles, loc="upper right", ncol=2)
fig.suptitle("Model1002 all macro observables: history + 12-quarter forecast", fontsize=14)
fig.text(0.5, 0.02, "Quarter", ha="center")
output_all = figure_dir / "macro_forecasts_all_observables.png"
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(output_all, dpi=160, bbox_inches="tight")
plt.close(fig)

# Build impulse response and decomposition matrices from model system
model = Model1002(settings={"date_forecast_start": "2018-Q4"})
system = compute_system(model)
horizon = forecast_data.shape[0]
shock_names = list(model.indexes.exogenous_shocks)
design = _observable_shock_design(system, horizon=horizon)

# design dimensions: (horizon, n_obs, horizon, n_shocks)
irf_t0 = design[:, :, 0, :]
shock_scores = np.abs(irf_t0).sum(axis=(0, 1))
order = np.argsort(shock_scores)[::-1]
top_k = min(6, shock_scores.size)
top_shocks = order[:top_k]

for obs_i, obs_name in enumerate(obs_labels):
    fig = plt.figure(figsize=(9, 5))
    h = np.arange(1, horizon + 1)
    for s in top_shocks:
        plt.plot(h, irf_t0[:, obs_i, s], label=shock_names[s], linewidth=1.6)
    plt.title(f"IRF: {obs_name} (period-0 shock)")
    plt.xlabel("Forecast quarter after shock")
    plt.ylabel("Response")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    out = irf_dir / f"irf_{obs_name}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

# Historical decomposition approximation (IRF-accumulated contributions)
cumulative_by_t = []
for t in range(horizon):
    cumulative_by_t.append(design[t, :, : t + 1, :].sum(axis=1))
decomp = np.stack(cumulative_by_t, axis=0)

for obs_i, obs_name in enumerate(obs_labels):
    fig, ax = plt.subplots(figsize=(11, 5))
    base = np.zeros(horizon)
    x = np.arange(horizon)
    for s in top_shocks:
        vals = decomp[:, obs_i, s]
        ax.bar(x, vals, bottom=base, width=0.85, label=shock_names[s])
        base = base + vals
    ax.set_xticks(x)
    ax.set_xticklabels(obs_dates, rotation=45, ha="right")
    ax.set_title(f"Historical decomposition (IRF-accumulated): {obs_name}")
    ax.set_ylabel("Accumulated contribution")
    ax.set_xlabel("Forecast quarter")
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    out = hd_dir / f"historical_decomposition_{obs_name}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

print(f"saved_all_macro={output_all}")
print(f"saved_irf_count={len(list(irf_dir.glob('*.png')))}")
print(f"saved_hd_count={len(list(hd_dir.glob('*.png')))}")
print(f"figure_dir={figure_dir}")
