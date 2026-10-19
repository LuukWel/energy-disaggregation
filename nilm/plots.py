"""Optional figures (python -m nilm.pipeline ... --plots). Nothing here feeds back into the results."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from nilm.evaluation import MAIN_MODEL
from nilm.features import STAT_FEATURES, dtw_matrix, statistical_features
from nilm.transients import sampling_rate, smooth_power

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots in fixed order; an appliance keeps its colour in every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def save_figures(fig_dir, signals, transients, predictions, metrics, importance, appliances):
    """Write all figures and return their paths."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    colours = {appliance: SERIES[i % len(SERIES)] for i, appliance in enumerate(appliances)}

    figures = {
        "step_correction.png": plot_step_correction(signals, transients),
        "turn_on_transients.png": plot_transients(transients, appliances, colours),
        "model_comparison_turn_on.png": plot_model_comparison(metrics),
        "confusion_matrix_turn_on_rf.png": plot_confusion_matrix(predictions, appliances),
        "feature_importance_turn_on_rf.png": plot_feature_importance(importance),
        "pca_overview_turn_on.png": plot_pca_overview(transients, appliances, colours),
        "dtw_dendrogram_turn_on.png": plot_dtw_dendrogram(transients),
    }
    paths = []
    for name, fig in figures.items():
        if fig is None:
            continue
        path = fig_dir / name
        fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        paths.append(path)
    return paths


def _figure(width, height):
    return Figure(figsize=(width, height), facecolor=SURFACE)


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def plot_step_correction(signals, transients):
    """Tag time vs detected step for one event.

    Shows the event with the largest turn-on offset among events whose baseline
    is stable on both sides, so the example isn't distorted by another load.
    """
    complete = transients.groupby("event_id").filter(
        lambda g: (g["status"] == "ok").sum() == 2 and g["baseline_stable"].astype(bool).all()
    )
    if complete.empty:
        return None
    on = complete[complete["transition"] == "on"]
    event_id = on.loc[on["tag_offset_sec"].abs().idxmax(), "event_id"]
    rows = complete[complete["event_id"] == event_id].set_index("transition")
    first = rows.iloc[0]
    signal = signals[(first["house"], first["segment"])]
    smooth, fs = smooth_power(signal.power), sampling_rate(signal.ts)

    fig = _figure(10, 3.4)
    axes = fig.subplots(1, 2)
    for ax, transition in zip(axes, ("on", "off")):
        row = rows.loc[transition]
        lo, hi = np.searchsorted(signal.ts, [row["tag_ts"] - 50, row["tag_ts"] + 50])
        t = signal.ts[lo:hi] - row["tag_ts"]
        ax.plot(t, smooth[lo:hi], color=SERIES[0], linewidth=2, label="smoothed power")

        # Shade the adaptive cut: 1 s before the step until the signal settled.
        cut_start = row["step_ts"] - row["tag_ts"] - int(fs) / fs
        ax.axvspan(cut_start, cut_start + row["transient_duration"], color=SERIES[1], alpha=0.12,
                   linewidth=0, label="extracted transient")
        ax.axvline(0, color=INK_MUTED, linewidth=1.2, label="tag")
        ax.axvline(row["step_ts"] - row["tag_ts"], color=SERIES[1], linewidth=2,
                   label=f"detected step ({row['tag_offset_sec']:+.1f} s)")

        _style(ax)
        ax.set_title(f"Turn-{transition}", loc="left", fontsize=10)
        ax.set_xlabel("Seconds from tag")
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_SECONDARY, loc="best")
    axes[0].set_ylabel("Aggregate power (W)")
    fig.suptitle(f"{first['appliance']} in {first['house']}: the tag is not the switching time",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def plot_transients(transients, appliances, colours, transition="on"):
    """Every extracted turn-on waveform, one panel per appliance."""
    subset = transients[transients["transition"] == transition]
    present = [a for a in appliances if a in set(subset["appliance"])]
    fig = _figure(2.6 * len(present), 3.0)
    axes = np.atleast_1d(fig.subplots(1, len(present), sharex=True))
    for ax, appliance in zip(axes, present):
        rows = subset[subset["appliance"] == appliance]
        for row in rows.itertuples():
            # The cut starts 1 s before the step; duration / samples gives the real spacing.
            t = np.arange(len(row.waveform)) * row.transient_duration / len(row.waveform) - 1
            ax.plot(t, row.waveform, color=colours[appliance], linewidth=1.5, alpha=0.6)
        _style(ax)
        ax.set_title(f"{appliance} (n={len(rows)})", loc="left", fontsize=9)
        ax.set_xlabel("Seconds from step")
    axes[0].set_ylabel("Power above baseline (W)")
    fig.suptitle(f"Extracted turn-{transition} transients (variable length; y-scales differ)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics, mode="turn_on"):
    """Mean accuracy per model with the range over seeds; the main model highlighted."""
    rows = metrics[metrics["mode"] == mode].iloc[::-1]
    if rows.empty:
        return None
    fig = _figure(6.5, 0.4 * len(rows) + 1.3)
    ax = fig.subplots()
    y = np.arange(len(rows))
    colours = [SERIES[0] if model == MAIN_MODEL else AXIS for model in rows["model"]]
    ax.barh(y, 100 * rows["accuracy_mean"], height=0.55, color=colours)
    spread = [100 * (rows["accuracy_mean"] - rows["accuracy_min"]), 100 * (rows["accuracy_max"] - rows["accuracy_mean"])]
    ax.errorbar(100 * rows["accuracy_mean"], y, xerr=spread, fmt="none", ecolor=INK_SECONDARY, elinewidth=1, capsize=2)
    for yi, value, top in zip(y, rows["accuracy_mean"], rows["accuracy_max"]):
        ax.text(100 * top + 1.5, yi, f"{100 * value:.1f}%", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(y, rows["model"])
    ax.set_xlim(0, 100)
    _style(ax, grid_axis="x")
    ax.set_xlabel("Leave-one-house-out accuracy (%)")
    n_seeds = int(rows["n_seeds"].max())
    ax.set_title(f"{mode.replace('_', '-')} transients: mean accuracy, bars show the range over {n_seeds} seeds",
                 loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(predictions, appliances, mode="turn_on", model=MAIN_MODEL):
    """Row-normalised confusion matrix, pooled over all seeds."""
    rows = predictions[(predictions["mode"] == mode) & (predictions["model"] == model)]
    if rows.empty:
        return None
    labels = [a for a in appliances if a in set(rows["appliance_true"]) | set(rows["appliance_predicted"])]
    counts = confusion_matrix(rows["appliance_true"], rows["appliance_predicted"], labels=labels)
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)

    fig = _figure(5.2, 4.4)
    ax = fig.subplots()
    # Zero stays the surface colour; anything above zero gets at least the lightest blue.
    cmap = ListedColormap(BLUES)
    cmap.set_under(SURFACE)
    ax.imshow(shares, cmap=cmap, vmin=1e-9, vmax=1.0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if counts[i, j]:
                ax.text(j, i, f"{shares[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color=SURFACE if shares[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    _style(ax, grid_axis=None)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    n_seeds = rows["seed"].nunique()
    accuracy = rows["correct"].mean()
    ax.set_title(f"{model}, {mode.replace('_', '-')}, leave-one-house-out\n"
                 f"accuracy {accuracy:.1%}; rows sum to 100%, pooled over {n_seeds} seed(s)",
                 loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance, mode="turn_on"):
    rows = importance[importance["mode"] == mode].sort_values("importance_mean")
    if rows.empty:
        return None
    fig = _figure(6, 0.28 * len(rows) + 1.2)
    ax = fig.subplots()
    y = np.arange(len(rows))
    ax.barh(y, rows["importance_mean"], height=0.6, color=SERIES[0])
    ax.errorbar(rows["importance_mean"], y, xerr=rows["importance_std"].fillna(0), fmt="none",
                ecolor=INK_SECONDARY, elinewidth=1, capsize=2)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_yticks(y, rows["feature"])
    _style(ax, grid_axis="x")
    ax.set_xlabel("Drop in held-out accuracy when the feature is shuffled")
    ax.set_title(f"Random Forest permutation importance, {mode.replace('_', '-')}\n"
                 "mean +/- std over leave-one-house-out folds and seeds", loc="left", fontsize=10)
    fig.tight_layout()
    return fig


def plot_pca_overview(transients, appliances, colours, transition="on"):
    """PCA of the statistical features (no prototypes involved). Exploratory only."""
    subset = transients[transients["transition"] == transition]
    if len(subset) < 3:
        return None
    features = pd.DataFrame([statistical_features(row) for _, row in subset.iterrows()])[STAT_FEATURES]
    xy = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(np.nan_to_num(features.to_numpy())))
    labels = subset["appliance"].to_numpy()

    present = [a for a in appliances if a in set(labels)]
    fig = _figure(2.4 * len(present), 2.7)
    axes = np.atleast_1d(fig.subplots(1, len(present), sharex=True, sharey=True))
    for ax, appliance in zip(axes, present):
        mine = labels == appliance
        ax.scatter(xy[~mine, 0], xy[~mine, 1], s=14, color=AXIS, linewidths=0)
        ax.scatter(xy[mine, 0], xy[mine, 1], s=40, color=colours[appliance], edgecolors=SURFACE, linewidths=1.5)
        _style(ax, grid_axis="both")
        ax.set_title(appliance, loc="left", fontsize=9)
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    fig.suptitle(f"PCA of turn-{transition} statistical features (all houses, exploratory)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def plot_dtw_dendrogram(transients, transition="on"):
    """Average-linkage clustering of the raw DTW distances. Exploratory only."""
    subset = transients[transients["transition"] == transition]
    if len(subset) < 3:
        return None
    distances = dtw_matrix(list(subset["waveform"]))
    # Average linkage, not Ward: Ward assumes Euclidean distances and DTW isn't one.
    tree = linkage(squareform(distances, checks=False), method="average")
    leaf_labels = [f"{r.appliance} {r.house}" for r in subset.itertuples()]

    fig = _figure(6, 0.2 * len(subset) + 1.2)
    ax = fig.subplots()
    dendrogram(tree, labels=leaf_labels, orientation="left", ax=ax, leaf_font_size=8,
               link_color_func=lambda _: INK_SECONDARY)
    _style(ax, grid_axis=None)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("DTW distance (sum of |difference| in W along the warping path)")
    ax.set_title(f"Turn-{transition} transients clustered by DTW distance", loc="left", fontsize=10)
    fig.tight_layout()
    return fig
