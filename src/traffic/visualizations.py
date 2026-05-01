"""Generate the figures used in the report.

Every chart is produced from the artifacts written by `profiling.py` and
`models.py` (CSV / JSON), so the visualization step can be run after the
fact without re-fitting models. Charts are saved as PNG into
`reports/figures/` and embedded into the report by `report_writer.py`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from traffic.config import FIGURES_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


# Consistent colors for the three congestion classes
CONGESTION_COLORS = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}


def _save(fig, name: str) -> Path:
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure %s", out)
    return out


def hourly_distribution_chart(results_dir: Path = RESULTS_DIR) -> Path:
    df = pd.read_csv(results_dir / "profile_hourly_volume.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["hour"], df["avg_volume"], color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average traffic volume (vehicles / hour)")
    ax.set_title("Average hourly traffic volume on I-94")
    ax.set_xticks(range(0, 24))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig_hourly_distribution.png")


def dow_distribution_chart(results_dir: Path = RESULTS_DIR) -> Path:
    df = pd.read_csv(results_dir / "profile_dow_volume.csv")
    labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#9467bd" if d in (1, 7) else "#1f77b4" for d in df["dow"]]
    ax.bar(labels, df["avg_volume"], color=colors, edgecolor="white")
    ax.set_ylabel("Average traffic volume")
    ax.set_title("Average traffic volume by day of week")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig_dow_distribution.png")


def weather_impact_chart(results_dir: Path = RESULTS_DIR) -> Path:
    df = pd.read_csv(results_dir / "profile_weather_volume.csv")
    df = df.sort_values("avg_volume", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(df["weather_main"], df["avg_volume"], color="#17becf", edgecolor="white")
    ax.set_xlabel("Average traffic volume")
    ax.set_title("Average traffic volume by weather category")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "fig_weather_impact.png")


def congestion_distribution_chart(results_dir: Path = RESULTS_DIR) -> Path:
    """Bar chart of class counts (low / medium / high)."""
    profile_path = results_dir / "profile.json"
    label_counts_path = results_dir / "class_counts.csv"
    if label_counts_path.exists():
        df = pd.read_csv(label_counts_path)
        labels = df["congestion"].tolist()
        counts = df["count"].tolist()
    else:
        # Fallback: read from raw quantiles -> not ideal, just skip
        return None  # pragma: no cover

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        labels,
        counts,
        color=[CONGESTION_COLORS[l] for l in labels],
        edgecolor="white",
    )
    ax.set_ylabel("Number of records")
    ax.set_title("Distribution of congestion classes")
    for bar, c in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{c:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig_congestion_distribution.png")


def model_comparison_chart(results_dir: Path = RESULTS_DIR) -> Path:
    with (results_dir / "model_results.json").open() as fh:
        results = json.load(fh)

    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = [r["name"] for r in results]
    width = 0.2
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["#1f77b4", "#2ca02c", "#d62728"]
    for i, r in enumerate(results):
        values = [r[m] for m in metrics]
        ax.bar(
            x + i * width,
            values,
            width,
            label=r["name"],
            color=palette[i % len(palette)],
            edgecolor="white",
        )
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison: weighted classification metrics")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "fig_model_comparison.png")


def confusion_matrices_chart(results_dir: Path = RESULTS_DIR) -> Path:
    with (results_dir / "model_results.json").open() as fh:
        results = json.load(fh)

    label_order = ["high", "low", "medium"]  # matches alphabetAsc StringIndexer
    fig, axes = plt.subplots(1, len(results), figsize=(15, 4.5))
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = np.array(r["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(r["name"])
        ax.set_xticks(range(len(label_order)))
        ax.set_yticks(range(len(label_order)))
        ax.set_xticklabels(label_order)
        ax.set_yticklabels(label_order)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrices on the held-out test set")
    fig.tight_layout()
    return _save(fig, "fig_confusion_matrices.png")


def feature_importance_chart(
    feature_names: List[str], results_dir: Path = RESULTS_DIR
) -> Path:
    with (results_dir / "model_results.json").open() as fh:
        results = json.load(fh)

    rf = next((r for r in results if r["name"] == "RandomForest"), None)
    if not rf or not rf["feature_importances"]:
        return None  # pragma: no cover

    importances = rf["feature_importances"]
    if len(importances) != len(feature_names):
        # OneHot expansion makes the assembled feature vector longer than
        # the source column list. Group OHE positions back into their
        # source column for a clean bar chart.
        # Falls back to "feature_i" labels if we can't reconcile.
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    pairs = sorted(zip(feature_names, importances), key=lambda p: p[1])
    names, vals = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, vals, color="#2ca02c", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest feature importance")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "fig_feature_importance.png")


def temperature_volume_scatter(processed_csv: Path) -> Path:
    """Hex-bin density of temperature vs. traffic volume.

    Reads from a sampled processed CSV to keep matplotlib responsive.
    """
    df = pd.read_csv(processed_csv)
    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hexbin(
        df["temp_c"], df["traffic_volume"], gridsize=40, cmap="viridis", mincnt=1
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Traffic volume")
    ax.set_title("Traffic volume vs. temperature (hex-bin density)")
    fig.colorbar(hb, ax=ax, label="Records")
    return _save(fig, "fig_temp_vs_volume.png")


def generate_all(feature_names: List[str], processed_sample_csv: Path) -> Dict[str, Path]:
    paths = {
        "hourly": hourly_distribution_chart(),
        "dow": dow_distribution_chart(),
        "weather": weather_impact_chart(),
        "congestion": congestion_distribution_chart(),
        "model_comparison": model_comparison_chart(),
        "confusion": confusion_matrices_chart(),
        "feature_importance": feature_importance_chart(feature_names),
        "temp_vs_volume": temperature_volume_scatter(processed_sample_csv),
    }
    return {k: v for k, v in paths.items() if v is not None}
