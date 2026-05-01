"""Generate the extra figures and tables the report needs:

* fig_architecture.png — system architecture diagram
* per_class_metrics.json — per-class precision/recall/F1 for each model
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "reports" / "results"
FIGURES = REPO / "reports" / "figures"


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------
LABEL_ORDER = ["high", "low", "medium"]  # matches alphabetAsc StringIndexer


def per_class_metrics(cm: list[list[int]]) -> dict:
    cm_arr = np.array(cm, dtype=float)
    metrics = {}
    for i, label in enumerate(LABEL_ORDER):
        tp = cm_arr[i, i]
        fp = cm_arr[:, i].sum() - tp
        fn = cm_arr[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = int(cm_arr[i, :].sum())
        metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
    return metrics


def write_per_class_metrics():
    with (RESULTS / "model_results.json").open() as fh:
        results = json.load(fh)
    out = {r["name"]: per_class_metrics(r["confusion_matrix"]) for r in results}
    with (RESULTS / "per_class_metrics.json").open("w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote {RESULTS / 'per_class_metrics.json'}")
    return out


# ---------------------------------------------------------------------------
# System architecture diagram
# ---------------------------------------------------------------------------
def write_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.set_title("System Architecture", fontsize=14, fontweight="bold", pad=10)

    def box(x, y, w, h, label, fc, ec="#444"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.1",
            facecolor=fc, edgecolor=ec, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=10, fontweight="bold", color="#222",
        )

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
        )
        if label:
            ax.text(
                (x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label,
                ha="center", va="bottom", fontsize=8, color="#444", style="italic",
            )

    # Row 1: Data sources
    box(0.2, 5.3, 2.2, 0.9, "UCI / Kaggle\nMetro Interstate", "#cfe8ff")
    box(2.7, 5.3, 1.8, 0.9, "Amazon S3\n(s3a://...)", "#ffe7b3")
    box(4.8, 5.3, 1.8, 0.9, "Synthetic\nFallback", "#e2e2e2")

    # Row 2: Spark layer
    box(0.5, 3.8, 6.0, 0.9, "Apache Spark (DataFrame API + ML Pipeline)", "#bde0a8")
    box(0.7, 2.7, 1.6, 0.7, "Profiling", "#e7f5dc")
    box(2.5, 2.7, 1.6, 0.7, "Preprocessing", "#e7f5dc")
    box(4.3, 2.7, 1.0, 0.7, "Logistic\nReg.", "#fff2cc")
    box(5.4, 2.7, 1.0, 0.7, "Decision\nTree", "#fff2cc")

    # Row 3: Bedrock
    box(7.2, 3.8, 3.4, 0.9, "Amazon Bedrock\n(Claude 3 Haiku) / Mock", "#f4cccc")

    # Row 4: outputs
    box(0.5, 1.3, 2.0, 0.8, "Profile JSON\n+ CSVs", "#fce5cd")
    box(2.7, 1.3, 2.0, 0.8, "Model artifacts\n(Spark ML)", "#fce5cd")
    box(4.9, 1.3, 2.0, 0.8, "Metrics + CMs", "#fce5cd")
    box(7.1, 1.3, 1.7, 0.8, "Alerts JSON", "#fce5cd")
    box(8.95, 1.3, 1.7, 0.8, "Figures (PNG)", "#fce5cd")

    # Row 5: Report
    box(2.5, 0.1, 6.0, 0.7, "Final Project Report (.docx)", "#d9d2e9")

    # Random Forest (couldn't fit in row above)
    box(6.5, 2.7, 1.0, 0.7, "Random\nForest", "#fff2cc")

    # Arrows: data sources -> spark
    arrow(1.3, 5.3, 1.5, 4.7)
    arrow(3.5, 5.3, 3.0, 4.7)
    arrow(5.6, 5.3, 4.5, 4.7)

    # Spark -> ML stages
    arrow(1.5, 3.8, 1.5, 3.4)
    arrow(3.3, 3.8, 3.3, 3.4)

    # Models -> Bedrock
    arrow(6.5, 3.4, 8.0, 3.8)

    # Spark + Bedrock -> outputs
    arrow(1.5, 2.7, 1.5, 2.1)
    arrow(3.3, 2.7, 3.7, 2.1)
    arrow(5.0, 2.7, 5.9, 2.1)
    arrow(8.5, 3.8, 7.95, 2.1)
    arrow(7.0, 3.4, 9.8, 2.1)

    # outputs -> report
    arrow(5.5, 1.3, 5.5, 0.8)

    out = FIGURES / "fig_architecture.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


# ---------------------------------------------------------------------------
# Class-balance pie + traffic-volume histogram (extra profiling figure)
# ---------------------------------------------------------------------------
def write_volume_histogram():
    import pandas as pd
    df = pd.read_csv(REPO / "data" / "processed" / "processed_sample.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(df["traffic_volume"], bins=50, color="#1f77b4", edgecolor="white")
    ax.axvline(1500, color="#2ca02c", linestyle="--", linewidth=1.5, label="low/medium boundary (1,500)")
    ax.axvline(3500, color="#d62728", linestyle="--", linewidth=1.5, label="medium/high boundary (3,500)")
    ax.set_xlabel("Hourly traffic volume (vehicles)")
    ax.set_ylabel("Frequency (sampled rows)")
    ax.set_title("Distribution of hourly traffic volume with class boundaries")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = FIGURES / "fig_volume_histogram.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    write_per_class_metrics()
    write_architecture_diagram()
    write_volume_histogram()
