#!/usr/bin/env python3
# This source file is part of the ARPA-H CARE LLM project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
CLI for evaluating LLM surgical decision predictions against ground truth.

Compares the LLM 'decision' column (Yes/No) from a results CSV against
the 'had_surgery' column (True/False) from the processed data CSV,
matched on llm_caseID.
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

DEFAULT_GROUND_TRUTH = "data/processed_data_20260202.csv"
DEFAULT_PREDICTIONS_DIR = "data/decision_results/final"
DEFAULT_PLOT_DIR = "data/evaluation_plots"

CONFIDENCE_BINS = {
    "Low (1-4)": (1, 4),
    "Medium (5-7)": (5, 7),
    "High (8-10)": (8, 10),
}


def load_predictions(path: str) -> pd.DataFrame:
    """Load LLM predictions and normalize the decision column to boolean."""
    df = pd.read_csv(path, usecols=["llm_caseID", "decision", "confidence"])
    df["predicted"] = df["decision"].str.strip().str.lower() == "yes"
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load ground truth and ensure had_surgery is boolean."""
    df = pd.read_csv(path, usecols=["llm_caseID", "had_surgery"])
    df["actual"] = df["had_surgery"].astype(bool)
    return df


def evaluate(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """Merge predictions with ground truth on llm_caseID and return merged df."""
    merged = predictions.merge(ground_truth, on="llm_caseID", how="inner")
    if len(merged) == 0:
        raise ValueError("No matching llm_caseID values found between predictions and ground truth.")
    return merged


def compute_metrics(y_true, y_pred, confidence=None) -> dict:
    """Compute all evaluation metrics from arrays."""
    if len(y_true) == 0 or y_true.nunique() < 2 or y_pred.nunique() < 1:
        return None

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    prevalence = y_true.sum() / len(y_true)

    auc = None
    if confidence is not None:
        try:
            auc = roc_auc_score(y_true, confidence)
        except ValueError:
            pass

    return {
        "n": len(y_true),
        "prevalence": prevalence,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "mcc": mcc,
        "auc_roc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def print_metrics(merged: pd.DataFrame, label: str = "") -> dict:
    """Compute and print evaluation metrics. Returns metrics dict."""
    y_true = merged["actual"].astype(int)
    y_pred = merged["predicted"].astype(int)

    metrics = compute_metrics(y_true, y_pred, merged.get("confidence"))

    header = f"=== Evaluation Results{f': {label}' if label else ''} ==="
    print(header)
    print(f"Total matched cases: {metrics['n']}")
    print(f"Prevalence (actual surgery rate): {metrics['prevalence']:.3f}")
    print()

    print("Confusion Matrix:")
    print(f"  {'':>20} Predicted No  Predicted Yes")
    print(f"  {'Actual No':>20}    {metrics['tn']:>7}        {metrics['fp']:>7}")
    print(f"  {'Actual Yes':>20}    {metrics['fn']:>7}        {metrics['tp']:>7}")
    print()

    print("Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}  (PPV)")
    print(f"  Recall:      {metrics['recall']:.4f}  (Sensitivity)")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  NPV:         {metrics['npv']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  MCC:         {metrics['mcc']:.4f}")
    if metrics["auc_roc"] is not None:
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}  (using confidence scores)")
    print()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Surgery", "Surgery"]))

    return metrics


def print_confidence_subgroups(merged: pd.DataFrame, label: str = "") -> list[dict]:
    """Compute and print metrics broken down by confidence bins."""
    print(f"--- Confidence Subgroup Analysis{f': {label}' if label else ''} ---")

    subgroup_metrics = []
    for bin_label, (lo, hi) in CONFIDENCE_BINS.items():
        subset = merged[(merged["confidence"] >= lo) & (merged["confidence"] <= hi)]
        if len(subset) == 0:
            print(f"\n  {bin_label}: no cases")
            continue

        y_true = subset["actual"].astype(int)
        y_pred = subset["predicted"].astype(int)

        m = compute_metrics(y_true, y_pred)
        if m is None:
            print(f"\n  {bin_label} (n={len(subset)}): insufficient class variation")
            continue

        m["confidence_bin"] = bin_label
        subgroup_metrics.append(m)

        surgery_rate = y_pred.sum() / len(y_pred)
        print(f"\n  {bin_label} (n={m['n']}, prevalence={m['prevalence']:.3f}, surgery_pred_rate={surgery_rate:.3f}):")
        print(f"    Accuracy={m['accuracy']:.3f}  Precision={m['precision']:.3f}  "
              f"Recall={m['recall']:.3f}  F1={m['f1']:.3f}  MCC={m['mcc']:.3f}")

    # Per-confidence-level accuracy
    print(f"\n  Per-confidence-level accuracy:")
    for conf_val in sorted(merged["confidence"].dropna().unique()):
        subset = merged[merged["confidence"] == conf_val]
        y_true = subset["actual"].astype(int)
        y_pred = subset["predicted"].astype(int)
        acc = accuracy_score(y_true, y_pred) if len(subset) > 0 else 0
        print(f"    Confidence {int(conf_val):>2}: n={len(subset):>6}  accuracy={acc:.3f}")

    print()
    return subgroup_metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _model_short_name(label: str) -> str:
    """Extract a short model name from a filename label."""
    name = label.replace("all_results_", "").replace(".csv", "")
    # Remove date suffix like _20260204
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]
    return name


def plot_confusion_matrices(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Plot confusion matrix for each model."""
    n = len(all_merged)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (label, merged) in zip(axes, all_merged.items()):
        y_true = merged["actual"].astype(int)
        y_pred = merged["predicted"].astype(int)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Surgery", "Surgery"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(_model_short_name(label))

    fig.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(plot_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _add_bar_labels(ax, bars, fmt=".3f"):
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f"{height:{fmt}}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)


def plot_metrics_comparison(all_metrics: list[dict], plot_dir: str) -> None:
    """Bar chart comparing key metrics across models."""
    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "mcc"]
    metric_labels = ["Accuracy", "Precision\n(PPV)", "Recall\n(Sens.)", "Specificity", "F1", "MCC"]

    models = [_model_short_name(m["model"]) for m in all_metrics]
    x = np.arange(len(metric_keys))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, name) in enumerate(zip(all_metrics, models)):
        values = [m[k] for k in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=name)
        _add_bar_labels(ax, bars)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Key Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "barplot_metrics_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_counts(all_metrics: list[dict], plot_dir: str) -> None:
    """Grouped bar chart of TP, FP, TN, FN counts across models."""
    models = [_model_short_name(m["model"]) for m in all_metrics]
    count_keys = ["tp", "fp", "tn", "fn"]
    count_labels = ["True Pos", "False Pos", "True Neg", "False Neg"]
    colors = ["#4CAF50", "#FF9800", "#2196F3", "#F44336"]

    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (key, clabel, color) in enumerate(zip(count_keys, count_labels, colors)):
        values = [m[key] for m in all_metrics]
        bars = ax.bar(x + i * width, values, width, label=clabel, color=color)
        _add_bar_labels(ax, bars, fmt=".0f")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.set_ylabel("Count")
    ax.set_title("Confusion Matrix Counts by Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "barplot_confusion_counts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metrics_by_confidence_bin(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Grouped bar charts of key metrics by confidence bin, one subplot per metric."""
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_titles = ["Accuracy", "Precision (PPV)", "Recall (Sensitivity)", "F1 Score"]
    bin_labels = list(CONFIDENCE_BINS.keys())

    # Gather data: {model: {bin: {metric: val}}}
    data = {}
    for label, merged in all_merged.items():
        name = _model_short_name(label)
        data[name] = {}
        for bl, (lo, hi) in CONFIDENCE_BINS.items():
            subset = merged[(merged["confidence"] >= lo) & (merged["confidence"] <= hi)]
            if len(subset) == 0:
                continue
            m = compute_metrics(subset["actual"].astype(int), subset["predicted"].astype(int))
            if m:
                data[name][bl] = m

    models = list(data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for idx, (mk, mt) in enumerate(zip(metric_keys, metric_titles)):
        ax = axes[idx]
        x = np.arange(len(bin_labels))
        width = 0.8 / len(models)

        for i, name in enumerate(models):
            values = [data[name].get(bl, {}).get(mk, 0) if isinstance(data[name].get(bl), dict) else 0
                      for bl in bin_labels]
            bars = ax.bar(x + i * width, values, width, label=name)
            _add_bar_labels(ax, bars)

        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(bin_labels, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score")
        ax.set_title(mt)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Metrics by Confidence Bin", fontsize=14)
    fig.tight_layout()
    path = os.path.join(plot_dir, "barplot_metrics_by_confidence_bin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accuracy_by_confidence_bars(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Grouped bar chart of accuracy at each confidence level (1-10)."""
    conf_range = range(1, 11)
    models = {_model_short_name(label): merged for label, merged in all_merged.items()}
    model_names = list(models.keys())

    x = np.arange(len(conf_range))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(model_names):
        merged = models[name]
        accs = []
        for c in conf_range:
            subset = merged[merged["confidence"] == c]
            if len(subset) > 0:
                accs.append(accuracy_score(subset["actual"].astype(int), subset["predicted"].astype(int)))
            else:
                accs.append(0)
        bars = ax.bar(x + i * width, accs, width, label=name)
        _add_bar_labels(ax, bars)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([str(c) for c in conf_range])
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.set_title("Accuracy by Confidence Level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "barplot_accuracy_by_confidence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_case_count_by_confidence(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Grouped bar chart of case counts at each confidence level."""
    conf_range = range(1, 11)
    models = {_model_short_name(label): merged for label, merged in all_merged.items()}
    model_names = list(models.keys())

    x = np.arange(len(conf_range))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(model_names):
        merged = models[name]
        counts = [len(merged[merged["confidence"] == c]) for c in conf_range]
        bars = ax.bar(x + i * width, counts, width, label=name)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([str(c) for c in conf_range])
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Number of Cases")
    ax.set_title("Case Count by Confidence Level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "barplot_case_count_by_confidence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_surgery_rates(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Grouped bar chart: predicted vs actual surgery rate for all models on one plot."""
    models = {_model_short_name(label): merged for label, merged in all_merged.items()}
    model_names = list(models.keys())

    pred_rates = [m["predicted"].mean() for m in models.values()]
    actual_rates = [m["actual"].mean() for m in models.values()]

    x = np.arange(len(model_names))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, pred_rates, width, label="Predicted Surgery Rate", color="#2196F3")
    bars2 = ax.bar(x + width / 2, actual_rates, width, label="Actual Surgery Rate", color="#FF9800")
    _add_bar_labels(ax, bars1)
    _add_bar_labels(ax, bars2)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(max(pred_rates), max(actual_rates)) * 1.25)
    ax.set_title("Predicted vs Actual Surgery Rate by Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "barplot_surgery_rates.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_accuracy_by_confidence(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Line plot of accuracy at each confidence level per model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, merged in all_merged.items():
        conf_levels = sorted(merged["confidence"].dropna().unique())
        accuracies = []
        counts = []
        for c in conf_levels:
            subset = merged[merged["confidence"] == c]
            acc = accuracy_score(subset["actual"].astype(int), subset["predicted"].astype(int))
            accuracies.append(acc)
            counts.append(len(subset))

        name = _model_short_name(label)
        ax.plot(conf_levels, accuracies, marker="o", label=name, linewidth=2)

        # Annotate counts
        for c, acc, n in zip(conf_levels, accuracies, counts):
            ax.annotate(f"n={n}", (c, acc), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=6, alpha=0.7)

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Confidence Level")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "accuracy_by_confidence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_f1_by_confidence(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Line plot of F1 score at each confidence level per model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, merged in all_merged.items():
        conf_levels = sorted(merged["confidence"].dropna().unique())
        f1_scores = []
        for c in conf_levels:
            subset = merged[merged["confidence"] == c]
            y_true = subset["actual"].astype(int)
            y_pred = subset["predicted"].astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

        name = _model_short_name(label)
        ax.plot(conf_levels, f1_scores, marker="s", label=name, linewidth=2)

    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by Confidence Level")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "f1_by_confidence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confidence_distribution(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Stacked bar chart: confidence distribution colored by correct/incorrect."""
    n = len(all_merged)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (label, merged) in zip(axes, all_merged.items()):
        merged = merged.copy()
        merged["correct"] = merged["actual"] == merged["predicted"]
        conf_levels = range(1, 11)

        correct_counts = [len(merged[(merged["confidence"] == c) & merged["correct"]]) for c in conf_levels]
        incorrect_counts = [len(merged[(merged["confidence"] == c) & ~merged["correct"]]) for c in conf_levels]

        ax.bar(conf_levels, correct_counts, label="Correct", color="#4CAF50", alpha=0.85)
        ax.bar(conf_levels, incorrect_counts, bottom=correct_counts, label="Incorrect", color="#F44336", alpha=0.85)

        ax.set_xlabel("Confidence Score")
        ax.set_title(_model_short_name(label))
        ax.set_xticks(range(1, 11))
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Number of Cases")
    fig.suptitle("Confidence Distribution: Correct vs Incorrect", fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(plot_dir, "confidence_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curves(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """ROC curves for each model using confidence as the score."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for label, merged in all_merged.items():
        y_true = merged["actual"].astype(int)
        conf = merged["confidence"]
        if conf.isna().all():
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, conf)
            auc = roc_auc_score(y_true, conf)
            name = _model_short_name(label)
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (using confidence as score)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(plot_dir, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_subgroup_heatmap(all_merged: dict[str, pd.DataFrame], plot_dir: str) -> None:
    """Heatmap of metrics by confidence bin and model."""
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    rows = []
    for label, merged in all_merged.items():
        name = _model_short_name(label)
        for bin_label, (lo, hi) in CONFIDENCE_BINS.items():
            subset = merged[(merged["confidence"] >= lo) & (merged["confidence"] <= hi)]
            if len(subset) == 0:
                continue
            y_true = subset["actual"].astype(int)
            y_pred = subset["predicted"].astype(int)
            m = compute_metrics(y_true, y_pred)
            if m is None:
                continue
            for mk in metric_keys:
                rows.append({"model": name, "confidence_bin": bin_label, "metric": mk, "value": m[mk]})

    if not rows:
        return

    df = pd.DataFrame(rows)

    for metric in metric_keys:
        mdf = df[df["metric"] == metric]
        pivot = mdf.pivot(index="model", columns="confidence_bin", values="value")
        # Reorder columns
        ordered_cols = [b for b in CONFIDENCE_BINS if b in pivot.columns]
        pivot = pivot[ordered_cols]

        fig, ax = plt.subplots(figsize=(6, max(3, len(pivot) * 0.8 + 1)))
        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=11)

        ax.set_title(f"{metric.capitalize()} by Confidence Bin")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"heatmap_{metric}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def generate_all_plots(all_merged: dict[str, pd.DataFrame], all_metrics: list[dict], plot_dir: str) -> None:
    """Generate all evaluation plots."""
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating plots in {plot_dir}/:")

    # Bar plots comparing all models
    plot_metrics_comparison(all_metrics, plot_dir)
    plot_confusion_counts(all_metrics, plot_dir)
    plot_metrics_by_confidence_bin(all_merged, plot_dir)
    plot_accuracy_by_confidence_bars(all_merged, plot_dir)
    plot_case_count_by_confidence(all_merged, plot_dir)
    plot_surgery_rates(all_merged, plot_dir)

    # Other plots
    plot_confusion_matrices(all_merged, plot_dir)
    plot_accuracy_by_confidence(all_merged, plot_dir)
    plot_f1_by_confidence(all_merged, plot_dir)
    plot_confidence_distribution(all_merged, plot_dir)
    plot_roc_curves(all_merged, plot_dir)
    plot_subgroup_heatmap(all_merged, plot_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def save_metrics(all_metrics: list[dict], output_path: str) -> None:
    """Save metrics for all models to a CSV file."""
    df = pd.DataFrame(all_metrics)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM surgical decision predictions against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single predictions file
  python cli_evaluate.py -p data/decision_results/final/all_results_gpt4.1_20260203.csv

  # Evaluate all models with subgroup analysis and plots
  python cli_evaluate.py --all --plot

  # Evaluate all and save metrics to CSV
  python cli_evaluate.py --all --output evaluation_metrics.csv --plot

  # Custom plot directory
  python cli_evaluate.py --all --plot --plot-dir my_plots
        """,
    )

    parser.add_argument(
        "--predictions",
        "-p",
        type=str,
        help="Path to LLM predictions CSV (must have llm_caseID and decision columns)",
    )

    parser.add_argument(
        "--ground-truth",
        "-g",
        type=str,
        default=DEFAULT_GROUND_TRUTH,
        help=f"Path to ground truth CSV (default: {DEFAULT_GROUND_TRUTH})",
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help=f"Evaluate all CSV files in {DEFAULT_PREDICTIONS_DIR}",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save evaluation metrics to this CSV file",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate evaluation plots",
    )

    parser.add_argument(
        "--plot-dir",
        type=str,
        default=DEFAULT_PLOT_DIR,
        help=f"Directory to save plots (default: {DEFAULT_PLOT_DIR})",
    )

    args = parser.parse_args()

    if not args.predictions and not args.all:
        parser.error("Either --predictions or --all is required")

    ground_truth = load_ground_truth(args.ground_truth)
    all_metrics = []
    all_merged = {}

    if args.all:
        csv_files = sorted(glob.glob(os.path.join(DEFAULT_PREDICTIONS_DIR, "*.csv")))
        if not csv_files:
            print(f"No CSV files found in {DEFAULT_PREDICTIONS_DIR}")
            return 1

        for csv_path in csv_files:
            label = os.path.basename(csv_path)
            predictions = load_predictions(csv_path)
            merged = evaluate(predictions, ground_truth)
            all_merged[label] = merged

            metrics = print_metrics(merged, label=label)
            metrics["model"] = label
            all_metrics.append(metrics)

            print_confidence_subgroups(merged, label=label)
            print()
    else:
        label = os.path.basename(args.predictions)
        predictions = load_predictions(args.predictions)
        merged = evaluate(predictions, ground_truth)
        all_merged[label] = merged

        metrics = print_metrics(merged)
        metrics["model"] = label
        all_metrics.append(metrics)

        print_confidence_subgroups(merged)

    if args.output:
        save_metrics(all_metrics, args.output)

    if args.plot:
        generate_all_plots(all_merged, all_metrics, args.plot_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
