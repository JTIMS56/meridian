"""
Visualization utilities for PII detection evaluation.
Generates confusion matrices, ROC curves, model comparison charts, and error analysis.
"""
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import OUTPUT_DIR, VISUALIZATION_DPI

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    model_name: str,
    save_path: Path = None,
):
    """Plot and save a heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) * 0.7)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    save_path = save_path or OUTPUT_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=VISUALIZATION_DPI)
    plt.close(fig)
    logger.info(f"Saved confusion matrix → {save_path}")
    return save_path


def plot_roc_curves(
    y_true: List[str],
    model_probas: Dict[str, np.ndarray],
    classes: List[str],
    save_path: Path = None,
):
    """
    Plot macro-averaged ROC curves for multiple models on one figure.
    model_probas: {model_name: (n_tokens, n_classes) array}
    """
    y_true_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, probas in model_probas.items():
        fpr_macro, tpr_macro = [], []
        for i in range(len(classes)):
            if y_true_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probas[:, i])
            fpr_macro.append(fpr)
            tpr_macro.append(tpr)

        # Interpolate to common FPR grid
        all_fpr = np.unique(np.concatenate(fpr_macro))
        mean_tpr = np.zeros_like(all_fpr)
        for fpr, tpr in zip(fpr_macro, tpr_macro):
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr /= len(fpr_macro) if fpr_macro else 1
        roc_auc = auc(all_fpr, mean_tpr)

        ax.plot(all_fpr, mean_tpr, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Macro-Averaged ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()

    save_path = save_path or OUTPUT_DIR / "roc_comparison.png"
    fig.savefig(save_path, dpi=VISUALIZATION_DPI)
    plt.close(fig)
    logger.info(f"Saved ROC comparison → {save_path}")
    return save_path


def plot_model_comparison_bar(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ("precision", "recall", "f1"),
    save_path: Path = None,
):
    """
    Bar chart comparing multiple models across metrics.
    results: {model_name: {metric_name: value}}
    """
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in model_names]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    save_path = save_path or OUTPUT_DIR / "model_comparison.png"
    fig.savefig(save_path, dpi=VISUALIZATION_DPI)
    plt.close(fig)
    logger.info(f"Saved model comparison → {save_path}")
    return save_path


def plot_error_analysis(
    y_true: List[str],
    y_pred: List[str],
    tokens: List[str],
    model_name: str,
    top_n: int = 20,
    save_path: Path = None,
):
    """Identify and plot the most common false-positive and false-negative patterns."""
    fp_patterns = {}
    fn_patterns = {}

    for true, pred, tok in zip(y_true, y_pred, tokens):
        if true == "O" and pred != "O":
            key = f"{tok[:20]} → {pred}"
            fp_patterns[key] = fp_patterns.get(key, 0) + 1
        elif true != "O" and pred == "O":
            key = f"{tok[:20]} ({true}) → O"
            fn_patterns[key] = fn_patterns.get(key, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # False positives
    fp_sorted = sorted(fp_patterns.items(), key=lambda x: -x[1])[:top_n]
    if fp_sorted:
        labels, counts = zip(*fp_sorted)
        axes[0].barh(range(len(labels)), counts, color="salmon")
        axes[0].set_yticks(range(len(labels)))
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].set_title(f"Top False Positives — {model_name}")
        axes[0].invert_yaxis()

    # False negatives
    fn_sorted = sorted(fn_patterns.items(), key=lambda x: -x[1])[:top_n]
    if fn_sorted:
        labels, counts = zip(*fn_sorted)
        axes[1].barh(range(len(labels)), counts, color="steelblue")
        axes[1].set_yticks(range(len(labels)))
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].set_title(f"Top False Negatives — {model_name}")
        axes[1].invert_yaxis()

    plt.tight_layout()
    save_path = save_path or OUTPUT_DIR / f"errors_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=VISUALIZATION_DPI)
    plt.close(fig)
    logger.info(f"Saved error analysis → {save_path}")
    return save_path
