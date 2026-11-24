"""Evaluation metrics, subgroup error analysis, and diagnostic plots for NSL-KDD models.

Attack recall (sensitivity on class 1) is treated as the primary security metric
alongside false-alarm rate (FPR) and macro F1-score.
"""

from pathlib import Path
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_proba: Optional[Iterable[float]] = None,
) -> dict[str, float]:
    """Compute binary intrusion-detection classification metrics.

    Args:
        y_true: Ground-truth binary labels (0 = normal, 1 = attack).
        y_pred: Predicted binary labels (0 = normal, 1 = attack).
        y_proba: Optional predicted probabilities for the attack class (class 1).

    Returns:
        Dictionary containing accuracy, attack precision, attack recall, attack F1,
        macro F1, false-alarm rate (FPR), optional ROC-AUC, and confusion counts.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if y_true_arr.size == 0 or y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must be non-empty 1D sequences with matching lengths."
        )

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    false_alarm_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "false_alarm_rate": false_alarm_rate,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba, dtype=float)
        if len(np.unique(y_true_arr)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba_arr))
        else:
            metrics["roc_auc"] = float("nan")

    return metrics


def evaluate_by_attack_family(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    attack_families: Iterable[str],
) -> pd.DataFrame:
    """Break down prediction performance by NSL-KDD intrusion family.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        attack_families: Intrusion family labels (`Normal`, `DoS`, `Probe`, `R2L`, `U2R`).

    Returns:
        DataFrame with record counts, correctly classified counts, and accuracy/recall per family.
    """
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "y_pred": np.asarray(y_pred, dtype=int),
            "attack_family": pd.Series(list(attack_families), dtype="object"),
        }
    )

    preferred_order = ["Normal", "DoS", "Probe", "R2L", "U2R", "Other"]
    present = [fam for fam in preferred_order if fam in set(df["attack_family"])]

    rows = []
    for family in present:
        sub = df[df["attack_family"] == family]
        total = int(len(sub))
        correct = int((sub["y_true"] == sub["y_pred"]).sum())
        flagged_attack = int((sub["y_pred"] == 1).sum())
        rate = float(correct / total) if total > 0 else 0.0
        rows.append(
            {
                "attack_family": family,
                "records": total,
                "correct": correct,
                "flagged_as_attack": flagged_attack,
                "detection_or_specificity_rate": round(rate, 4),
            }
        )

    return pd.DataFrame(rows)


def evaluate_seen_vs_unseen(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    attack_types: Iterable[str],
    train_attack_types: set[str],
) -> pd.DataFrame:
    """Compare test performance across Normal, Seen Attacks, and Unseen (Zero-Day) Attacks.

    Args:
        y_true: Ground-truth binary test labels.
        y_pred: Predicted binary test labels.
        attack_types: Specific NSL-KDD attack type strings for each test record.
        train_attack_types: Set of attack type strings observed in KDDTrain+.

    Returns:
        Summary DataFrame quantifying the generalization gap on unseen attack types.
    """
    df = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "y_pred": np.asarray(y_pred, dtype=int),
            "attack_type": pd.Series(list(attack_types), dtype="object").str.lower(),
        }
    )

    def _bucket(row: pd.Series) -> str:
        if row["y_true"] == 0:
            return "Normal"
        if row["attack_type"] in train_attack_types:
            return "Seen Attacks (in KDDTrain+)"
        return "Unseen Attacks (KDDTest+ only)"

    df["group"] = df.apply(_bucket, axis=1)

    order = [
        "Normal",
        "Seen Attacks (in KDDTrain+)",
        "Unseen Attacks (KDDTest+ only)",
    ]
    rows = []
    for group in order:
        sub = df[df["group"] == group]
        if sub.empty:
            continue
        total = int(len(sub))
        correct = int((sub["y_true"] == sub["y_pred"]).sum())
        rows.append(
            {
                "group": group,
                "records": total,
                "correct": correct,
                "missed": total - correct,
                "rate": round(float(correct / total), 4),
            }
        )

    return pd.DataFrame(rows)


def evaluate_thresholds(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    thresholds: Iterable[float] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70),
) -> pd.DataFrame:
    """Evaluate attack recall, false-alarm rate, precision, and macro F1 across thresholds."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_proba_arr = np.asarray(y_proba, dtype=float)

    rows = []
    for thr in thresholds:
        y_pred_thr = (y_proba_arr >= thr).astype(int)
        m = compute_classification_metrics(y_true_arr, y_pred_thr)
        rows.append(
            {
                "threshold": round(float(thr), 2),
                "accuracy": round(m["accuracy"], 4),
                "precision": round(m["precision"], 4),
                "attack_recall": round(m["recall"], 4),
                "attack_f1": round(m["f1"], 4),
                "macro_f1": round(m["macro_f1"], 4),
                "false_alarm_rate": round(m["false_alarm_rate"], 4),
            }
        )
    return pd.DataFrame(rows)


def plot_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    title: str,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot and optionally save an annotated binary confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100.0

    annot = np.array(
        [
            [f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)" for j in range(2)]
            for i in range(2)
        ]
    )

    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=130)
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Predicted Normal (0)", "Predicted Attack (1)"],
        yticklabels=["Actual Normal (0)", "Actual Attack (1)"],
        linewidths=0.8,
        linecolor="#d0d7de",
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_experiment_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot validation metrics across experiments E1-E4 side by side."""
    plot_cols = ["val_accuracy", "val_precision", "val_recall", "val_f1", "val_macro_f1"]
    rename_map = {
        "val_accuracy": "Accuracy",
        "val_precision": "Attack Precision",
        "val_recall": "Attack Recall",
        "val_f1": "Attack F1",
        "val_macro_f1": "Macro F1",
    }

    melted = results_df[["id"] + plot_cols].melt(
        id_vars="id", var_name="Metric", value_name="Score"
    )
    melted["Metric"] = melted["Metric"].map(rename_map)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=130)
    sns.barplot(data=melted, x="id", y="Score", hue="Metric", palette="viridis", ax=ax)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7.5, padding=2, rotation=90)

    ymin = max(0.85, float(melted["Score"].min()) - 0.03)
    ax.set_ylim(ymin, 1.015)
    ax.set_title(
        "Validation Performance Across Experiments E1-E4 (20% Stratified KDDTrain+ Split)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Experiment ID", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.legend(title="Validation Metric", loc="lower right", frameon=True)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_test_error_breakdown(
    family_df: pd.DataFrame,
    seen_unseen_df: pd.DataFrame,
    model_label: str,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot KDDTest+ detection rate by intrusion family and seen vs. unseen attack groups."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=130)

    # Left panel: Attack families (including Normal specificity)
    fam_colors = ["#2b8cbe" if f == "Normal" else "#e34a33" for f in family_df["attack_family"]]
    bars_left = axes[0].bar(
        family_df["attack_family"],
        family_df["detection_or_specificity_rate"],
        color=fam_colors,
        edgecolor="#222222",
        linewidth=0.6,
    )
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Correct Rate (Specificity / Recall)", fontsize=10)
    axes[0].set_xlabel("Traffic Category", fontsize=10)
    axes[0].set_title(f"{model_label}: KDDTest+ Rate by Family", fontsize=11, fontweight="bold")
    for bar, (_, row) in zip(bars_left, family_df.iterrows()):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.025,
            f"{row['detection_or_specificity_rate']:.1%}\n(n={row['records']:,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    # Right panel: Normal vs Seen Attacks vs Unseen Attacks
    grp_colors = ["#2b8cbe", "#31a354", "#d95f0e"]
    bars_right = axes[1].bar(
        seen_unseen_df["group"],
        seen_unseen_df["rate"],
        color=grp_colors[: len(seen_unseen_df)],
        edgecolor="#222222",
        linewidth=0.6,
    )
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("Correct Rate (Specificity / Recall)", fontsize=10)
    axes[1].set_xlabel("Generalization Bucket", fontsize=10)
    axes[1].set_title(
        f"{model_label}: Seen vs. Unseen Attacks on KDDTest+", fontsize=11, fontweight="bold"
    )
    axes[1].tick_params(axis="x", labelsize=8.5)
    for bar, (_, row) in zip(bars_right, seen_unseen_df.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.025,
            f"{row['rate']:.1%}\n(n={row['records']:,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


def plot_threshold_tradeoff(
    threshold_df: pd.DataFrame,
    model_label: str,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot attack recall, macro F1, precision, and false-alarm rate across thresholds."""
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=130)

    ax.plot(
        threshold_df["threshold"],
        threshold_df["attack_recall"],
        marker="o",
        linewidth=2.2,
        color="#d95f0e",
        label="Attack Recall (Primary Metric)",
    )
    ax.plot(
        threshold_df["threshold"],
        threshold_df["macro_f1"],
        marker="s",
        linewidth=2.0,
        color="#2b8cbe",
        label="Macro F1",
    )
    ax.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        marker="^",
        linewidth=1.8,
        linestyle="--",
        color="#31a354",
        label="Attack Precision",
    )
    ax.plot(
        threshold_df["threshold"],
        threshold_df["false_alarm_rate"],
        marker="D",
        linewidth=2.0,
        color="#756bb1",
        label="False-Alarm Rate (FPR)",
    )
    ax.axhline(0.80, color="#999999", linestyle=":", linewidth=1.2, label="0.80 Recall Target")
    ax.axvline(0.50, color="#444444", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Decision Threshold (Attack Probability Cutoff)", fontsize=10)
    ax.set_ylabel("Score / Rate", fontsize=10)
    ax.set_title(
        f"{model_label}: KDDTest+ Recall vs. False-Alarm Trade-Off by Threshold",
        fontsize=11.5,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="center right", frameon=True, fontsize=8.5)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig
