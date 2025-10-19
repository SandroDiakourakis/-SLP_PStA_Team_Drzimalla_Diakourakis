# Custom metrics
"""
Evaluation metrics and visualization utilities
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from typing import Dict, List
from pathlib import Path

log = logging.getLogger(__name__)


def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        class_names: List[str] = None
) -> Dict:
    """
    Compute comprehensive evaluation metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC)
        class_names: List of class names

    Returns:
        Dictionary of metrics
    """
    # Core metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Build results dictionary
    metrics = {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
    }

    # Add per-class metrics
    if class_names is None:
        class_names = [f"class_{i}" for i in range(len(per_class_f1))]

    metrics['per_class_f1'] = {
        class_name: float(f1)
        for class_name, f1 in zip(class_names, per_class_f1)
    }

    metrics['per_class_precision'] = {
        class_name: float(prec)
        for class_name, prec in zip(class_names, per_class_precision)
    }

    metrics['per_class_recall'] = {
        class_name: float(rec)
        for class_name, rec in zip(class_names, per_class_recall)
    }

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics['classification_report'] = report

    # AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            # Multi-class AUC (one-vs-rest)
            auc_ovr = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')
            metrics['auc_ovr'] = float(auc_ovr)
        except Exception as e:
            log.warning(f"Could not compute AUC: {e}")

    return metrics


def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str],
        save_path: Path = None,
        normalize: bool = True,
        figsize: tuple = (10, 8)
):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        normalize: Normalize by true labels
        figsize: Figure size
    """
    if isinstance(cm, list):
        cm = np.array(cm)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    else:
        cm_normalized = cm

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(
        importance_df,
        save_path: Path = None,
        top_n: int = 20,
        figsize: tuple = (10, 8)
):
    """
    Plot feature importance

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        save_path: Path to save figure
        top_n: Number of top features to plot
        figsize: Figure size
    """
    # Get top N features
    top_features = importance_df.head(top_n)

    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()

    plt.close()