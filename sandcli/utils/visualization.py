# Plotting functions
"""
Visualization utilities for SAND Task 1.
Generates confusion matrices, training curves, and feature importance plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        output_path: Path,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        figsize: tuple = (10, 8),
        cmap: str = "Blues"
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        y_true: Ground truth labels (numeric)
        y_pred: Predicted labels (numeric)
        class_names: List of class names in order
        output_path: Path to save figure
        normalize: If True, normalize by true label counts
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
    """
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'

    # Create figure
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    plt.close()


def plot_per_class_metrics(
        metrics_dict: Dict[str, float],
        class_names: List[str],
        output_path: Path,
        metric_name: str = "F1-Score",
        figsize: tuple = (12, 6)
) -> None:
    """
    Plot per-class metrics as bar chart.

    Args:
        metrics_dict: Dict mapping class names to metric values
        class_names: List of class names in order
        output_path: Path to save figure
        metric_name: Name of metric being plotted
        figsize: Figure size
    """
    # Extract values in order
    values = [metrics_dict.get(cls, 0.0) for cls in class_names]

    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(class_names)), values, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.01,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Formatting
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class {metric_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics saved to: {output_path}")
    plt.close()


def plot_feature_importance(
        feature_names: List[str],
        importances: np.ndarray,
        output_path: Path,
        top_n: int = 30,
        figsize: tuple = (10, 12)
) -> None:
    """
    Plot feature importance for tree-based models.

    Args:
        feature_names: List of feature names
        importances: Array of importance scores
        output_path: Path to save figure
        top_n: Number of top features to display
        figsize: Figure size
    """
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(top_features))

    ax.barh(y_pos, top_importances, color='coral', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance plot saved to: {output_path}")
    plt.close()


def plot_training_curves(
        history: Dict[str, List[float]],
        output_path: Path,
        figsize: tuple = (14, 5)
) -> None:
    """
    Plot training and validation curves (for deep learning models).

    Args:
        history: Dict with keys like 'train_loss', 'val_loss', 'train_f1', 'val_f1'
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot F1
    if 'train_f1' in history:
        axes[1].plot(history['train_f1'], label='Train F1', linewidth=2)
    if 'val_f1' in history:
        axes[1].plot(history['val_f1'], label='Val F1', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Macro F1', fontsize=11)
    axes[1].set_title('Training & Validation F1', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {output_path}")
    plt.close()


def generate_results_summary(
        metrics: Dict[str, Any],
        output_path: Path,
        model_name: str = "Model",
        additional_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generate a markdown summary of results.

    Args:
        metrics: Dictionary of computed metrics
        output_path: Path to save markdown file
        model_name: Name of the model
        additional_info: Additional information to include
    """
    lines = [
        f"# {model_name} - Results Summary\n",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "---\n",
        "## Overall Performance\n",
    ]

    # Overall metrics
    if 'macro_f1' in metrics:
        lines.append(f"- **Macro F1**: {metrics['macro_f1']:.4f}")
    if 'accuracy' in metrics:
        lines.append(f"- **Accuracy**: {metrics['accuracy']:.4f}")
    if 'weighted_f1' in metrics:
        lines.append(f"- **Weighted F1**: {metrics['weighted_f1']:.4f}")

    lines.append("\n---\n## Per-Class Performance\n")

    # Per-class metrics
    if 'per_class_f1' in metrics:
        lines.append("| Class | F1-Score | Precision | Recall |")
        lines.append("|-------|----------|-----------|--------|")

        per_class_f1 = metrics['per_class_f1']
        per_class_precision = metrics.get('per_class_precision', {})
        per_class_recall = metrics.get('per_class_recall', {})

        for cls in sorted(per_class_f1.keys()):
            f1 = per_class_f1[cls]
            prec = per_class_precision.get(cls, 0.0)
            rec = per_class_recall.get(cls, 0.0)
            lines.append(f"| {cls} | {f1:.4f} | {prec:.4f} | {rec:.4f} |")

    # Additional info
    if additional_info:
        lines.append("\n---\n## Model Configuration\n")
        for key, value in additional_info.items():
            lines.append(f"- **{key}**: {value}")

    lines.append("\n---\n")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Results summary saved to: {output_path}")


def save_metrics_json(
        metrics: Dict[str, Any],
        output_path: Path
) -> None:
    """
    Save metrics dictionary as JSON.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    metrics_clean = convert_types(metrics)

    with open(output_path, 'w') as f:
        json.dump(metrics_clean, f, indent=2)

    print(f"✓ Metrics saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Visualization utilities for SAND Task 1")
    print("Import this module to use plotting functions.")