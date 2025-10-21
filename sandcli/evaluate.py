"""
Evaluation script for SAND Task 1.
Computes Macro F1, per-class metrics, and generates visualizations.
"""

import argparse
import pickle
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import sys

# Import from our utils
from sandcli.utils.metrics import compute_metrics
from sandcli.utils.visualization import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    generate_results_summary,
    save_metrics_json
)
from omegaconf import DictConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def load_model(model_path: Path) -> Any:
    """
    Load trained model from disk.

    Args:
        model_path: Path to model file (.pkl for ML, .pt for PyTorch)

    Returns:
        Loaded model object
    """
    log.info(f"Loading model from: {model_path}")

    if model_path.suffix == '.pkl':
        # ML model (XGBoost/LightGBM/SVM)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        log.info("✓ ML model loaded successfully")
        return model

    elif model_path.suffix == '.pt' or model_path.suffix == '.pth':
        # PyTorch model (for Phase 2 - SSL)
        try:
            import torch
            model = torch.load(model_path, map_location='cpu')
            model.eval()
            log.info("✓ PyTorch model loaded successfully")
            return model
        except ImportError:
            log.error("PyTorch not installed. Cannot load .pt model.")
            sys.exit(1)

    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


def load_features_and_labels(
        manifest_path: Path,
        feature_dir: Path,
        label_encoder_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Load features and labels for evaluation.

    Args:
        manifest_path: Path to test manifest CSV
        feature_dir: Directory containing feature arrays (.npy files)
        label_encoder_path: Path to label encoder pickle

    Returns:
        X: Feature matrix (n_samples, n_features)
        y_true: True labels (encoded as integers)
        subject_ids: Array of subject IDs
        file_ids: List of file identifiers
    """
    # Load manifest
    log.info(f"Loading manifest from: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    log.info(f"✓ Loaded manifest with {len(manifest)} samples")

    # Load label encoder
    log.info(f"Loading label encoder from: {label_encoder_path}")
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    log.info(f"✓ Label encoder loaded. Classes: {list(label_encoder.classes_)}")

    # Load features
    X_list = []
    y_list = []
    subject_ids_list = []
    file_ids = []
    skipped = 0

    log.info("Loading features...")
    for idx, row in manifest.iterrows():
        file_path = Path(row['file_path'])
        file_id = file_path.stem

        # Construct feature file path
        feature_path = feature_dir / f"{file_id}.npy"

        if not feature_path.exists():
            log.warning(f"Feature not found: {feature_path}, skipping...")
            skipped += 1
            continue

        # Load feature array
        features = np.load(feature_path)

        # Handle 2D features (e.g., MFCC with time dimension)
        # Shape could be (n_mfcc, n_frames) -> flatten with statistics
        if len(features.shape) > 1:
            # Compute statistics over time dimension (axis=1 for shape (n_mfcc, n_frames))
            if features.shape[0] < features.shape[1]:
                # Likely (n_features, n_frames)
                time_axis = 1
            else:
                # Likely (n_frames, n_features)
                time_axis = 0

            features_flat = np.concatenate([
                np.mean(features, axis=time_axis),
                np.std(features, axis=time_axis),
                np.min(features, axis=time_axis),
                np.max(features, axis=time_axis)
            ])
        else:
            features_flat = features

        X_list.append(features_flat)

        # Encode label
        label = row['label']
        y_list.append(label_encoder.transform([label])[0])

        # Store metadata
        subject_ids_list.append(row.get('subject_id', 'unknown'))
        file_ids.append(file_id)

    if skipped > 0:
        log.warning(f"⚠ Skipped {skipped} samples due to missing features")

    X = np.array(X_list)
    y_true = np.array(y_list)
    subject_ids = np.array(subject_ids_list)

    log.info(f"✓ Loaded features: {X.shape}")
    log.info(f"✓ Loaded labels: {y_true.shape}")
    log.info(f"✓ Feature dimension: {X.shape[1]}")

    return X, y_true, subject_ids, file_ids


def get_predictions(model: Any, X: np.ndarray, model_type: str = 'ml') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Get predictions from model.

    Args:
        model: Trained model
        X: Feature matrix
        model_type: 'ml' for sklearn-like models, 'pytorch' for PyTorch models

    Returns:
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (if available)
    """
    log.info("Generating predictions...")

    if model_type == 'ml':
        # Standard sklearn-like interface
        y_pred = model.predict(X)

        # Try to get probabilities
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)

        return y_pred, y_pred_proba

    elif model_type == 'pytorch':
        import torch

        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(X_tensor)

            # Get probabilities
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            y_pred_proba = torch.softmax(logits, dim=1).numpy()
            y_pred = np.argmax(y_pred_proba, axis=1)

        return y_pred, y_pred_proba

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_model(
        model: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        class_names: list,
        output_dir: Path,
        model_name: str = "Model",
        model_type: str = 'ml'
) -> Dict[str, Any]:
    """
    Evaluate model and generate all metrics and visualizations.

    Args:
        model: Trained model
        X: Feature matrix
        y_true: True labels (encoded integers)
        class_names: List of class names in order
        output_dir: Directory to save outputs
        model_name: Name of model for reporting
        model_type: 'ml' or 'pytorch'

    Returns:
        Dictionary of all metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"\n{'=' * 60}")
    log.info(f"EVALUATING: {model_name}")
    log.info(f"{'=' * 60}\n")

    # Get predictions
    y_pred, y_pred_proba = get_predictions(model, X, model_type)

    # Compute all metrics using our metrics.py function
    log.info("Computing metrics...")
    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        class_names=class_names
    )

    # Print primary metric
    log.info(f"\n🎯 PRIMARY METRIC")
    log.info(f"   Macro F1: {metrics['macro_f1']:.4f}\n")

    # Print additional metrics
    log.info(f"📊 ADDITIONAL METRICS")
    log.info(f"   Accuracy:    {metrics['accuracy']:.4f}")
    log.info(f"   Micro F1:    {metrics['micro_f1']:.4f}")
    log.info(f"   Weighted F1: {metrics['weighted_f1']:.4f}")

    if 'auc_ovr' in metrics:
        log.info(f"   AUC (OvR):   {metrics['auc_ovr']:.4f}")

    # Print per-class F1
    log.info(f"\n📋 PER-CLASS F1 SCORES")
    for cls, f1 in metrics['per_class_f1'].items():
        log.info(f"   {cls:20s}: {f1:.4f}")

    # Generate visualizations
    log.info("\n🎨 Generating visualizations...")

    # 1. Confusion matrix (raw counts)
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=cm_path,
        normalize=False,
        title=f"{model_name} - Confusion Matrix"
    )

    # 2. Normalized confusion matrix
    cm_norm_path = output_dir / "confusion_matrix_normalized.png"
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=cm_norm_path,
        normalize=True,
        title=f"{model_name} - Normalized Confusion Matrix"
    )

    # 3. Per-class F1 bar chart
    f1_plot_path = output_dir / "per_class_f1.png"
    plot_per_class_metrics(
        metrics_dict=metrics['per_class_f1'],
        class_names=class_names,
        output_path=f1_plot_path,
        metric_name="F1-Score"
    )

    # 4. Per-class Precision bar chart
    precision_plot_path = output_dir / "per_class_precision.png"
    plot_per_class_metrics(
        metrics_dict=metrics['per_class_precision'],
        class_names=class_names,
        output_path=precision_plot_path,
        metric_name="Precision"
    )

    # 5. Per-class Recall bar chart
    recall_plot_path = output_dir / "per_class_recall.png"
    plot_per_class_metrics(
        metrics_dict=metrics['per_class_recall'],
        class_names=class_names,
        output_path=recall_plot_path,
        metric_name="Recall"
    )

    # Save metrics as JSON
    metrics_path = output_dir / "metrics.json"
    save_metrics_json(metrics, metrics_path)

    # Generate markdown summary
    summary_path = output_dir / "results_summary.md"
    generate_results_summary(
        metrics=metrics,
        output_path=summary_path,
        model_name=model_name,
        additional_info={
            'num_samples': len(y_true),
            'num_classes': len(class_names),
            'feature_dim': X.shape[1] if len(X.shape) > 1 else 'N/A',
            'class_distribution': {
                class_names[i]: int(np.sum(y_true == i))
                for i in range(len(class_names))
            }
        }
    )

    log.info(f"\n{'=' * 60}")
    log.info(f"✅ EVALUATION COMPLETE")
    log.info(f"📁 Results saved to: {output_dir}")
    log.info(f"{'=' * 60}\n")

    return metrics


def main():
    """Standalone CLI entry point (for direct execution)"""
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test set for SAND Task 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model file (.pkl for ML, .pt for PyTorch)'
    )

    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to test manifest CSV'
    )

    parser.add_argument(
        '--feature_dir',
        type=str,
        required=True,
        help='Directory containing extracted features (.npy files)'
    )

    parser.add_argument(
        '--label_encoder',
        type=str,
        required=True,
        help='Path to label encoder pickle file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save evaluation results'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='Model',
        help='Name of model for reporting'
    )

    parser.add_argument(
        '--model_type',
        type=str,
        choices=['ml', 'pytorch'],
        default='ml',
        help='Type of model: "ml" for sklearn-like, "pytorch" for PyTorch models'
    )

    args = parser.parse_args()

    # Convert to Path objects
    model_path = Path(args.model_path)
    manifest_path = Path(args.manifest)
    feature_dir = Path(args.feature_dir)
    label_encoder_path = Path(args.label_encoder)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not model_path.exists():
        log.error(f"Model not found at {model_path}")
        sys.exit(1)

    if not manifest_path.exists():
        log.error(f"Manifest not found at {manifest_path}")
        sys.exit(1)

    if not feature_dir.exists():
        log.error(f"Feature directory not found at {feature_dir}")
        sys.exit(1)

    if not label_encoder_path.exists():
        log.error(f"Label encoder not found at {label_encoder_path}")
        sys.exit(1)

    # Load model
    model = load_model(model_path)

    # Load label encoder for class names
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)
    log.info(f"Classes: {class_names}")

    # Load features and labels
    X, y_true, subject_ids, file_ids = load_features_and_labels(
        manifest_path, feature_dir, label_encoder_path
    )

    # Evaluate
    metrics = evaluate_model(
        model=model,
        X=X,
        y_true=y_true,
        class_names=class_names,
        output_dir=output_dir,
        model_name=args.model_name,
        model_type=args.model_type
    )

    # Final summary
    print(f"\n{'🎯' * 20}")
    print(f"PRIMARY METRIC (Macro F1): {metrics['macro_f1']:.4f}")
    print(f"{'🎯' * 20}\n")


def run(cfg: DictConfig) -> None:
    """
    Hydra-compatible evaluation function

    This is called by main.py when using: python -m sandcli.main command=evaluate
    """
    from omegaconf import DictConfig

    log.info("=" * 80)
    log.info("STEP 5: MODEL EVALUATION")
    log.info("=" * 80)

    # Parse required arguments from config
    model_path = Path(cfg.get('model_path'))
    manifest_path = Path(cfg.get('manifest'))
    feature_dir = Path(cfg.get('feature_dir'))
    label_encoder_path = Path(cfg.get('label_encoder'))
    output_dir = Path(cfg.get('output_dir'))
    model_name = cfg.get('model_name', 'Model')
    model_type = cfg.get('model_type', 'ml')

    # Validate inputs
    if not model_path.exists():
        log.error(f"Model not found at {model_path}")
        sys.exit(1)

    if not manifest_path.exists():
        log.error(f"Manifest not found at {manifest_path}")
        sys.exit(1)

    if not feature_dir.exists():
        log.error(f"Feature directory not found at {feature_dir}")
        sys.exit(1)

    if not label_encoder_path.exists():
        log.error(f"Label encoder not found at {label_encoder_path}")
        sys.exit(1)

    # Load model
    model = load_model(model_path)

    # Load label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    # Load features and labels
    X, y_true, subject_ids, file_ids = load_features_and_labels(
        manifest_path, feature_dir, label_encoder_path
    )

    # Evaluate
    metrics = evaluate_model(
        model=model,
        X=X,
        y_true=y_true,
        class_names=class_names,
        output_dir=output_dir,
        model_name=model_name,
        model_type=model_type
    )

    log.info(f"\n✅ Evaluation complete!")
    log.info(f"\n🎯 PRIMARY METRIC (Macro F1): {metrics['macro_f1']:.4f}\n")

if __name__ == "__main__":
    main()