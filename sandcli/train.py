# Training orchestrator
"""
Training script for ML models
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import DictConfig
import json
from datetime import datetime

from models.ml_models import create_ml_model, cross_validate_model
from sandcli.utils.metrics import compute_metrics, plot_confusion_matrix

log = logging.getLogger(__name__)


def load_features_from_manifest(manifest_path: Path) -> tuple:
    """Load features and labels from manifest"""
    df = pd.read_csv(manifest_path)

    # Load all features
    features = []
    labels = []

    log.info(f"Loading features from {len(df)} samples...")

    for idx, row in df.iterrows():
        feature_path = row['feature_path']
        label = row['label']

        # Load feature vector
        feature = np.load(feature_path)
        features.append(feature)
        labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    log.info(f"Loaded features: {X.shape}")
    log.info(f"Label distribution: {pd.Series(y).value_counts().to_dict()}")

    return X, y, df


def run(cfg: DictConfig) -> None:
    """
    Main training function for ML models

    Steps:
    1. Load features from manifest
    2. Create model (XGBoost or LightGBM)
    3. Optionally perform cross-validation
    4. Train final model on full training set
    5. Evaluate on test set
    6. Save model and results
    """
    log.info("=" * 80)
    log.info("STEP 4: MODEL TRAINING (ML)")
    log.info("=" * 80)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{cfg.experiment_name}_{timestamp}"
    exp_dir = Path(cfg.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Experiment directory: {exp_dir}")

    # Save config
    with open(exp_dir / "config.yaml", 'w') as f:
        f.write(str(cfg))

    # Determine feature directory
    feature_type = cfg.features.type.replace('+', '_')
    feature_dir = Path("features") / feature_type

    # Load training data
    train_manifest = feature_dir / "train" / "manifest.csv"
    test_manifest = feature_dir / "test" / "manifest.csv"

    log.info("Loading training data...")
    X_train, y_train, train_df = load_features_from_manifest(train_manifest)

    log.info("Loading test data...")
    X_test, y_test, test_df = load_features_from_manifest(test_manifest)

    # Get class names
    class_names = cfg.labels.class_names

    # Create model
    model_name = cfg.model.name
    log.info(f"Creating {model_name} model...")

    # Cross-validation (optional)
    if cfg.training.get('cv_folds', 0) > 0:
        log.info(f"\nPerforming {cfg.training.cv_folds}-fold cross-validation...")

        if model_name.lower() == 'xgboost':
            from models.ml_models import XGBoostModel
            model_class = XGBoostModel
        elif model_name.lower() == 'lightgbm':
            from models.ml_models import LightGBMModel
            model_class = LightGBMModel
        else:
            raise ValueError(f"Unknown model: {model_name}")

        cv_results = cross_validate_model(
            model_class,
            X_train,
            y_train,
            cfg.model.params,
            class_names,
            n_folds=cfg.training.cv_folds,
            random_state=cfg.seed
        )

        # Save CV results
        with open(exp_dir / "cv_results.json", 'w') as f:
            json.dump(cv_results, f, indent=2)

    # Train final model on full training set
    log.info("\nTraining final model on full training set...")
    model = create_ml_model(model_name, cfg.model.params, class_names)
    model.fit(X_train, y_train, X_test, y_test)

    # Save model
    model_path = exp_dir / "model.pkl"
    model.save(model_path)

    # Evaluate on test set
    log.info("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_pred_proba, class_names)

    # Log metrics
    log.info("\n" + "=" * 80)
    log.info("TEST SET RESULTS")
    log.info("=" * 80)
    log.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    log.info(f"Accuracy: {metrics['accuracy']:.4f}")
    log.info(f"\nPer-class F1 scores:")
    for class_name, f1 in metrics['per_class_f1'].items():
        log.info(f"  {class_name}: {f1:.4f}")

    # Save metrics
    with open(exp_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot and save confusion matrix
    cm_path = exp_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
    log.info(f"\n✅ Confusion matrix saved to {cm_path}")

    # Save predictions
    predictions_df = test_df.copy()
    predictions_df['predicted_label'] = y_pred
    predictions_df['true_label'] = y_test

    # Add probabilities
    for i, class_name in enumerate(class_names):
        predictions_df[f'prob_{class_name}'] = y_pred_proba[:, i]

    predictions_path = exp_dir / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    log.info(f"✅ Predictions saved to {predictions_path}")

    # Feature importance
    if hasattr(model, 'get_feature_importance'):
        log.info("\nComputing feature importance...")
        importance_df = model.get_feature_importance()
        importance_path = exp_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        log.info(f"✅ Feature importance saved to {importance_path}")

    log.info("\n✅ Training complete!")
    log.info(f"All results saved to: {exp_dir}")


if __name__ == "__main__":
    from sandcli.main import cli

    cli()