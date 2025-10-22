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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import pickle

from models.ml_models import create_ml_model, cross_validate_model
from sandcli.utils.metrics import compute_metrics, plot_confusion_matrix

log = logging.getLogger(__name__)


def apply_smote(X, y, cfg):
    """Apply SMOTE oversampling for minority classes"""
    if not cfg.training.get('use_smote', False):
        return X, y

    try:
        from imblearn.over_sampling import SMOTE
        log.info("\n⚖️  Applying SMOTE for class balancing...")

        smote = SMOTE(
            k_neighbors=cfg.training.get('smote_k_neighbors', 5),
            random_state=cfg.seed
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)

        log.info(f"   Before SMOTE: {X.shape[0]} samples")
        log.info(f"   After SMOTE:  {X_resampled.shape[0]} samples")

        return X_resampled, y_resampled
    except ImportError:
        log.warning("⚠️  imbalanced-learn not installed. Skipping SMOTE.")
        log.warning("   Install with: pip install imbalanced-learn")
        return X, y


def tune_hyperparameters(X_train, y_train, cfg, class_names):
    """Perform GridSearch for hyperparameter tuning"""
    log.info("\n" + "=" * 80)
    log.info("HYPERPARAMETER TUNING (GridSearch)")
    log.info("=" * 80)

    import xgboost as xgb

    # Get tuning grid from config
    param_grid = {
        'max_depth': cfg.training.tuning_grid.get('max_depth', [8, 10, 12]),
        'learning_rate': cfg.training.tuning_grid.get('learning_rate', [0.01, 0.03, 0.05]),
        'n_estimators': cfg.training.tuning_grid.get('n_estimators', [500, 1000]),
        'min_child_weight': cfg.training.tuning_grid.get('min_child_weight', [1, 3]),
        'gamma': cfg.training.tuning_grid.get('gamma', [0.1, 0.2]),
    }

    total_combinations = np.prod([len(v) for v in param_grid.values()])
    log.info(f"Testing {total_combinations} parameter combinations")
    log.info(f"⏱️  Estimated time: {total_combinations * 0.5:.0f}-{total_combinations:.0f} minutes")

    # Base model
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(class_names),
        random_state=cfg.seed,
        tree_method='hist'
    )

    # Sample weights for imbalance
    sample_weights = compute_sample_weight('balanced', y_train)

    # GridSearch with CV
    cv = StratifiedKFold(n_splits=cfg.training.cv_folds, shuffle=True, random_state=cfg.seed)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    log.info("\n🚀 Starting GridSearch...")
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    log.info("\n" + "=" * 80)
    log.info("✅ GRIDSEARCH COMPLETE!")
    log.info("=" * 80)
    log.info(f"\n🏆 Best CV Macro F1: {grid_search.best_score_:.4f}")
    log.info(f"\n🎯 Best Parameters:")
    for param, value in grid_search.best_params_.items():
        log.info(f"   {param}: {value}")

    return grid_search.best_params_


def create_ensemble_model(cfg, class_names):
    """Create ensemble of multiple models"""
    log.info("\n🎭 Creating Ensemble Model...")

    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import VotingClassifier
    from sklearn.svm import SVC

    # XGBoost

    # ========================================================================
    # OPTIONAL: SMOTE for class imbalance
    # ========================================================================
    if cfg.training.get('use_smote', False):
        X_train, y_train_encoded = apply_smote(X_train, y_train_encoded, cfg)

    # ========================================================================
    # OPTIONAL: Hyperparameter Tuning
    # ========================================================================
    if cfg.training.get('tune_hyperparameters', False):
        best_params = tune_hyperparameters(X_train, y_train_encoded, cfg, class_names)
        # Update config with best params
        for param, value in best_params.items():
            cfg.model.params[param] = value

        # Save best params
        with open(exp_dir / "best_hyperparameters.json", 'w') as f:
            json.dump(best_params, f, indent=2)
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(class_names),
    log.info(f"\nCreating {model_name} model...")
    )

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(class_names),
        max_depth=cfg.model.params.max_depth,
        learning_rate=cfg.model.params.learning_rate,
        n_estimators=cfg.model.params.n_estimators,
    # ========================================================================
    # TRAINING: Regular or Ensemble
    # ========================================================================

        random_state=cfg.seed

    # SVM
    svm_model = SVC(
        kernel='rbf',
        C=10.0,
        probability=True,
        class_weight='balanced',
    # Check if Ensemble mode is enabled
    if cfg.model.get('use_ensemble', False):
        log.info("\n🎭 ENSEMBLE MODE ACTIVATED")
        model = create_ensemble_model(cfg, class_names)

        # Train ensemble
        log.info("\nTraining ensemble model...")
        if sample_weights is not None:
            model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    if cfg.model.get('use_ensemble', False):
        # Save ensemble directly with pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        log.info(f"Ensemble model saved to: {model_path}")
    else:
        model.save(model_path)
        log.info(f"Model saved to: {model_path}")
            model.fit(X_train, y_train_encoded)
    else:
        # Regular single model training
        log.info("\nTraining final model on full training set...")
        model = create_ml_model(model_name, cfg.model.params, class_names)

        # Pass sample weights if available
        if sample_weights is not None and hasattr(model, 'set_sample_weights'):
            model.set_sample_weights(sample_weights)

        model.fit(X_train, y_train_encoded, X_test, y_test_encoded)
        ],
        voting='soft',
        weights=[2, 2, 1]
    )

    log.info("   ✓ XGBoost")
    log.info("   ✓ LightGBM")
    log.info("   ✓ SVM (RBF)")
    log.info("   Voting: Soft (weighted probabilities)")

    return ensemble


def load_features_from_manifest(manifest_path: Path) -> tuple:
    """Load features and labels from manifest"""
    df = pd.read_csv(manifest_path)

    # Load all features
    features = []
    labels = []

    log.info(f"Loading features from {len(df)} samples...")

    # Feature importance (only for single models)
    if not cfg.model.get('use_ensemble', False):
        if hasattr(model, 'get_feature_importance'):
            log.info("\nComputing feature importance...")
            importance_df = model.get_feature_importance()
            importance_path = exp_dir / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            log.info(f"✅ Feature importance saved to {importance_path}")
    else:
        log.info("\n⚠️  Feature importance not available for ensemble models")
        labels.append(label)

    X = np.array(features)
    y = np.array(labels)

    log.info(f"Loaded features: {X.shape}")
    log.info(f"Label distribution (before encoding):\n{pd.Series(y).value_counts().to_dict()}")

    return X, y, df


def run(cfg: DictConfig) -> None:
    """
    Main training function for ML models

    Steps:
    1. Load features from manifest
    2. Encode labels (text to integers)
    3. Create model (XGBoost or LightGBM)
    4. Optionally perform cross-validation
    5. Train final model on full training set
    6. Evaluate on test set
    7. Save model and results
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

    # Encode labels (convert text labels to integers)
    log.info("\nEncoding labels...")
    label_encoder = LabelEncoder()

    # Fit on all unique labels from both train and test
    all_labels = np.concatenate([y_train, y_test])
    label_encoder.fit(all_labels)

    # Transform labels
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    log.info(f"Label classes: {label_encoder.classes_}")
    log.info(f"Encoded train labels: {np.unique(y_train_encoded)}")
    log.info(f"Label distribution (train): {pd.Series(y_train_encoded).value_counts().to_dict()}")

    # Save label encoder
    label_encoder_path = exp_dir / "label_encoder.pkl"
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    log.info(f"Label encoder saved to: {label_encoder_path}")

    # Get class names
    class_names = list(label_encoder.classes_)

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
            y_train_encoded,
            cfg.model.params,
            class_names,
            n_folds=cfg.training.cv_folds,
            random_state=cfg.seed
        )

        # Save CV results
        with open(exp_dir / "cv_results.json", 'w') as f:
            json.dump(cv_results, f, indent=2)

    # Compute sample weights for imbalanced classes
    from sklearn.utils.class_weight import compute_sample_weight
    if cfg.training.get('use_class_weights', False):
        sample_weights = compute_sample_weight('balanced', y_train_encoded)
        log.info(f"\n⚖️  Using class weights for imbalanced data")
        log.info(f"   Weight range: {sample_weights.min():.2f} - {sample_weights.max():.2f}")
    else:
        sample_weights = None

    # Train final model on full training set
    log.info("\nTraining final model on full training set...")
    model = create_ml_model(model_name, cfg.model.params, class_names)

    # Pass sample weights if available
    if sample_weights is not None and hasattr(model, 'set_sample_weights'):
        model.set_sample_weights(sample_weights)

    model.fit(X_train, y_train_encoded, X_test, y_test_encoded)

    # Save model
    model_path = exp_dir / "model.pkl"
    model.save(model_path)

    # Evaluate on test set
    log.info("\nEvaluating on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Compute metrics (using encoded labels)
    metrics = compute_metrics(y_test_encoded, y_pred, y_pred_proba, class_names)

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
    predictions_df['predicted_label'] = [label_encoder.inverse_transform([p])[0] for p in y_pred]
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

