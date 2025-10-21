# XGBoost, LightGBM wrappers
"""
Classical machine learning models: XGBoost, LightGBM, SVM
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib

import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

log = logging.getLogger(__name__)


class MLModelWrapper:
    """Base wrapper for ML models"""

    def __init__(self, model_name: str, params: Dict, class_names: List[str]):
        self.model_name = model_name
        self.params = params
        self.class_names = class_names
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        raise NotImplementedError

    def save(self, path: Path):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'class_names': self.class_names,
            'model_name': self.model_name
        }, path)
        log.info(f"Model saved to {path}")

    def load(self, path: Path):
        """Load model from disk"""
        checkpoint = joblib.load(path)
        self.model = checkpoint['model']
        self.label_encoder = checkpoint['label_encoder']
        self.scaler = checkpoint['scaler']
        self.class_names = checkpoint['class_names']
        self.model_name = checkpoint['model_name']
        log.info(f"Model loaded from {path}")


class XGBoostModel(MLModelWrapper):
    """XGBoost classifier wrapper"""

    def __init__(self, params: Dict, class_names: List[str]):
        super().__init__("xgboost", params, class_names)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train XGBoost model"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Compute class weights for imbalance
        class_weights = self._compute_class_weights(y_encoded)

        # Update params with num_class
        # Convert to dict if it's a DictConfig (from Hydra)
        if hasattr(self.params, 'items'):
            params = dict(self.params)
        else:
            params = self.params.copy()

        params['num_class'] = len(self.class_names)
        params['scale_pos_weight'] = class_weights  # Only works for binary; use sample_weight for multi-class

        # Create DMatrix
        dtrain = xgb.DMatrix(X_scaled, label=y_encoded)

        # Validation set
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_scaled = self.scaler.transform(X_val)
            dval = xgb.DMatrix(X_val_scaled, label=y_val_encoded)
            evals.append((dval, 'val'))

        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 300),
            evals=evals,
            early_stopping_rounds=params.get('early_stopping_rounds', 30),
            verbose_eval=False
        )

        log.info(f"✅ XGBoost training complete. Best iteration: {self.model.best_iteration}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled)
        y_pred = self.model.predict(dtest)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred.astype(int))
        return y_pred_labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled)

        # XGBoost returns probabilities directly for multi-class
        proba = self.model.predict(dtest)

        # If output is 1D (class indices), convert to one-hot
        if len(proba.shape) == 1:
            n_classes = len(self.class_names)
            proba_matrix = np.zeros((len(proba), n_classes))
            proba_matrix[np.arange(len(proba)), proba.astype(int)] = 1.0
            return proba_matrix

        return proba

    def _compute_class_weights(self, y: np.ndarray) -> float:
        """Compute class weights for imbalanced data"""
        unique, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(unique) * counts)
        return float(np.mean(weights))

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        importance = self.model.get_score(importance_type='gain')
        df = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        })
        df = df.sort_values('importance', ascending=False)
        return df


class LightGBMModel(MLModelWrapper):
    """LightGBM classifier wrapper"""

    def __init__(self, params: Dict, class_names: List[str]):
        super().__init__("lightgbm", params, class_names)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train LightGBM model"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Compute sample weights for class imbalance
        class_weights = self._compute_sample_weights(y_encoded)

        # Create dataset
        train_data = lgb.Dataset(X_scaled, label=y_encoded, weight=class_weights)

        # Validation set
        valid_sets = [train_data]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            X_val_scaled = self.scaler.transform(X_val)
            val_weights = self._compute_sample_weights(y_val_encoded)
            val_data = lgb.Dataset(X_val_scaled, label=y_val_encoded, weight=val_weights, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')

        # Train
        params = self.params.copy()
        params['num_class'] = len(self.class_names)
        params['objective'] = 'multiclass'
        params['metric'] = 'multi_logloss'

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 300),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=params.get('early_stopping_rounds', 30)),
                lgb.log_evaluation(period=0)
            ]
        )

        log.info(f"✅ LightGBM training complete. Best iteration: {self.model.best_iteration}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        X_scaled = self.scaler.transform(X)
        y_pred_proba = self.model.predict(X_scaled)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        return y_pred_labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict(X_scaled)
        return proba

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute sample weights for imbalanced classes"""
        unique, counts = np.unique(y, return_counts=True)
        class_weights = len(y) / (len(unique) * counts)
        weight_dict = dict(zip(unique, class_weights))
        sample_weights = np.array([weight_dict[label] for label in y])
        return sample_weights

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        importance = self.model.feature_importance(importance_type='gain')
        df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(importance))],
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False)
        return df


def create_ml_model(model_name: str, params: Dict, class_names: List[str]) -> MLModelWrapper:
    """Factory function to create ML model"""
    if model_name.lower() == 'xgboost':
        return XGBoostModel(params, class_names)
    elif model_name.lower() == 'lightgbm':
        return LightGBMModel(params, class_names)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def cross_validate_model(
        model_class,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict,
        class_names: List[str],
        n_folds: int = 5,
        random_state: int = 42
) -> Dict:
    """Perform cross-validation"""
    log.info(f"Performing {n_folds}-fold cross-validation...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        log.info(f"Fold {fold + 1}/{n_folds}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create and train model
        model = model_class(params, class_names)
        model.fit(X_train, y_train, X_val, y_val)

        # Evaluate
        y_pred = model.predict(X_val)

        # Compute metrics
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        accuracy = accuracy_score(y_val, y_pred)

        fold_results.append({
            'fold': fold + 1,
            'macro_f1': macro_f1,
            'accuracy': accuracy
        })

        log.info(f"Fold {fold + 1} - Macro F1: {macro_f1:.4f}, Accuracy: {accuracy:.4f}")

    # Aggregate results
    results = {
        'mean_macro_f1': np.mean([r['macro_f1'] for r in fold_results]),
        'std_macro_f1': np.std([r['macro_f1'] for r in fold_results]),
        'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'fold_results': fold_results
    }

    log.info(f"\n{'=' * 60}")
    log.info(f"Cross-Validation Results:")
    log.info(f"Macro F1: {results['mean_macro_f1']:.4f} ± {results['std_macro_f1']:.4f}")
    log.info(f"Accuracy: {results['mean_accuracy']:.4f}")
    log.info(f"{'=' * 60}\n")

    return results