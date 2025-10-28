import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


class ModelTrainer:
    """Trainiert und evaluiert ML-Modelle"""

    AVAILABLE_MODELS = {
        'SVC': SVC,
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier
    }

    def __init__(self, grid_file: str = "grids/model_grids.json"):
        self.grid_file = Path(grid_file)
        self.grids = self._load_grids()

    def _load_grids(self) -> Dict:
        """Lädt Grid-Konfigurationen aus JSON"""
        if not self.grid_file.exists():
            raise FileNotFoundError(f"Grid-Datei nicht gefunden: {self.grid_file}")

        with open(self.grid_file, 'r') as f:
            return json.load(f)

    def train_model(self,
                    model_name: str,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    hyperparameters: Dict[str, Any] = None) -> Tuple[Any, Dict, float]:
        """
        Trainiert ein einzelnes Modell mit gegebenen Hyperparametern

        Returns:
            (model, metrics, training_time)
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modell {model_name} nicht verfügbar")

        # Modell initialisieren
        model_class = self.AVAILABLE_MODELS[model_name]

        if hyperparameters:
            model = model_class(**hyperparameters)
        else:
            model = model_class()

        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Evaluation
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }

        return model, metrics, training_time

    def grid_search(self,
                    model_name: str,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    cv: int = 5,
                    scoring: str = 'f1_weighted') -> Tuple[Any, Dict, Dict, float]:
        """
        Führt Grid Search für ein Modell durch

        Returns:
            (best_model, best_params, metrics, training_time)
        """
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Modell {model_name} nicht verfügbar")

        if model_name not in self.grids:
            raise ValueError(f"Keine Grid-Konfiguration für {model_name} gefunden")

        model_class = self.AVAILABLE_MODELS[model_name]
        param_grid = self.grids[model_name]

        print(f"🔍 Starte Grid Search für {model_name}...")
        print(f"   Parameter-Kombinationen: {self._count_combinations(param_grid)}")

        # Grid Search
        grid_search = GridSearchCV(
            estimator=model_class(),
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Best Model evaluieren
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'cv_best_score': grid_search.best_score_,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }

        print(f"✅ Bestes F1-Score: {metrics['f1_score']:.4f}")
        print(f"   Best Params: {grid_search.best_params_}")

        return best_model, grid_search.best_params_, metrics, training_time

    def train_all_models(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_test: np.ndarray,
                         y_test: np.ndarray,
                         use_grid_search: bool = True) -> Dict:
        """
        Trainiert alle verfügbaren Modelle

        Returns:
            Dictionary mit Ergebnissen für jedes Modell
        """
        results = {}

        for model_name in self.grids.keys():
            print(f"\n{'=' * 60}")
            print(f"Trainiere: {model_name}")
            print(f"{'=' * 60}")

            try:
                if use_grid_search:
                    model, params, metrics, train_time = self.grid_search(
                        model_name, X_train, y_train, X_test, y_test
                    )
                else:
                    model, metrics, train_time = self.train_model(
                        model_name, X_train, y_train, X_test, y_test
                    )
                    params = {}

                results[model_name] = {
                    'model': model,
                    'hyperparameters': params,
                    'metrics': metrics,
                    'training_time': train_time
                }

            except Exception as e:
                print(f"❌ Fehler bei {model_name}: {e}")
                results[model_name] = {'error': str(e)}

        return results

    @staticmethod
    def _count_combinations(param_grid: Dict) -> int:
        """Zählt die Anzahl möglicher Hyperparameter-Kombinationen"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count

    def get_available_models(self) -> list:
        """Gibt Liste verfügbarer Modelle zurück"""
        return list(self.grids.keys())