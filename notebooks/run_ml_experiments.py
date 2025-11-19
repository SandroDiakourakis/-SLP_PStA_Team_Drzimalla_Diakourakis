#!/usr/bin/env python3
"""
Automatisierte Ausführung von ML-Experimenten mit verschiedenen Konfigurationen.

Dieses Skript führt das SAND_TASK_1 Notebook mit verschiedenen Wav2Vec2-Konfigurationen aus
und speichert die Ergebnisse jeweils in separaten Experiment-Ordnern.

Usage:
    python run_ml_experiments.py
"""

import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from tqdm import tqdm
from scipy.signal import resample
from typing import List, Optional, Tuple
import json
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, cohen_kappa_score
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# EXPERIMENT KONFIGURATIONEN
# ============================================================================

EXPERIMENT_CONFIGS = [
    {
        "name": "config_1_mean_pooling_mean_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "mean",
        "layer_fusion": "concat",
        "description": "Multi-layer (5 layers), mean pooling, concat fusion"
    },
    {
        "name": "config_1_mean_pooling_mean_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "mean",
        "layer_fusion": "mean",
        "description": "Multi-layer (5 layers), mean pooling, mean fusion"
    },
    {
        "name": "config_1_mean_pooling_weighted_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "mean",
        "layer_fusion": "weighted",
        "description": "Multi-layer (5 layers), mean pooling, weighted fusion"
    },
    {
        "name": "config_2_max_pooling_concat_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "max",
        "layer_fusion": "concat",
        "description": "Multi-layer (5 layers), max pooling, concat fusion"
    },
    {
        "name": "config_2_max_pooling_mean_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "max",
        "layer_fusion": "mean",
        "description": "Multi-layer (5 layers), max pooling, mean fusion"
    },
    {
        "name": "config_2_max_pooling_weighted_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "max",
        "layer_fusion": "weighted",
        "description": "Multi-layer (5 layers), max pooling, weighted fusion"
    },
    {
        "name": "config_3_last_pooling_concat_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "last",
        "layer_fusion": "concat",
        "description": "Multi-layer (5 layers), last token pooling, concat fusion"
    },
    {
        "name": "config_3_last_pooling_mean_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "last",
        "layer_fusion": "mean",
        "description": "Multi-layer (5 layers), last token pooling, mean fusion"
    },
    {
        "name": "config_3_last_pooling_weighted_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "last",
        "layer_fusion": "weighted",
        "description": "Multi-layer (5 layers), last token pooling, weighted fusion"
    },
    {
        "name": "config_4_first_pooling_concat_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "first",
        "layer_fusion": "concat",
        "description": "Multi-layer (5 layers), first token pooling, concat fusion"
    },
    {
        "name": "config_4_first_pooling_mean_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "first",
        "layer_fusion": "mean",
        "description": "Multi-layer (5 layers), first token pooling, mean fusion"
    },
    {
        "name": "config_4_first_pooling_weighted_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [6, 9, 12, 15, 18],
        "pooling": "first",
        "layer_fusion": "weighted",
        "description": "Multi-layer (5 layers), first token pooling, weighted fusion"
    },
    {
        "name": "config_5_fewer_layers",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [12, 18, 24],
        "pooling": "mean",
        "layer_fusion": "concat",
        "description": "Fewer layers (3 layers: 12, 18, 24), concat fusion"
    },
    {
        "name": "config_6_more_layers",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [3, 6, 9, 12, 15, 18, 21, 24],
        "pooling": "mean",
        "layer_fusion": "concat",
        "description": "More layers (8 layers), concat fusion"
    },
    {
        "name": "config_7_first_pooling_concat_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [19, 20, 21, 22, 23, 24],
        "pooling": "max",
        "layer_fusion": "concat",
        "description": "Multi-layer (5 layers), first token pooling, concat fusion"
    },
    {
        "name": "config_7_first_pooling_mean_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [19, 20, 21, 22, 23, 24],
        "pooling": "max",
        "layer_fusion": "mean",
        "description": "Multi-layer (5 layers), first token pooling, mean fusion"
    },
    {
        "name": "config_7_first_pooling_weighted_fusion",
        "model_name": "facebook/wav2vec2-large-960h",
        "layers": [19, 20, 21, 22, 23, 24],
        "pooling": "max",
        "layer_fusion": "weighted",
        "description": "Multi-layer (5 layers), first token pooling, weighted fusion"
    },
]

# Globale Pfade
DATA_PATH = Path('/Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis/data/task1')
EXCEL_PATH = DATA_PATH / 'sand_task_1.xlsx'
TASK_DIR = DATA_PATH / 'training'

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_from_file(audio_path: str, 
                               processor: Wav2Vec2Processor, 
                               model: Wav2Vec2Model, 
                               device: torch.device,
                               layers: List[int],
                               pooling: str = "mean",
                               layer_fusion: str = "concat") -> Optional[np.ndarray]:
    """Extrahiere Multi-Layer Wav2Vec2 Features von einer Audio-Datei.
    
    Args:
        layer_fusion: "concat" (concatenate), "mean" (average), or "weighted" (learned weights)
    """
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = resample(audio, num_samples)
        
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            layer_features = []
            
            for layer_idx in layers:
                layer_output = hidden_states[layer_idx]
                
                if pooling == "mean":
                    pooled = layer_output.mean(dim=1)
                elif pooling == "max":
                    pooled = layer_output.max(dim=1)[0]
                elif pooling == "first":
                    pooled = layer_output[:, 0, :]
                elif pooling == "last":
                    pooled = layer_output[:, -1, :]
                else:
                    pooled = layer_output.mean(dim=1)
                
                layer_features.append(pooled)
            
            # Layer Fusion Strategy
            if layer_fusion == "mean":
                # Mittele über alle Layer (gleiche Gewichtung)
                fused_features = torch.stack(layer_features).mean(dim=0)
            elif layer_fusion == "weighted":
                # Gewichtete Kombination (einfache Version: exponentiell ansteigend)
                # Höhere Layer bekommen mehr Gewicht
                weights = torch.softmax(torch.arange(len(layer_features), dtype=torch.float32), dim=0)
                weights = weights.view(-1, 1, 1).to(device)
                stacked = torch.stack(layer_features)
                fused_features = (stacked * weights).sum(dim=0)
            else:  # "concat" (default)
                # Konkateniere alle Layer
                fused_features = torch.cat(layer_features, dim=1)
            
        return fused_features.squeeze(0).cpu().numpy()
        
    except Exception as e:
        print(f"❌ Fehler beim Verarbeiten von {audio_path}: {e}")
        return None

# ============================================================================
# DATA LOADING
# ============================================================================

def load_train_val_data(excel_path: Path, task_dir: Path) -> Tuple[List, List, List, List]:
    """Lädt Training und Validation Daten."""
    audio_folders = {
        "phonationA": "phonationA", "phonationE": "phonationE",
        "phonationI": "phonationI", "phonationO": "phonationO",
        "phonationU": "phonationU", "rhythmKA": "rhythmKA",
        "rhythmPA": "rhythmPA", "rhythmTA": "rhythmTA",
    }
    
    def load_dataset(sheet_name: str):
        df_labels = pd.read_excel(excel_path, sheet_name=sheet_name)
        file_groups, labels = [], []
        
        for _, row in df_labels.iterrows():
            individual_id = row['ID']
            label = row['Class'] - 1
            file_paths = []
            
            for folder_name, file_pattern in audio_folders.items():
                folder_path = task_dir / folder_name
                audio_file = folder_path / f"{individual_id}_{file_pattern}.wav"
                if audio_file.exists():
                    file_paths.append(str(audio_file))
                else:
                    break
            
            if len(file_paths) == 8:
                file_groups.append(file_paths)
                labels.append(label)
        
        return file_groups, labels
    
    X_train_files, y_train = load_dataset('Training Baseline - Task 1')
    X_val_files, y_val = load_dataset('Validation Baseline - Task 1')
    
    return X_train_files, y_train, X_val_files, y_val

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(config: dict):
    """Führe ein einzelnes Experiment mit gegebener Konfiguration aus."""
    
    print("\n" + "="*80)
    print(f"STARTE EXPERIMENT: {config['name']}")
    print("="*80)
    print(f"Beschreibung: {config['description']}")
    print(f"Model: {config['model_name']}")
    print(f"Layers: {config['layers']}")
    print(f"Pooling: {config['pooling']}")
    print(f"Layer Fusion: {config['layer_fusion']}")
    print("="*80 + "\n")
    
    # Lade Modell
    print("Lade Wav2Vec2 Modell...")
    processor = Wav2Vec2Processor.from_pretrained(config['model_name'])
    wav2vec_model = Wav2Vec2Model.from_pretrained(config['model_name'])
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    wav2vec_model = wav2vec_model.to(device)
    wav2vec_model.eval()
    print(f"✓ Modell geladen auf Device: {device}\n")
    
    # Lade Daten
    print("Lade Daten...")
    X_train_files, y_train_list, X_val_files, y_val_list = load_train_val_data(EXCEL_PATH, TASK_DIR)
    print(f"✓ Training: {len(X_train_files)} samples, Validation: {len(X_val_files)} samples\n")
    
    # Feature Extraction
    print("Extrahiere Features...")
    X_train_features = []
    for file_group in tqdm(X_train_files, desc="Training"):
        sample_features = []
        for audio_file in file_group:
            features = extract_features_from_file(
                audio_file, processor, wav2vec_model, device,
                layers=config['layers'], pooling=config['pooling'],
                layer_fusion=config['layer_fusion']
            )
            if features is not None:
                sample_features.append(features)
        
        if len(sample_features) == 8:
            X_train_features.append(np.concatenate(sample_features))
    
    X_val_features = []
    for file_group in tqdm(X_val_files, desc="Validation"):
        sample_features = []
        for audio_file in file_group:
            features = extract_features_from_file(
                audio_file, processor, wav2vec_model, device,
                layers=config['layers'], pooling=config['pooling'],
                layer_fusion=config['layer_fusion']
            )
            if features is not None:
                sample_features.append(features)
        
        if len(sample_features) == 8:
            X_val_features.append(np.concatenate(sample_features))
    
    X_train = np.array(X_train_features)
    y_train = np.array(y_train_list)
    X_test = np.array(X_val_features)
    y_test = np.array(y_val_list)
    
    print(f"\n✓ Features extrahiert: Train {X_train.shape}, Val {X_test.shape}\n")
    
    # Train SVM
    print("Trainiere SVM...")
    svm_pipeline = Pipeline([('svm', SVC(random_state=42, probability=True))])
    svm_param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__kernel': ['rbf', 'linear'],
        'svm__gamma': ['scale', 'auto']
    }
    svm_grid = GridSearchCV(svm_pipeline, svm_param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    svm_grid.fit(X_train, y_train)
    svm_pred = svm_grid.predict(X_test)
    svm_f1 = f1_score(y_test, svm_pred, average='macro')
    print(f"✓ SVM: F1-Macro = {svm_f1:.4f}, Best Params: {svm_grid.best_params_}\n")
    
    # Train Logistic Regression
    print("Trainiere Logistic Regression...")
    lr_pipeline = Pipeline([('lr', LogisticRegression(random_state=42, max_iter=2000))])
    lr_param_grid = {
        'lr__C': [0.01, 0.1, 1, 10, 100],
        'lr__penalty': ['l2'],
        'lr__solver': ['lbfgs', 'liblinear']
    }
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    lr_pred = lr_grid.predict(X_test)
    lr_f1 = f1_score(y_test, lr_pred, average='macro')
    print(f"✓ LR: F1-Macro = {lr_f1:.4f}, Best Params: {lr_grid.best_params_}\n")
    
    # Wähle bestes Modell
    best_model_name = "SVM" if svm_f1 > lr_f1 else "Logistic Regression"
    best_pred = svm_pred if svm_f1 > lr_f1 else lr_pred
    best_f1 = max(svm_f1, lr_f1)
    
    print(f"🏆 Bestes Modell: {best_model_name} (F1-Macro: {best_f1:.4f})\n")
    
    # Speichere Ergebnisse
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"ml_experiment_{config['name']}_{timestamp}"
    experiment_dir = Path(f"../experiments/{experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = experiment_dir / "validation_results"
    validation_dir.mkdir(exist_ok=True)
    
    # Config
    exp_config = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "classical_ml_automated",
        "config_name": config['name'],
        "config_description": config['description'],
        "feature_extractor": {
            "type": "Wav2Vec2Extractor",
            "model_name": config['model_name'],
            "layers": config['layers'],
            "layer_fusion": config['layer_fusion'],
            "pooling": config['pooling'],
            "feature_dim_per_layer": wav2vec_model.config.hidden_size,
            "feature_dim_total": wav2vec_model.config.hidden_size * len(config['layers']) if config['layer_fusion'] == 'concat' else wav2vec_model.config.hidden_size,
        },
        "dataset": {
            "train_samples": len(X_train),
            "val_samples": len(X_test),
            "num_classes": len(np.unique(y_train)),
        },
        "models_evaluated": [
            {"name": "SVM", "best_params": svm_grid.best_params_, "cv_f1_macro": float(svm_grid.best_score_), "val_f1_macro": float(svm_f1)},
            {"name": "Logistic Regression", "best_params": lr_grid.best_params_, "cv_f1_macro": float(lr_grid.best_score_), "val_f1_macro": float(lr_f1)}
        ],
        "best_model": {"name": best_model_name, "val_f1_macro": float(best_f1)}
    }
    
    with open(experiment_dir / "config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, best_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, best_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_test, best_pred)),
        "f1_macro": float(f1_score(y_test, best_pred, average='macro')),
        "f1_weighted": float(f1_score(y_test, best_pred, average='weighted')),
    }
    
    with open(validation_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Classification Report
    class_names = [f"Class_{i}" for i in range(len(np.unique(y_test)))]
    report = classification_report(y_test, best_pred, target_names=class_names, digits=4)
    with open(validation_dir / "classification_report.txt", 'w') as f:
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_pred)
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    cm_df.to_csv(validation_dir / "confusion_matrix.csv")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {best_model_name}\nF1-Macro: {best_f1:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(validation_dir / "confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Experiment gespeichert: {experiment_dir}\n")
    
    return {
        "config_name": config['name'],
        "best_model": best_model_name,
        "f1_macro": best_f1,
        "experiment_dir": str(experiment_dir)
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Führe alle Experimente aus."""
    print("="*80)
    print("AUTOMATISIERTE ML-EXPERIMENTE")
    print("="*80)
    print(f"Anzahl Konfigurationen: {len(EXPERIMENT_CONFIGS)}")
    print("="*80 + "\n")
    
    results = []
    
    for i, config in enumerate(EXPERIMENT_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(EXPERIMENT_CONFIGS)}")
        print(f"{'='*80}")
        
        result = run_experiment(config)
        results.append(result)
        
        # Cleanup GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Zusammenfassung
    print("\n" + "="*80)
    print("ALLE EXPERIMENTE ABGESCHLOSSEN")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    print("\nZusammenfassung:")
    print(results_df.to_string(index=False))
    
    # Speichere Zusammenfassung
    summary_path = Path("../experiments/experiment_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"\n✓ Zusammenfassung gespeichert: {summary_path}")
    
    best_idx = results_df['f1_macro'].idxmax()
    best_config = results_df.loc[best_idx]
    print(f"\n🏆 BESTES EXPERIMENT:")
    print(f"  Config: {best_config['config_name']}")
    print(f"  Modell: {best_config['best_model']}")
    print(f"  F1-Macro: {best_config['f1_macro']:.4f}")
    print(f"  Directory: {best_config['experiment_dir']}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
