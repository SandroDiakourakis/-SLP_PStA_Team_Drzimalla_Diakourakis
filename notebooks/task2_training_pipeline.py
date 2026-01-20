#!/usr/bin/env python3
"""
SAND Challenge Task 2: ALS Progression Prediction
Complete Training Pipeline with 5-Fold Cross-Validation

Quick Run Mode:
- Baseline (XGBoost)
- Sample Weights
- Ensemble (XGBoost + Gradient Boosting)

Optimized for GPU (CUDA) execution.
"""

import numpy as np
import pandas as pd
import torch
import random
import os
import sys
from transformers import Wav2Vec2Model, Wav2Vec2Processor, HubertModel, Wav2Vec2FeatureExtractor
import soundfile as sf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from pathlib import Path
import warnings
from tqdm import tqdm
from scipy.signal import resample
from typing import List, Optional, Tuple, Dict, Any
import json
from datetime import datetime
import pickle
from dataclasses import dataclass, field, asdict
import argparse

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Zentrale Konfiguration für alle Experimente."""
    
    # Pfade (für Server anpassen!)
    data_path: Path = Path('~/slp_project/data/task2')  # GPU: Server-Pfad
    excel_path: Path = None
    task_dir: Path = None
    output_dir: Path = Path('./task2_results')
    
    # Cross-Validation
    n_folds: int = 5
    cv_random_state: int = 42
    
    # Feature Extraction
    wav2vec2_model: str = "facebook/wav2vec2-large-960h"
    wav2vec2_layers: List[int] = field(default_factory=lambda: [6, 9, 12, 15, 18])
    wav2vec2_pooling: str = "mean"
    wav2vec2_fusion: str = "concat"
    
    hubert_model: str = "facebook/hubert-large-ls960-ft"
    hubert_layers: List[int] = field(default_factory=lambda: [6, 9, 12, 15, 18])
    hubert_pooling: str = "mean"
    hubert_fusion: str = "concat"
    
    model_fusion_strategy: str = "concat"
    
    # Dimensionality Reduction
    use_pca: bool = False
    pca_variance: float = 0.95
    
    # Audio Types
    audio_types: List[str] = field(default_factory=lambda: [
        'phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
        'rhythmKA', 'rhythmPA', 'rhythmTA'
    ])
    
    # Model Parameters
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_early_stopping: int = 20
    xgb_eval_size: float = 0.2
    
    # GPU Settings
    use_gpu_for_xgboost: bool = True  # GPU: Aktiviert für Server
    gpu_id: int = 0
    
    global_seed: int = 42
    
    def __post_init__(self):
        # Expandiere home directory
        self.data_path = Path(self.data_path).expanduser()
        
        if self.excel_path is None:
            self.excel_path = self.data_path / 'sand_task_2.xlsx'
        if self.task_dir is None:
            self.task_dir = self.data_path / 'training'
        
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d
    
    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

# ============================================================================
# EXPERIMENT TRACKER
# ============================================================================

class ExperimentTracker:
    """Verwaltet Experimente und verhindert Überschreibung von Ergebnissen."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []
    
    def log_experiment(self, 
                      name: str, 
                      experiment_config: dict, 
                      results: dict, 
                      artifacts: Optional[List[str]] = None) -> str:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        experiment = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "config": experiment_config,
            "results": results,
            "artifacts": artifacts or []
        }
        
        self.experiments.append(experiment)
        
        exp_file = self.base_dir / f"{run_id}_{name}.json"
        with open(exp_file, 'w') as f:
            json.dump(experiment, f, indent=2)
        
        print(f"Experiment geloggt: {run_id}_{name}")
        return run_id

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_global_seed(seed: int = 42):
    """Setzt alle Random Seeds für vollständige Reproduzierbarkeit."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Global seed gesetzt: {seed}")

# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(df: pd.DataFrame, config: Config) -> Tuple[List[List[str]], np.ndarray, np.ndarray]:
    """Bereite Audio-Pfade, Labels und Metadata vor."""
    audio_files = []
    labels = []
    metadata = []
    
    for _, row in df.iterrows():
        patient_id = row['ID']
        patient_files = []
        all_exist = True
        
        for audio_type in config.audio_types:
            file_path = config.task_dir / audio_type / f"{patient_id}_{audio_type}.wav"
            if file_path.exists():
                patient_files.append(str(file_path))
            else:
                print(f"Warnung: Fehlende Datei {file_path}")
                all_exist = False
                break
        
        if all_exist and len(patient_files) == 8:
            audio_files.append(patient_files)
            labels.append(row['ALSFRS--R_end'] - 1)  # Transform to [0,1,2,3]
            sex_encoded = 1 if row['Sex'] == 'M' else 0
            metadata.append([row['Age'], sex_encoded, row['Months'], row['ALSFRS--R_start']])
    
    return audio_files, np.array(labels), np.array(metadata)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_wav2vec2_features(audio_path: str, processor, model, device, config) -> Optional[np.ndarray]:
    """Extrahiere Wav2Vec2 Features."""
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = resample(audio, num_samples)
        
        inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        layer_features = []
        for layer_idx in config.wav2vec2_layers:
            hidden_state = outputs.hidden_states[layer_idx].cpu().numpy()[0]
            pooled = np.mean(hidden_state, axis=0) if config.wav2vec2_pooling == "mean" else hidden_state[0]
            layer_features.append(pooled)
        
        features = np.concatenate(layer_features) if config.wav2vec2_fusion == "concat" else np.mean(layer_features, axis=0)
        return features
    
    except Exception as e:
        print(f"Fehler bei {audio_path}: {e}")
        return None

def extract_hubert_features(audio_path: str, processor, model, device, config) -> Optional[np.ndarray]:
    """Extrahiere HuBERT Features."""
    try:
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        target_sr = 16000
        if sr != target_sr:
            num_samples = int(len(audio) * target_sr / sr)
            audio = resample(audio, num_samples)
        
        inputs = processor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        layer_features = []
        for layer_idx in config.hubert_layers:
            hidden_state = outputs.hidden_states[layer_idx].cpu().numpy()[0]
            pooled = np.mean(hidden_state, axis=0) if config.hubert_pooling == "mean" else hidden_state[0]
            layer_features.append(pooled)
        
        features = np.concatenate(layer_features) if config.hubert_fusion == "concat" else np.mean(layer_features, axis=0)
        return features
    
    except Exception as e:
        print(f"Fehler bei {audio_path}: {e}")
        return None

def extract_features_for_files(file_group, wav2vec_processor, wav2vec_model, 
                               hubert_processor, hubert_model, device, config) -> Optional[np.ndarray]:
    """Extrahiere Features für eine Gruppe von Audio-Dateien."""
    wav2vec_features = []
    hubert_features = []
    
    for audio_file in file_group:
        w2v_feat = extract_wav2vec2_features(audio_file, wav2vec_processor, wav2vec_model, device, config)
        hub_feat = extract_hubert_features(audio_file, hubert_processor, hubert_model, device, config)
        
        if w2v_feat is not None and hub_feat is not None:
            wav2vec_features.append(w2v_feat)
            hubert_features.append(hub_feat)
    
    if len(wav2vec_features) == 8 and len(hubert_features) == 8:
        wav2vec_combined = np.concatenate(wav2vec_features)
        hubert_combined = np.concatenate(hubert_features)
        
        if config.model_fusion_strategy == "concat":
            return np.concatenate([wav2vec_combined, hubert_combined])
        else:
            return np.mean([wav2vec_combined, hubert_combined], axis=0)
    
    return None

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def create_xgboost_model(config: Config):
    """Erstelle XGBoost Modell mit GPU-Unterstützung wenn verfügbar."""
    if config.use_gpu_for_xgboost:
        return xgb.XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_subsample,
            tree_method="hist",   # bleibt "hist"
            device="cuda",        # DAS ist der GPU-Schalter (XGBoost >= 2.0)

            random_state=config.global_seed,
            eval_metric="mlogloss",
            early_stopping_rounds=config.xgb_early_stopping,
            n_jobs=1              # bei GPU sinnvoll (CPU-Threads sonst unnötig)
        )
    else:
        return xgb.XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            subsample=config.xgb_subsample,
            colsample_bytree=config.xgb_subsample,
            random_state=config.global_seed,
            eval_metric='mlogloss',
            early_stopping_rounds=config.xgb_early_stopping,
            n_jobs=-1
        )

def run_cv_experiment(audio_files, y, metadata, cv, wav2vec_processor, wav2vec_model,
                     hubert_processor, hubert_model, device, config, 
                     experiment_name: str, use_sample_weights: bool = False,
                     train_second_model: bool = False):
    """Führe ein vollständiges CV-Experiment durch."""
    
    print(f"\n{'='*80}")
    print(f"{experiment_name}")
    print(f"{'='*80}\n")
    
    fold_scores = []
    fold_details = []
    oof_predictions = np.zeros(len(y))
    
    # Für Ensemble: Speichere auch zweites Modell
    oof_predictions_model2 = np.zeros(len(y)) if train_second_model else None
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(audio_files, y), 1):
        print(f"\n{'-'*80}")
        print(f"FOLD {fold_idx}/{config.n_folds}")
        print(f"{'-'*80}")
        
        # SPLIT
        train_files = [audio_files[i] for i in train_idx]
        test_files = [audio_files[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        metadata_train, metadata_test = metadata[train_idx], metadata[test_idx]
        
        # FEATURE EXTRACTION
        print("Extrahiere Features...")
        X_train_audio = []
        for file_group in tqdm(train_files, desc="Train", file=sys.stdout):
            feat = extract_features_for_files(
                file_group, wav2vec_processor, wav2vec_model,
                hubert_processor, hubert_model, device, config
            )
            if feat is not None:
                X_train_audio.append(feat)
        X_train_audio = np.array(X_train_audio)
        
        X_test_audio = []
        for file_group in tqdm(test_files, desc="Test", file=sys.stdout):
            feat = extract_features_for_files(
                file_group, wav2vec_processor, wav2vec_model,
                hubert_processor, hubert_model, device, config
            )
            if feat is not None:
                X_test_audio.append(feat)
        X_test_audio = np.array(X_test_audio)
        
        # PCA + METADATA
        X_train_full = np.concatenate([X_train_audio, metadata_train], axis=1)
        X_test_full = np.concatenate([X_test_audio, metadata_test], axis=1)
        
        if config.use_pca:
            pca = PCA(n_components=config.pca_variance, random_state=config.global_seed)
            X_train_audio_pca = pca.fit_transform(X_train_audio)
            X_test_audio_pca = pca.transform(X_test_audio)
            X_train_full = np.concatenate([X_train_audio_pca, metadata_train], axis=1)
            X_test_full = np.concatenate([X_test_audio_pca, metadata_test], axis=1)
        
        # SCALING
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test_full)
        
        # SAMPLE WEIGHTS (fold-spezifisch!)
        fold_weights = None
        if use_sample_weights:
            fold_weights = compute_sample_weight('balanced', y_train)
        
        # SPLIT FÜR EARLY STOPPING
        if use_sample_weights:
            X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
                X_train_scaled, y_train, fold_weights,
                test_size=config.xgb_eval_size, stratify=y_train, random_state=config.global_seed
            )
        else:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_scaled, y_train,
                test_size=config.xgb_eval_size, stratify=y_train, random_state=config.global_seed
            )
            w_tr, w_val = None, None
        
        # TRAINING MODELL 1 (XGBoost)
        print("Trainiere XGBoost...")
        model1 = create_xgboost_model(config)
        
        if use_sample_weights:
            model1.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_val, y_val)], 
                      sample_weight_eval_set=[w_val], 
                      verbose=False)
        else:
            model1.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred = model1.predict(X_test_scaled)
        oof_predictions[test_idx] = y_pred
        
        # MODELL 2 (Gradient Boosting für Ensemble)
        # if train_second_model:
        #     print("Trainiere Gradient Boosting...")
        #     model2 = GradientBoostingClassifier(
        #         n_estimators=200, max_depth=5, learning_rate=0.1,
        #         subsample=0.8, random_state=config.global_seed
        #     )
        #     model2.fit(X_train_scaled, y_train)
            
        #     # Ensemble Prediction (Equal Weights)
        #     pred_proba1 = model1.predict_proba(X_test_scaled)
        #     pred_proba2 = model2.predict_proba(X_test_scaled)
        #     ensemble_proba = (pred_proba1 + pred_proba2) / 2
        #     y_pred = np.argmax(ensemble_proba, axis=1)
        #     oof_predictions[test_idx] = y_pred
        #     oof_predictions_model2[test_idx] = model2.predict(X_test_scaled)
        
        # MODELL 2 (HistGradient Boosting für Ensemble)
        if train_second_model:
            print("Trainiere HistGradient Boosting...")
            model2 = HistGradientBoostingClassifier(
                max_iter=500,              # mehr Iterationen (günstiger als bei GB)
                max_depth=None,            # None = automatisch (~31 leaf nodes)
                learning_rate=0.05,        # etwas niedriger als XGBoost (0.1)
                l2_regularization=0.1,     # Ridge-Regularisierung
                max_leaf_nodes=31,         # Standard, sehr effizient
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,       # wie XGBoost early_stopping_rounds
                random_state=config.global_seed,
                verbose=0
            )
            model2.fit(X_train_scaled, y_train)
            
            # Ensemble Prediction (Equal Weights)
            pred_proba1 = model1.predict_proba(X_test_scaled)
            pred_proba2 = model2.predict_proba(X_test_scaled)
            ensemble_proba = (pred_proba1 + pred_proba2) / 2
            y_pred = np.argmax(ensemble_proba, axis=1)
            oof_predictions[test_idx] = y_pred
            oof_predictions_model2[test_idx] = model2.predict(X_test_scaled)
        
        # EVALUATION
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        
        fold_scores.append(f1)
        fold_details.append({
            'fold': fold_idx,
            'f1_macro': f1,
            'accuracy': acc
        })
        
        print(f"F1-Macro: {f1:.4f}")
    
    # AGGREGIERTE ERGEBNISSE
    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    
    print(f"\n{'='*80}")
    print(f"ERGEBNISSE: {experiment_name}")
    print(f"{'='*80}")
    print(f"F1-Macro: {mean_f1:.4f} ± {std_f1:.4f}")
    
    return {
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'fold_scores': fold_scores,
        'fold_details': fold_details,
        'oof_predictions': oof_predictions
    }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SAND Task 2 Training Pipeline')
    parser.add_argument('--data-path', type=str, default='~/slp_project/data/task2',
                       help='Pfad zu den Daten')
    parser.add_argument('--output-dir', type=str, default='./task2_results',
                       help='Output-Verzeichnis')
    parser.add_argument('--use-gpu', action='store_true', default=True,
                       help='Verwende GPU für XGBoost')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Überspringe Ensemble Training')
    args = parser.parse_args()
    
    # SETUP
    config = Config(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        use_gpu_for_xgboost=args.use_gpu
    )
    
    set_global_seed(config.global_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    tracker = ExperimentTracker(config.output_dir / 'experiments')
    
    # LOAD DATA
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df_full = pd.read_excel(config.excel_path, sheet_name='SAND - TRAINING set - Task 2')
    audio_files, y, metadata = prepare_data(df_full, config)
    
    print(f"Samples: {len(audio_files)}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # LOAD MODELS
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(config.wav2vec2_model)
    wav2vec_model = Wav2Vec2Model.from_pretrained(config.wav2vec2_model).to(device)
    wav2vec_model.eval()
    
    hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(config.hubert_model)
    hubert_model = HubertModel.from_pretrained(config.hubert_model).to(device)
    hubert_model.eval()
    
    print("Modelle geladen")
    
    # CV SETUP
    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.cv_random_state)
    
    # EXPERIMENT 1: BASELINE
    baseline_results = run_cv_experiment(
        audio_files, y, metadata, cv,
        wav2vec_processor, wav2vec_model,
        hubert_processor, hubert_model,
        device, config,
        experiment_name="BASELINE: XGBoost",
        use_sample_weights=False,
        train_second_model=False
    )
    
    tracker.log_experiment(
        name="XGBoost_Baseline",
        experiment_config={"model": "XGBoost", "use_gpu": config.use_gpu_for_xgboost},
        results={"f1_macro_mean": baseline_results['mean_f1'], "f1_macro_std": baseline_results['std_f1']}
    )
    
    # EXPERIMENT 2: SAMPLE WEIGHTS
    sw_results = run_cv_experiment(
        audio_files, y, metadata, cv,
        wav2vec_processor, wav2vec_model,
        hubert_processor, hubert_model,
        device, config,
        experiment_name="SAMPLE WEIGHTS: XGBoost",
        use_sample_weights=True,
        train_second_model=False
    )
    
    tracker.log_experiment(
        name="XGBoost_SampleWeights",
        experiment_config={"model": "XGBoost", "sample_weights": "balanced"},
        results={"f1_macro_mean": sw_results['mean_f1'], "f1_macro_std": sw_results['std_f1']}
    )
    
    # EXPERIMENT 3: ENSEMBLE
    if not args.skip_ensemble:
        ensemble_results = run_cv_experiment(
            audio_files, y, metadata, cv,
            wav2vec_processor, wav2vec_model,
            hubert_processor, hubert_model,
            device, config,
            experiment_name="ENSEMBLE: XGBoost + Gradient Boosting",
            use_sample_weights=False,
            train_second_model=True
        )
        
        tracker.log_experiment(
            name="Ensemble_XGB_GB",
            experiment_config={"model": "Ensemble", "weights": [1, 1]},
            results={"f1_macro_mean": ensemble_results['mean_f1'], "f1_macro_std": ensemble_results['std_f1']}
        )
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Baseline:      {baseline_results['mean_f1']:.4f} ± {baseline_results['std_f1']:.4f}")
    print(f"Sample Weights: {sw_results['mean_f1']:.4f} ± {sw_results['std_f1']:.4f}")
    if not args.skip_ensemble:
        print(f"Ensemble:      {ensemble_results['mean_f1']:.4f} ± {ensemble_results['std_f1']:.4f}")
    
    print(f"\nErgebnisse gespeichert in: {config.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
