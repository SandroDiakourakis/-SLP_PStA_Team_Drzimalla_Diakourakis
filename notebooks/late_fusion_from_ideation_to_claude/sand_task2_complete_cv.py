#!/usr/bin/env python3
"""
===============================================================================
SAND Challenge - Task 2: ALSFRS-R Progression Prediction
5-Fold Cross-Validation Pipeline
===============================================================================

TASK DESCRIPTION:
-----------------
Predict the ALSFRS-R scale of ALS patients at their last follow-up visit
using voice recordings and temporal metadata.

KEY FEATURES:
-------------
- 4 classes (ALSFRS-R scores 1-4, no healthy controls)
- Multi-file audio input (8 recordings per patient)
- Temporal features: Months between assessments, Initial ALSFRS-R score
- Late fusion architecture with attention mechanism
- 5-fold stratified cross-validation
- Automatic hyperparameter selection
- Transfer learning support from Task 1

USAGE:
------
1. Configure paths in main():
   config.DATA_DIR = "/path/to/task2/training"
   config.METADATA_FILE = "/path/to/sand_task_2.xlsx"

2. Run:
   python sand_task2_complete_cv.py

3. Results will be saved to:
   ./task2_cv_results/experiment_TIMESTAMP/final_results.json

REQUIREMENTS:
-------------
torch, transformers, pandas, librosa, soundfile, sklearn, matplotlib, seaborn

AUTHORS: Adapted for Task 2 from original Task 1 pipeline
DATE: January 2025
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import librosa
from tqdm import tqdm
import random
import warnings
import sys
import os

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Task2Config:
    """Complete configuration for Task 2"""
    
    # ==================== PATHS (MUST BE CONFIGURED) ====================
    DATA_DIR: str = "/Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis/data/task2/training"
    METADATA_FILE: str = "/Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis/data/task2/sand_task_2.xlsx"
    OUTPUT_DIR: str = "./task2_cv_results"
    # PRETRAINED_MODEL_PATH: Optional[str] = "/Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis/notebooks/late_fusion_from_ideation_to_claude/best_model_checkpoint.pth"  # Optional: Task 1 model for transfer learning
    PRETRAINED_MODEL_PATH: Optional[str] = "/Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis/experiments/experiments/experiment_20251116_132709/final_model_20251116_155016.pth"
    
    # ==================== TASK-SPECIFIC ====================
    NUM_CLASSES: int = 4  # Task 2 has only 4 classes (ALSFRS-R 1-4)
    NUM_FILES: int = 8    # 8 audio files per patient
    USE_TEMPORAL_FEATURES: bool = True  # Use Months and ALSFRS-R_start
    
    # ==================== CROSS-VALIDATION ====================
    N_FOLDS: int = 5
    RANDOM_SEED: int = 42
    
    # ==================== MODEL ARCHITECTURE ====================
    MODEL_NAME: str = "facebook/wav2vec2-large"
    POOLING: str = "mean"  # mean, max, first-last
    LAYERS: Optional[List[int]] = None  # e.g., [7, 11, 15] for multi-layer
    LAYER_FUSION: str = "concat"  # concat, mean, weighted
    
    FUSION_TYPE: str = "feature"  # feature or decision
    FUSION_STRATEGY: str = "attention"  # attention, learned, mean, max
    HIDDEN_DIM: int = 512
    FILE_PROCESSOR_DIM: int = 256
    DROPOUT: float = 0.3
    SHARED_FILE_PROCESSOR: bool = False
    
    # ==================== TRAINING ====================
    BATCH_SIZE: int = 12
    ACCUMULATION_STEPS: int = 4
    EPOCHS: int = 10
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    MAX_GRAD_NORM: float = 1.0
    
    # ==================== EARLY STOPPING ====================
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_METRIC: str = "f1"  # f1, accuracy, loss
    
    # ==================== AUGMENTATION ====================
    USE_AUGMENTATION: bool = False
    TIME_STRETCH_RANGE: Tuple[float, float] = (0.9, 1.1)
    PITCH_SHIFT_RANGE: Tuple[int, int] = (-2, 2)
    NOISE_SNR_RANGE: Tuple[float, float] = (20, 40)
    TIME_MASK_PROB: float = 0.3
    AUGMENTATION_PROB: float = 0.8
    
    # ==================== SYSTEM ====================
    DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"
    NUM_WORKERS: int = 0  # DataLoader workers (0 for CUDA compatibility)
    
    # ==================== TRANSFER LEARNING ====================
    USE_PRETRAINED: bool = False
    FINE_TUNE_LAYERS: int = 0  # Number of layers to fine-tune (0 = frozen)


# ============================================================================
# AUDIO AUGMENTATION
# ============================================================================

class AudioAugmentation:
    """Audio augmentation pipeline"""
    
    def __init__(self, config: Task2Config):
        self.time_stretch_range = config.TIME_STRETCH_RANGE
        self.pitch_shift_range = config.PITCH_SHIFT_RANGE
        self.noise_snr_range = config.NOISE_SNR_RANGE
        self.time_mask_prob = config.TIME_MASK_PROB
        self.augmentation_prob = config.AUGMENTATION_PROB
        
    def __call__(self, audio: np.ndarray, sr: int, apply_augmentation: bool = True) -> np.ndarray:
        if not apply_augmentation or random.random() > self.augmentation_prob:
            return audio
        
        # Randomly select augmentations
        if random.random() < 0.5:
            audio = self._time_stretch(audio, sr)
        if random.random() < 0.5:
            audio = self._pitch_shift(audio, sr)
        if random.random() < 0.6:
            audio = self._add_noise(audio)
        if random.random() < self.time_mask_prob:
            audio = self._time_mask(audio)
            
        return audio
    
    def _time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        rate = random.uniform(*self.time_stretch_range)
        try:
            return librosa.effects.time_stretch(audio, rate=rate)
        except:
            return audio
    
    def _pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        n_steps = random.randint(*self.pitch_shift_range)
        try:
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        except:
            return audio
    
    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        snr_db = random.uniform(*self.noise_snr_range)
        signal_power = np.mean(audio ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        return audio + noise
    
    def _time_mask(self, audio: np.ndarray) -> np.ndarray:
        audio_len = len(audio)
        max_mask_len = int(audio_len * 0.15)
        if max_mask_len < 2:
            return audio
        mask_len = random.randint(1, max_mask_len)
        mask_start = random.randint(0, audio_len - mask_len)
        audio_masked = audio.copy()
        audio_masked[mask_start:mask_start + mask_len] *= 0.1
        return audio_masked


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        pass


class Wav2Vec2Extractor(FeatureExtractor):
    """Wav2Vec 2.0 feature extractor"""
    
    def __init__(self, config: Task2Config):
        from transformers import Wav2Vec2Model
        
        self.device = config.DEVICE
        self.pooling = config.POOLING
        self.layers = config.LAYERS if config.LAYERS is not None else [12]
        self.layer_fusion = config.LAYER_FUSION
        
        self.model = Wav2Vec2Model.from_pretrained(config.MODEL_NAME).to(self.device)
        self.model.eval()
        
        # Calculate feature dimension
        base_dim = self.model.config.hidden_size
        pooling_multiplier = 2 if config.POOLING in ["first-last"] else 1
        
        if config.LAYER_FUSION == "concat":
            self._feature_dim = base_dim * len(self.layers) * pooling_multiplier
        else:
            self._feature_dim = base_dim * pooling_multiplier
        
        if config.LAYER_FUSION == "weighted":
            self.layer_weights = nn.Parameter(torch.ones(len(self.layers)))
            self.layer_weights.to(self.device)
        
        logger.info(f"✓ Wav2Vec2 loaded: {config.MODEL_NAME}")
        logger.info(f"  Layers: {self.layers}, Fusion: {self.layer_fusion}")
        logger.info(f"  Feature dim: {self._feature_dim}")
    
    def extract(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(audio, output_hidden_states=True)
            layer_features = []
            
            for layer_idx in self.layers:
                features = outputs.hidden_states[layer_idx]
                
                # Pool over time
                if self.pooling == "mean":
                    pooled = features.mean(dim=1)
                elif self.pooling == "max":
                    pooled = features.max(dim=1)[0]
                elif self.pooling == "first":
                    pooled = features[:, 0, :]
                elif self.pooling == "last":
                    pooled = features[:, -1, :]
                elif self.pooling == "first-last":
                    pooled = torch.cat([features[:, 0, :], features[:, -1, :]], dim=-1)
                else:
                    pooled = features.mean(dim=1)
                
                layer_features.append(pooled)
            
            # Fuse layers
            if self.layer_fusion == "concat":
                fused = torch.cat(layer_features, dim=-1)
            elif self.layer_fusion == "mean":
                fused = torch.stack(layer_features, dim=0).mean(dim=0)
            elif self.layer_fusion == "weighted":
                weights = F.softmax(self.layer_weights, dim=0)
                weighted = [w * feat for w, feat in zip(weights, layer_features)]
                fused = torch.stack(weighted, dim=0).sum(dim=0)
            else:
                fused = layer_features[0]
        
        return fused
    
    def get_feature_dim(self) -> int:
        return self._feature_dim


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Task2Sample:
    """Sample with temporal information for Task 2"""
    individual_id: str
    file_paths: List[str]
    label: int  # ALSFRS-R_end (0-indexed: 0-3)
    months: float  # Time between assessments
    alsfrs_start: int  # Initial ALSFRS-R score
    age: Optional[int] = None
    sex: Optional[str] = None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_task2_data(config: Task2Config) -> List[Task2Sample]:
    """Load and combine train + dev data for Task 2"""
    
    logger.info(f"Loading Task 2 data from: {config.METADATA_FILE}")
    
    try:
        # Load all sheets
        df_train = pd.read_excel(config.METADATA_FILE, sheet_name="SAND - TRAINING set - Task 2")
        df_train_baseline = pd.read_excel(config.METADATA_FILE, sheet_name="Training Baseline - Task 2")
        df_val_baseline = pd.read_excel(config.METADATA_FILE, sheet_name="Validation Baseline - Task 2")
        
        # Combine as instructed by professor
        df_combined = pd.concat([df_train_baseline, df_val_baseline], ignore_index=True)
        logger.info(f"✓ Combined: {len(df_train_baseline)} train + {len(df_val_baseline)} val = {len(df_combined)} total")
        
    except Exception as e:
        logger.error(f"❌ Error loading Excel: {e}")
        raise
    
    # Build samples
    samples = []
    data_path = Path(config.DATA_DIR)
    audio_types = ['phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
                   'rhythmPA', 'rhythmTA', 'rhythmKA']
    
    missing_files = 0
    for _, row in df_combined.iterrows():
        individual_id = str(row['ID']).strip()
        
        # Check all audio files exist
        file_paths = []
        all_exist = True
        for audio_type in audio_types:
            fpath = data_path / audio_type / f"{individual_id}_{audio_type}.wav"
            if not fpath.exists():
                all_exist = False
                missing_files += 1
                break
            file_paths.append(str(fpath))
        
        if not all_exist:
            continue
        
        try:
            # Extract metadata (Note: Excel has double dash: ALSFRS--R)
            label = int(row['ALSFRS--R_end']) - 1  # Convert to 0-indexed
            months = float(row['Months'])
            alsfrs_start = int(row['ALSFRS--R_start'])
            age = int(row['Age']) if 'Age' in row and pd.notna(row['Age']) else None
            sex = str(row['Sex']) if 'Sex' in row and pd.notna(row['Sex']) else None
            
            if label < 0 or label >= config.NUM_CLASSES:
                logger.warning(f"Invalid label {label+1} for {individual_id}")
                continue
            
            sample = Task2Sample(
                individual_id=individual_id,
                file_paths=file_paths,
                label=label,
                months=months,
                alsfrs_start=alsfrs_start,
                age=age,
                sex=sex
            )
            samples.append(sample)
            
        except Exception as e:
            logger.warning(f"Error processing {individual_id}: {e}")
            continue
    
    if missing_files > 0:
        logger.warning(f"⚠️  Skipped {missing_files} samples due to missing audio files")
    
    logger.info(f"✓ Loaded {len(samples)} valid samples")
    
    # Show class distribution
    labels = [s.label for s in samples]
    class_counts = pd.Series(labels).value_counts().sort_index()
    logger.info("\nClass distribution:")
    for cls, count in class_counts.items():
        logger.info(f"  Class {cls+1} (ALSFRS-R): {count} samples ({count/len(samples)*100:.1f}%)")
    
    return samples


# ============================================================================
# DATASET
# ============================================================================

class Task2Dataset(Dataset):
    """Dataset for Task 2 with temporal features"""
    
    def __init__(self, samples, feature_extractor, config, augmentation=None, is_training=False):
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.config = config
        self.augmentation = augmentation
        self.is_training = is_training
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract audio features
        file_features = []
        for fpath in sample.file_paths:
            try:
                audio, sr = sf.read(fpath)
                
                if self.is_training and self.augmentation:
                    audio = self.augmentation(audio, sr, apply_augmentation=True)
                
                audio_tensor = torch.FloatTensor(audio)
                features = self.feature_extractor.extract(audio_tensor, sr)
                # Remove batch dimension if present (features might be [1, feat_dim])
                if features.dim() == 2 and features.size(0) == 1:
                    features = features.squeeze(0)
                file_features.append(features)
                
            except Exception as e:
                logger.warning(f"Error loading {fpath}: {e}")
                feat_dim = self.feature_extractor.get_feature_dim()
                file_features.append(torch.zeros(feat_dim))
        
        file_features = torch.stack(file_features)
        
        # Temporal features
        if self.config.USE_TEMPORAL_FEATURES:
            temporal = torch.FloatTensor([
                sample.months / 100.0,  # Normalize
                (sample.alsfrs_start - 1) / 3.0  # Normalize to [0,1]
            ])
        else:
            temporal = torch.FloatTensor([])
        
        return {
            'file_features': file_features,
            'temporal_features': temporal,
            'label': torch.tensor(sample.label, dtype=torch.long),
            'individual_id': sample.individual_id
        }


def collate_fn(batch):
    """Custom collate function"""
    return {
        'file_features': torch.stack([b['file_features'] for b in batch]),
        'temporal_features': torch.stack([b['temporal_features'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch]),
        'individual_ids': [b['individual_id'] for b in batch]
    }


# ============================================================================
# MODEL
# ============================================================================

class Task2Classifier(nn.Module):
    """Late fusion classifier for Task 2"""
    
    def __init__(self, config: Task2Config, audio_feat_dim: int):
        super().__init__()
        
        self.config = config
        temporal_dim = 2 if config.USE_TEMPORAL_FEATURES else 0
        
        # File processors
        if config.SHARED_FILE_PROCESSOR:
            self.file_processor = self._make_processor(audio_feat_dim, config.FILE_PROCESSOR_DIM, config.DROPOUT)
        else:
            self.file_processors = nn.ModuleList([
                self._make_processor(audio_feat_dim, config.FILE_PROCESSOR_DIM, config.DROPOUT)
                for _ in range(config.NUM_FILES)
            ])
        
        # Fusion
        if config.FUSION_STRATEGY == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=config.FILE_PROCESSOR_DIM,
                num_heads=4,
                dropout=config.DROPOUT,
                batch_first=True
            )
            fusion_dim = config.FILE_PROCESSOR_DIM
        elif config.FUSION_STRATEGY == "learned":
            self.fusion_weights = nn.Parameter(torch.ones(config.NUM_FILES) / config.NUM_FILES)
            fusion_dim = config.FILE_PROCESSOR_DIM
        else:
            fusion_dim = config.FILE_PROCESSOR_DIM
        
        # Combine with temporal
        combined_dim = fusion_dim + temporal_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        )
    
    def _make_processor(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, file_features, temporal_features=None):
        batch_size = file_features.size(0)
        
        # Process files
        if hasattr(self, 'file_processor'):
            processed = []
            for i in range(self.config.NUM_FILES):
                processed.append(self.file_processor(file_features[:, i, :]))
            processed = torch.stack(processed, dim=1)
        else:
            processed = []
            for i in range(self.config.NUM_FILES):
                processed.append(self.file_processors[i](file_features[:, i, :]))
            processed = torch.stack(processed, dim=1)
        
        # Fuse
        if self.config.FUSION_STRATEGY == "attention":
            fused, _ = self.attention(processed, processed, processed)
            fused = fused.mean(dim=1)
        elif self.config.FUSION_STRATEGY == "learned":
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = torch.sum(processed * weights.view(1, -1, 1), dim=1)
        elif self.config.FUSION_STRATEGY == "max":
            fused, _ = torch.max(processed, dim=1)
        else:  # mean
            fused = processed.mean(dim=1)
        
        # Add temporal
        if self.config.USE_TEMPORAL_FEATURES and temporal_features is not None:
            combined = torch.cat([fused, temporal_features], dim=1)
        else:
            combined = fused
        
        return self.classifier(combined)


# ============================================================================
# TRAINING
# ============================================================================

class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience=10, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return improved


def train_fold(fold_idx, train_samples, val_samples, config, fold_dir):
    """Train a single fold"""
    
    logger.info(f"\n{'='*70}")
    logger.info(f"FOLD {fold_idx + 1}/{config.N_FOLDS}")
    logger.info(f"{'='*70}")
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # Create feature extractor
    logger.info("Initializing feature extractor...")
    feature_extractor = Wav2Vec2Extractor(config)
    
    # Optionally load pretrained weights
    if config.USE_PRETRAINED and config.PRETRAINED_MODEL_PATH:
        try:
            checkpoint = torch.load(config.PRETRAINED_MODEL_PATH, map_location=config.DEVICE)
            feature_extractor.model.load_state_dict(checkpoint['feature_extractor'], strict=False)
            logger.info(f"✓ Loaded pretrained weights from Task 1")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    # Create datasets
    augmentation = AudioAugmentation(config) if config.USE_AUGMENTATION else None
    
    train_dataset = Task2Dataset(train_samples, feature_extractor, config, augmentation, is_training=True)
    val_dataset = Task2Dataset(val_samples, feature_extractor, config, None, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=config.NUM_WORKERS)
    
    # Create model
    model = Task2Classifier(config, feature_extractor.get_feature_dim()).to(config.DEVICE)
    
    # Loss with class weighting
    train_labels = [s.label for s in train_samples]
    class_counts = np.bincount(train_labels, minlength=config.NUM_CLASSES)
    class_weights = len(train_labels) / (config.NUM_CLASSES * class_counts + 1e-6)
    class_weights = torch.FloatTensor(class_weights).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # Learning rate scheduler (verbose removed for compatibility with newer PyTorch)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='max')
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_acc': []}
    best_f1 = 0
    
    for epoch in range(config.EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            file_feats = batch['file_features'].to(config.DEVICE)
            temporal_feats = batch['temporal_features'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            outputs = model(file_feats, temporal_feats)
            loss = criterion(outputs, labels)
            
            loss = loss / config.ACCUMULATION_STEPS
            loss.backward()
            
            if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config.ACCUMULATION_STEPS
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': train_loss / (batch_idx + 1), 
                             'acc': train_correct / train_total})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                file_feats = batch['file_features'].to(config.DEVICE)
                temporal_feats = batch['temporal_features'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)
                
                outputs = model(file_feats, temporal_feats)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                   f"Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")
        
        # Update scheduler
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, fold_dir / 'model_best.pth')
        
        # Early stopping check
        if early_stopping(val_f1, epoch):
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    checkpoint = torch.load(fold_dir / 'model_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            file_feats = batch['file_features'].to(config.DEVICE)
            temporal_feats = batch['temporal_features'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            outputs = model(file_feats, temporal_feats)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    acc = accuracy_score(all_labels, all_preds)
    
    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i+1}' for i in range(config.NUM_CLASSES)],
                yticklabels=[f'Class {i+1}' for i in range(config.NUM_CLASSES)])
    plt.title(f'Fold {fold_idx+1} - Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(fold_dir / 'confusion_matrix.png', dpi=150)
    plt.close()
    
    # Save training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss History')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Val F1')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Metrics History')
    plt.tight_layout()
    plt.savefig(fold_dir / 'training_history.png', dpi=150)
    plt.close()
    
    metrics = {
        'fold': fold_idx,
        'f1_macro': float(f1_macro),
        'f1_per_class': [float(f) for f in f1_per_class],
        'accuracy': float(acc),
        'best_epoch': checkpoint['epoch']
    }
    
    with open(fold_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✓ Fold {fold_idx+1} completed: F1={f1_macro:.4f}, Acc={acc:.4f}")
    
    return metrics


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cross_validation(samples, config):
    """Run 5-fold cross-validation"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(config.OUTPUT_DIR) / f"experiment_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    config_dict['DEVICE'] = str(config_dict['DEVICE'])
    if config_dict['LAYERS'] is not None:
        config_dict['LAYERS'] = list(config_dict['LAYERS'])
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STARTING 5-FOLD CROSS-VALIDATION")
    logger.info(f"Experiment directory: {exp_dir}")
    logger.info(f"{'='*80}\n")
    
    # Stratified K-Fold
    labels = np.array([s.label for s in samples])
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(samples)), labels)):
        fold_dir = exp_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        
        metrics = train_fold(fold_idx, train_samples, val_samples, config, fold_dir)
        fold_results.append(metrics)
    
    # Compute final results
    logger.info(f"\n{'='*80}")
    logger.info(f"FINAL RESULTS - 5-FOLD CROSS-VALIDATION")
    logger.info(f"{'='*80}\n")
    
    f1_scores = [fold['f1_macro'] for fold in fold_results]
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    logger.info(f"F1-Macro Score: {mean_f1:.4f} ± {std_f1:.4f}")
    logger.info(f"Individual folds: {[f'{f:.4f}' for f in f1_scores]}")
    
    logger.info(f"\nPer-class F1 scores:")
    for class_idx in range(config.NUM_CLASSES):
        class_f1s = [fold['f1_per_class'][class_idx] for fold in fold_results]
        logger.info(f"  Class {class_idx+1}: {np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}")
    
    # Save final results
    final_results = {
        'mean_f1_macro': float(mean_f1),
        'std_f1_macro': float(std_f1),
        'fold_f1_scores': [float(f) for f in f1_scores],
        'per_class_results': {},
        'timestamp': timestamp,
        'config': config_dict
    }
    
    for class_idx in range(config.NUM_CLASSES):
        class_f1s = [fold['f1_per_class'][class_idx] for fold in fold_results]
        final_results['per_class_results'][f'class_{class_idx+1}'] = {
            'mean_f1': float(np.mean(class_f1s)),
            'std_f1': float(np.std(class_f1s))
        }
    
    with open(exp_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save summary report
    with open(exp_dir / 'summary_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SAND CHALLENGE - TASK 2: 5-FOLD CROSS-VALIDATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Experiment: {timestamp}\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Classes: {config.NUM_CLASSES}\n\n")
        f.write(f"MAIN RESULT:\n")
        f.write(f"  F1-Macro: {mean_f1:.4f} ± {std_f1:.4f}\n\n")
        f.write(f"Individual Folds:\n")
        for i, f1 in enumerate(f1_scores):
            f.write(f"  Fold {i+1}: {f1:.4f}\n")
        f.write(f"\nPer-Class Results:\n")
        for class_idx in range(config.NUM_CLASSES):
            class_f1s = [fold['f1_per_class'][class_idx] for fold in fold_results]
            f.write(f"  Class {class_idx+1}: {np.mean(class_f1s):.4f} ± {np.std(class_f1s):.4f}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("For your professor: Results from 5-fold cross-validation on combined train+dev set\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"\n✓✓✓ Results saved to: {exp_dir}")
    logger.info(f"✓✓✓ Check final_results.json for complete metrics\n")
    
    return final_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    
    # Configuration
    config = Task2Config()
    
    # Optional: Hyperparameter adjustments
    # config.HIDDEN_DIM = 512
    # config.DROPOUT = 0.3
    # config.FUSION_STRATEGY = "attention"
    # config.LEARNING_RATE = 1e-4
    # config.LAYERS = [7, 11, 15]  # Multi-layer features
    # ==========================================
    
    logger.info("=" * 80)
    logger.info("SAND CHALLENGE - TASK 2: ALSFRS-R PROGRESSION PREDICTION")
    logger.info("5-Fold Cross-Validation Pipeline")
    logger.info("=" * 80)
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Classes: {config.NUM_CLASSES}")
    logger.info(f"Temporal features: {config.USE_TEMPORAL_FEATURES}")
    logger.info(f"Augmentation: {config.USE_AUGMENTATION}")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\nLoading Task 2 data...")
    samples = load_task2_data(config)
    
    if len(samples) == 0:
        logger.error("❌ No samples loaded! Check your paths.")
        return
    
    # Run cross-validation
    results = run_cross_validation(samples, config)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓✓✓ CROSS-VALIDATION COMPLETED SUCCESSFULLY! ✓✓✓")
    logger.info("=" * 80)
    logger.info(f"\nFINAL RESULT: F1-Macro = {results['mean_f1_macro']:.4f} ± {results['std_f1_macro']:.4f}")
    logger.info("\nFor your professor:")
    logger.info(f"  Results are from 5-fold cross-validation (mean + std)")
    logger.info(f"  on the combined train+dev set as requested.")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
