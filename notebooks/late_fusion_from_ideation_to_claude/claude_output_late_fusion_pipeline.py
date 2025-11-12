# f1-macro war bei dem Evaluationset bei der Challenge bei 0.6 -> in die Nähe sollen wir auch kommen
"""
Late Fusion Pipeline for Multi-File Audio Classification

CHALLENGE TARGET: F1-Macro ≥ 0.6

Features:
- Multi-file audio classification (8 files per individual)
- Late fusion (feature-level or decision-level)
- Class-balanced loss with automatic weighting
- Early stopping with configurable metrics (F1, Accuracy, Loss)
- Learning rate scheduling
- Gradient clipping for stable training

This pipeline extracts features from neural audio models (wav2vec2, HuBERT, WavLM)
and performs late fusion to classify individuals based on multiple audio files.

Example use case: Classify individuals based on 8 audio files per person
(vowels: a, e, i, o, u + syllables: pa, ta, ka)
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, cohen_kappa_score, balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/terminal environments
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import librosa
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Feature Extractors - Modular and swappable
# ============================================================================

class FeatureExtractor(ABC):
    """Abstract base class for neural feature extractors."""

    @abstractmethod
    def extract(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract features from audio.

        Args:
            audio: Audio tensor of shape (batch_size, time) or (time,)
            sr: Sample rate

        Returns:
            Features of shape (batch_size, feature_dim) - already pooled over time
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        pass

hidden_states = 6

class Wav2Vec2Extractor(FeatureExtractor):
    """Wav2Vec 2.0 feature extractor."""

    def __init__(self, model_name: str = "facebook/wav2vec2-large", # You can also try "facebook/wav2vec2-base"
                 pooling: str = "mean", device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"):
        """
        Args:
            model_name: HuggingFace model name or fairseq model name
            pooling: Pooling strategy ("mean", "max", "first", "last")
            device: Device to run model on
        """
        from transformers import Wav2Vec2Model

        self.device = device
        self.pooling = pooling
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        self._feature_dim = self.model.config.hidden_size

        logger.info(f"Loaded Wav2Vec2 model: {model_name} on {device}")

    def extract(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Extract Wav2Vec2 features."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        with torch.no_grad():
            outputs = self.model(audio, output_hidden_states=True)
            # Extract hidden states from layer 12 (0=embedding, 1-24=layers)
            features = outputs.hidden_states[hidden_states]  # (batch, time, hidden_dim)

            # Pool over time dimension
            if self.pooling == "mean":
                pooled = features.mean(dim=1)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled

    def get_feature_dim(self) -> int:
        return self._feature_dim


class HuBERTExtractor(FeatureExtractor):
    """HuBERT feature extractor."""

    def __init__(self, model_name: str = "facebook/hubert-large-ll60k", #"facebook/hubert-large-ls960-ft",
                 pooling: str = "mean", device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"):
        from transformers import HubertModel

        self.device = device
        self.pooling = pooling
        self.model = HubertModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self._feature_dim = self.model.config.hidden_size

        logger.info(f"Loaded HuBERT model: {model_name} on {device}")

    def extract(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        with torch.no_grad():
            outputs = self.model(audio, output_hidden_states=True)
            # Extract hidden states from layer 12 (0=embedding, 1-24=layers)
            features = outputs.hidden_states[hidden_states]  # (batch, time, hidden_dim)

            if self.pooling == "mean":
                pooled = features.mean(dim=1)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled

    def get_feature_dim(self) -> int:
        return self._feature_dim


class WavLMExtractor(FeatureExtractor):
    """WavLM feature extractor."""

    def __init__(self, model_name: str = "microsoft/wavlm-large",
                 pooling: str = "mean", device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"):
        from transformers import WavLMModel

        self.device = device
        self.pooling = pooling
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self._feature_dim = self.model.config.hidden_size

        logger.info(f"Loaded WavLM model: {model_name} on {device}")

    def extract(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.device)

        with torch.no_grad():
            outputs = self.model(audio, output_hidden_states=True)
            # Extract hidden states from layer 12 (0=embedding, 1-24=layers)
            features = outputs.hidden_states[hidden_states]  # (batch, time, hidden_dim)

            if self.pooling == "mean":
                pooled = features.mean(dim=1)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled

    def get_feature_dim(self) -> int:
        return self._feature_dim

# ============================================================================
# Neural Network Components
# ============================================================================

class FileProcessor(nn.Module):
    """Process features from a single audio file with improved architecture."""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 output_dim: int = 128, dropout: float = 0.3):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # Z.B. 768 → 256
            nn.LayerNorm(hidden_dim),              # Normalisierung
            nn.ReLU(),                             # Aktivierungsfunktion
            nn.Dropout(dropout),                   # Regularisierung
            nn.Linear(hidden_dim, output_dim),     # 256 → 128
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features of shape (batch_size, input_dim)

        Returns:
            Processed features of shape (batch_size, output_dim)
        """
        return self.network(features)


class FileProcessorWithDecision(nn.Module):
    """Process features and make per-file predictions (decision-level fusion) (everything per file)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 intermediate_dim: int = 128, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # Z.B. 768 → 256
            nn.LayerNorm(hidden_dim),              # Normalisierung
            nn.ReLU(),                             # Aktivierungsfunktion
            nn.Dropout(dropout),                    # Regularisierung
            nn.Linear(hidden_dim, intermediate_dim),     # 256 → 128
            nn.LayerNorm(intermediate_dim),         # Normalisierung
            nn.ReLU(),                              # Normalisierung
            nn.Dropout(dropout)                     # Regularisierung
        )

        self.classifier = nn.Linear(intermediate_dim, num_classes)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Input features of shape (batch_size, input_dim)

        Returns:
            (processed_features, logits):
                - processed_features of shape (batch_size, intermediate_dim)
                - logits of shape (batch_size, num_classes)
        """
        processed = self.network(features)
        logits = self.classifier(processed)
        return processed, logits


class LateFusionClassifier(nn.Module):
    """
    Late fusion classifier that combines features/decisions from multiple files.

    Supports two fusion strategies:
    1. Feature-level fusion: Concatenate file-level features, then classify
    2. Decision-level fusion: Average file-level predictions
    """

    def __init__(self, input_dim: int, num_files: int, num_classes: int,
                 fusion_type: str = "feature", hidden_dim: int = 256,
                 file_processor_dim: int = 128, dropout: float = 0.3,
                 shared_file_processor: bool = False):
        """
        Args:
            input_dim: Dimension of features from neural extractor
            num_files: Number of audio files per individual
            num_classes: Number of classes to predict
            fusion_type: "feature" or "decision"
            hidden_dim: Hidden dimension for file processors
            file_processor_dim: Output dimension of file processors
            dropout: Dropout probability
            shared_file_processor: If True, share weights across file processors
        """
        super().__init__()

        self.num_files = num_files
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.shared_file_processor = shared_file_processor

        # Create file processors
        if fusion_type == "feature":
            if shared_file_processor:
                # Single shared processor
                self.file_processor = FileProcessor(
                    input_dim, hidden_dim, file_processor_dim, dropout
                )
            else:
                # Separate processor for each file
                self.file_processors = nn.ModuleList([
                    FileProcessor(input_dim, hidden_dim, file_processor_dim, dropout)
                    for _ in range(num_files)
                ])

            # Final classifier: Concatenate features + classify
            fusion_dim = file_processor_dim * num_files
            self.fusion_classifier = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )

        elif fusion_type == "decision":
            if shared_file_processor:
                # Single shared processor with decision head
                self.file_processor = FileProcessorWithDecision(
                    input_dim, hidden_dim, file_processor_dim, num_classes, dropout
                )
            else:
                # Separate processor for each file
                self.file_processors = nn.ModuleList([
                    FileProcessorWithDecision(input_dim, hidden_dim, file_processor_dim,
                                             num_classes, dropout)
                    for _ in range(num_files)
                ])

            # Optional fusion network (can also just average decisions), Fuse multiple decisions
            fusion_dim = num_classes * num_files
            self.fusion_network = nn.Sequential(
                nn.Linear(fusion_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, file_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            file_features: List of tensors, each of shape (batch_size, input_dim)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        if len(file_features) != self.num_files:
            raise ValueError(f"Expected {self.num_files} files, got {len(file_features)}")

        if self.fusion_type == "feature":
            # Process each file's features
            if self.shared_file_processor:
                processed = [self.file_processor(feat) for feat in file_features]
            else:
                processed = [proc(feat) for proc, feat in zip(self.file_processors, file_features)]

            # Concatenate and classify
            fused = torch.cat(processed, dim=-1)
            logits = self.fusion_classifier(fused)

        elif self.fusion_type == "decision":
            # Get decisions from each file
            if self.shared_file_processor:
                outputs = [self.file_processor(feat) for feat in file_features]
            else:
                outputs = [proc(feat) for proc, feat in zip(self.file_processors, file_features)]

            # Extract logits
            file_logits = [logits for _, logits in outputs]

            # Fuse decisions
            fused_logits = torch.cat(file_logits, dim=-1)
            logits = self.fusion_network(fused_logits)

        return logits

# ============================================================================
# Dataset
# ============================================================================

@dataclass
class AudioSample:
    """Represents a single individual with multiple audio files."""
    file_paths: List[Path]
    label: int
    individual_id: str
    sex: Optional[int] = None  # 0=male, 1=female, None=unknown


class MultiFileAudioDataset(Dataset):
    """Dataset for multi-file audio classification."""

    def __init__(self, samples: List[AudioSample], feature_extractor: FeatureExtractor,
                 target_sr: int = 16000):
        """
        Args:
            samples: List of AudioSample objects
            feature_extractor: Feature extractor to use
            target_sr: Target sample rate
        """
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int, Optional[int]]:
        """
        Returns:
            (file_features, label, sex):
                - file_features: List of feature tensors, one per file
                - label: Class label
                - sex: Sex attribute (0=male, 1=female, None=unknown)
        """
        sample = self.samples[idx]
        file_features = []

        if idx == 0:
            logger.info(f"Loading first sample (ID: {sample.individual_id})...")

        for file_idx, file_path in enumerate(sample.file_paths):
            # Load audio
            audio, sr = sf.read(str(file_path), dtype="float32")

            if idx == 0 and file_idx == 0:
                logger.info(f"  First audio file loaded: {file_path.name}, sr={sr}, duration={len(audio)/sr:.2f}s")

            # Resample if needed using librosa
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
                if idx == 0 and file_idx == 0:
                    logger.info(f"  Resampled from {sr}Hz to {self.target_sr}Hz")

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio)

            # Extract features
            features = self.feature_extractor.extract(audio_tensor, sr)
            file_features.append(features.squeeze(0))  # Remove batch dim

        if idx == 0:
            logger.info(f"  Extracted features for all {len(file_features)} files")

        return file_features, sample.label, sample.sex

def collate_fn(batch: List[Tuple[List[torch.Tensor], int, Optional[int]]]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Custom collate function for batching multi-file samples with sex attribute.
    
    Returns:
        (batched_features, labels, sex_tensor):
            - batched_features: List of tensors, one per file
            - labels: Tensor of class labels (long)
            - sex_tensor: Tensor of sex attributes (long), -1 for unknown/None
    """
    num_files = len(batch[0][0])
    batch_size = len(batch)

    # Organize by file index
    batched_features = []
    for file_idx in range(num_files):
        file_batch = torch.stack([sample[0][file_idx] for sample in batch])
        batched_features.append(file_batch)

    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    
    # Convert sex to tensor: None -> -1, 0 -> 0, 1 -> 1
    sex_list = [sample[2] if sample[2] is not None else -1 for sample in batch]
    sex_tensor = torch.tensor(sex_list, dtype=torch.long)

    return batched_features, labels, sex_tensor


# ============================================================================
# Training Pipeline
# ============================================================================

class LateFusionPipeline:
    """Complete pipeline for late fusion classification."""

    def __init__(self, feature_extractor: FeatureExtractor, model: LateFusionClassifier,
                 device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                 experiment_name: Optional[str] = None):
        self.feature_extractor = feature_extractor
        self.model = model.to(device)
        self.device = device
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_f1': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
        }
        
        # Experiment tracking
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = Path("experiments") / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Store configuration
        self.config = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'feature_extractor': {
                'type': feature_extractor.__class__.__name__,
                'feature_dim': feature_extractor.get_feature_dim(),
            },
            'model': {
                'num_files': model.num_files,
                'num_classes': model.num_classes,
                'fusion_type': model.fusion_type,
                'shared_file_processor': model.shared_file_processor,
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
        }

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 10, lr: float = 1e-3, weight_decay: float = 1e-4, max_grad_norm: float = 1.0,
              early_stopping_patience: int = 2, early_stopping_metric: str = "f1", lambda_fair: float = 0.1):
        """
        Train the model with early stopping and optional fairness regularization.

        Args:
            train_loader: DataLoader for the training set
            val_loader: DataLoader for the validation set
            epochs: Number of training epochs
            lr: Learning rate
            weight_decay: Weight decay for the optimizer
            max_grad_norm: Maximum norm for gradient clipping
            early_stopping_patience: Number of epochs without improvement before stopping
            early_stopping_metric: Metric for early stopping ("f1", "accuracy", "loss")
            lambda_fair: Weight for fairness regularization term (0.0 = disabled)
        """
        # Store training hyperparameters
        self.config['training'] = {
            'epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'max_grad_norm': max_grad_norm,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_metric': early_stopping_metric,
            'lambda_fair': lambda_fair,
            'batch_size': train_loader.batch_size,
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset) if val_loader else 0,
        }
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Scheduler mode depends on metric (max for f1/accuracy, min for loss)
        scheduler_mode = 'min' if early_stopping_metric == 'loss' else 'max'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=scheduler_mode, factor=0.5, patience=3
        )
        
        # Class weights to handle imbalanced data
        all_labels = []
        for _, labels, _ in train_loader:  # Now includes sex tensor
            all_labels.extend(labels.numpy())
        
        # Get unique classes and their counts
        unique_classes = np.unique(all_labels)
        num_classes = self.model.num_classes
        
        # Calculate class weights only for present classes
        class_counts = np.bincount(all_labels, minlength=num_classes)
        
        # Check if all classes are present
        if np.any(class_counts == 0):
            missing_classes = np.where(class_counts == 0)[0]
            logger.warning(f"⚠️  Classes with 0 samples in training: {missing_classes}")
            logger.warning(f"⚠️  Class distribution: {class_counts}")
            logger.warning(f"⚠️  Disabling class weights to avoid NaN")
            criterion = nn.CrossEntropyLoss()
        else:
            # Safe calculation: only divide by non-zero counts
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_counts)
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
        
        logger.info(f"Early stopping: patience={early_stopping_patience}, metric={early_stopping_metric}")
        
        # Fairness regularization setup
        if lambda_fair > 0:
            logger.info(f"Fairness regularization enabled: lambda_fair={lambda_fair}")
            logger.info("  Will balance loss across sex groups (male=0, female=1)")
        else:
            logger.info("Fairness regularization disabled (lambda_fair=0)")

        # Early stopping variables
        best_metric = 0.0 if early_stopping_metric != "loss" else float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(epochs):
            # === TRAINING ===
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Epoch {epoch+1}/{epochs}")
            logger.info(f"{'='*60}")
            
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
             
            # For F1 calculation
            train_all_preds = []
            train_all_labels = []
            
            # Fairness tracking
            epoch_fairness_losses = []

            # Progress bar for training
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                            leave=False, ncols=100)

            batch_num = 0
            for file_features, labels, sex in train_pbar:
                batch_num += 1
                if batch_num == 1:
                    logger.info(f"Processing first batch of epoch {epoch+1}...")
                if batch_num % 10 == 0:
                    logger.info(f"Epoch {epoch+1}: Processed {batch_num} batches...")
                # Move to device
                file_features = [feat.to(self.device) for feat in file_features]
                labels = labels.to(self.device)
                sex = sex.to(self.device)

                # Forward
                logits = self.model(file_features)
                loss_main = criterion(logits, labels)
                
                # Fairness regularization (if enabled)
                loss_fairness = torch.tensor(0.0, device=self.device)
                if lambda_fair > 0:
                    # Compute per-sample losses (reduction='none')
                    per_sample_losses = F.cross_entropy(logits, labels, reduction='none')
                    
                    # Separate by sex (0=male, 1=female, -1=unknown)
                    male_mask = (sex == 0)
                    female_mask = (sex == 1)
                    
                    # Calculate group losses if groups exist
                    loss_male = per_sample_losses[male_mask].mean() if male_mask.sum() > 0 else None
                    loss_female = per_sample_losses[female_mask].mean() if female_mask.sum() > 0 else None
                    
                    # Fairness penalty: |loss_male - loss_female|
                    if loss_male is not None and loss_female is not None:
                        loss_gap = torch.abs(loss_male - loss_female)
                        loss_fairness = lambda_fair * loss_gap
                        epoch_fairness_losses.append({
                            'loss_male': loss_male.item(),
                            'loss_female': loss_female.item(),
                            'loss_gap': loss_gap.item()
                        })
                
                # Total loss
                loss = loss_main + loss_fairness

                # Backward
                optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                # Collect predictions and labels for F1 calculation
                train_all_preds.extend(preds.cpu().numpy())
                train_all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                current_acc = train_correct / train_total if train_total > 0 else 0
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})

            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Calculate F1 score from accumulated predictions
            train_f1 = f1_score(train_all_labels, train_all_preds, average='macro', zero_division=0)

            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(avg_train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['train_f1'].append(train_f1)
            
            # Log fairness metrics if available
            if lambda_fair > 0 and epoch_fairness_losses:
                avg_loss_male = np.mean([f['loss_male'] for f in epoch_fairness_losses])
                avg_loss_female = np.mean([f['loss_female'] for f in epoch_fairness_losses])
                avg_loss_gap = np.mean([f['loss_gap'] for f in epoch_fairness_losses])
                logger.info(f"  Fairness: loss_male={avg_loss_male:.4f}, loss_female={avg_loss_female:.4f}, gap={avg_loss_gap:.4f}")

            # === VALIDATION ===
            if val_loader is not None:
                val_acc, val_loss, val_f1 = self.evaluate(val_loader)
                self.train_history['val_loss'].append(val_loss)
                self.train_history['val_acc'].append(val_acc)
                self.train_history['val_f1'].append(val_f1)

                # Select metric for scheduler and early stopping
                if early_stopping_metric == "f1":
                    current_metric = val_f1
                    scheduler.step(val_f1)
                elif early_stopping_metric == "accuracy":
                    current_metric = val_acc
                    scheduler.step(val_acc)
                else:  # loss
                    current_metric = val_loss
                    scheduler.step(val_loss)

                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

                # Early Stopping Check
                improved = False
                if early_stopping_metric == "loss":
                    improved = current_metric < best_metric
                else:
                    improved = current_metric > best_metric
                
                if improved:
                    best_metric = current_metric
                    patience_counter = 0
                    best_epoch = epoch + 1
                    
                    # Auto-save best model
                    self.save("best_model_checkpoint.pth")
                    logger.info(f"✓ New best {early_stopping_metric}: {best_metric:.4f} (saved checkpoint)")
                else:
                    patience_counter += 1
                    logger.info(f"⚠️  No improvement for {patience_counter}/{early_stopping_patience} epochs")
                    
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"\n{'='*70}")
                        logger.info(f"🛑 EARLY STOPPING after epoch {epoch+1}")
                        logger.info(f"   Best {early_stopping_metric}: {best_metric:.4f} at epoch {best_epoch}")
                        logger.info(f"   Loading best model from checkpoint...")
                        logger.info(f"{'='*70}\n")
                        
                        # Load best model back
                        self.load("best_model_checkpoint.pth")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}")
        
        if val_loader is not None:
            logger.info(f"\n✓ Training completed. Best model from epoch {best_epoch} loaded.")
        else:
            logger.info(f"\n✓ Training completed.")
        
        # Save experiment results
        self._save_experiment_results(best_epoch, best_metric)

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluate the model.
        
        Returns:
            Tuple[float, float, float]: (accuracy, avg_loss, f1_macro)
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad(): # Keine Gradients berechnen
            for file_features, labels, sex in data_loader:  # Now includes sex tensor
                file_features = [feat.to(self.device) for feat in file_features]
                labels = labels.to(self.device)
                # sex is not used in evaluation, only for training fairness

                logits = self.model(file_features)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(data_loader)
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        return accuracy, avg_loss, f1_macro

    def _save_experiment_results(self, best_epoch: int, best_metric: float):
        """Save experiment configuration and training history."""
        # Add training results to config
        self.config['results'] = {
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'final_train_loss': self.train_history['train_loss'][-1] if self.train_history['train_loss'] else None,
            'final_train_acc': self.train_history['train_acc'][-1] if self.train_history['train_acc'] else None,
            'final_train_f1': self.train_history['train_f1'][-1] if self.train_history['train_f1'] else None,
            'final_val_loss': self.train_history['val_loss'][-1] if self.train_history['val_loss'] else None,
            'final_val_acc': self.train_history['val_acc'][-1] if self.train_history['val_acc'] else None,
            'final_val_f1': self.train_history['val_f1'][-1] if self.train_history['val_f1'] else None,
        }
        
        # Save config as JSON
        config_path = self.experiment_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
        
        # Save training history as CSV
        history_df = pd.DataFrame(self.train_history)
        history_path = self.experiment_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info(f"Training history saved to {history_path}")
        
        # Save training history plot
        self._save_training_plots()
    
    def _save_training_plots(self):
        """Save training history plots to experiment directory."""
        if not self.train_history['epoch']:
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss plot
        ax1.plot(self.train_history['epoch'], self.train_history['train_loss'], 'b-o', label='Train Loss')
        if self.train_history['val_loss']:
            ax1.plot(self.train_history['epoch'], self.train_history['val_loss'], 'r-o', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_history['epoch'], self.train_history['train_acc'], 'b-o', label='Train Acc')
        if self.train_history['val_acc']:
            ax2.plot(self.train_history['epoch'], self.train_history['val_acc'], 'r-o', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # F1 Score plot
        ax3.plot(self.train_history['epoch'], self.train_history['train_f1'], 'b-o', label='Train F1')
        if self.train_history['val_f1']:
            ax3.plot(self.train_history['epoch'], self.train_history['val_f1'], 'r-o', label='Val F1')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score (Macro)')
        ax3.set_title('Training and Validation F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.experiment_dir / "training_history.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training plots saved to {plot_path}")
    
    def save_evaluation_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: Optional[List[str]] = None,
                               dataset_name: str = "validation"):
        """Save evaluation results including metrics, confusion matrix, and predictions."""
        eval_dir = self.experiment_dir / f"{dataset_name}_results"
        eval_dir.mkdir(exist_ok=True)
        
        # Calculate all metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        # Save metrics as JSON
        metrics_path = eval_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        # Save predictions as CSV
        predictions_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'correct': y_true == y_pred
        })
        predictions_path = eval_dir / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")
        
        # Save classification report
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        report_path = eval_dir / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Classification report saved to {report_path}")
        
        # Save confusion matrix as CSV and plot
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, 
                            index=[f"True_{c}" for c in (class_names or range(len(cm)))],
                            columns=[f"Pred_{c}" for c in (class_names or range(len(cm)))])
        cm_path = eval_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")
        
        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names or range(len(cm)),
                   yticklabels=class_names or range(len(cm)))
        plt.title(f'Confusion Matrix - {dataset_name.capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_plot_path = eval_dir / "confusion_matrix.png"
        plt.savefig(cm_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix plot saved to {cm_plot_path}")
        
        return metrics

    def predict(self, file_paths: List[Path]) -> int:
        """Predict class for a single individual."""
        self.model.eval()

        file_features = []
        for file_path in file_paths:
            audio, sr = sf.read(str(file_path), dtype="float32")
            audio_tensor = torch.from_numpy(audio)
            features = self.feature_extractor.extract(audio_tensor, sr)
            file_features.append(features)

        with torch.no_grad():
            logits = self.model(file_features)
            pred = logits.argmax(dim=1).item()

        return pred
    
    def predict_batch(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for entire dataset (for evaluation)."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for file_features, labels, sex in data_loader:  # Now includes sex tensor
                file_features = [feat.to(self.device) for feat in file_features]
                labels = labels.to(self.device)
                # sex is not used for prediction

                logits = self.model(file_features)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_preds), np.array(all_labels)

    def save(self, path: Union[str, Path]):
        """Save model checkpoint."""
        # If path is relative, save in experiment directory
        if not Path(path).is_absolute():
            path = self.experiment_dir / path
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.file_processors[0].network[0].in_features if not self.model.shared_file_processor else self.model.file_processor.network[0].in_features,
                'num_files': self.model.num_files,
                'num_classes': self.model.num_classes,
                'fusion_type': self.model.fusion_type,
            },
            'train_history': self.train_history,
            'experiment_name': self.experiment_name,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]):
        """Load model checkpoint."""
        # If path is relative, load from experiment directory
        if not Path(path).is_absolute():
            path = self.experiment_dir / path
        
        # Use weights_only=False to load training history and other metadata
        # This is safe for our own checkpoints
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

    def plot_training_history(self):
        """Plot training and validation metrics over epochs."""
        if not self.train_history['epoch']:
            logger.warning("No training history available. Train the model first.")
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Loss plot
        ax1.plot(self.train_history['epoch'], self.train_history['train_loss'], 'b-o', label='Train Loss')
        if self.train_history['val_loss']:
            ax1.plot(self.train_history['epoch'], self.train_history['val_loss'], 'r-o', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(self.train_history['epoch'], self.train_history['train_acc'], 'b-o', label='Train Acc')
        if self.train_history['val_acc']:
            ax2.plot(self.train_history['epoch'], self.train_history['val_acc'], 'r-o', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # F1 Score plot
        ax3.plot(self.train_history['epoch'], self.train_history['train_f1'], 'b-o', label='Train F1')
        if self.train_history['val_f1']:
            ax3.plot(self.train_history['epoch'], self.train_history['val_f1'], 'r-o', label='Val F1')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score (Macro)')
        ax3.set_title('Training and Validation F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # Save plot instead of showing it (for non-interactive environments)
        plot_path = self.experiment_dir / "training_history_final.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Training history plot saved to {plot_path}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: Optional[List[str]] = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save plot instead of showing it (for non-interactive environments)
        plot_path = self.experiment_dir / "confusion_matrix_final.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix plot saved to {plot_path}")

    def print_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                               class_names: Optional[List[str]] = None,
                               train_samples: Optional[List] = None,
                               val_samples: Optional[List] = None):
        """Print detailed evaluation metrics with model configuration."""
        
        print("\n" + "="*80)
        print("📊 LATE FUSION PIPELINE - DETAILED EVALUATION REPORT")
        print("="*80)

        # =====================================================================
        # 1. MODEL CONFIGURATION
        # =====================================================================
        print("\n🔧 MODEL CONFIGURATION:")
        print("-" * 80)
        
        # Feature Extractor
        extractor_name = self.feature_extractor.__class__.__name__
        print(f"{'Feature Extractor':<30} {extractor_name}")
        
        # Fusion Type
        fusion_type = self.model.fusion_type
        print(f"{'Fusion Type':<30} {fusion_type.upper()}")
        
        # Number of files
        num_files = self.model.num_files
        print(f"{'Audio Files per Individual':<30} {num_files}")
        
        # Number of classes
        num_classes = self.model.num_classes
        print(f"{'Number of Classes':<30} {num_classes}")
        
        # Model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{'Total Parameters':<30} {total_params:,}")
        print(f"{'Trainable Parameters':<30} {trainable_params:,}")
        
        # Shared file processor
        shared = self.model.shared_file_processor
        print(f"{'Shared File Processor':<30} {'Yes' if shared else 'No'}")
        
        # Device
        device = self.device
        print(f"{'Device':<30} {device.upper()}")

        # =====================================================================
        # 2. DATASET INFORMATION
        # =====================================================================
        print("\n📁 DATASET INFORMATION:")
        print("-" * 80)
        
        total_val_samples = len(y_true)
        print(f"{'Validation Samples':<30} {total_val_samples}")
        
        # Class distribution in validation set
        unique, counts = np.unique(y_true, return_counts=True)
        print(f"\n{'Class Distribution (Validation)':<30}")
        for cls, count in zip(unique, counts):
            percentage = (count / len(y_true)) * 100
            class_label = class_names[cls] if class_names else str(cls)
            print(f"  Class {class_label:<5} {count:3d} samples ({percentage:5.1f}%)")
        
        # Training set info if provided
        if train_samples is not None:
            print(f"\n{'Training Samples':<30} {len(train_samples)}")
            train_labels = [s.label for s in train_samples]
            train_unique, train_counts = np.unique(train_labels, return_counts=True)
            print(f"{'Class Distribution (Training)':<30}")
            for cls, count in zip(train_unique, train_counts):
                percentage = (count / len(train_labels)) * 100
                class_label = class_names[cls] if class_names else str(cls)
                print(f"  Class {class_label:<5} {count:3d} samples ({percentage:5.1f}%)")

        # =====================================================================
        # 3. OVERALL METRICS
        # =====================================================================
        print("\n" + "="*80)
        print("📊 OVERALL METRICS:")
        print("="*80)

        # Calculate all metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print(f"\n{'Metric':<30} {'Score':<15} {'Interpretation':<30}")
        print("-" * 80)
        
        print(f"{'Accuracy':<30} {accuracy:<15.4f} {self._interpret_score(accuracy):<30}")
        print(f"{'Balanced Accuracy':<30} {balanced_acc:<15.4f} {'(Fair for imbalanced data)':<30}")
        print(f"{'Cohens Kappa':<30} {kappa:<15.4f} {self._interpret_kappa(kappa):<30}")
        
        print(f"\n{'Macro-averaged Metrics':<30}")
        print(f"{'  Precision':<28} {precision_macro:<15.4f}")
        print(f"{'  Recall':<28} {recall_macro:<15.4f}")
        print(f"{'  F1-Score':<28} {f1_macro:<15.4f}")
        
        print(f"\n{'Weighted-averaged Metrics':<30}")
        print(f"{'  Precision':<28} {precision_weighted:<15.4f}")
        print(f"{'  Recall':<28} {recall_weighted:<15.4f}")
        print(f"{'  F1-Score':<28} {f1_weighted:<15.4f}")

        # =====================================================================
        # 4. PER-CLASS METRICS
        # =====================================================================
        print("\n" + "="*80)
        print("📋 PER-CLASS METRICS:")
        print("="*80)
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        # =====================================================================
        # 5. SUMMARY & RECOMMENDATIONS
        # =====================================================================
        print("\n" + "="*80)
        print("💡 SUMMARY & RECOMMENDATIONS:")
        print("="*80)
        
        # Check for class imbalance
        if len(counts) > 1:
            imbalance_ratio = max(counts) / min(counts)
            if imbalance_ratio > 2:
                print(f"\n⚠️  Class Imbalance Detected: Ratio = {imbalance_ratio:.2f}x")
                print("   Recommendation: Consider using weighted loss (already used in CrossEntropyLoss)")
            else:
                print(f"\n✓ Balanced dataset: Ratio = {imbalance_ratio:.2f}x")
        
        # Performance assessment
        print(f"\n📈 Performance Assessment:")
        if f1_macro >= 0.6:
            print(f"   ✓ F1-Macro ({f1_macro:.4f}) meets challenge target!")
        else:
            gap = 0.6 - f1_macro
            print(f"   ⚠️  F1-Macro ({f1_macro:.4f}) - Gap to target: {gap:.4f}")
            print("   Suggestions:")
            print("   - Increase training epochs")
            print("   - Adjust learning rate")
            print("   - Try different feature extractors (HuBERT, WavLM)")
            print("   - Increase model capacity (hidden_dim, file_processor_dim)")

        # Kappa interpretation
        print(f"\n🎯 Agreement Quality (Cohen's Kappa = {kappa:.4f}):")
        if kappa >= 0.81:
            print("   ✓ Almost Perfect Agreement")
        elif kappa >= 0.61:
            print("   ✓ Substantial Agreement")
        elif kappa >= 0.41:
            print("   ⚠️  Moderate Agreement")
        else:
            print("   ⚠️  Fair/Poor Agreement")

        print("\n" + "="*80 + "\n")

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'cohen_kappa': kappa,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted
        }

    @staticmethod
    def _interpret_score(score: float) -> str:
        """Interpret score quality."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        else:
            return "Poor"

    @staticmethod
    def _interpret_kappa(kappa: float) -> str:
        """Interpret Cohen's Kappa."""
        if kappa >= 0.81:
            return "Almost Perfect"
        elif kappa >= 0.61:
            return "Substantial"
        elif kappa >= 0.41:
            return "Moderate"
        elif kappa >= 0.21:
            return "Fair"
        else:
            return "Slight/Poor"

# ============================================================================
# Create Train- and Val-Dataset from Excel
# ============================================================================
def create_dataset_from_excel(sheet_name: str, excel_path: Path) -> List[AudioSample]:
    """
    Create dataset from Excel sheet specification.
    
    Args:
        sheet_name: Name of the Excel sheet (e.g., 'SAND - TRAINING set - Task 1')
        excel_path: Path to the Excel file
    
    Returns:
        List of AudioSample objects
    """
    samples = []
    
    # Definiere die Task-Ordner und deren erwartete Audio-Dateien
    task_dir = Path("data/task1/training")
    
    if not task_dir.exists():
        logger.error(f"Task directory not found: {task_dir}")
        return []
    
    # Ordner-Namen müssen mit Audio-Datei-Namen übereinstimmen
    audio_folders = {
        "phonationA": "phonationA",
        "phonationE": "phonationE",
        "phonationI": "phonationI",
        "phonationO": "phonationO",
        "phonationU": "phonationU",
        "rhythmKA": "rhythmKA",
        "rhythmPA": "rhythmPA",
        "rhythmTA": "rhythmTA",
    }
    
    # Lese Excel-Sheet
    try:
        df_labels = pd.read_excel(excel_path, sheet_name=sheet_name)
        logger.info(f"Loaded {len(df_labels)} individuals from sheet '{sheet_name}'")
    except Exception as e:
        logger.error(f"Error loading Excel sheet '{sheet_name}': {e}")
        return []
    
    # Erstelle Dictionary für schnelle Label-Abfrage: ID -> Class
    id_to_label = dict(zip(df_labels['ID'], df_labels['Class']))
    
    # Erstelle Dictionary für Sex-Abfrage: ID -> Sex (0=male, 1=female, None=unknown)
    id_to_sex = {}
    if 'Sex' in df_labels.columns:
        for idx, row in df_labels.iterrows():
            individual_id = row['ID']
            sex_value = row['Sex']
            
            # Map sex to integer: m/male/M -> 0, w/female/f/F/W -> 1
            if pd.notna(sex_value):
                sex_str = str(sex_value).strip().lower()
                if sex_str in ['m', 'male', 'männlich']:
                    id_to_sex[individual_id] = 0
                elif sex_str in ['w', 'f', 'female', 'weiblich']:
                    id_to_sex[individual_id] = 1
                else:
                    logger.warning(f"Unknown sex value '{sex_value}' for {individual_id}, setting to None")
                    id_to_sex[individual_id] = None
            else:
                id_to_sex[individual_id] = None
    else:
        logger.warning("'Sex' column not found in Excel sheet - all sex values will be None")
    
    # Für jeden Individuum aus dem Sheet: Sammle alle 8 Audio-Dateien
    for individual_id in df_labels['ID']:
        file_paths = []
        
        # Sammle alle 8 Audio-Dateien für diesen Individuum
        for folder_name, file_pattern in audio_folders.items():
            folder_path = task_dir / folder_name
            
            # Suche Datei mit Pattern: ID000_phonationA.wav
            audio_file = folder_path / f"{individual_id}_{file_pattern}.wav"
            
            if audio_file.exists():
                file_paths.append(audio_file)
            else:
                logger.warning(f"Missing audio file: {audio_file}")
                break  # Falls eine Datei fehlt, überspringe diesen Individuum
        
        # Nur hinzufügen, wenn alle 8 Dateien vorhanden sind
        if len(file_paths) == 8:
            label = id_to_label[individual_id]
            
            # Convert label from 1-based (1,2,3,4,5) to 0-based (0,1,2,3,4) for PyTorch
            label = label - 1
            
            # Get sex attribute (can be 0, 1, or None)
            sex = id_to_sex.get(individual_id, None)
            
            sample = AudioSample(
                file_paths=file_paths,
                label=label,
                individual_id=individual_id,
                sex=sex
            )
            samples.append(sample)
            sex_str = "male" if sex == 0 else "female" if sex == 1 else "unknown"
            logger.info(f"Added {individual_id}: label={label}, sex={sex_str}, files={len(file_paths)}")
        else:
            logger.warning(f"Incomplete data for {individual_id}: only {len(file_paths)}/8 files found")
    
    logger.info(f"Total samples from '{sheet_name}': {len(samples)}")
    return samples


def load_train_val_datasets(excel_path: Optional[Path] = None) -> Tuple[List[AudioSample], List[AudioSample]]:
    """
    Load separate training and validation datasets from Excel sheets.
    
    Args:
        excel_path: Path to Excel file. If None, uses default path.
    
    Returns:
        (train_samples, val_samples)
    """
    if excel_path is None:
        # Default path - adjust this to your project structure
        excel_path = Path("data/task1/sand_task_1.xlsx")
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        logger.info("Please provide the correct path to your Excel file.")
        return [], []
    
    # Lade Training-Set
    train_samples = create_dataset_from_excel(
        sheet_name='Training Baseline - Task 1',
        excel_path=excel_path
    )
    
    # Lade Validation-Set
    val_samples = create_dataset_from_excel(
        sheet_name='Validation Baseline - Task 1',
        excel_path=excel_path
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Final Dataset Summary:")
    logger.info(f"  Training samples: {len(train_samples)}")
    logger.info(f"  Validation samples: {len(val_samples)}")
    logger.info(f"  Total samples: {len(train_samples) + len(val_samples)}")
    logger.info(f"{'='*60}\n")
    
    return train_samples, val_samples

def main():
    """Example usage of the late fusion pipeline."""

    # Configuration
    NUM_FILES = 8
    NUM_CLASSES = 5
    BATCH_SIZE = 4
    EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    logger.info(f"Device Selection:")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    logger.info(f"  MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"  MPS built: {torch.backends.mps.is_built()}")
    logger.info(f"  Selected device: {DEVICE}")
    
    if DEVICE == "cpu":
        logger.warning("⚠️  WARNING: Using CPU! Training will be VERY slow (hours instead of minutes)")
        logger.warning("⚠️  For M1/M2 Mac: Install PyTorch with MPS support:")
        logger.warning("⚠️    pip3 install torch torchvision torchaudio")
    
    logger.info(f"Using device: {DEVICE}")

    # 1. Choose and initialize feature extractor
    #logger.info("Initializing feature extractor...")
    #feature_extractor = Wav2Vec2Extractor(
     #   model_name="facebook/wav2vec2-base-960h", #"facebook/wav2vec2-base-960h", "facebook/wav2vec2-large"
      #  pooling="mean",
       # device=DEVICE
    #)

    # feature_extractor = HuBERTExtractor(
    #     model_name="facebook/hubert-large-ll60k",
    #     pooling="mean",
    #     device=DEVICE
    # )

    feature_extractor = WavLMExtractor(
         model_name="microsoft/wavlm-large",
         pooling="mean",
         device=DEVICE
    )

    logger.info("✓ Feature extractor initialized.")

    feature_dim = feature_extractor.get_feature_dim()
    logger.info(f"Feature dimension: {feature_dim}")

    # 2. Load dataset with predefined train/val split from Excel
    logger.info("Loading dataset from Excel sheets...")
    train_samples, val_samples = load_train_val_datasets()
    logger.info("✓ Dataset loaded.")

    if len(train_samples) == 0 or len(val_samples) == 0:
        logger.error("Failed to load train or validation samples!")
        logger.info("\nExpected Excel structure:")
        logger.info("  Sheet 1: 'Training Baseline - Task 1'")
        logger.info("    Columns: ID, Age, Sex, Class")
        logger.info("  Sheet 2: 'Validation Baseline - Task 1'")
        logger.info("    Columns: ID, Age, Sex, Class")
        return

    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # 3. Create datasets and dataloaders
    logger.info("Creating datasets and dataloaders...")
    train_dataset = MultiFileAudioDataset(train_samples, feature_extractor)
    val_dataset = MultiFileAudioDataset(val_samples, feature_extractor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate_fn)
    logger.info("✓ Dataloaders created.")

    # 4. Create model
    logger.info("Creating model...")
    model = LateFusionClassifier(
        input_dim=feature_dim,
        num_files=NUM_FILES,
        num_classes=NUM_CLASSES,
        fusion_type="decision",  # "feature = " or "decision"
        hidden_dim=256,
        file_processor_dim=128,
        dropout=0.3,
        shared_file_processor=True # True or False
    )
    logger.info("✓ Model created.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 5. Create pipeline and train
    pipeline = LateFusionPipeline(feature_extractor, model, device=DEVICE)

    logger.info("Starting training...")
    pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=5e-4,
        weight_decay=1e-4
    )
    logger.info("✓ Training completed.")

    # 6. Plot training history
    logger.info("\n📊 Plotting training history...")
    pipeline.plot_training_history()
    logger.info("✓ Training history plotted.")

    # 7. Detailed evaluation on validation set
    logger.info("\n📋 Evaluating on validation set...")
    val_preds, val_labels = pipeline.predict_batch(val_loader)
    logger.info("✓ Evaluation completed.")

    # Class names basierend auf den Labels
    class_names = [str(i) for i in sorted(set(val_labels))]

    # Print detailed metrics
    metrics = pipeline.print_evaluation_report(
        val_labels, 
        val_preds, 
        class_names=class_names,
        train_samples=train_samples,
        val_samples=val_samples      
    )
    
    # Save evaluation results
    pipeline.save_evaluation_results(val_labels, val_preds, class_names=class_names)

    # Plot confusion matrix
    logger.info("\n📊 Plotting confusion matrix...")
    pipeline.plot_confusion_matrix(val_labels, val_preds, class_names=class_names)
    logger.info("✓ Confusion matrix plotted.")

    # 8. Save model
    logger.info("\n💾Saving model...")
    pipeline.save("final_model.pth")
    logger.info("✓ Model saved.")

    # 9. Example inference
    logger.info("\n🔮 Example predictions on validation set:")
    if len(val_samples) > 0:
        test_sample = val_samples[23]
        prediction = pipeline.predict(test_sample.file_paths)
        logger.info(f"\nExample prediction for {test_sample.individual_id}:")
        logger.info(f"  Predicted: {prediction}, True: {test_sample.label}")

    logger.info("\n" + "="*70)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("="*70)

if __name__ == "__main__":
    main()
