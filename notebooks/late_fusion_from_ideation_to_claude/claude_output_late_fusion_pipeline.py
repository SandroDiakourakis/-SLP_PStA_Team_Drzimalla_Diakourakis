# TODO Loss in NN mit einbauen, weil bei Trainingsdaten 3 Frauen und 1 Mann und im Testset 1 Mann/Frau
# f1-macro war bei dem Evaluationset bei der Challenge bei 0.6 -> in die Nähe sollen wir auch kommen
"""
Late Fusion Pipeline for Multi-File Audio Classification

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
import matplotlib.pyplot as plt
import seaborn as sns

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
            outputs = self.model(audio)
            # features = outputs.last_hidden_state  # (batch, time, hidden_dim)
            features = outputs[:,:,12]# (batch, time, hidden_dim)

            # Pool over time dimension
            if self.pooling == "mean":
                pooled = features.mean(dim=1)
            # elif self.pooling == "max":
            #     pooled = features.max(dim=1)[0]
            # elif self.pooling == "first":
            #     pooled = features[:, 0, :]
            # elif self.pooling == "last":
            #     pooled = features[:, -1, :]
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
            outputs = self.model(audio)
            # features = outputs.last_hidden_state
            features = outputs[:,:,12]# (batch, time, hidden_dim)

            if self.pooling == "mean":
                pooled = features.mean(dim=1)
            # elif self.pooling == "max":
            #     pooled = features.max(dim=1)[0]
            # elif self.pooling == "first":
            #     pooled = features[:, 0, :]
            # elif self.pooling == "last":
            #     pooled = features[:, -1, :]
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
            outputs = self.model(audio)
            # features = outputs.last_hidden_state
            features = outputs[:,:,12]# (batch, time, hidden_dim)

            if self.pooling == "mean":
                pooled = features.mean(dim=1)
            # elif self.pooling == "max":
            #     pooled = features.max(dim=1)[0]
            # elif self.pooling == "first":
            #     pooled = features[:, 0, :]
            # elif self.pooling == "last":
            #     pooled = features[:, -1, :]
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

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int]:
        """
        Returns:
            (file_features, label):
                - file_features: List of feature tensors, one per file
                - label: Class label
        """
        sample = self.samples[idx]
        file_features = []

        for file_path in sample.file_paths:
            # Load audio
            audio, sr = sf.read(str(file_path), dtype="float32")

            # Resample if needed (simplified - use librosa for proper resampling)
            if sr != self.target_sr:
                logger.warning(f"Sample rate mismatch: {sr} vs {self.target_sr}. "
                             "Consider using librosa.resample()")

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio)

            # Extract features
            features = self.feature_extractor.extract(audio_tensor, sr)
            file_features.append(features.squeeze(0))  # Remove batch dim

        return file_features, sample.label

def collate_fn(batch: List[Tuple[List[torch.Tensor], int]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Custom collate function for batching multi-file samples."""
    num_files = len(batch[0][0])
    batch_size = len(batch)

    # Organize by file index
    batched_features = []
    for file_idx in range(num_files):
        file_batch = torch.stack([sample[0][file_idx] for sample in batch])
        batched_features.append(file_batch)

    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)

    return batched_features, labels


# ============================================================================
# Training Pipeline
# ============================================================================

class LateFusionPipeline:
    """Complete pipeline for late fusion classification."""

    def __init__(self, feature_extractor: FeatureExtractor, model: LateFusionClassifier,
                 device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"):
        self.feature_extractor = feature_extractor
        self.model = model.to(device)
        self.device = device

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 10, lr: float = 1e-3, weight_decay: float = 1e-4):
        """Train the model."""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        for epoch in range(epochs):
            # === TRAINING ===
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for file_features, labels in train_loader:
                # Move to device
                file_features = [feat.to(self.device) for feat in file_features]
                labels = labels.to(self.device)

                # Forward
                logits = self.model(file_features)
                loss = criterion(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                preds = logits.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # === VALIDATION ===
            if val_loader is not None:
                val_acc, val_loss = self.evaluate(val_loader)
                scheduler.step(val_acc)

                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                          f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}")

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): # Keine Gradients berechnen
            for file_features, labels in data_loader:
                file_features = [feat.to(self.device) for feat in file_features]
                labels = labels.to(self.device)

                logits = self.model(file_features)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        # TODO F1-average Score hinzufügen, falls ungleich avg_loss

        return accuracy, avg_loss

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
            for file_features, labels in data_loader:
                file_features = [feat.to(self.device) for feat in file_features]
                labels = labels.to(self.device)

                logits = self.model(file_features)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_preds), np.array(all_labels)

    def save(self, path: Union[str, Path]):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.file_processors[0].network[0].in_features if not self.model.shared_file_processor else self.model.file_processor.network[0].in_features,
                'num_files': self.model.num_files,
                'num_classes': self.model.num_classes,
                'fusion_type': self.model.fusion_type,
            }
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

    def plot_training_history(self):
        """Plot training and validation metrics over epochs."""
        if not self.train_history['epoch']:
            logger.warning("No training history available. Train the model first.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(self.train_history['epoch'], self.train_history['train_loss'], 'b-o', label='Train Loss')
        ax1.plot(self.train_history['epoch'], self.train_history['val_loss'], 'r-o', label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(self.train_history['epoch'], self.train_history['train_acc'], 'b-o', label='Train Acc')
        ax2.plot(self.train_history['epoch'], self.train_history['val_acc'], 'r-o', label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

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
        plt.show()

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
        print(f"{'Cohen\'s Kappa':<30} {kappa:<15.4f} {self._interpret_kappa(kappa):<30}")
        
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
        "rythmKA": "rythmKA",
        "rythmPA": "rythmPA",
        "rythmTA": "rythmTA",
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
            
            sample = AudioSample(
                file_paths=file_paths,
                label=label,
                individual_id=individual_id
            )
            samples.append(sample)
            logger.info(f"Added {individual_id}: label={label}, files={len(file_paths)}")
        else:
            logger.warning(f"Incomplete data for {individual_id}: only {len(file_paths)}/8 files found")
    
    logger.info(f"Total samples from '{sheet_name}': {len(samples)}")
    return samples


def load_train_val_datasets() -> Tuple[List[AudioSample], List[AudioSample]]:
    """
    Load separate training and validation datasets from Excel sheets.
    
    Returns:
        (train_samples, val_samples)
    """
    excel_path = Path("/Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis/data/task1/sand_task_1.xlsx")
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
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

    logger.info(f"Using device: {DEVICE}")

    # 1. Choose and initialize feature extractor
    logger.info("Initializing feature extractor...")
    feature_extractor = Wav2Vec2Extractor(
        model_name="facebook/wav2vec2-base-960h", # or use default "facebook/wav2vec2-large"
        pooling="mean",
        device=DEVICE
    )
    logger.info("✓ Feature extractor initialized.")

    # feature_extractor = HuBERTExtractor(
    #     model_name="facebook/hubert-large-ll60k",
    #     pooling="mean",
    #     device=DEVICE
    # )
    # feature_extractor = WavLMExtractor(
    #     model_name="microsoft/wavlm-large",
    #     pooling="mean",
    #     device=DEVICE
    # )

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
        fusion_type="feature",  # or "decision"
        hidden_dim=256,
        file_processor_dim=128,
        dropout=0.3,
        shared_file_processor=False
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
        lr=1e-3,
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

    # Plot confusion matrix
    logger.info("\n📊 Plotting confusion matrix...")
    pipeline.plot_confusion_matrix(val_labels, val_preds, class_names=class_names)
    logger.info("✓ Confusion matrix plotted.")

    # 8. Save model
    logger.info("\n💾Saving model...")
    pipeline.save("late_fusion_model.pth")
    logger.info("✓ Model saved.")

    # 9. Example inference
    logger.info("\n🔮 Example predictions on validation set:")
    if len(val_samples) > 0:
        test_sample = val_samples[0]
        prediction = pipeline.predict(test_sample.file_paths)
        logger.info(f"\nExample prediction for {test_sample.individual_id}:")
        logger.info(f"  Predicted: {prediction}, True: {test_sample.label}")

    logger.info("\n" + "="*70)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("="*70)

if __name__ == "__main__":
    main()
