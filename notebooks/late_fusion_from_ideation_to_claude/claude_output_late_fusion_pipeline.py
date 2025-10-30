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


# ============================================================================
# Example Usage
# ============================================================================

import pandas as pd  # Füge am Anfang hinzu

def create_example_dataset() -> List[AudioSample]:
    """
    Create dataset from your custom structure.

    Expected directory structure:
        data/task1/training/
            phonationA/
                ID000_phonationA.wav
                ID001_phonationA.wav
                ...
            phonationE/
                ID000_phonationE.wav
                ...
            ... (insgesamt 8 Ordner)
    
    Labels werden aus der Excel-Datei geladen.
    """
    samples = []
    
    # Laden der Label aus Excel
    excel_path = Path("/Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis/data/task1/sand_task_1.xlsx")
    
    if not excel_path.exists():
        logger.error(f"Excel file not found: {excel_path}")
        return []
    
    # Lese Excel-Datei
    try:
        df_labels = pd.read_excel(excel_path, sheet_name='SAND - TRAINING set - Task 1')
        logger.info(f"Loaded labels for {len(df_labels)} individuals")
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        return []
    
    # Erstelle Dictionary für schnelle Label-Abfrage: ID -> Class
    id_to_label = dict(zip(df_labels['ID'], df_labels['Class']))
    
    # Definiere die Task-Ordner und deren erwartete Audio-Dateien
    task_dir = Path("data/task1/training")
    
    if not task_dir.exists():
        logger.error(f"Task directory not found: {task_dir}")
        return []
    
    # Ordner-Namen müssen mit Audio-Datei-Namen übereinstimmen
    # Z.B. Ordner "phonationA" → Datei "ID000_phonationA.wav"
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
    
    # Sammle alle eindeutigen IDs aus den Dateien
    all_ids = set()
    
    for folder_name in audio_folders.keys():
        folder_path = task_dir / folder_name
        if folder_path.exists():
            for wav_file in folder_path.glob("*.wav"):
                # Extrahiere ID aus Dateinamen (z.B. "ID000_phonationA.wav" → "ID000")
                file_stem = wav_file.stem  # "ID000_phonationA"
                id_part = file_stem.split("_")[0]  # "ID000"
                all_ids.add(id_part)
    
    logger.info(f"Found {len(all_ids)} individuals in audio files")
    
    # Für jeden Individuum: Sammle alle 8 Audio-Dateien
    for individual_id in sorted(all_ids):
        # Prüfe ob Label in Excel existiert
        if individual_id not in id_to_label:
            logger.warning(f"No label found for {individual_id}, skipping...")
            continue
        
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
            logger.warning(f"Incomplete data for {individual_id}: only {len(file_paths)}/8 files")
    
    logger.info(f"Total samples created: {len(samples)}")
    return samples

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
    # Alternative: HuBERTExtractor(...) or WavLMExtractor(...)

    feature_dim = feature_extractor.get_feature_dim()
    logger.info(f"Feature dimension: {feature_dim}")

    # 2. Create dataset
    logger.info("Loading dataset...")
    samples = create_example_dataset()

    if len(samples) == 0:
        logger.warning("No samples found. Please prepare your data.")
        logger.info("\nExpected directory structure:")
        logger.info("data/task1/training/")
        logger.info("    phonationA/")
        logger.info("        ID000_phonationA.wav")
        logger.info("        ID001_phonationA.wav")
        logger.info("        ...")
        logger.info("    phonationE/")
        logger.info("        ID000_phonationE.wav")
        logger.info("        ...")
        logger.info("    ... (insgesamt 8 Ordner)")
        return

    # Split into train/val
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # Create datasets and dataloaders
    train_dataset = MultiFileAudioDataset(train_samples, feature_extractor)
    val_dataset = MultiFileAudioDataset(val_samples, feature_extractor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate_fn)

    # 3. Create model
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

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Create pipeline and train
    pipeline = LateFusionPipeline(feature_extractor, model, device=DEVICE)

    logger.info("Starting training...")
    pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=1e-3,
        weight_decay=1e-4
    )
    logger.info("Training completed.")

    # 5. Save model
    logger.info("Saving model...")
    pipeline.save("late_fusion_model.pth")

    # 6. Example inference
    if len(val_samples) > 0:
        test_sample = val_samples[0]
        prediction = pipeline.predict(test_sample.file_paths)
        logger.info(f"\nExample prediction for {test_sample.individual_id}:")
        logger.info(f"  Predicted: {prediction}, True: {test_sample.label}")


if __name__ == "__main__":
    main()
