# Feature extraction
"""
Feature extraction: MFCC, eGeMAPS, and SSL embeddings
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
import librosa
from scipy import stats
import opensmile
import joblib
from typing import Dict, List

from sandcli.utils.audio import load_audio, normalize_audio, preemphasis, remove_silence

log = logging.getLogger(__name__)


class MFCCExtractor:
    """Extract MFCC features with deltas and statistics"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.features.mfcc
        self.sample_rate = cfg.data.sample_rate

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio"""
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.cfg.n_mfcc,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.hop_length,
            n_mels=self.cfg.n_mels
        )

        features = [mfcc]

        # Add deltas
        if self.cfg.include_deltas:
            delta = librosa.feature.delta(mfcc)
            features.append(delta)

        # Add delta-deltas
        if self.cfg.include_delta_deltas:
            delta2 = librosa.feature.delta(mfcc, order=2)
            features.append(delta2)

        # Concatenate along feature axis
        features = np.vstack(features)

        # Compute statistics over time
        feature_vector = self._compute_statistics(features)

        return feature_vector

    def _compute_statistics(self, features: np.ndarray) -> np.ndarray:
        """Compute statistical functionals over time axis"""
        statistics = []

        for stat_name in self.cfg.get('statistics', ['mean', 'std']):
            if stat_name == 'mean':
                statistics.append(np.mean(features, axis=1))
            elif stat_name == 'std':
                statistics.append(np.std(features, axis=1))
            elif stat_name == 'min':
                statistics.append(np.min(features, axis=1))
            elif stat_name == 'max':
                statistics.append(np.max(features, axis=1))
            elif stat_name == 'skewness':
                statistics.append(stats.skew(features, axis=1))
            elif stat_name == 'kurtosis':
                statistics.append(stats.kurtosis(features, axis=1))

        return np.concatenate(statistics)


class EGeMAPSExtractor:
    """Extract eGeMAPS features using openSMILE"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.features.egemaps
        self.sample_rate = cfg.data.sample_rate

        # Initialize openSMILE
        self.smile = opensmile.Smile(
            feature_set=self.cfg.feature_set,
            feature_level=self.cfg.feature_level,
        )

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extract eGeMAPS features from audio"""
        # openSMILE expects audio as pandas Series or file path
        # We'll pass it as array with sample rate
        features = self.smile.process_signal(audio, self.sample_rate)

        # Convert to numpy array
        feature_vector = features.values.flatten()

        return feature_vector


def extract_features_for_file(
        file_path: str,
        extractors: Dict,
        sample_rate: int,
        preprocess: bool = True
) -> Dict[str, np.ndarray]:
    """Extract all features for a single audio file"""
    try:
        # Load audio
        audio, sr = load_audio(file_path, sample_rate=sample_rate)

        # Preprocessing
        if preprocess:
            audio = normalize_audio(audio, method='peak')
            audio = remove_silence(audio, sr)
            audio = preemphasis(audio)

        # Extract features from each extractor
        features = {}
        for name, extractor in extractors.items():
            try:
                features[name] = extractor.extract(audio)
            except Exception as e:
                log.error(f"Error extracting {name} from {file_path}: {e}")
                features[name] = None

        return features

    except Exception as e:
        log.error(f"Error processing {file_path}: {e}")
        return {name: None for name in extractors.keys()}


def run(cfg: DictConfig) -> None:
    """
    Main feature extraction function

    Steps:
    1. Load train/test manifests
    2. Initialize feature extractors (MFCC, eGeMAPS)
    3. Extract features for all audio files
    4. Save features to disk (NPY format for efficiency)
    5. Create feature manifests
    """
    log.info("=" * 80)
    log.info("STEP 3: FEATURE EXTRACTION")
    log.info("=" * 80)

    # Determine feature type
    feature_type = cfg.features.type
    log.info(f"Feature type: {feature_type}")

    # Paths
    manifest_dir = Path(cfg.data.manifest_dir)
    feature_dir = Path("features")
    feature_dir.mkdir(exist_ok=True)

    # Initialize extractors based on feature type
    extractors = {}

    if 'mfcc' in feature_type.lower():
        log.info("Initializing MFCC extractor...")
        extractors['mfcc'] = MFCCExtractor(cfg)

    if 'egemaps' in feature_type.lower():
        log.info("Initializing eGeMAPS extractor...")
        extractors['egemaps'] = EGeMAPSExtractor(cfg)

    if len(extractors) == 0:
        log.error(f"No valid extractors for feature type: {feature_type}")
        return

    # Process train and test sets
    for split in ['train', 'test']:
        log.info(f"\nProcessing {split} set...")

        manifest_path = manifest_dir / f"{split}_manifest.csv"
        if not manifest_path.exists():
            log.warning(f"Manifest not found: {manifest_path}, skipping")
            continue

        # Load manifest
        df = pd.read_csv(manifest_path)
        log.info(f"Loaded {len(df)} samples")

        # Create output directory for this split
        split_feature_dir = feature_dir / feature_type.replace('+', '_') / split
        split_feature_dir.mkdir(parents=True, exist_ok=True)

        # Extract features
        feature_paths = []
        failed_files = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
            file_path = Path(row['file_path'])

            # Make path absolute by prepending data root
            # Paths in manifest are relative to data root (e.g., "training/phonationA/ID001.wav")
            # We need to resolve them to "data/task1/training/phonationA/ID001.wav"
            if not file_path.is_absolute():
                # Get the parent of the first directory in the path
                # If path is "training/phonationA/ID001.wav", we need "data/task1/"
                root_dir = Path(cfg.data.root_dir).parent  # data/task1/training -> data/task1
                full_path = root_dir / file_path
            else:
                full_path = file_path

            # Extract features
            features = extract_features_for_file(
                str(full_path),
                extractors,
                cfg.data.sample_rate,
                preprocess=True
            )

            # Check if extraction succeeded
            if any(v is None for v in features.values()):
                failed_files.append(file_path)
                feature_paths.append(None)
                continue

            # Concatenate all features
            if len(features) > 1:
                feature_vector = np.concatenate([v for v in features.values()])
            else:
                feature_vector = list(features.values())[0]

            # Save feature vector
            feature_filename = f"{Path(file_path).stem}.npy"
            feature_path = split_feature_dir / feature_filename
            np.save(feature_path, feature_vector)

            feature_paths.append(str(feature_path))

        # Update manifest with feature paths
        df['feature_path'] = feature_paths

        # Remove failed samples
        if failed_files:
            log.warning(f"⚠️  Failed to extract features for {len(failed_files)} files")
            df = df[df['feature_path'].notna()].copy()

        # Save updated manifest
        feature_manifest_path = split_feature_dir / "manifest.csv"
        df.to_csv(feature_manifest_path, index=False)
        log.info(f"✅ Saved feature manifest to {feature_manifest_path}")

        # Log feature statistics
        sample_features = np.load(df.iloc[0]['feature_path'])
        log.info(f"Feature dimension: {sample_features.shape[0]}")

    log.info("\n✅ Feature extraction complete!")


if __name__ == "__main__":
    from sandcli.main import cli

    cli()