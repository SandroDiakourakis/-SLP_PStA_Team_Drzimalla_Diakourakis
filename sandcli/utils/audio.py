# Audio loading utilities
"""
Audio loading and preprocessing utilities
"""

import logging
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional

log = logging.getLogger(__name__)


def load_audio(
        file_path: str,
        sample_rate: int = 16000,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        duration: Load only specified duration (seconds)
        offset: Start reading after this time (seconds)

    Returns:
        audio: Audio time series as numpy array
        sr: Sample rate
    """
    try:
        audio, sr = librosa.load(
            file_path,
            sr=sample_rate,
            mono=mono,
            duration=duration,
            offset=offset
        )
        return audio, sr
    except Exception as e:
        log.error(f"Error loading {file_path}: {e}")
        raise


def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
    """
    Normalize audio signal

    Args:
        audio: Input audio signal
        method: Normalization method ('peak', 'rms', 'lufs')

    Returns:
        Normalized audio
    """
    if method == 'peak':
        # Normalize to [-1, 1] based on peak amplitude
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio

    elif method == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            return audio / rms
        return audio

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter to audio signal

    Emphasizes high frequencies for better feature extraction
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def remove_silence(
        audio: np.ndarray,
        sr: int,
        top_db: int = 30,
        frame_length: int = 2048,
        hop_length: int = 512
) -> np.ndarray:
    """
    Remove silent portions from audio

    Args:
        audio: Input audio
        sr: Sample rate
        top_db: Threshold for silence detection (dB below peak)
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis

    Returns:
        Audio with silence removed
    """
    # Trim leading and trailing silence
    audio_trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )

    return audio_trimmed


def augment_audio(
        audio: np.ndarray,
        sr: int,
        time_stretch: Optional[Tuple[float, float]] = None,
        pitch_shift: Optional[Tuple[int, int]] = None,
        add_noise: bool = False,
        noise_factor: float = 0.005
) -> np.ndarray:
    """
    Apply data augmentation to audio

    Args:
        audio: Input audio
        sr: Sample rate
        time_stretch: Range for time stretching (e.g., (0.9, 1.1))
        pitch_shift: Range for pitch shifting in semitones (e.g., (-2, 2))
        add_noise: Whether to add random noise
        noise_factor: Magnitude of noise to add

    Returns:
        Augmented audio
    """
    augmented = audio.copy()

    # Time stretching
    if time_stretch is not None:
        rate = np.random.uniform(*time_stretch)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)

    # Pitch shifting
    if pitch_shift is not None:
        n_steps = np.random.randint(*pitch_shift)
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

    # Add noise
    if add_noise:
        noise = np.random.randn(len(augmented))
        augmented = augmented + noise_factor * noise

    return augmented


def pad_or_truncate(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate audio to target length

    Args:
        audio: Input audio
        target_length: Target number of samples

    Returns:
        Audio with target length
    """
    current_length = len(audio)

    if current_length < target_length:
        # Pad with zeros
        padding = target_length - current_length
        audio = np.pad(audio, (0, padding), mode='constant')
    elif current_length > target_length:
        # Truncate
        audio = audio[:target_length]

    return audio