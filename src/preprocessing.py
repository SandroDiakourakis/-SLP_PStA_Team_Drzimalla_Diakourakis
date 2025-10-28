"""
Feature Extraction und Preprocessing für ALS Audio-Daten
"""

import numpy as np
import pandas as pd
import librosa
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def extract_audio_features(audio_path, n_mfcc=13):
    """
    Extrahiert Features aus einer einzelnen Audio-Datei

    Parameters:
    -----------
    audio_path : str or Path
        Pfad zur .wav Datei
    n_mfcc : int
        Anzahl der MFCC Koeffizienten

    Returns:
    --------
    dict : Dictionary mit allen Features oder None bei Fehler
    """
    features = {}

    try:
        # === LIBROSA Features ===
        y, sr = librosa.load(audio_path, sr=None)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for i in range(n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])

        # Spektrale Features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_std'] = np.std(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)

        # Pitch Features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=400)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_min'] = np.min(pitch_values)
            features['pitch_max'] = np.max(pitch_values)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_min'] = 0
            features['pitch_max'] = 0
            features['pitch_range'] = 0

        # === PARSELMOUTH/PRAAT Features ===
        sound = parselmouth.Sound(str(audio_path))

        # Point Process für Jitter/Shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 400)

        # Jitter
        try:
            features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_rap'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            features['jitter_ppq5'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        except:
            features['jitter_local'] = 0
            features['jitter_rap'] = 0
            features['jitter_ppq5'] = 0

        # Shimmer
        try:
            features['shimmer_local'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3,
                                             1.6)
            features['shimmer_apq3'] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_apq5'] = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except:
            features['shimmer_local'] = 0
            features['shimmer_apq3'] = 0
            features['shimmer_apq5'] = 0

        # Harmonics-to-Noise Ratio (HNR)
        try:
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            features['hnr_mean'] = call(harmonicity, "Get mean", 0, 0)
            features['hnr_std'] = call(harmonicity, "Get standard deviation", 0, 0)
        except:
            features['hnr_mean'] = 0
            features['hnr_std'] = 0

        # Formanten
        try:
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            features['f1_mean'] = call(formant, "Get mean", 1, 0, 0, "hertz")
            features['f1_std'] = call(formant, "Get standard deviation", 1, 0, 0, "hertz")
            features['f2_mean'] = call(formant, "Get mean", 2, 0, 0, "hertz")
            features['f2_std'] = call(formant, "Get standard deviation", 2, 0, 0, "hertz")
            features['f3_mean'] = call(formant, "Get mean", 3, 0, 0, "hertz")
            features['f3_std'] = call(formant, "Get standard deviation", 3, 0, 0, "hertz")
        except:
            features['f1_mean'] = features['f1_std'] = 0
            features['f2_mean'] = features['f2_std'] = 0
            features['f3_mean'] = features['f3_std'] = 0

        # Intensity
        try:
            intensity = call(sound, "To Intensity", 75, 0.0, "yes")
            features['intensity_mean'] = call(intensity, "Get mean", 0, 0, "energy")
            features['intensity_std'] = call(intensity, "Get standard deviation", 0, 0)
        except:
            features['intensity_mean'] = 0
            features['intensity_std'] = 0

    except Exception as e:
        print(f"Fehler bei {audio_path}: {str(e)}")
        return None

    return features


def load_and_extract_features(data_path, task='task1', phonation_type='A', n_mfcc=13, verbose=True):
    """
    Lädt Daten und extrahiert Features für einen Task

    Parameters:
    -----------
    data_path : str or Path
        Pfad zum data Ordner
    task : str
        'task1' oder 'task2'
    phonation_type : str
        Typ der Phonation ('A', 'E', 'I', 'O', 'U')
    n_mfcc : int
        Anzahl der MFCC Koeffizienten
    verbose : bool
        Ausgabe von Fortschritt

    Returns:
    --------
    tuple : (X, y) - Features und Labels als numpy arrays
    """
    data_path = Path(data_path)
    task_path = data_path / task

    # Excel-Datei laden
    excel_files = list(task_path.glob('*.xlsx'))
    if not excel_files:
        raise FileNotFoundError(f"Keine .xlsx Datei in {task_path} gefunden")

    df_labels = pd.read_excel(excel_files[0])

    if verbose:
        print(f"📊 Lade {task} mit Phonation {phonation_type}")
        print(f"   Labels aus: {excel_files[0].name}")
        print(f"   Anzahl Patienten: {len(df_labels)}")

    # Audio-Ordner
    audio_dir = task_path / 'training' / f'phonation{phonation_type}'

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio-Ordner nicht gefunden: {audio_dir}")

    # Features extrahieren
    data = []
    failed_files = []

    for idx, row in df_labels.iterrows():
        patient_id = row['ID']

        # Audio-Datei finden
        audio_file = audio_dir / f"{patient_id}_phonation{phonation_type}.wav"

        if not audio_file.exists():
            failed_files.append(audio_file.name)
            continue

        if verbose and (idx + 1) % 20 == 0:
            print(f"   Verarbeitet: {idx + 1}/{len(df_labels)}")

        # Features extrahieren
        features = extract_audio_features(audio_file, n_mfcc=n_mfcc)

        if features is not None:
            # Metadaten hinzufügen
            features['Age'] = row['Age']

            # Sex encodieren
            features['Sex'] = 0 if row['Sex'] == 'M' else 1

            # Label hinzufügen
            if 'Class' in row:
                features['label'] = row['Class']
            elif 'ALSFRS--R_start' in row:
                # Für task2: verwende Start-Score als Label
                features['label'] = row['ALSFRS--R_start']

            data.append(features)

    if len(failed_files) > 0 and verbose:
        print(f"⚠️  Fehlgeschlagen: {len(failed_files)} Dateien")

    # DataFrame erstellen
    df = pd.DataFrame(data)

    if verbose:
        print(f"✅ Erfolgreich: {len(df)}/{len(df_labels)} Dateien")
        if 'label' in df.columns:
            print(f"   Klassenverteilung:")
            for cls, count in sorted(df['label'].value_counts().items()):
                print(f"   Klasse {cls}: {count}")

    # Features und Labels trennen
    y = df['label'].values
    X = df.drop(['label'], axis=1).values

    if verbose:
        print(f"   Feature-Dimension: {X.shape}")

    return X, y


def preprocess_data(X, y=None, scaler=None, fit_scaler=True):
    """
    Skaliert Features mit StandardScaler

    Parameters:
    -----------
    X : np.ndarray
        Feature Matrix
    y : np.ndarray, optional
        Labels
    scaler : StandardScaler, optional
        Vortrainierter Scaler (für Test-Daten)
    fit_scaler : bool
        Ob der Scaler gefittet werden soll

    Returns:
    --------
    tuple : (X_scaled, y, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler