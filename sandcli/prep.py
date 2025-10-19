# Data validation & manifest
"""
Data preparation: scan audio files, build manifest, validate data
"""

import logging
from pathlib import Path
import pandas as pd
from typing import List, Dict, Optional
from omegaconf import DictConfig
from tqdm import tqdm
import soundfile as sf
import regex as re

log = logging.getLogger(__name__)

# Direct mapping of numeric class → descriptive text label
CLASS_MAPPING: Dict[int, str] = {
    1: "ALS_Severe (1)",
    2: "ALS_Moderate (2)",
    3: "ALS_Mild (3)",
    4: "ALS_None (4)",
    5: "Healthy (5)"
}


def extract_subject_id(filename: str) -> str:
    """
    Extract subject ID from filename

    Expected format: <subject_id>_<task>_<timestamp>.wav
    Adjust parsing logic based on actual filename convention
    """
    # Example: "P001_phonationA_001.wav" -> "P001"
    return filename.split('_')[0]


def prepare_label_mapping(metadata_path: Path) -> Optional[Dict[str, str]]:
    """
    Loads the SAND Task 1 Excel metadata once and creates an efficient
    lookup dictionary: {subject_id (str) : text label (str)}.

    Args:
        metadata_path: Path to the Excel metadata file.

    Returns:
        A dictionary mapping subject IDs to textual labels, or None on failure.
    """

    log.info(f"💾 Reading metadata from: {metadata_path}")

    try:
        # 1. Load the Excel sheet containing subject information
        metadata_df = pd.read_excel(
            metadata_path,
            sheet_name="SAND - TRAINING set - Task 1"
        )
    except FileNotFoundError:
        log.error(f"❌ Metadata file not found: {metadata_path}. Aborting.")
        return None
    except Exception as e:
        log.error(f"❌ Critical error reading Excel file: {e}")
        return None

    # 2. Clean and prepare the data
    # Convert the 'Class' column to integer values; invalid entries become NaN
    metadata_df["Class"] = pd.to_numeric(metadata_df["Class"], errors='coerce').astype('Int64')

    # Build the final {ID → Label} dictionary
    id_to_label_map: Dict[str, str] = {}
    for index, row in metadata_df.iterrows():
        subject_id = str(row.get("ID")).strip()
        class_num = row.get("Class")

        # Only include valid entries
        if subject_id.startswith("ID") and pd.notna(class_num):
            label = CLASS_MAPPING.get(int(class_num), "UNKNOWN_CLASS")
            id_to_label_map[subject_id] = label

    if not id_to_label_map:
        log.warning("⚠️ Das Mapping-Dictionary ist leer. Prüfen Sie die Excel-Datei.")
        return None

    log.info(f"✅ Mapping für {len(id_to_label_map)} Subjekte erfolgreich erstellt.")
    return id_to_label_map


def extract_label_from_path(file_path: Path, id_to_label_map: Dict[str, str]) -> str:
    """
    Extracts the corresponding label for a given WAV file using 
    the previously prepared lookup dictionary.

    Args:
        file_path: Path to the WAV file (e.g. Path('data/ID002_phonationA.wav')).
        id_to_label_map: The pre-built dictionary {ID: Label} returned by prepare_label_mapping().

    Returns:
        The corresponding textual label (e.g. 'ALS_Mild') or 'UNKNOWN' if not found.
    """

    match = re.match(r".*(ID\d{3})_.*\.wav$", str(file_path))
    if match:
        file_id = match.group(1)
        return id_to_label_map.get(file_id, "UNKNOWN")
    return "UNKNOWN"


def validate_audio_file(file_path: Path, expected_sr: int = 16000) -> Dict:
    """Validate audio file and return metadata"""
    try:
        info = sf.info(str(file_path))

        return {
            'valid': True,
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'frames': info.frames,
        }
    except Exception as e:
        log.error(f"Error reading {file_path}: {e}")
        return {'valid': False, 'error': str(e)}


def scan_audio_files(root_dir: Path, tasks: List[str], id_to_label_map: Dict[str, str]) -> List[Dict]:
    """Scan all audio files and collect metadata"""
    audio_files = []

    log.info(f"Scanning audio files in {root_dir}")

    for task in tasks:
        task_dir = root_dir / task

        if not task_dir.exists():
            log.warning(f"Task directory not found: {task_dir}")
            continue

        # Find all audio files
        patterns = ['*.wav', '*.flac', '*.mp3']
        files = []
        for pattern in patterns:
            files.extend(task_dir.glob(pattern))

        log.info(f"Found {len(files)} audio files in {task}")

        for file_path in tqdm(files, desc=f"Processing {task}"):
            # Validate audio
            audio_info = validate_audio_file(file_path)

            if not audio_info['valid']:
                log.warning(f"Skipping invalid file: {file_path}")
                continue

            # Extract metadata
            subject_id = extract_subject_id(file_path.name)
            label = extract_label_from_path(file_path, id_to_label_map)  # ← ÜBERGIB MAPPING

            audio_files.append({
                'file_path': str(file_path.relative_to(root_dir.parent)),
                'filename': file_path.name,
                'subject_id': subject_id,
                'task': task,
                'label': label,
                'duration': audio_info['duration'],
                'sample_rate': audio_info['sample_rate'],
                'channels': audio_info['channels'],
            })

    return audio_files


def run(cfg: DictConfig) -> None:
    """
    Main preparation function

    Steps:
    1. Scan all audio files
    2. Validate audio format and quality
    3. Extract metadata (subject ID, task, label)
    4. Build manifest CSV
    5. Generate summary statistics
    """
    log.info("=" * 80)
    log.info("STEP 1: DATA PREPARATION")
    log.info("=" * 80)

    # Paths
    root_dir = Path(cfg.data.root_dir)
    manifest_dir = Path(cfg.data.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Lade Metadaten-Mapping
    metadata_path = Path(cfg.data.get('metadata_file', 'data/task1/metadata.xlsx'))
    id_to_label_map = prepare_label_mapping(metadata_path)

    if id_to_label_map is None:
        log.error("❌ Konnte Metadaten nicht laden. Abbruch.")
        return

    # Scan audio files
    audio_files = scan_audio_files(root_dir, cfg.data.tasks, id_to_label_map)

    if len(audio_files) == 0:
        log.error("No valid audio files found!")
        return

    # Create DataFrame
    df = pd.DataFrame(audio_files)

    # Summary statistics
    log.info("\n" + "=" * 80)
    log.info("DATA SUMMARY")
    log.info("=" * 80)
    log.info(f"Total audio files: {len(df)}")
    log.info(f"Unique subjects: {df['subject_id'].nunique()}")
    log.info(f"Tasks: {df['task'].unique().tolist()}")
    log.info(f"\nLabel distribution:")
    log.info(df['label'].value_counts().to_string())
    log.info(f"\nDuration statistics (seconds):")
    log.info(df['duration'].describe().to_string())
    log.info(f"\nSample rate distribution:")
    log.info(df['sample_rate'].value_counts().to_string())

    # Check for issues
    unknown_labels = df[df['label'] == 'UNKNOWN']
    if len(unknown_labels) > 0:
        log.warning(f"\n⚠️  Found {len(unknown_labels)} files with UNKNOWN labels:")
        log.warning(unknown_labels[['filename', 'subject_id']].head(10).to_string())

    # Save manifest
    manifest_path = manifest_dir / "full_manifest.csv"
    df.to_csv(manifest_path, index=False)
    log.info(f"\n✅ Manifest saved to: {manifest_path}")

    # Additional validation
    log.info("\n" + "=" * 80)
    log.info("VALIDATION CHECKS")
    log.info("=" * 80)

    # Check for duplicate files
    duplicates = df[df.duplicated(subset=['filename'], keep=False)]
    if len(duplicates) > 0:
        log.warning(f"⚠️  Found {len(duplicates)} duplicate filenames")

    # Check sample rate consistency
    if df['sample_rate'].nunique() > 1:
        log.warning(f"⚠️  Multiple sample rates detected: {df['sample_rate'].unique()}")
        log.warning("Consider resampling all audio to a consistent rate")

    # Check for very short/long files
    short_files = df[df['duration'] < 0.5]
    long_files = df[df['duration'] > 30]

    if len(short_files) > 0:
        log.warning(f"⚠️  Found {len(short_files)} very short files (<0.5s)")

    if len(long_files) > 0:
        log.warning(f"⚠️  Found {len(long_files)} very long files (>30s)")

    log.info("\n✅ Data preparation complete!")


if __name__ == "__main__":
    from sandcli.main import cli

    cli()
