# Subject-wise splitting
"""
Subject-wise train/test splitting to prevent data leakage
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig
from collections import Counter

log = logging.getLogger(__name__)


def subject_wise_split(
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify: bool = True,
        random_state: int = 42,
        min_samples: int = 1
) -> tuple:
    """
    Perform subject-wise splitting

    Ensures no subject appears in both train and test sets
    """
    log.info("Performing subject-wise split...")

    # Group by subject
    subject_labels = df.groupby('subject_id')['label'].agg(lambda x: x.mode()[0])
    subject_counts = df.groupby('subject_id').size()

    log.info(f"Total subjects: {len(subject_labels)}")
    log.info(
        f"Samples per subject - min: {subject_counts.min()}, max: {subject_counts.max()}, mean: {subject_counts.mean():.2f}")

    # Filter subjects with minimum samples
    valid_subjects = subject_counts[subject_counts >= min_samples].index
    subject_labels = subject_labels[valid_subjects]

    log.info(f"Subjects after filtering (>={min_samples} samples): {len(subject_labels)}")

    # Check if we have enough samples per class
    label_counts = Counter(subject_labels)
    log.info(f"Subject distribution by label: {dict(label_counts)}")

    # Prepare for stratified split
    subjects = subject_labels.index.tolist()
    labels = subject_labels.values

    # Stratified split if requested and possible
    if stratify and all(count >= 2 for count in label_counts.values()):
        train_subjects, test_subjects = train_test_split(
            subjects,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
        log.info("✅ Stratified split successful")
    else:
        if stratify:
            log.warning("⚠️  Cannot perform stratified split (insufficient samples per class)")
        train_subjects, test_subjects = train_test_split(
            subjects,
            test_size=test_size,
            random_state=random_state
        )
        log.info("✅ Random split performed")

    # Create train/test DataFrames
    train_df = df[df['subject_id'].isin(train_subjects)].copy()
    test_df = df[df['subject_id'].isin(test_subjects)].copy()

    return train_df, test_df


def validate_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Validate that split has no leakage and is well-balanced"""
    log.info("\n" + "=" * 80)
    log.info("SPLIT VALIDATION")
    log.info("=" * 80)

    # Check for subject leakage
    train_subjects = set(train_df['subject_id'].unique())
    test_subjects = set(test_df['subject_id'].unique())
    overlap = train_subjects & test_subjects

    if len(overlap) > 0:
        log.error(f"❌ LEAKAGE DETECTED! {len(overlap)} subjects appear in both train and test:")
        log.error(f"Overlapping subjects: {overlap}")
        raise ValueError("Data leakage detected in split!")
    else:
        log.info("✅ No subject leakage detected")

    # Distribution statistics
    log.info(f"\nTrain set:")
    log.info(f"  - Samples: {len(train_df)}")
    log.info(f"  - Subjects: {len(train_subjects)}")
    log.info(f"  - Label distribution:\n{train_df['label'].value_counts().to_string()}")

    log.info(f"\nTest set:")
    log.info(f"  - Samples: {len(test_df)}")
    log.info(f"  - Subjects: {len(test_subjects)}")
    log.info(f"  - Label distribution:\n{test_df['label'].value_counts().to_string()}")

    # Check for class balance
    train_ratios = train_df['label'].value_counts(normalize=True)
    test_ratios = test_df['label'].value_counts(normalize=True)

    log.info("\nClass proportions:")
    log.info(f"Train: {train_ratios.to_dict()}")
    log.info(f"Test:  {test_ratios.to_dict()}")

    # Warn if distributions are very different
    max_diff = max(abs(train_ratios - test_ratios))
    if max_diff > 0.15:
        log.warning(f"⚠️  Train/test distributions differ significantly (max diff: {max_diff:.2%})")


def run(cfg: DictConfig) -> None:
    """
    Main splitting function

    Steps:
    1. Load full manifest
    2. Perform subject-wise split
    3. Validate split (no leakage, balanced)
    4. Save train/test manifests
    """
    log.info("=" * 80)
    log.info("STEP 2: TRAIN/TEST SPLITTING")
    log.info("=" * 80)

    # Paths
    manifest_dir = Path(cfg.data.manifest_dir)
    full_manifest_path = manifest_dir / "full_manifest.csv"

    if not full_manifest_path.exists():
        log.error(f"Manifest not found: {full_manifest_path}")
        log.error("Please run 'prep' command first")
        return

    # Load manifest
    log.info(f"Loading manifest from {full_manifest_path}")
    df = pd.read_csv(full_manifest_path)

    # Remove samples with UNKNOWN labels
    unknown_mask = df['label'] == 'UNKNOWN'
    if unknown_mask.any():
        log.warning(f"Removing {unknown_mask.sum()} samples with UNKNOWN labels")
        df = df[~unknown_mask].copy()

    # Perform split
    train_df, test_df = subject_wise_split(
        df,
        test_size=cfg.split.test_size,
        stratify=cfg.split.stratify,
        random_state=cfg.seed,
        min_samples=cfg.split.min_samples_per_subject
    )

    # Validate split
    validate_split(train_df, test_df)

    # Save manifests
    train_path = manifest_dir / "train_manifest.csv"
    test_path = manifest_dir / "test_manifest.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    log.info(f"\n✅ Train manifest saved to: {train_path}")
    log.info(f"✅ Test manifest saved to: {test_path}")
    log.info("\n✅ Splitting complete!")


if __name__ == "__main__":
    from sandcli.main import cli

    cli()