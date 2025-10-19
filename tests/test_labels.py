"""
Unit tests for label preparation and extraction functions
used in the SAND Task 1 baseline pipeline.

Run with:
    pytest -v tests/test_labels.py
"""

from pathlib import Path
import pandas as pd
import tempfile
import pytest
import sys

# Ensure the project root (parent of 'tests') is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sandcli.prep import prepare_label_mapping, extract_label_from_path, CLASS_MAPPING


# -----------------------------------------------------------------------------
# Helper: create a temporary fake Excel metadata file for testing
# -----------------------------------------------------------------------------
@pytest.fixture
def fake_metadata_file(tmp_path: Path) -> Path:
    """
    Creates a temporary Excel file with a small subset of SAND Task 1 metadata
    for testing the label mapping functions.
    """
    data = {
        "ID": ["ID000", "ID001", "ID002", "ID003", "ID004"],
        "Age": [80, 61, 51, 44, 70],
        "Sex": ["M", "F", "F", "M", "F"],
        "Class": [5, 5, 4, 1, 3],
    }

    df = pd.DataFrame(data)
    file_path = tmp_path / "sand_task_1.xlsx"
    df.to_excel(file_path, index=False, sheet_name="SAND - TRAINING set - Task 1")

    return file_path


# -----------------------------------------------------------------------------
# Tests for prepare_label_mapping()
# -----------------------------------------------------------------------------
def test_prepare_label_mapping_valid(fake_metadata_file):
    """✅ Should correctly create mapping dictionary from Excel metadata"""
    mapping = prepare_label_mapping(fake_metadata_file)
    assert mapping is not None, "Mapping should not be None"
    assert isinstance(mapping, dict), "Mapping should be a dictionary"
    assert mapping["ID000"] == "Healthy (5)"
    assert mapping["ID003"] == "ALS_Severe (1)"
    assert len(mapping) == 5


def test_prepare_label_mapping_missing_file(tmp_path):
    """⚠️ Should return None and log an error if file not found"""
    missing_file = tmp_path / "nonexistent.xlsx"
    mapping = prepare_label_mapping(missing_file)
    assert mapping is None


def test_prepare_label_mapping_invalid_class(fake_metadata_file):
    """⚠️ Should handle invalid (non-numeric) class entries gracefully"""
    # Overwrite Excel with an invalid class value
    df = pd.DataFrame({"ID": ["ID999"], "Class": ["invalid"]})
    with pd.ExcelWriter(fake_metadata_file, mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, index=False, sheet_name="SAND - TRAINING set - Task 1")

    mapping = prepare_label_mapping(fake_metadata_file)
    # Even with invalid entry, function should still return a dict (empty or partial)
    assert isinstance(mapping, (dict, type(None)))


# -----------------------------------------------------------------------------
# Tests for extract_label_from_path()
# -----------------------------------------------------------------------------
def test_extract_label_from_path_valid(tmp_path, fake_metadata_file):
    """✅ Should correctly extract labels for valid file paths"""
    id_to_label_map = prepare_label_mapping(fake_metadata_file)

    # Create fake audio file paths
    file_healthy = tmp_path / "ID000_phonationA.wav"
    file_none = tmp_path / "ID002_phonationE.wav"
    file_severe = tmp_path / "ID003_phonationI.wav"

    assert extract_label_from_path(file_healthy, id_to_label_map) == "Healthy (5)"
    assert extract_label_from_path(file_none, id_to_label_map) == "ALS_None (4)"
    assert extract_label_from_path(file_severe, id_to_label_map) == "ALS_Severe (1)"


def test_extract_label_from_path_unknown_id(tmp_path, fake_metadata_file):
    """⚠️ Should return 'UNKNOWN' for IDs not present in metadata"""
    id_to_label_map = prepare_label_mapping(fake_metadata_file)
    file_unknown = tmp_path / "ID999_phonationA.wav"
    result = extract_label_from_path(file_unknown, id_to_label_map)
    assert result == "UNKNOWN"


def test_extract_label_from_path_invalid_filename(tmp_path, fake_metadata_file):
    """⚠️ Should return 'UNKNOWN' if filename has no valid ID"""
    id_to_label_map = prepare_label_mapping(fake_metadata_file)
    file_invalid = tmp_path / "subjectA_phonation.wav"
    result = extract_label_from_path(file_invalid, id_to_label_map)
    assert result == "UNKNOWN"


def test_class_mapping_integrity():
    """✅ CLASS_MAPPING dictionary should cover all 5 classes"""
    expected_keys = {1, 2, 3, 4, 5}
    assert set(CLASS_MAPPING.keys()) == expected_keys
    assert "ALS_Severe (1)" in CLASS_MAPPING.values()
