# Final Model Training and Inference Pipeline

This pipeline trains the final model using the complete training dataset and generates predictions for the validation set (test set) in the correct submission format.

## Overview

Based on the best performing experiment (`experiment_20251116_132709`) which achieved:
- **F1-macro: 0.594**
- **Accuracy: 71.7%**
- Best epoch: 28

## Pipeline Steps

1. **Load Configuration**: Loads the exact configuration from the best experiment
2. **Train on Complete Dataset**: Trains on all 272 training samples (no train/validation split)
3. **Generate Predictions**: Runs inference on 67 validation samples
4. **Create Submission File**: Generates `submission.csv` in the required format

## Files

- `final_model_pipeline.py` - Main pipeline script
- `test_final_pipeline.py` - Test script to verify setup
- `run_final_model.sh` - Convenient shell script to run the pipeline
- `FINAL_MODEL_README.md` - This file

## Data Structure

### Training Data
- **Location**: `data/task1/training/`
- **Metadata**: `data/task1/sand_task_1.xlsx` (sheet: "SAND - TRAINING set - Task 1")
- **Samples**: 272 individuals
- **Files per sample**: 8 audio files
  - Vowels: phonationA, phonationE, phonationI, phonationO, phonationU
  - Syllables: rhythmKA, rhythmPA, rhythmTA
- **Classes**: 5 classes (1-5)
  - Class 1: 6 samples
  - Class 2: 26 samples
  - Class 3: 57 samples
  - Class 4: 76 samples
  - Class 5: 107 samples

### Validation Data (Test Set)
- **Location**: `data/task1/validation/`
- **Metadata**: `data/task1/sand_task1_test.xlsx` (sheet: "SAND - TESTING set - Task 1")
- **Samples**: 67 individuals
- **Files per sample**: 8 audio files (same structure as training)

## Model Configuration

Based on `experiment_20251116_132709`:

### Feature Extraction
- **Model**: Wav2Vec 2.0 Large (facebook/wav2vec2-large-960h)
- **Layers**: [6, 9, 12, 15, 18] (multi-layer feature extraction)
- **Fusion**: Concatenation
- **Feature dim**: 5120

### Model Architecture
- **Type**: Late Fusion (feature-level)
- **Files**: 8 files per individual
- **Classes**: 5
- **Parameters**: 11,021,317 (all trainable)
- **File processor**: Not shared (separate processor for each file type)

### Training Configuration
- **Epochs**: 40
- **Learning rate**: 0.0005
- **Weight decay**: 0.0001
- **Batch size**: 24
- **Early stopping**: 10 epochs patience (on F1-macro)
- **Gradient clipping**: max norm 1.0

## Usage

### Option 1: Using the shell script (recommended)
```bash
./run_final_model.sh
```

### Option 2: Direct Python execution
```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Run the pipeline
python final_model_pipeline.py
```

### Option 3: Test setup first
```bash
# Test that all data can be loaded correctly
python test_final_pipeline.py

# Then run the main pipeline
python final_model_pipeline.py
```

## Output

The pipeline creates a directory `final_model/final_model_YYYYMMDD_HHMMSS/` containing:

1. **submission.csv** - Predictions in submission format
   - Columns: `ID`, `CLASS`
   - 67 rows (one per test sample)
   - IDs match those in `sand_task1_test.xlsx`
   - CLASS values are integers [1-5]

2. **best_model_checkpoint.pth** - Best model weights during training

3. **final_model.pth** - Final model weights after all epochs

4. **training_history.csv** - Training metrics per epoch

5. **config.json** - Complete configuration used for training

## Submission File Format

The `submission.csv` file follows the exact requirements:

```csv
ID,CLASS
4,4
11,2
14,3
19,1
20,5
...
```

### Format Requirements ✓
- ✅ File extension: `.csv`
- ✅ Two columns: `ID` and `CLASS` (uppercase)
- ✅ Separator: comma (`,`)
- ✅ 67 rows (excluding header)
- ✅ IDs match test set exactly
- ✅ CLASS values are integers in range [1, 5]

## Runtime

Expected runtime: **1-3 hours** (depending on hardware)
- Training: ~1-2 hours for 40 epochs
- Inference: ~10-20 minutes for 67 samples

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (will use MPS on Mac, CPU otherwise)
- **RAM**: At least 16 GB recommended
- **Disk**: ~2 GB for model checkpoints and results

## Dependencies

All dependencies are already installed in the virtual environment:
- torch
- transformers
- librosa
- pandas
- openpyxl
- tqdm
- scikit-learn
- matplotlib

## Troubleshooting

### Issue: Out of memory during training
**Solution**: The pipeline will automatically reduce batch size if OOM occurs. If it still fails, you can manually edit `final_model_pipeline.py` and reduce the batch size in the config.

### Issue: Missing audio files
**Solution**: Run `test_final_pipeline.py` to check which files are missing. All 272 training samples and 67 validation samples should have all 8 audio files.

### Issue: Import errors
**Solution**: Make sure you're in the project root directory and the virtual environment is activated:
```bash
cd /Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis
source venv/bin/activate
```

## Validation Results (from original experiment)

The model configuration being used achieved these results on a held-out validation set:

- **Accuracy**: 71.7%
- **Balanced Accuracy**: 59.5%
- **F1-Macro**: 0.594
- **Cohen's Kappa**: 0.605
- **Precision (macro)**: 0.593
- **Recall (macro)**: 0.595

## Notes

- The pipeline trains on the **complete training set** (all 272 samples) without holding out a validation set
- This maximizes the amount of training data and should improve performance on the test set
- The model uses the exact same architecture and hyperparameters that achieved F1-macro = 0.594 in the original experiment
- Predictions are made on the validation set which the model has never seen during training

## Next Steps

After running the pipeline:

1. Check the generated `submission.csv` file in the output directory
2. Verify the format matches requirements (run checks in the log output)
3. Submit the file to the challenge platform
4. The training history and model checkpoints are saved for future reference
