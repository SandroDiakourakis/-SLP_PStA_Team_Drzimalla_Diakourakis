"""
Final Model Training and Inference Pipeline

This script:
1. Loads the best model configuration from experiment_20251116_132709
2. Trains the model on the COMPLETE training dataset (no train/test split)
3. Runs inference on the validation dataset
4. Generates a submission CSV file with predictions in the required format:
   - Columns: ID, CLASS
   - IDs from sand_task1_test.xlsx
   - CLASS predictions as integers [1-5]
"""

# Fix for TensorFlow import issues on server
import os
import sys

# CRITICAL: Remove system TensorFlow paths BEFORE any imports
sys.path = [p for p in sys.path if '/usr/lib/python3/dist-packages' not in p]

# Set environment variables to suppress TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_NO_TF'] = '1'  # Tell transformers to not use TensorFlow

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import logging
from torch.utils.data import DataLoader

# Add the notebooks directory to the path to import the pipeline
sys.path.insert(0, str(Path(__file__).parent / 'notebooks' / 'late_fusion_from_ideation_to_claude'))

from claude_output_late_fusion_pipeline import (
    LateFusionPipeline,
    LateFusionClassifier,
    AudioSample,
    Wav2Vec2Extractor,
    HuBERTExtractor,
    WavLMExtractor,
    MultiFileAudioDataset,
    collate_fn
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinalModelTrainer:
    """Train final model on complete training set and generate predictions for validation set."""
    
    def __init__(self, experiment_dir: Path, training_data_dir: Path, 
                 validation_data_dir: Path, test_metadata_path: Path):
        """
        Args:
            experiment_dir: Path to the best experiment directory (e.g., experiment_20251116_132709)
            training_data_dir: Path to the training data directory
            validation_data_dir: Path to the validation data directory
            test_metadata_path: Path to sand_task1_test.xlsx
        """
        self.experiment_dir = experiment_dir
        self.training_data_dir = training_data_dir
        self.validation_data_dir = validation_data_dir
        self.test_metadata_path = test_metadata_path
        self.config = None
        self.pipeline = None
        
    def load_config(self) -> Dict:
        """Load configuration from the best experiment."""
        config_path = self.experiment_dir / 'config.json'
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        logger.info(f"Loaded config for {self.config['experiment_name']}")
        logger.info(f"Best epoch: {self.config['results']['best_epoch']}")
        logger.info(f"Best F1-macro: {self.config['results']['best_metric']:.4f}")
        
        return self.config
    
    def load_training_samples(self) -> List[AudioSample]:
        """Load ALL training samples (no split) from training directory and sand_task_1.xlsx."""
        logger.info("Loading training samples from complete dataset...")
        
        # Load metadata from Excel
        df = pd.read_excel(
            self.training_data_dir.parent / 'sand_task_1.xlsx',
            sheet_name='SAND - TRAINING set - Task 1'
        )
        
        logger.info(f"Found {len(df)} samples in training metadata")
        
        samples = []
        file_types = ['phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
                     'rhythmKA', 'rhythmPA', 'rhythmTA']
        
        missing_files = []
        for idx, row in df.iterrows():
            sample_id = row['ID']
            age = row['Age']
            sex = row['Sex']
            label = int(row['Class']) - 1  # Convert to 0-indexed
            
            # Collect paths for all 8 audio files
            file_paths = []
            all_exist = True
            
            for file_type in file_types:
                file_path = self.training_data_dir / file_type / f"{sample_id}_{file_type}.wav"
                if file_path.exists():
                    file_paths.append(file_path)  # Keep as Path object
                else:
                    all_exist = False
                    missing_files.append(str(file_path))
                    break
            
            if all_exist:
                sample = AudioSample(
                    file_paths=file_paths,
                    label=label,
                    individual_id=sample_id,
                    sex=1 if sex == 'F' else 0 if sex == 'M' else None
                )
                samples.append(sample)
        
        if missing_files:
            logger.warning(f"Missing {len(missing_files)} files, examples: {missing_files[:5]}")
        
        logger.info(f"Successfully loaded {len(samples)} complete training samples")
        
        # Log class distribution
        class_counts = {}
        for sample in samples:
            class_counts[sample.label] = class_counts.get(sample.label, 0) + 1
        logger.info(f"Class distribution: {dict(sorted(class_counts.items()))}")
        
        return samples
    
    def load_validation_samples(self) -> List[AudioSample]:
        """Load validation samples from validation directory and sand_task1_test.xlsx."""
        logger.info("Loading validation samples...")
        
        # Load metadata from test Excel
        df = pd.read_excel(
            self.test_metadata_path,
            sheet_name='SAND - TESTING set - Task 1'
        )
        
        logger.info(f"Found {len(df)} samples in test metadata")
        
        samples = []
        file_types = ['phonationA', 'phonationE', 'phonationI', 'phonationO', 'phonationU',
                     'rhythmKA', 'rhythmPA', 'rhythmTA']
        
        missing_files = []
        for idx, row in df.iterrows():
            # IDs in test file are integers (4, 11, 14, ...), need to format as ID004, ID011, etc.
            sample_id_num = int(row['ID'])
            sample_id = f"ID{sample_id_num:03d}"
            age = row['Age']
            sex = row['Sex']
            
            # Collect paths for all 8 audio files
            file_paths = []
            all_exist = True
            
            for file_type in file_types:
                file_path = self.validation_data_dir / file_type / f"{sample_id}_{file_type}.wav"
                if file_path.exists():
                    file_paths.append(file_path)  # Keep as Path object
                else:
                    all_exist = False
                    missing_files.append(str(file_path))
                    break
            
            if all_exist:
                sample = AudioSample(
                    file_paths=file_paths,
                    label=-1,  # No label for test set
                    individual_id=sample_id,
                    sex=1 if sex == 'F' else 0 if sex == 'M' else None
                )
                samples.append(sample)
        
        if missing_files:
            logger.warning(f"Missing {len(missing_files)} files, examples: {missing_files[:5]}")
        
        logger.info(f"Successfully loaded {len(samples)} validation samples")
        
        return samples
    
    def create_pipeline(self) -> LateFusionPipeline:
        """Create pipeline with the same configuration as the best experiment."""
        logger.info("Creating pipeline with best experiment configuration...")
        
        # Create feature extractor based on config
        extractor_config = self.config['feature_extractor']
        extractor_type = extractor_config['type']
        
        if extractor_type == 'Wav2Vec2Extractor':
            feature_extractor = Wav2Vec2Extractor(
                model_name=extractor_config['model_name'],
                pooling="mean",
                layers=extractor_config['layers'],
                layer_fusion=extractor_config['layer_fusion']
            )
        elif extractor_type == 'HuBERTExtractor':
            feature_extractor = HuBERTExtractor(
                model_name=extractor_config['model_name'],
                pooling="mean",
                layers=extractor_config['layers'],
                layer_fusion=extractor_config['layer_fusion']
            )
        elif extractor_type == 'WavLMExtractor':
            feature_extractor = WavLMExtractor(
                model_name=extractor_config['model_name'],
                pooling="mean",
                layers=extractor_config['layers'],
                layer_fusion=extractor_config['layer_fusion']
            )
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        
        # Create model
        training_config = self.config['training']
        model_config = self.config['model']
        
        model = LateFusionClassifier(
            input_dim=feature_extractor.get_feature_dim(),
            num_files=model_config['num_files'],
            num_classes=model_config['num_classes'],
            fusion_type=model_config['fusion_type'],
            shared_file_processor=model_config['shared_file_processor']
        )
        
        # Create pipeline
        self.pipeline = LateFusionPipeline(
            feature_extractor=feature_extractor,
            model=model,
            device=self.config['device'],
            experiment_name="final_model"
        )
        
        logger.info(f"Pipeline created with {model_config['total_parameters']:,} parameters")
        
        return self.pipeline
    
    def train_final_model(self, train_samples: List[AudioSample], output_dir: Path):
        """Train the final model on the complete training set."""
        logger.info("=" * 80)
        logger.info("TRAINING FINAL MODEL ON COMPLETE TRAINING SET")
        logger.info("=" * 80)
        
        # Use same training config as best experiment
        training_config = self.config['training']
        
        logger.info(f"Training configuration:")
        logger.info(f"  Epochs: {training_config['epochs']}")
        logger.info(f"  Learning rate: {training_config['lr']}")
        logger.info(f"  Weight decay: {training_config['weight_decay']}")
        logger.info(f"  Batch size: {training_config['physical_batch_size']}")
        logger.info(f"  Early stopping patience: {training_config['early_stopping_patience']}")
        
        # Create DataLoader for training
        logger.info(f"Creating DataLoader with batch_size={training_config['physical_batch_size']}...")
        train_dataset = MultiFileAudioDataset(
            samples=train_samples,
            feature_extractor=self.pipeline.feature_extractor,
            target_sr=16000,
            augmentation=None,
            is_training=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['physical_batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Create minimal validation set (5 samples) to avoid pipeline errors
        # The pipeline expects val_loader to save history correctly
        # Using 5 samples (one per class) for more stable validation metrics
        logger.info(f"Creating minimal validation DataLoader (5 samples for technical reasons)...")
        
        # Select one sample per class for validation
        val_samples_minimal = []
        classes_seen = set()
        for sample in train_samples:
            if sample.label not in classes_seen:
                val_samples_minimal.append(sample)
                classes_seen.add(sample.label)
            if len(val_samples_minimal) == 5:
                break
        
        logger.info(f"Selected {len(val_samples_minimal)} samples for minimal validation (classes: {sorted(classes_seen)})")
        
        val_dataset = MultiFileAudioDataset(
            samples=val_samples_minimal,
            feature_extractor=self.pipeline.feature_extractor,
            target_sr=16000,
            augmentation=None,
            is_training=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Train with minimal validation set (just for technical reasons, not for evaluation)
        results = self.pipeline.train(
            train_loader=train_loader,
            val_loader=val_loader,  # Minimal val set to avoid history save errors
            epochs=training_config['epochs'],
            lr=training_config['lr'],
            weight_decay=training_config['weight_decay'],
            max_grad_norm=training_config['max_grad_norm'],
            early_stopping_patience=training_config['early_stopping_patience'],
            early_stopping_metric=training_config['early_stopping_metric'],
            lambda_fair=training_config.get('lambda_fair', 0.0),
            accumulation_steps=training_config.get('accumulation_steps', 1)
        )
        
        logger.info("=" * 80)
        logger.info("FINAL MODEL TRAINING COMPLETED")
        logger.info("=" * 80)
        
        return results
    
    def generate_predictions(self, validation_samples: List[AudioSample], 
                           output_csv: Path):
        """Generate predictions for validation samples and save to CSV."""
        logger.info("=" * 80)
        logger.info("GENERATING PREDICTIONS FOR VALIDATION SET")
        logger.info("=" * 80)
        
        # Create DataLoader for validation (batch_size=1 to preserve order)
        logger.info(f"Creating DataLoader for {len(validation_samples)} validation samples...")
        val_dataset = MultiFileAudioDataset(
            samples=validation_samples,
            feature_extractor=self.pipeline.feature_extractor,
            target_sr=16000,
            augmentation=None,
            is_training=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Get predictions
        logger.info(f"Running inference...")
        predictions, _ = self.pipeline.predict_batch(val_loader)
        
        # predictions are 0-indexed (0-4), need to convert to 1-indexed (1-5)
        predictions = [pred + 1 for pred in predictions]
        
        # Load test metadata to get original ID format
        df = pd.read_excel(
            self.test_metadata_path,
            sheet_name='SAND - TESTING set - Task 1'
        )
        
        # Create submission dataframe
        # Match samples to IDs from the metadata
        submission_data = []
        for sample, pred in zip(validation_samples, predictions):
            # Extract numeric ID from sample ID (e.g., ID004 -> 4)
            sample_id_num = int(sample.individual_id.replace('ID', ''))
            submission_data.append({
                'ID': sample_id_num,
                'CLASS': pred
            })
        
        submission_df = pd.DataFrame(submission_data)
        
        # Sort by ID to match the order in the metadata file
        submission_df = submission_df.sort_values('ID').reset_index(drop=True)
        
        # Verify we have the correct number of predictions
        expected_count = len(df)
        actual_count = len(submission_df)
        
        if actual_count != expected_count:
            logger.warning(f"Prediction count mismatch: expected {expected_count}, got {actual_count}")
        else:
            logger.info(f"✓ Prediction count matches test set: {actual_count}")
        
        # Verify all IDs are present
        expected_ids = set(df['ID'].tolist())
        actual_ids = set(submission_df['ID'].tolist())
        missing_ids = expected_ids - actual_ids
        extra_ids = actual_ids - expected_ids
        
        if missing_ids:
            logger.warning(f"Missing IDs: {sorted(missing_ids)}")
        if extra_ids:
            logger.warning(f"Extra IDs: {sorted(extra_ids)}")
        if not missing_ids and not extra_ids:
            logger.info(f"✓ All IDs match test set")
        
        # Verify class range
        min_class = submission_df['CLASS'].min()
        max_class = submission_df['CLASS'].max()
        logger.info(f"Class range: [{min_class}, {max_class}]")
        
        if min_class < 1 or max_class > 5:
            logger.warning(f"CLASS values outside expected range [1, 5]")
        else:
            logger.info(f"✓ All CLASS values in valid range [1, 5]")
        
        # Log class distribution
        class_dist = submission_df['CLASS'].value_counts().sort_index()
        logger.info(f"Prediction class distribution:\n{class_dist}")
        
        # Save to CSV with exact format: ID,CLASS
        submission_df.to_csv(output_csv, index=False)
        logger.info(f"✓ Submission file saved to: {output_csv}")
        
        # Display first few rows
        logger.info(f"\nFirst 10 predictions:")
        logger.info(f"\n{submission_df.head(10).to_string(index=False)}")
        
        logger.info("=" * 80)
        logger.info("PREDICTION GENERATION COMPLETED")
        logger.info("=" * 80)
        
        return submission_df


def main():
    """Main execution function."""
    
    # Paths configuration
    BASE_DIR = Path(__file__).parent
    EXPERIMENT_DIR = BASE_DIR / 'experiments' / 'experiments' / 'experiment_20251116_132709'
    TRAINING_DATA_DIR = BASE_DIR / 'data' / 'task1' / 'training'
    VALIDATION_DATA_DIR = BASE_DIR / 'data' / 'task1' / 'validation'
    TEST_METADATA_PATH = BASE_DIR / 'data' / 'task1' / 'sand_task1_test.xlsx'
    
    # Output directory for final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = BASE_DIR / 'final_model' / f'final_model_{timestamp}'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    SUBMISSION_CSV = OUTPUT_DIR / 'submission.csv'
    
    logger.info("=" * 80)
    logger.info("FINAL MODEL TRAINING AND INFERENCE PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Experiment directory: {EXPERIMENT_DIR}")
    logger.info(f"Training data: {TRAINING_DATA_DIR}")
    logger.info(f"Validation data: {VALIDATION_DATA_DIR}")
    logger.info(f"Test metadata: {TEST_METADATA_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("=" * 80)
    
    # Create trainer
    trainer = FinalModelTrainer(
        experiment_dir=EXPERIMENT_DIR,
        training_data_dir=TRAINING_DATA_DIR,
        validation_data_dir=VALIDATION_DATA_DIR,
        test_metadata_path=TEST_METADATA_PATH
    )
    
    # Step 1: Load configuration
    logger.info("\n[STEP 1/5] Loading configuration from best experiment...")
    config = trainer.load_config()
    
    # Step 2: Load training samples (complete dataset)
    logger.info("\n[STEP 2/5] Loading complete training dataset...")
    train_samples = trainer.load_training_samples()
    
    # Step 3: Load validation samples
    logger.info("\n[STEP 3/5] Loading validation dataset...")
    validation_samples = trainer.load_validation_samples()
    
    # Step 4: Create pipeline and train final model
    logger.info("\n[STEP 4/5] Training final model on complete training set...")
    pipeline = trainer.create_pipeline()
    results = trainer.train_final_model(train_samples, OUTPUT_DIR)
    
    # Step 5: Generate predictions and submission file
    logger.info("\n[STEP 5/5] Generating predictions for validation set...")
    submission_df = trainer.generate_predictions(validation_samples, SUBMISSION_CSV)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Final model saved to: {OUTPUT_DIR}")
    logger.info(f"Submission file: {SUBMISSION_CSV}")
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(validation_samples)}")
    logger.info(f"Predictions generated: {len(submission_df)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
