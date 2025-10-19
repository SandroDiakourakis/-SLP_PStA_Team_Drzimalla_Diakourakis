# Inference
"""
Prediction script for SAND Task 1.
Generates predictions on unlabeled data and exports submission file.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional
import sys
from tqdm import tqdm


def load_model(model_path: Path) -> Any:
    """Load trained model from disk."""
    print(f"Loading model from: {model_path}")

    if model_path.suffix == '.pkl':
        # ML model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    elif model_path.suffix == '.pt':
        # PyTorch model
        import torch
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        return model
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")


def load_features(
        manifest_path: Path,
        feature_dir: Path
) -> Tuple[np.ndarray, list]:
    """
    Load features for prediction.

    Args:
        manifest_path: Path to manifest CSV
        feature_dir: Directory containing feature arrays

    Returns:
        X: Feature matrix
        file_ids: List of file identifiers
    """
    # Load manifest
    manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(manifest)} samples")

    X_list = []
    file_ids = []

    print("Loading features...")
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest)):
        file_path = Path(row['file_path'])
        file_id = file_path.stem

        # Load feature array
        feature_path = feature_dir / f"{file_id}.npy"

        if not feature_path.exists():
            print(f"Warning: Feature not found for {file_id}, skipping...")
            continue

        features = np.load(feature_path)

        # Handle 2D features
        if len(features.shape) > 1:
            features_flat = np.concatenate([
                np.mean(features, axis=0),
                np.std(features, axis=0),
                np.min(features, axis=0),
                np.max(features, axis=0)
            ])
        else:
            features_flat = features

        X_list.append(features_flat)
        file_ids.append(file_id)

    X = np.array(X_list)
    print(f"Loaded features: {X.shape}")

    return X, file_ids


def predict_and_save(
        model: Any,
        X: np.ndarray,
        file_ids: list,
        label_encoder_path: Path,
        output_path: Path,
        include_probabilities: bool = False
) -> pd.DataFrame:
    """
    Generate predictions and save to CSV.

    Args:
        model: Trained model
        X: Feature matrix
        file_ids: List of file identifiers
        label_encoder_path: Path to label encoder
        output_path: Path to save predictions CSV
        include_probabilities: If True, include class probabilities

    Returns:
        DataFrame with predictions
    """
    # Load label encoder
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    print("Generating predictions...")
    y_pred_encoded = model.predict(X)
    y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

    # Create DataFrame
    results = pd.DataFrame({
        'id': file_ids,
        'label': y_pred_labels
    })

    # Add probabilities if requested
    if include_probabilities and hasattr(model, 'predict_proba'):
        print("Computing class probabilities...")
        y_proba = model.predict_proba(X)

        for i, class_name in enumerate(label_encoder.classes_):
            results[f'prob_{class_name}'] = y_proba[:, i]

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"✓ Predictions saved to: {output_path}")

    # Print summary
    print("\nPrediction summary:")
    print(results['label'].value_counts().sort_index())

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for SAND Task 1"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model file (.pkl or .pt)'
    )

    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to manifest CSV (test or validation set)'
    )

    parser.add_argument(
        '--feature_dir',
        type=str,
        required=True,
        help='Directory containing extracted features'
    )

    parser.add_argument(
        '--label_encoder',
        type=str,
        required=True,
        help='Path to label encoder pickle file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save predictions CSV'
    )

    parser.add_argument(
        '--include_probabilities',
        action='store_true',
        help='Include class probabilities in output'
    )

    args = parser.parse_args()

    # Convert to Path objects
    model_path = Path(args.model_path)
    manifest_path = Path(args.manifest)
    feature_dir = Path(args.feature_dir)
    label_encoder_path = Path(args.label_encoder)
    output_path = Path(args.output)

    # Validate inputs
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    if not feature_dir.exists():
        print(f"Error: Feature directory not found at {feature_dir}")
        sys.exit(1)

    if not label_encoder_path.exists():
        print(f"Error: Label encoder not found at {label_encoder_path}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("SAND TASK 1 - PREDICTION")
    print(f"{'=' * 60}\n")

    # Load model
    model = load_model(model_path)

    # Load features
    X, file_ids = load_features(manifest_path, feature_dir)

    # Generate predictions
    results = predict_and_save(
        model, X, file_ids, label_encoder_path, output_path,
        include_probabilities=args.include_probabilities
    )

    print(f"\n{'=' * 60}")
    print("PREDICTION COMPLETE")
    print(f"Total predictions: {len(results)}")
    print(f"Output saved to: {output_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()