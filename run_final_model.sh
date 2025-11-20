#!/bin/bash

# Final Model Training and Inference Script
# This script trains the final model on the complete training dataset
# and generates predictions for the validation set.

echo "=================================="
echo "Final Model Training & Inference"
echo "=================================="
echo ""
echo "This will:"
echo "1. Train the model on ALL 272 training samples"
echo "2. Generate predictions for 67 validation samples"
echo "3. Create a submission CSV file"
echo ""
echo "Expected runtime: 1-3 hours (depending on hardware)"
echo ""

# Run the pipeline with python3
echo "Starting pipeline..."
python3 final_model_pipeline.py

echo ""
echo "=================================="
echo "Pipeline completed!"
echo "Check the final_model/ directory for results"
echo "=================================="
