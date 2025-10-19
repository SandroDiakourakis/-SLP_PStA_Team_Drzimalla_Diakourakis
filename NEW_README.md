# 🎯 SAND Challenge Task 1 - Multi-Class Classification Pipeline

**Fully local, cost-free implementation for neurodegenerative disease detection from speech.**

Designed for **Apple M2 Max MacBook Pro** with no cloud dependencies or paid APIs.

---

## 📋 Overview

This pipeline implements both **classical machine learning** (Phase 1) and **self-supervised deep learning** (Phase 2) approaches for SAND Challenge Task 1.

### Primary Metric
**Macro F1-Score** - ensures balanced performance across all disease severity classes.

### Key Features
✅ Subject-wise train/test splitting (no data leakage)  
✅ Class imbalance handling (weighted loss, scale_pos_weight)  
✅ MFCC and eGeMAPS feature extraction  
✅ XGBoost and LightGBM with hyperparameter tuning  
✅ SSL models: WavLM, HuBERT, Wav2Vec2  
✅ Comprehensive evaluation and visualization  
✅ Reproducible (deterministic seeding)  

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sand-task1.git
cd sand-task1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install package
pip install -e .

# Verify installation
sandcli --help
```

### 2. Prepare Data

Place your SAND Task 1 training data in:
```
data/task1/training/
├── phonationA/
├── phonationE/
├── phonationI/
├── phonationO/
├── phonationU/
├── rhythmKA/
├── rhythmPA/
└── rhythmTA/
```

### 3. Run Pipeline (Phase 1 - ML)

```bash
# Step 1: Create manifest
python -m sandcli.main command=prep

# Step 2: Subject-wise split
python -m sandcli.main command=split

# Step 3: Extract MFCC features
python -m sandcli.main command=featurize

# Step 4: Train XGBoost
python -m sandcli.main command=train

# Step 5: Evaluate
python -m sandcli.main command=eval \
    model_path=runs/baseline_ml_xgboost_YYYYMMDD_HHMMSS/model.pkl \
    manifest=data/manifests/test_manifest.csv \
    feature_dir=features/mfcc_egemaps/test \
    label_encoder=runs/baseline_ml_xgboost_YYYYMMDD_HHMMSS/label_encoder.pkl \
    output_dir=runs/baseline_ml_xgboost_YYYYMMDD_HHMMSS/eval

# Step 6: Generate predictions
python -m sandcli.main command=predict \
    model_path=runs/baseline_ml_xgboost_YYYYMMDD_HHMMSS/model.pkl \
    manifest=data/manifests/test_manifest.csv \
    feature_dir=features/mfcc_egemaps/test \
    label_encoder=runs/baseline_ml_xgboost_YYYYMMDD_HHMMSS/label_encoder.pkl \
    output=runs/baseline_ml_xgboost_YYYYMMDD_HHMMSS/predictions.csv
```
---

## 📂 Project Structure

```
sand_task1/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
│
├── conf/                        # Hydra configuration files
│   ├── config.yaml              # Base config
│   ├── baseline_ml.yaml         # XGBoost/LightGBM
│   ├── ssl_wavlm.yaml           # WavLM
│   ├── ssl_hubert.yaml          # HuBERT
│   └── ssl_wav2vec2.yaml        # Wav2Vec2
│
├── data/                        # Data directory
│   ├── task1/training/          # Audio files
│   └── manifests/               # CSV manifests
│
├── features/                    # Extracted features
│   ├── mfcc/
│   ├── egemaps/
│   └── ssl_embeddings/
│
├── sandcli/                     # Main CLI package
│   ├── main.py                  # Entry point
│   ├── prep.py                  # Data preparation
│   ├── split.py                 # Train/test splitting
│   ├── featurize.py             # Feature extraction
│   ├── train.py                 # Model training
│   ├── eval.py                  # Evaluation
│   ├── predict.py               # Inference
│   ├── ensemble.py              # Model ensembling
│   └── utils/                   # Utilities
│       ├── audio.py
│       ├── metrics.py
│       └── visualization.py
│
├── models/                      # Model implementations
│   ├── ml_models.py             # XGBoost, LightGBM
│   ├── ssl_classifier.py        # SSL models
│   └── ensemble_model.py
│
├── runs/                        # Experiment outputs
└── notebooks/                   # Jupyter notebooks
```

---

## 🔧 Configuration

All configurations use **Hydra** for easy experimentation.

### Example: Modify XGBoost Parameters

Edit `conf/baseline_ml.yaml`:

```yaml
model:
  params:
    max_depth: 8              # Increase tree depth
    learning_rate: 0.03       # Lower learning rate
    n_estimators: 500         # More trees
```

Then run:
```bash
python -m sandcli.main command=train model.params.max_depth=10
```

---

## 📊 Evaluation Metrics

### Primary Metric
- **Macro F1-Score**: Average of per-class F1 scores (handles class imbalance)

### Additional Metrics
- Accuracy
- Per-class Precision, Recall, F1
- Confusion Matrix (raw and normalized)
- ROC-AUC (if probabilities available)

### Outputs
Each experiment generates:
```
runs/experiment_YYYYMMDD_HHMMSS/
├── config.yaml                      # Experiment config
├── model.pkl                        # Trained model
├── label_encoder.pkl                # Label encoder
├── metrics.json                     # All metrics
├── confusion_matrix.png             # Confusion matrix plot
├── confusion_matrix_normalized.png
├── per_class_f1.png                 # Per-class F1 bar chart
├── per_class_precision.png
├── per_class_recall.png
├── predictions.csv                  # Predictions with probabilities
├── feature_importance.csv           # Feature importances (ML models)
└── results_summary.md               # Human-readable summary
```

---

## 🎓 Phase 1: Classical ML

### Models
- **XGBoost**: Gradient boosting with histogram-based algorithm
- **LightGBM**: Fast, memory-efficient boosting

### Features
- **MFCC**: Mel-frequency cepstral coefficients (13 coefficients + deltas)
- **eGeMAPS**: Extended Geneva Minimalistic Acoustic Parameter Set (88 features)

### Class Balancing
- `scale_pos_weight` (XGBoost)
- Sample weighting (LightGBM)

### Cross-Validation
5-fold stratified CV for robust evaluation.

---

## 🧠 Phase 2: Self-Supervised Learning (SSL)

### Supported Models
- **WavLM-Base** (microsoft/wavlm-base)
- **HuBERT-Base** (facebook/hubert-base-ls960)
- **Wav2Vec2-Base** (facebook/wav2vec2-base)

### Training Modes
1. **Frozen backbone** → Train only classifier head
2. **Fine-tuning** → Update backbone + classifier

### Loss Functions
- Weighted Cross-Entropy
- Focal Loss (for severe imbalance)

---

## 🐛 Troubleshooting

### Common Issues

**1. openSMILE installation fails**
```bash
# Install via conda instead
conda install -c conda-forge opensmile
```

**2. MPS (Apple Silicon GPU) errors**
```bash
# Fallback to CPU in config.yaml
device: "cpu"
```

**3. Out of memory during SSL training**
```bash
# Reduce batch size in ssl_wavlm.yaml
training:
  batch_size: 8  # or lower
```

**4. Label mismatch errors**
Check that your data follows the expected structure and that `prep.py` correctly extracts subject IDs.

---

## 📈 Expected Performance

### Phase 1 (XGBoost + MFCC+eGeMAPS)
- **Macro F1**: ~0.65-0.75 (baseline)
- Training time: ~5-10 minutes on M2 Max

### Phase 2 (WavLM fine-tuning)
- **Macro F1**: ~0.75-0.85 (target)
- Training time: ~1-2 hours on M2 Max

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact:
- **Email**: your.email@example.com
- **SAND Challenge**: [Official Website](https://sand-challenge.org)

---

## 🙏 Acknowledgments

- SAND Challenge organizers
- Hugging Face for pretrained models
- scikit-learn, XGBoost, LightGBM communities
- openSMILE team for acoustic features

---

**Built with ❤️ for the neurodegenerative disease research community**