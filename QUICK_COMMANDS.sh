#!/bin/bash
# Quick Command Reference - SAND Task 1 Pipeline

# ===========================================
# SETUP
# ===========================================

# Virtual Environment aktivieren
source venv/bin/activate

# Dependencies installieren
pip install -r requirements.txt

# Umgebung testen
./test_pipeline.sh

# ===========================================
# PIPELINE BEFEHLE (Richtige Syntax!)
# ===========================================

# ❌ FALSCH (funktioniert nicht):
# python sandcli/main.py --dry-run command=prep

# ✅ RICHTIG:

# Config anzeigen (statt --dry-run)
python -m sandcli.main --cfg job

# 1. Data Preparation
python -m sandcli.main command=prep

# 2. Train/Test Split
python -m sandcli.main command=split

# 3. Feature Extraction
python -m sandcli.main command=featurize features.type=mfcc

# 4. Model Training (mit baseline config)
python -m sandcli.main --config-name=baseline_ml command=train

# ===========================================
# PARAMETER ÜBERSCHREIBEN
# ===========================================

# Seed ändern
python -m sandcli.main command=prep seed=123

# Experiment-Name ändern
python -m sandcli.main command=train experiment_name="my_experiment"

# Test-Size ändern (30% statt 20%)
python -m sandcli.main command=split split.test_size=0.3

# Verschiedene Features
python -m sandcli.main command=featurize features.type=egemaps
python -m sandcli.main command=featurize features.type=mfcc+egemaps

# Model-Parameter ändern
python -m sandcli.main command=train \
  model.name=xgboost \
  model.params.max_depth=10 \
  model.params.n_estimators=500

# ===========================================
# VERSCHIEDENE CONFIGS VERWENDEN
# ===========================================

# Default config.yaml
python -m sandcli.main command=train

# Baseline ML config
python -m sandcli.main --config-name=baseline_ml command=train

# SSL configs (Phase 2)
python -m sandcli.main --config-name=ssl_wav2vec2 command=train

# ===========================================
# DEBUGGING & VALIDATION
# ===========================================

# Zeige vollständige Config
python -m sandcli.main --cfg all

# Zeige nur Job-Config
python -m sandcli.main --cfg job

# Zeige Hydra-Config
python -m sandcli.main --cfg hydra

# Python-Module testen
python3 -c "from sandcli import prep, split, featurize, train; print('OK')"

# Manifests überprüfen
ls -lh data/manifests/
head data/manifests/full_manifest.csv

# Features überprüfen
ls -lh features/mfcc/train/ | head

# Experimente überprüfen
ls -ltr runs/

# ===========================================
# COMPLETE END-TO-END EXAMPLE
# ===========================================

# Aktiviere Umgebung
source venv/bin/activate

# Schritt für Schritt
python -m sandcli.main command=prep
python -m sandcli.main command=split
python -m sandcli.main command=featurize features.type=mfcc
python -m sandcli.main --config-name=baseline_ml command=train

# Ergebnisse anschauen
python3 << 'EOF'
import json
from pathlib import Path

runs = sorted(Path("runs").glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)
if runs:
    latest = runs[0]
    with open(latest / "metrics.json") as f:
        m = json.load(f)
    print(f"Latest: {latest.name}")
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"Macro F1: {m['macro_f1']:.4f}")
EOF

