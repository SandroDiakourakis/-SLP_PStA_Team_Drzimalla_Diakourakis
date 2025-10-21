# SAND Task 1 Pipeline - Vollständiger Leitfaden

## 📋 Inhaltsverzeichnis
1. [Gefundene Probleme & Lösungen](#gefundene-probleme--lösungen)
2. [Umgebung Setup](#umgebung-setup)
3. [Pipeline-Befehle](#pipeline-befehle)
4. [Validierung & Tests](#validierung--tests)
5. [Training & Evaluation](#training--evaluation)
6. [Troubleshooting](#troubleshooting)

---

## 🔍 Gefundene Probleme & Lösungen

### Problem 1: `--dry-run` Flag existiert nicht
**Fehler:** `main.py: error: unrecognized arguments: --dry-run`

**Grund:** Hydra unterstützt keinen `--dry-run` Flag. Sie verwenden Hydra als CLI-Framework, daher müssen Sie die Hydra-Syntax verwenden.

**Lösung:** 
- Um die Config anzuzeigen: `python -m sandcli.main --cfg job`
- Um einen Test zu machen: `python -m sandcli.main command=prep --cfg job` (zeigt nur Config)

### Problem 2: Fehlende Abhängigkeiten
**Fehler:** `ModuleNotFoundError: No module named 'regex'` und `openpyxl`

**Lösung:** ✅ `requirements.txt` wurde aktualisiert mit:
- `regex` (für Pattern Matching in prep.py)
- `openpyxl` (für Excel-Dateien)

### Problem 3: Falsche Befehlssyntax
**Falsch:** `python sandcli/main.py`
**Richtig:** `python -m sandcli.main`

### Problem 4: Fehlende Config-Sektionen
**Lösung:** ✅ `config.yaml` wurde ergänzt mit:
- `features` Sektion
- `model` Sektion  
- `training` Sektion

---

## 🚀 Umgebung Setup

### Schritt 1: Virtual Environment aktivieren
```bash
cd "/Users/fabian.drzimalla/Master Projects/SLP_PStA_Team_Drzimalla_Diakourakis"

# Wenn venv existiert:
source venv/bin/activate

# Wenn nicht, erstelle es:
python3 -m venv venv
source venv/bin/activate
```

### Schritt 2: Abhängigkeiten installieren
```bash
# Upgrade pip
pip install --upgrade pip

# Installiere alle Dependencies
pip install -r requirements.txt

# Optional: Installiere Paket im Development Mode
pip install -e .
```

### Schritt 3: Umgebung verifizieren
```bash
# Teste Python-Imports
python3 << 'EOF'
import sys
print(f"✓ Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")

try:
    import hydra
    print(f"✓ Hydra: {hydra.__version__}")
except ImportError as e:
    print(f"✗ Hydra: {e}")

try:
    import librosa
    print(f"✓ Librosa: {librosa.__version__}")
except ImportError as e:
    print(f"✗ Librosa: {e}")

try:
    import opensmile
    print(f"✓ OpenSMILE: {opensmile.__version__}")
except ImportError as e:
    print(f"✗ OpenSMILE: {e}")

try:
    import regex
    print(f"✓ Regex: installed")
except ImportError as e:
    print(f"✗ Regex: {e}")

try:
    import openpyxl
    print(f"✓ OpenPyXL: installed")
except ImportError as e:
    print(f"✗ OpenPyXL: {e}")
EOF
```

---

## 📊 Datenstruktur Validierung

### Erwartete Struktur:
```
data/task1/
├── sand_task_1.xlsx          # Metadata mit Labels
└── training/
    ├── phonationA/
    │   ├── ID000_phonationA.wav
    │   ├── ID001_phonationA.wav
    │   └── ...
    ├── phonationE/
    ├── phonationI/
    ├── phonationO/
    ├── phonationU/
    ├── rhythmKA/
    ├── rhythmPA/
    └── rhythmTA/
```

### Validiere Datenstruktur:
```bash
# Prüfe ob alle Verzeichnisse existieren
python3 << 'EOF'
from pathlib import Path

data_dir = Path("data/task1/training")
tasks = ["phonationA", "phonationE", "phonationI", "phonationO", "phonationU", 
         "rhythmKA", "rhythmPA", "rhythmTA"]

print("📁 Überprüfe Datenstruktur...")
for task in tasks:
    task_path = data_dir / task
    if task_path.exists():
        wav_files = list(task_path.glob("*.wav"))
        print(f"✓ {task}: {len(wav_files)} WAV-Dateien")
    else:
        print(f"✗ {task}: Verzeichnis nicht gefunden!")

metadata = Path("data/task1/sand_task_1.xlsx")
if metadata.exists():
    print(f"✓ Metadata-Datei gefunden: {metadata}")
else:
    print(f"✗ Metadata-Datei fehlt: {metadata}")
EOF
```

---

## 🎯 Pipeline-Befehle

### Schritt 0: Config anzeigen (Dry-Run Alternative)
```bash
# Zeige die gesamte Konfiguration an
python -m sandcli.main --cfg job

# Zeige nur Hydra-Config
python -m sandcli.main --cfg hydra

# Zeige alles
python -m sandcli.main --cfg all
```

### Schritt 1: Data Preparation
**Was macht dieser Schritt:**
- Scannt alle Audio-Dateien in den Task-Verzeichnissen
- Extrahiert Subject-IDs aus Dateinamen (z.B. `ID001`)
- Lädt Labels aus `sand_task_1.xlsx`
- Validiert Audio-Dateien (Sample Rate, Dauer, Channels)
- Erstellt `full_manifest.csv` mit allen Metadaten

**Befehl:**
```bash
python -m sandcli.main command=prep
```

**Erwartete Ausgabe:**
- `data/manifests/full_manifest.csv` erstellt
- Log zeigt Anzahl gefundener Dateien pro Task
- Warnung bei invaliden Dateien

**Validierung:**
```bash
# Prüfe ob Manifest existiert und korrekt ist
python3 << 'EOF'
import pandas as pd
from pathlib import Path

manifest = Path("data/manifests/full_manifest.csv")
if manifest.exists():
    df = pd.read_csv(manifest)
    print(f"✓ Manifest erstellt: {len(df)} Samples")
    print(f"\nLabel-Verteilung:")
    print(df['label'].value_counts())
    print(f"\nTask-Verteilung:")
    print(df['task'].value_counts())
    print(f"\nAnzahl Subjects: {df['subject_id'].nunique()}")
else:
    print("✗ Manifest nicht gefunden!")
EOF
```

---

### Schritt 2: Train/Test Split
**Was macht dieser Schritt:**
- Lädt `full_manifest.csv`
- Führt Subject-wise Split durch (keine Subjects in Train UND Test)
- Stratifiziert nach Labels (wenn möglich)
- Validiert Split (kein Data Leakage)
- Erstellt `train_manifest.csv` und `test_manifest.csv`

**Befehl:**
```bash
python -m sandcli.main command=split
```

**Parameter überschreiben:**
```bash
# Ändere Test-Größe auf 30%
python -m sandcli.main command=split split.test_size=0.3

# Deaktiviere Stratifizierung
python -m sandcli.main command=split split.stratify=false

# Ändere Random Seed
python -m sandcli.main command=split seed=123
```

**Erwartete Ausgabe:**
- `data/manifests/train_manifest.csv`
- `data/manifests/test_manifest.csv`
- Log zeigt Split-Statistiken und Leakage-Check

**Validierung:**
```bash
python3 << 'EOF'
import pandas as pd
from pathlib import Path

train_df = pd.read_csv("data/manifests/train_manifest.csv")
test_df = pd.read_csv("data/manifests/test_manifest.csv")

train_subjects = set(train_df['subject_id'].unique())
test_subjects = set(test_df['subject_id'].unique())
overlap = train_subjects & test_subjects

print(f"✓ Train: {len(train_df)} samples, {len(train_subjects)} subjects")
print(f"✓ Test: {len(test_df)} samples, {len(test_subjects)} subjects")

if len(overlap) > 0:
    print(f"✗ LEAKAGE! {len(overlap)} subjects in both sets")
else:
    print("✓ Kein Leakage - Split ist valide!")
EOF
```

---

### Schritt 3: Feature Extraction
**Was macht dieser Schritt:**
- Lädt Train/Test Manifests
- Extrahiert Features (MFCC, eGeMAPS oder beide)
- Preprocessiert Audio (Normalisierung, Silence Removal)
- Speichert Features als `.npy` Dateien
- Erstellt Feature-Manifests mit Pfaden zu Features

**Befehl (MFCC):**
```bash
python -m sandcli.main command=featurize features.type=mfcc
```

**Befehl (eGeMAPS):**
```bash
python -m sandcli.main command=featurize features.type=egemaps
```

**Befehl (MFCC + eGeMAPS):**
```bash
python -m sandcli.main command=featurize features.type=mfcc+egemaps
```

**Mit Konfigurationsdatei:**
```bash
# Verwende baseline_ml.yaml Config
python -m sandcli.main --config-name=baseline_ml command=featurize
```

**Erwartete Ausgabe:**
- `features/<feature_type>/train/` mit `.npy` Dateien
- `features/<feature_type>/test/` mit `.npy` Dateien
- `features/<feature_type>/train/manifest.csv`
- `features/<feature_type>/test/manifest.csv`

**Validierung:**
```bash
python3 << 'EOF'
import numpy as np
import pandas as pd
from pathlib import Path

feature_type = "mfcc"  # oder "egemaps" oder "mfcc_egemaps"
train_manifest = Path(f"features/{feature_type}/train/manifest.csv")

if train_manifest.exists():
    df = pd.read_csv(train_manifest)
    print(f"✓ Feature-Manifest: {len(df)} samples")
    
    # Lade erstes Feature
    first_feature = np.load(df.iloc[0]['feature_path'])
    print(f"✓ Feature-Dimension: {first_feature.shape[0]}")
else:
    print(f"✗ Feature-Manifest nicht gefunden: {train_manifest}")
EOF
```

---

### Schritt 4: Model Training
**Was macht dieser Schritt:**
- Lädt Features von Train/Test Manifests
- Erstellt ML Model (XGBoost oder LightGBM)
- Optional: Cross-Validation
- Trainiert finales Modell
- Evaluiert auf Test-Set
- Speichert Modell, Metriken, Confusion Matrix

**Befehl (mit baseline config):**
```bash
python -m sandcli.main --config-name=baseline_ml command=train
```

**Befehl (mit Custom Parameters):**
```bash
python -m sandcli.main command=train \
  model.name=xgboost \
  model.params.max_depth=8 \
  model.params.n_estimators=500 \
  training.cv_folds=5
```

**Erwartete Ausgabe:**
- `runs/<experiment_name>_<timestamp>/`
  - `config.yaml` - Verwendete Konfiguration
  - `model.pkl` - Trainiertes Modell
  - `metrics.json` - Test-Metriken
  - `confusion_matrix.png` - Confusion Matrix Plot
  - `cv_results.json` - Cross-Validation Ergebnisse (optional)

**Monitoring während Training:**
```bash
# In separatem Terminal: Watch für neue Experiment-Verzeichnisse
watch -n 2 'ls -lt runs/ | head -10'
```

**Validierung:**
```bash
python3 << 'EOF'
import json
from pathlib import Path

# Finde neuestes Experiment
runs_dir = Path("runs")
experiments = sorted(runs_dir.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)

if experiments:
    latest = experiments[0]
    print(f"✓ Neuestes Experiment: {latest.name}")
    
    metrics_file = latest / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        print(f"✓ Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"✓ Macro F1: {metrics['macro_f1']:.4f}")
else:
    print("✗ Keine Experimente gefunden")
EOF
```

---

### Schritt 5: Evaluation
**Was macht dieser Schritt:**
- Lädt trainiertes Modell
- Lädt Test-Daten
- Generiert Predictions
- Berechnet detaillierte Metriken
- Erstellt Visualisierungen

**Befehl:**
```bash
python -m sandcli.main command=evaluate \
  model_path=runs/<experiment_name>/model.pkl
```

---

### Schritt 6: Prediction (auf neuen Daten)
**Was macht dieser Schritt:**
- Lädt trainiertes Modell
- Lädt neue Audio-Dateien
- Extrahiert Features
- Generiert Predictions
- Exportiert Submission-File

**Befehl:**
```bash
python -m sandcli.main command=predict \
  model_path=runs/<experiment_name>/model.pkl \
  input_dir=data/task1/test
```

---

## 🧪 Kompletter Pipeline-Durchlauf

### Quick-Start (End-to-End):
```bash
# 1. Umgebung aktivieren
source venv/bin/activate

# 2. Data Prep
python -m sandcli.main command=prep

# 3. Split
python -m sandcli.main command=split

# 4. Features (MFCC)
python -m sandcli.main command=featurize features.type=mfcc

# 5. Training (mit baseline config)
python -m sandcli.main --config-name=baseline_ml command=train

# 6. Alle Schritte auf einmal (NICHT EMPFOHLEN für ersten Durchlauf)
# python -m sandcli.main --config-name=baseline_ml command=prep,split,featurize,train
```

### Mit verschiedenen Experimenten:
```bash
# Experiment 1: MFCC + XGBoost
python -m sandcli.main command=featurize features.type=mfcc
python -m sandcli.main command=train \
  experiment_name=exp01_mfcc_xgb \
  model.name=xgboost

# Experiment 2: eGeMAPS + LightGBM
python -m sandcli.main command=featurize features.type=egemaps
python -m sandcli.main command=train \
  experiment_name=exp02_egemaps_lgbm \
  model.name=lightgbm \
  features.type=egemaps

# Experiment 3: MFCC+eGeMAPS Fusion
python -m sandcli.main command=featurize features.type=mfcc+egemaps
python -m sandcli.main --config-name=baseline_ml command=train \
  experiment_name=exp03_fusion \
  features.type=mfcc+egemaps
```

---

## ✅ Validierung & Sanity Checks

### Pre-Training Checks:
```bash
# 1. Überprüfe ob alle Manifests existieren
ls -lh data/manifests/

# 2. Überprüfe Feature-Dateien
ls -lh features/mfcc/train/ | head

# 3. Überprüfe Config-Dateien
python -m sandcli.main --cfg job | head -50

# 4. Test-Import aller Module
python3 << 'EOF'
print("Testing imports...")
from sandcli import prep, split, featurize, train, evaluate, predict
print("✓ All modules imported successfully")
EOF
```

### Post-Training Checks:
```bash
# 1. Überprüfe Experiment-Output
python3 << 'EOF'
import json
from pathlib import Path

runs_dir = Path("runs")
for exp_dir in sorted(runs_dir.glob("*/"), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
    print(f"\n📊 {exp_dir.name}")
    
    metrics_file = exp_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  Macro F1: {m['macro_f1']:.4f}")
EOF
```

---

## 🐛 Troubleshooting

### Problem: "Command not found: python"
**Lösung:** Verwende `python3` statt `python`

### Problem: "ModuleNotFoundError"
**Lösung:**
```bash
pip install -r requirements.txt
# oder für spezifisches Modul:
pip install <module_name>
```

### Problem: "File not found" bei Manifests
**Lösung:** Führe die Pipeline-Schritte in der richtigen Reihenfolge aus:
1. prep → 2. split → 3. featurize → 4. train

### Problem: "UNKNOWN" Labels im Manifest
**Lösung:** 
- Überprüfe `data/task1/sand_task_1.xlsx`
- Stelle sicher, dass alle Subject-IDs im Excel vorhanden sind
- Überprüfe Dateinamen-Format: `ID###_<task>.wav`

### Problem: Hydra Konfigurationsfehler
**Lösung:**
```bash
# Zeige Config an um Fehler zu finden
python -m sandcli.main --cfg job

# Verwende absolute Pfade für config-name
python -m sandcli.main --config-name=baseline_ml --cfg job
```

### Problem: Out of Memory bei Feature Extraction
**Lösung:**
```bash
# Prozessiere nur ein Subset (für Tests)
# Bearbeite config.yaml und reduziere data.tasks temporär
```

---

## 📈 Erwartete Performance-Benchmarks

### Baseline (MFCC + XGBoost):
- **Accuracy:** ~0.60-0.75
- **Macro F1:** ~0.55-0.70
- **Training Zeit:** 2-5 Minuten (auf CPU)

### Optimiert (eGeMAPS + XGBoost + Tuning):
- **Accuracy:** ~0.70-0.80
- **Macro F1:** ~0.65-0.75
- **Training Zeit:** 5-10 Minuten

### Fusion (MFCC+eGeMAPS + Ensemble):
- **Accuracy:** ~0.75-0.85
- **Macro F1:** ~0.70-0.80
- **Training Zeit:** 10-15 Minuten

---

## 🎯 Nächste Schritte nach Phase 1

1. **Hyperparameter Tuning:**
   ```bash
   # Grid Search oder Bayesian Optimization
   python -m sandcli.main command=train \
     training.cv_folds=10 \
     model.params.max_depth=8,10,12
   ```

2. **Ensemble Learning:**
   ```bash
   python -m sandcli.main command=ensemble \
     models=runs/exp01_*/model.pkl,runs/exp02_*/model.pkl
   ```

3. **SSL Models (Phase 2):**
   ```bash
   python -m sandcli.main --config-name=ssl_wav2vec2 command=train
   ```

---

**Viel Erfolg! 🚀**

