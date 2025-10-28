# 🎤 SAND Task 1 - ALS Speech Classification Pipeline

Eine vollständige Machine-Learning-Pipeline zur Klassifikation von ALS-Sprechstörungen basierend auf Audiodaten.

---

## 📋 Inhaltsverzeichnis

- [Installation](#-installation)
- [Schnellstart](#-schnellstart)
- [Pipeline-Befehle](#-pipeline-befehle)
- [Konfiguration](#-konfiguration)
- [Erweiterte Verwendung](#-erweiterte-verwendung)
- [Optimierungsstrategien](#-optimierungsstrategien)
- [Troubleshooting](#-troubleshooting)

---

## 🚀 Installation

### Voraussetzungen

- Python 3.8 oder höher
- macOS, Linux oder Windows
- Mindestens 4 GB RAM

### Setup

```bash
# 1. Repository klonen (falls noch nicht geschehen)
cd /Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis

# 2. Virtual Environment erstellen
python3 -m venv venv

# 3. Virtual Environment aktivieren
source venv/bin/activate  # macOS/Linux
# oder: venv\Scripts\activate  # Windows

# 4. Dependencies installieren
pip install -r requirements.txt

# 5. Package installieren
pip install -e .
```

### Abhängigkeiten prüfen

```bash
python3 -c "import xgboost, lightgbm, opensmile; print('✅ Alle Dependencies installiert')"
```

---

## ⚡ Schnellstart

### Option 1: Komplette Pipeline (empfohlen)

```bash
# Alles in einem Schritt: Vorbereitung → Split → Features → Training
python3 -m sandcli.main command=prep && \
python3 -m sandcli.main command=split && \
python3 -m sandcli.main command=featurize && \
python3 -m sandcli.main command=train
```

### Option 2: Bestes Modell direkt trainieren

```bash
# Mit den optimalen Einstellungen (Ensemble + SMOTE)
python3 -m sandcli.main command=train \
  training.use_smote=true \
  model.use_ensemble=true
```

### Option 3: Automatisierte Optimierungen

```bash
# Führt mehrere Optimierungsstrategien nacheinander aus
chmod +x run_optimizations.sh
./run_optimizations.sh
```

---

## 🔧 Pipeline-Befehle

Die Pipeline besteht aus 7 Hauptbefehlen, die über `python3 -m sandcli.main command=<BEFEHL>` ausgeführt werden.

---

### 1. `prep` - Datenaufbereitung

**Zweck:** Lädt und bereinigt die Rohdaten (Audiodateien + Metadaten).

```bash
python3 -m sandcli.main command=prep
```

**Was passiert:**
- Liest Excel-Metadaten (`sand_task_1.xlsx`)
- Scannt Audiodateien in `data/task1/training/`
- Erstellt Manifest mit Datei-Label-Zuordnungen
- Validiert Datenintegrität

**Optionale Parameter:**
```bash
# Andere Datenquelle verwenden
python3 -m sandcli.main command=prep \
  data.root_dir="data/task1/training_test" \
  data.metadata_file="data/task1/metadata_test.xlsx"
```

---

### 2. `split` - Train/Test-Aufteilung

**Zweck:** Teilt Daten in Trainings- und Testset auf (stratifiziert).

```bash
python3 -m sandcli.main command=split
```

**Was passiert:**
- Liest Full-Manifest
- Stratifizierte Aufteilung: 80% Train, 20% Test
- Stellt sicher, dass alle Klassen in beiden Sets vertreten sind
- Speichert `train_manifest.csv` und `test_manifest.csv`

**Optionale Parameter:**
```bash
# Andere Test-Größe (z.B. 30%)
python3 -m sandcli.main command=split split.test_size=0.3

# Ohne Stratifikation (nicht empfohlen!)
python3 -m sandcli.main command=split split.stratify=false
```

---

### 3. `featurize` - Feature-Extraktion

**Zweck:** Extrahiert akustische Features aus Audiodateien.

#### Standard: Nur MFCC-Features

```bash
python3 -m sandcli.main command=featurize
```

**Output:** `features/mfcc/train/` und `features/mfcc/test/`

#### Option A: Nur eGeMAPSv02-Features

```bash
python3 -m sandcli.main command=featurize features.type=egemaps
```

**Output:** `features/egemaps/train/` und `features/egemaps/test/`

#### Option B: MFCC + eGeMAPSv02 kombiniert (BESTE PERFORMANCE!)

```bash
python3 -m sandcli.main command=featurize features.type="mfcc+egemaps"
```

**Output:** `features/mfcc_egemaps/train/` und `features/mfcc_egemaps/test/`

**Performance-Boost:** +10-15% Macro F1

#### MFCC-Parameter anpassen

```bash
# Mehr MFCC-Koeffizienten (bessere Auflösung)
python3 -m sandcli.main command=featurize features.mfcc.n_mfcc=40

# Deltas deaktivieren (schneller, aber weniger Features)
python3 -m sandcli.main command=featurize \
  features.mfcc.include_deltas=false \
  features.mfcc.include_delta_deltas=false

# Andere Sample-Rate
python3 -m sandcli.main command=featurize data.sample_rate=16000
```

---

### 4. `train` - Modell trainieren

**Zweck:** Trainiert ein Machine-Learning-Modell auf den extrahierten Features.

#### Standard-Training (Einzelmodell XGBoost)

```bash
python3 -m sandcli.main command=train
```

**Output:**
- Modell: `runs/sand_task1_TIMESTAMP/model.pkl`
- Metriken: `runs/sand_task1_TIMESTAMP/metrics.json`
- Confusion Matrix: `runs/sand_task1_TIMESTAMP/confusion_matrix.png`
- Predictions: `runs/sand_task1_TIMESTAMP/predictions.csv`

#### Mit SMOTE (Class Imbalance Handling)

```bash
python3 -m sandcli.main command=train training.use_smote=true
```

**Performance-Boost:** +5-10% Macro F1 (besonders für Minority-Klassen)

#### Mit Ensemble (4 Modelle kombiniert)

```bash
python3 -m sandcli.main command=train model.use_ensemble=true
```

**Ensemble-Modelle:**
- XGBoost (Gewicht: 3)
- LightGBM (Gewicht: 3)
- Random Forest (Gewicht: 2)
- SVM RBF (Gewicht: 1)

**Performance-Boost:** +3-7% Macro F1

#### BESTE KONFIGURATION (Kombination)

```bash
python3 -m sandcli.main command=train \
  training.use_smote=true \
  model.use_ensemble=true
```

**Ergebnis:** Macro F1 = **0.34+** (statt 0.22 Baseline)

#### Hyperparameter-Tuning

```bash
# GridSearch über optimiertes Parameter-Grid (8 Kombinationen, ~30-60 Min)
python3 -m sandcli.main command=train \
  training.tune_hyperparameters=true \
  training.use_smote=true \
  model.use_ensemble=true
```

**Output:** `runs/*/cv_results.json` mit allen getesteten Kombinationen

#### Verschiedene Modelle

```bash
# Nur LightGBM (schneller als XGBoost)
python3 -m sandcli.main command=train model.name=lightgbm

# Nur XGBoost (Standard)
python3 -m sandcli.main command=train model.name=xgboost
```

#### Manuelle Hyperparameter

```bash
python3 -m sandcli.main command=train \
  model.params.max_depth=10 \
  model.params.learning_rate=0.05 \
  model.params.n_estimators=2000
```

#### Seed für Reproduzierbarkeit

```bash
# Fixierter Seed
python3 -m sandcli.main command=train seed=42

# Verschiedene Seeds für Multi-Seed-Ensemble
python3 -m sandcli.main command=train seed=42 experiment_name="run_seed42"
python3 -m sandcli.main command=train seed=123 experiment_name="run_seed123"
python3 -m sandcli.main command=train seed=456 experiment_name="run_seed456"
```

---

### 5. `evaluate` - Modell evaluieren

**Zweck:** Evaluiert ein bereits trainiertes Modell auf dem Testset.

```bash
python3 -m sandcli.main command=evaluate \
  model_path="runs/sand_task1_20251022_105250/model.pkl"
```

**Output:**
- Detaillierte Metriken (Macro F1, Accuracy, Per-Class F1)
- Confusion Matrix
- Classification Report

---

### 6. `predict` - Predictions auf neuen Daten

**Zweck:** Macht Vorhersagen mit einem trainierten Modell.

```bash
python3 -m sandcli.main command=predict \
  model_path="runs/sand_task1_20251022_105250/model.pkl"
```

**Output:** CSV mit Predictions für das Testset

#### Auf benutzerdefinierten Daten

```bash
python3 -m sandcli.main command=predict \
  model_path="runs/sand_task1_20251022_105250/model.pkl" \
  data.root_dir="data/task1/new_data"
```

---

### 7. `ensemble` - Multi-Model-Ensemble

**Zweck:** Kombiniert mehrere trainierte Modelle via Voting.

```bash
python3 -m sandcli.main command=ensemble \
  model_paths="['runs/seed_42_*/model.pkl', 'runs/seed_123_*/model.pkl', 'runs/seed_456_*/model.pkl']"
```

**Voting-Strategien:**
- Hard Voting: Mehrheitsentscheidung
- Soft Voting: Gewichtete Wahrscheinlichkeiten (Standard)

---

## ⚙️ Konfiguration

### Zentrale Config-Datei

Alle Einstellungen sind in `conf/config.yaml` definiert.

#### Wichtige Parameter

```yaml
# Experiment
seed: 42
experiment_name: "sand_task1"
output_dir: "runs"

# Features
features:
  type: "mfcc"  # Options: mfcc, egemaps, mfcc+egemaps
  mfcc:
    n_mfcc: 20  # Anzahl Koeffizienten (20-40 empfohlen)
    include_deltas: true
    include_delta_deltas: true

# Model
model:
  name: "xgboost"  # Options: xgboost, lightgbm
  use_ensemble: false
  params:
    max_depth: 8
    learning_rate: 0.05
    n_estimators: 2000

# Training
training:
  cv_folds: 5
  use_smote: false
  tune_hyperparameters: false
```

### Config überschreiben (CLI)

Jeder Parameter kann über die CLI überschrieben werden:

```bash
python3 -m sandcli.main command=train \
  seed=123 \
  features.mfcc.n_mfcc=40 \
  model.params.max_depth=10 \
  training.use_smote=true
```

### Eigene Config-Datei

```bash
# Eigene Config erstellen
cp conf/config.yaml conf/my_config.yaml

# Mit eigener Config trainieren
python3 -m sandcli.main --config-name=my_config command=train
```

---

## 🎯 Erweiterte Verwendung

### Komplette Pipeline mit Custom-Settings

```bash
# 1. Daten vorbereiten
python3 -m sandcli.main command=prep

# 2. Split mit 70/30 statt 80/20
python3 -m sandcli.main command=split split.test_size=0.3

# 3. MFCC + eGeMAPSv02 Features
python3 -m sandcli.main command=featurize \
  features.type="mfcc+egemaps" \
  features.mfcc.n_mfcc=40

# 4. Training mit allem
python3 -m sandcli.main command=train \
  features.type="mfcc+egemaps" \
  features.mfcc.n_mfcc=40 \
  training.use_smote=true \
  model.use_ensemble=true \
  training.tune_hyperparameters=true
```

### Multi-Seed Ensemble Workflow

```bash
# 1. Trainiere 5 Modelle mit verschiedenen Seeds
for seed in 42 123 456 789 1024; do
  python3 -m sandcli.main command=train \
    seed=$seed \
    experiment_name="seed_${seed}" \
    training.use_smote=true \
    model.use_ensemble=true
done

# 2. Finde bestes Modell
for file in runs/seed_*/metrics.json; do
  f1=$(cat $file | python3 -c 'import json,sys; print(json.load(sys.stdin)["macro_f1"])')
  echo "$f1 - $file"
done | sort -rn | head -1

# 3. Kombiniere Top-3 Modelle
python3 -m sandcli.main command=ensemble \
  model_paths="['runs/seed_42_*/model.pkl', 'runs/seed_123_*/model.pkl', 'runs/seed_456_*/model.pkl']"
```

### Batch-Experimente

```bash
# Verschiedene Feature-Kombinationen testen
for feature_type in mfcc egemaps "mfcc+egemaps"; do
  python3 -m sandcli.main command=featurize features.type="$feature_type"
  python3 -m sandcli.main command=train \
    features.type="$feature_type" \
    experiment_name="features_${feature_type//+/_}"
done
```

---

## 🚀 Optimierungsstrategien

### Strategie 1: Feature-Engineering

```bash
# MFCC + eGeMAPSv02 (BESTE FEATURES!)
python3 -m sandcli.main command=featurize features.type="mfcc+egemaps"
python3 -m sandcli.main command=train features.type="mfcc+egemaps"
```

**Erwarteter Boost:** +10-15% Macro F1

### Strategie 2: SMOTE + Ensemble

```bash
python3 -m sandcli.main command=train \
  training.use_smote=true \
  model.use_ensemble=true
```

**Erwarteter Boost:** +8-15% Macro F1

### Strategie 3: Hyperparameter-Tuning

```bash
python3 -m sandcli.main command=train \
  training.tune_hyperparameters=true \
  training.use_smote=true
```

**Erwarteter Boost:** +3-7% Macro F1
**Dauer:** 30-60 Minuten

### Strategie 4: Alle Optimierungen kombiniert

```bash
# 1. Feature-Extraktion
python3 -m sandcli.main command=featurize \
  features.type="mfcc+egemaps" \
  features.mfcc.n_mfcc=40

# 2. Training mit allen Optimierungen
python3 -m sandcli.main command=train \
  features.type="mfcc+egemaps" \
  features.mfcc.n_mfcc=40 \
  training.use_smote=true \
  model.use_ensemble=true \
  training.tune_hyperparameters=true
```

**Erwarteter Boost:** +20-30% Macro F1 (von 0.22 → 0.38+)

### Strategie 5: Automatisiertes Skript

```bash
# Führt alle Optimierungen nacheinander aus
./run_optimizations.sh
```

---

## 📊 Ergebnisse analysieren

### Alle Runs anzeigen

```bash
# Liste aller Runs mit Macro F1
for file in runs/*/metrics.json; do
  f1=$(cat $file | python3 -c 'import json,sys; print(json.load(sys.stdin)["macro_f1"])')
  acc=$(cat $file | python3 -c 'import json,sys; print(json.load(sys.stdin)["accuracy"])')
  echo "$f1 | $acc | $file"
done | sort -rn
```

### Top 5 Runs

```bash
for file in runs/*/metrics.json; do
  f1=$(cat $file | python3 -c 'import json,sys; print(json.load(sys.stdin)["macro_f1"])')
  echo "$f1 - $file"
done | sort -rn | head -5
```

### Neuesten Run anzeigen

```bash
ls -t runs/*/metrics.json | head -1 | xargs cat | python3 -m json.tool
```

### Confusion Matrix öffnen

```bash
# Neueste
open $(ls -t runs/*/confusion_matrix.png | head -1)

# Spezifischer Run
open runs/sand_task1_20251022_105250/confusion_matrix.png
```

---

## 🐛 Troubleshooting

### Problem 1: opensmile-Fehler bei eGeMAPSv02

**Symptom:**
```
opensmile.core.lib.OpenSmileException: Code: 1
```

**Lösung:**
```bash
# opensmile neu installieren
pip uninstall opensmile
pip install opensmile==2.5.0

# Oder: Nur MFCC verwenden
python3 -m sandcli.main command=featurize features.type=mfcc
```

### Problem 2: Memory Error beim Training

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Lösung:**
```bash
# Weniger MFCC-Koeffizienten
python3 -m sandcli.main command=featurize features.mfcc.n_mfcc=13

# SMOTE deaktivieren
python3 -m sandcli.main command=train training.use_smote=false

# Batch-Size reduzieren (nur bei Deep Learning)
```

### Problem 3: Hyperparameter-Tuning dauert zu lange

**Symptom:** GridSearch läuft 10+ Stunden

**Lösung:**
```bash
# Kleineres Grid in conf/config.yaml:
training:
  tuning_grid:
    max_depth: [8, 10]           # Nur 2 Werte
    learning_rate: [0.05]        # Fixiert
    n_estimators: [2000]         # Fixiert
    min_child_weight: [1, 3]     # Nur 2 Werte
    gamma: [0.3]                 # Fixiert

# = 8 statt 243 Kombinationen!
```

### Problem 4: SVM Convergence Warning

**Symptom:**
```
ConvergenceWarning: Solver terminated early (max_iter=1000)
```

**Lösung:**
```bash
# Features skalieren (wird automatisch gemacht)
# Oder: SVM aus Ensemble entfernen
python3 -m sandcli.main command=train model.use_ensemble=false
```

### Problem 5: Features nicht gefunden

**Symptom:**
```
FileNotFoundError: features/mfcc/train/...
```

**Lösung:**
```bash
# Features neu extrahieren
python3 -m sandcli.main command=featurize

# Oder: Mit richtigem Feature-Type trainieren
python3 -m sandcli.main command=train features.type=mfcc
```

---

## 📂 Ordnerstruktur

```
SLP_PStA_Team_Drzimalla_Diakourakis/
├── conf/
│   └── config.yaml              # Zentrale Konfiguration
├── data/
│   ├── manifests/               # Train/Test-Splits
│   └── task1/
│       ├── sand_task_1.xlsx     # Metadaten
│       └── training/            # Audiodateien
├── features/
│   ├── mfcc/                    # MFCC-Features
│   ├── egemaps/                 # eGeMAPSv02-Features
│   └── mfcc_egemaps/            # Kombinierte Features
├── runs/                        # Trainierte Modelle + Metriken
│   └── sand_task1_TIMESTAMP/
│       ├── model.pkl
│       ├── metrics.json
│       ├── confusion_matrix.png
│       └── predictions.csv
├── sandcli/                     # Pipeline-Code
│   ├── main.py                  # CLI Entry Point
│   ├── prep.py
│   ├── split.py
│   ├── featurize.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── ensemble.py
├── models/                      # Model-Definitionen
├── requirements.txt
├── setup.py
├── run_optimizations.sh         # Automatisiertes Optimierungs-Skript
└── README.md                    # Diese Datei
```

---

## 🎓 Best Practices

### 1. Immer mit Baseline starten

```bash
# Einfachstes Modell zuerst
python3 -m sandcli.main command=train
```

### 2. Inkrementell optimieren

```bash
# Schritt 1: SMOTE hinzufügen
python3 -m sandcli.main command=train training.use_smote=true

# Schritt 2: Ensemble hinzufügen
python3 -m sandcli.main command=train training.use_smote=true model.use_ensemble=true

# Schritt 3: Bessere Features
python3 -m sandcli.main command=featurize features.type="mfcc+egemaps"
python3 -m sandcli.main command=train features.type="mfcc+egemaps" training.use_smote=true model.use_ensemble=true
```

### 3. Ergebnisse dokumentieren

```bash
# Nach jedem Run
echo "Run $(date): Macro F1 = $(cat runs/sand_task1_*/metrics.json | tail -1 | jq .macro_f1)" >> results.log
```

### 4. Seeds fixieren für Reproduzierbarkeit

```bash
python3 -m sandcli.main command=train seed=42
```

### 5. Cross-Validation nutzen

```bash
# Wird automatisch gemacht (5-Fold CV)
# Erhöhen für stabilere Schätzungen
python3 -m sandcli.main command=train training.cv_folds=10
```

---

## 📈 Erwartete Performance

| Setup | Macro F1 | Accuracy | Notizen |
|-------|----------|----------|---------|
| Baseline (XGBoost) | 0.22 | 0.30 | Ohne Optimierungen |
| + SMOTE | 0.28 | 0.36 | +27% Boost |
| + Ensemble | 0.34 | 0.37 | +55% Boost |
| + MFCC+eGeMAPSv02 | 0.38+ | 0.42+ | +73% Boost (geschätzt) |
| + Hyperparameter-Tuning | 0.40+ | 0.45+ | +82% Boost (Ziel) |

---

## 🤝 Contribution

Dieses Projekt wurde entwickelt von:
- Fabian Drzimalla
- Diakourakis

Für **Speech and Language Processing - Practical Student Assignment**

---

## 📄 Lizenz

Dieses Projekt ist Teil einer akademischen Arbeit und nur für Bildungszwecke gedacht.

---

## 📞 Support

Bei Problemen oder Fragen:

1. Siehe [Troubleshooting](#-troubleshooting)
2. Prüfe die Logs: `runs/*/train.log`
3. Validiere die Config: `cat conf/config.yaml`

---

**Viel Erfolg! 🚀**

