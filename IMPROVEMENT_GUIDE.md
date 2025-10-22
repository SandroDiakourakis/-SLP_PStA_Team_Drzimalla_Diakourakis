# 🚀 VERBESSERUNGSSTRATEGIE - Vollständiger Leitfaden

## 📊 **AKTUELLE SITUATION:**

Ihre Ergebnisse:
- **Macro F1: 0.2216** ❌ (Ziel: >0.60)
- **Accuracy: 0.3295** ❌ (nur 33%!)
- **ALS_Severe F1: 0.0000** ❌ (komplett nicht erkannt!)

**Hauptprobleme:**
1. ⚠️ **Extremes Klassenungleichgewicht** (~17:1 - Healthy vs. ALS_Severe)
2. ⚠️ **Zu wenig Features** (nur 78 MFCC Features)
3. ⚠️ **Keine Hyperparameter-Optimierung**
4. ⚠️ **Einzelnes Modell** (kein Ensemble)

---

## 🎯 **VERBESSERUNGSSTRATEGIE (3 STUFEN):**

### **STUFE 1: Erweiterte Features 🔥 (SCHNELL - 10 Min)**

**Was wird verbessert:**
- ✅ 20 MFCCs (statt 13)
- ✅ 80 Mel-Bins (statt 40)
- ✅ 12 Statistiken pro Feature (statt 2)
- ✅ Class Weighting für Imbalance
- ✅ Bessere Regularisierung

**Erwartete Features:** ~720 (statt 78!)

**Erwartete Verbesserung:**
```
Macro F1:  0.22 → 0.45-0.55 ⬆️
Accuracy:  0.33 → 0.50-0.60 ⬆️
```

**Ausführen:**
```bash
source venv/bin/activate
./stage1_improve_features.sh
```

**ODER manuell:**
```bash
source venv/bin/activate

# Extrahiere verbesserte Features
python3 -m sandcli.main --config-name=improved_features command=featurize

# Trainiere mit neuen Features
python3 -m sandcli.main --config-name=improved_features command=train
```

---

### **STUFE 2: Hyperparameter-Tuning 🔍 (MITTEL - 1-2 Std)**

**Was wird optimiert:**
- ✅ GridSearch über 2187 Parameter-Kombinationen
- ✅ Findet optimale `max_depth`, `learning_rate`, `n_estimators`
- ✅ 5-Fold Cross-Validation
- ✅ Maximiert Macro F1

**Erwartete Verbesserung:**
```
Macro F1:  0.45 → 0.55-0.65 ⬆️
Accuracy:  0.50 → 0.60-0.70 ⬆️
```

**Ausführen:**
```bash
source venv/bin/activate
python3 stage2_gridsearch.py
```

**Nach Abschluss:**
1. Öffne `runs/gridsearch_*/best_params.json`
2. Kopiere die besten Parameter in `conf/improved_features.yaml`
3. Trainiere erneut:
```bash
python3 -m sandcli.main --config-name=improved_features command=train
```

---

### **STUFE 3: Ensemble Learning 🎭 (FORTGESCHRITTEN - 15 Min)**

**Was wird kombiniert:**
- ✅ XGBoost (Gradient Boosting)
- ✅ LightGBM (schnelleres Gradient Boosting)
- ✅ SVM mit RBF Kernel (für nicht-lineare Patterns)
- ✅ Soft Voting (verwendet Wahrscheinlichkeiten)

**Erwartete Verbesserung:**
```
Macro F1:  0.55 → 0.60-0.70 ⬆️
Accuracy:  0.60 → 0.65-0.75 ⬆️
```

**Ausführen:**
```bash
source venv/bin/activate
python3 stage3_ensemble.py
```

---

## 📈 **ERWARTETER GESAMT-FORTSCHRITT:**

| Stufe | Methode | Macro F1 | Accuracy | Dauer |
|-------|---------|----------|----------|-------|
| **Baseline** | MFCC (13) + XGBoost | 0.22 | 0.33 | ✅ Fertig |
| **Stufe 1** | MFCC (20) + 12 Stats + Weighting | 0.45-0.55 | 0.50-0.60 | 10 Min |
| **Stufe 2** | + GridSearch | 0.55-0.65 | 0.60-0.70 | 1-2 Std |
| **Stufe 3** | + Ensemble | 0.60-0.70 | 0.65-0.75 | 15 Min |

---

## 🎯 **EMPFOHLENER WORKFLOW:**

### **Quick Win (10 Minuten):**
```bash
source venv/bin/activate
./stage1_improve_features.sh
```

### **Vollständige Optimierung (2-3 Stunden):**
```bash
source venv/bin/activate

# 1. Verbesserte Features (10 Min)
./stage1_improve_features.sh

# 2. Hyperparameter-Tuning (1-2 Std)
python3 stage2_gridsearch.py

# 3. Beste Parameter übernehmen
# Bearbeite conf/improved_features.yaml mit besten Parametern
# Trainiere neu:
python3 -m sandcli.main --config-name=improved_features command=train

# 4. Ensemble (15 Min)
python3 stage3_ensemble.py
```

---

## 💡 **WEITERE VERBESSERUNGEN (Optional):**

### **A. Mehr Feature-Typen:**
```bash
# eGeMAPS (akustische Features)
python3 -m sandcli.main command=featurize features.type=egemaps

# Fusion (MFCC + eGeMAPS)
python3 -m sandcli.main command=featurize features.type=mfcc+egemaps
```

### **B. Data Augmentation:**
- Time stretching
- Pitch shifting
- Adding noise
→ Erhöht Training-Daten künstlich

### **C. Deep Learning (SSL):**
```bash
# Wav2Vec2 Pre-trained Features
python3 -m sandcli.main --config-name=ssl_wav2vec2 command=train
```

---

## 🔍 **PROBLEM-ANALYSE Tools:**

### **Feature Importance anschauen:**
```bash
cat runs/baseline_ml_xgboost_20251021_190616/feature_importance.csv | head -20
```

### **Confusion Matrix anschauen:**
```bash
open runs/baseline_ml_xgboost_20251021_190616/confusion_matrix.png
```

### **Klassenverteilung prüfen:**
```bash
python3 << 'EOF'
import pandas as pd
train = pd.read_csv('data/manifests/train_manifest.csv')
print(train['label'].value_counts())
EOF
```

---

## ✅ **ERFOLGSKRITERIEN:**

### **Minimum (akzeptabel):**
- Macro F1: > 0.50
- Accuracy: > 0.55
- Alle Klassen F1 > 0.30

### **Gut:**
- Macro F1: > 0.60
- Accuracy: > 0.65
- Alle Klassen F1 > 0.40

### **Exzellent:**
- Macro F1: > 0.70
- Accuracy: > 0.75
- Alle Klassen F1 > 0.50

---

## 🚨 **HÄUFIGE FEHLER:**

1. ❌ **Features nicht neu extrahiert** → Immer `command=featurize` nach Config-Änderung!
2. ❌ **Class Weights vergessen** → Wichtig bei Imbalance!
3. ❌ **Zu wenig Epochs** → XGBoost braucht oft 500-1000+ Trees
4. ❌ **Overfitting** → Verwende Cross-Validation zur Validierung

---

## 📞 **NEXT STEPS:**

**Sofort starten (empfohlen):**
```bash
source venv/bin/activate
./stage1_improve_features.sh
```

**Dann Results vergleichen:**
```bash
# Alte Results:
cat runs/baseline_ml_xgboost_20251021_190616/metrics.json

# Neue Results:
cat runs/improved_features_*/metrics.json
```

**Bei Fragen:**
- Schaue in `PIPELINE_GUIDE.md` für Details
- Checke Logs in `outputs/` bei Fehlern
- Vergleiche Confusion Matrices zwischen Runs

---

**Viel Erfolg bei der Verbesserung! 🚀**

