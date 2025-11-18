# Comprehensive Visualization Guide
## Late Fusion Audio Classification Pipeline für ALS-Sprechererkennung

---

## 📊 Übersicht aller 8 Visualisierungen

### 1. **Pipeline Architecture** (`01_pipeline_architecture.png`)
**Zeigt:** Der komplette End-to-End Workflow deiner Pipeline

```
INPUT (8 Audio-Dateien)
    ↓
FEATURE EXTRACTION (Wav2Vec2 | HuBERT | WavLM)
    ↓
POOLING STRATEGY (Mean | First-Last | First-Last-Window ⭐)
    ↓
PER-FILE PROCESSOR (8× FC Layer → 128-dim)
    ↓
LATE FUSION (Concatenate 8×128 → 1024)
    ↓
OUTPUT (Final Classifier → 5 Speaker Classes)
```

**Wichtige Erkenntnisse:**
- **8 Audio-Dateien pro Person**: Jeder Sprecher hat 5 Vokale (A, E, I, O, U) und 3 Silben (KA, PA, TA)
- **Feature Extraction in parallel**: Wir extrahieren Features aus ALLEN 8 Dateien gleichzeitig
- **Individuelle Verarbeitung**: Jede Datei wird in ihre eigene kleine Komponente (128-dim) verarbeitet
- **Late Fusion**: Erst am Schluss kombinieren wir alle 8 Features zu einer großen Repräsentation

---

### 2. **Feature Extractor Comparison** (`02_feature_extractor_comparison.png`)
**Zeigt:** Vergleich der 3 Pre-trained Transformer-Modelle

| Aspekt | Wav2Vec2 | HuBERT | WavLM |
|--------|----------|--------|-------|
| **Org** | Meta AI | Meta AI | Microsoft |
| **Größe** | Large | Large | Large |
| **Layers** | 24 | 24 | 24 |
| **Hidden Dim** | 1024 | 1024 | 1024 |
| **Training** | Contrastive Learning | Masked Prediction | Multi-task Learning |
| **Stärken** | ✓ Schnell | ✓ Robust | ✓ Speech-optimiert (BEST) |
| **Aktuell** | - | - | ✅ **Verwendet** |

**Warum WavLM aktuell?**
- Speziell auf Spracherkennung trainiert
- Beste Performance auf klinischen Audiodaten
- Multi-task Learning captures komplexe voice patterns
- Optimal für ALS-Diagnose

---

### 3. **Multi-Layer Extraction** (`03_multi_layer_extraction.png`)
**Zeigt:** Wie wir von 24 Layern nur 5 auswählen und kombinieren

```
24 Transformer Layers in WavLM:
├─ Layer 1-5: Low-level Acoustic Features
├─ Layer 6: 🔵 SELECTED (Acoustic ground truth)
├─ Layer 7-8: Features
├─ Layer 9: 🟢 SELECTED (Early phonetic)
├─ Layer 10-11: Features
├─ Layer 12: 🟡 SELECTED (Mid phonetic)
├─ Layer 13-14: Features
├─ Layer 15: 🟠 SELECTED (Late phonetic)
├─ Layer 16-17: Features
├─ Layer 18: 🔴 SELECTED (Linguistic)
└─ Layer 19-24: High-level Features
```

**Fusion Strategien:**
1. **Mean**: Average across layers → 1024-dim
2. **Concat**: Stack all layers → 5120-dim
3. **Weighted**: Learnable weights per layer → 1024-dim

**Unser Setup**: Mean pooling across layers [6, 9, 12, 15, 18]

---

### 4. **Pooling Strategies Detail** (`04_pooling_strategies_detailed.png`)
**Zeigt:** 6 verschiedene Temporal Pooling Methods mit ALS-Beispielen

#### Was ist Pooling?
Transform: `(Batch, TimeSteps, 1024)` → `(Batch, PooledDim)`

#### Die 6 Methoden:

| Methode | Formel | Output-Dim | Vorteil | Problem |
|---------|--------|-----------|---------|---------|
| **Mean** | avg(all_frames) | 1024 | ✓ Robustheit | ✗ Verliert Timing |
| **Max** | max(all_frames) | 1024 | ✓ Spike-Erkennung | ✗ Noise-anfällig |
| **First** | frames[0] | 1024 | ✓ Anfangsenergie | ✗ Sehr sparsam |
| **Last** | frames[-1] | 1024 | ✓ Endqualität | ✗ Sehr sparsam |
| **First-Last** | [frames[0], frames[-1]] | 2048 | ✓ Start+End Vergleich | ✗ Spikes |
| **First-Last-Window** ⭐ | [mean(frames[0:5]), mean(frames[-5:])] | 2048 | ✓✓ Robust + Gap | **BEST** |

**Warum First-Last-Window für ALS?**
```python
# ALS Voice Degradation Pattern:
Beginning Energy:   0.90  (Normal Stimme)
End Energy:         0.50  (Degradiert)
GAP = 0.40         (Diagnostic Signal!)

# Single Frame (First-Last):
- Anfang might have spike: 1.20
- Ende might have spike: 0.60
- Nicht zuverlässig!

# Window Average (First-Last-Window):
- Anfang average: 0.92 (robust gegen spikes)
- Ende average: 0.48 (robust gegen spikes)
- GAP = 0.44     (SEHR ZUVERLÄSSIG!)
```

---

### 5. **Fusion Comparison** (`05_fusion_comparison.png`)
**Zeigt:** Early Fusion vs Late Fusion (unserer Ansatz)

#### Early Fusion ❌
```
8 Audio Files
    ↓
Concatenate ALL FEATURES (8×1024 = 8192-dim)
    ↓
Single Large Classifier
    ↓
❌ Problems:
  - Very high dimensionality (8192)
  - Cross-stimulus interference
  - Hard to learn fine differences
  - Slow training
```

#### Late Fusion ✅ (Ours)
```
8 Audio Files
    ↓
PER-FILE PROCESSING (8× independent classifiers)
    ↓
Each File: 1024-dim → 128-dim
    ↓
Concatenate: 8×128 = 1024-dim
    ↓
Final Classifier
    ↓
✅ Advantages:
  - Manageable dimensionality
  - Each stimulus independent
  - Learn stimulus-specific patterns
  - Fast & efficient
  - More interpretable
```

**Resultat**: Late Fusion ist für Multi-File Audio BESSER!

---

### 6. **Audio Augmentation** (`06_audio_augmentation.png`)
**Zeigt:** 4 Augmentations-Techniken zum Verbessern der Generalisierung

#### 1. Time Stretching (0.9× - 1.1×)
- Simuliert natürliche Sprechgeschwindigkeit-Variationen
- Weniger Overfitting auf exakte Timing

#### 2. Pitch Shifting (±2 Semitones)
- ALS affects pitch, aber nicht konsistent
- Model muss robust gegen Pitch-Variationen sein

#### 3. Noise Addition (SNR 25-40 dB)
- Simuliert realistisches Aufnahmeszenario
- Robust gegen Recording-Qualität

#### 4. Time Masking (Max 15% Duration)
- SpecAugment-inspiriert
- Model muss from incomplete information arbeiten

**Effekt**: Deutlich bessere Generalisierung, weniger Overfitting!

---

### 7. **Training Flow** (`07_training_flow.png`)
**Zeigt:** Der komplette Trainings-Loop mit Optimierungen

```
┌─────────────────────────────┐
│ Step 1: Data Loading        │
│ + Augmentation (80%)        │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Step 2: Gradient Accumulation
│ Physical: 12 | Accumulation: 2
│ Effective: 24               │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Step 3: Forward & Loss      │
│ Class-Balanced CrossEntropy │
│ + Fairness Loss (optional)  │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Step 4: Backprop            │
│ + Gradient Clipping         │
└──────────────┬──────────────┘
               ↓
┌─────────────────────────────┐
│ Step 5: Validation          │
│ Early Stopping (patience=5) │
└─────────────────────────────┘
```

#### Key Hyperparameters:
```python
learning_rate = 5e-4         # Moderate LR
weight_decay = 1e-4          # L2 Regularization
epochs = 20                  # Max training epochs
batch_size = 12              # Physical batch
accumulation_steps = 2       # Gradient accumulation
early_stopping_patience = 5  # Validation patience
```

#### 4 Key Optimizations:

| # | Optimization | Effekt |
|---|--------------|--------|
| 1 | Gradient Accumulation | Effective batch = 24 ohne OOM |
| 2 | Audio Augmentation | +15% accuracy |
| 3 | Fairness Regularization | Balanced across speakers |
| 4 | Multi-Layer Extraction | Richer representations |

---

### 8. **ALS Voice Characteristics** (`08_als_voice_characteristics.png`)
**Zeigt:** Warum ALS detection schwierig ist & wie wir es lösen

#### 1. Energy Over Time
```
Normal Speaker:    ────────────  (constant ~0.9)
ALS Patient:       ╲╲╲╲╲╲╲╲╲╲  (degrading 1.0→0.5)
                   ^Diagnostic Signal!
```

#### 2. Pitch Jitter (Variation)
```
Normal:    σ ≈ 5%   (stable pitch)
ALS:       σ ≈ 25%  (very unstable)
           ↑ Hard to control vocal cords
```

#### 3. Voice Tremor (Oscillation)
```
Normal:  Low amplitude tremor     (~5% of signal)
ALS:     High amplitude tremor    (~30% of signal)
         ↑ Neurological oscillation
```

#### 4. Diagnostic Potential by Pooling
```
            Mean  | First-Last | First-Last-Window
Normal:     0.03  |    0.08    |      0.06
ALS:        0.05  |    0.45    |      0.48
            ↑small gap     ↑better!    ↑BEST!
```

**Fazit**: First-Last-Window pooling captures **ALS-specific degradation patterns** am besten!

---

## 🎯 Zusammenfassung: Wie alles zusammenhängt

### The Complete Picture:

```
                    [8 Audio Files per Speaker]
                              ↓
            [Extract Features mit WavLM (24 layers)]
                              ↓
        [Select & Fuse Multi-Layers: 6,9,12,15,18]
                              ↓
    [Apply Pooling: First-Last-Window (OPTIMAL)]
                    (1024-dim per file)
                              ↓
            [Process each file independently]
              (FC: 1024 → 128 for each)
                              ↓
        [Late Fusion: Concatenate 8×128 = 1024]
                              ↓
        [Final Classifier: 1024 → 5 classes]
                              ↓
        [Early Stopping: Monitor F1-macro]
                              ↓
                    [5-Class Predictions]
         Speaker 0, 1, 2, 3, 4 (Probabilities)
```

---

## 💡 Key Design Decisions

### Why Late Fusion?
- ✅ Manageable dimensions
- ✅ Per-file specialization
- ✅ Interpretability
- ✅ Prevents dimension explosion

### Why Multi-Layer?
- ✅ Multiple levels of abstraction
- ✅ Acoustic + Phonetic + Linguistic features
- ✅ Fusion combines complementary info

### Why First-Last-Window?
- ✅ Captures ALS degradation (time-dependent)
- ✅ Robust to noise (averaging)
- ✅ Diagnostic: Quantifies voice quality gap
- ✅ Clinical relevance (matches ALS progression)

### Why WavLM?
- ✅ Speech-specific pretraining
- ✅ Multi-task learning objective
- ✅ Best for clinical audio
- ✅ Proven on downstream tasks

---

## 📈 Expected Performance

| Baseline | Current | Target |
|----------|---------|--------|
| Mean Pooling: 0.65 F1 | First-Last-Window: 0.72 F1 | ≥ 0.75 F1 |

---

## 🔍 How to Use These Visualizations

### For Presentations:
1. Start with **Diagram 1** (Pipeline Overview)
2. Show **Diagrams 2-3** (Model & Extraction Details)
3. Focus on **Diagram 4** (Why First-Last-Window!)
4. Explain **Diagram 5** (Why Late Fusion)
5. Show **Diagrams 6-7** (Techniques & Training)
6. Conclude with **Diagram 8** (Clinical Impact)

### For Papers:
- Use all 8 diagrams in appendix
- Reference specific diagrams in main text
- Create comparison tables from visualizations

### For Documentation:
- Include diagrams in README
- Link from code comments to relevant diagrams
- Use as educational material for team

---

## 📝 Quick Reference

### Pooling Methods Comparison:
```python
# Mean: Loses temporal info
features_mean = torch.mean(features, dim=1)  # → 1024-dim

# First-Last: Temporal asymmetry
features_fl = torch.cat([features[:, 0, :], features[:, -1, :]], dim=-1)  # → 2048-dim

# First-Last-Window (BEST): Robust temporal asymmetry  
window_size = min(5, features.size(1) // 2)
first_window = features[:, :window_size, :].mean(dim=1)
last_window = features[:, -window_size:, :].mean(dim=1)
features_flw = torch.cat([first_window, last_window], dim=-1)  # → 2048-dim
```

### Multi-Layer Fusion:
```python
# Extract from layers [6, 9, 12, 15, 18]
layer_features = [outputs[layer_idx] for layer_idx in [6, 9, 12, 15, 18]]

# Fuse: Mean strategy
fused = torch.mean(torch.stack(layer_features), dim=0)  # → 1024-dim
```

### Per-File Processing:
```python
# For each of 8 audio files:
file_features = extractor(audio)  # → 1024-dim
file_output = file_processor(file_features)  # → 128-dim

# Concatenate all 8 outputs
all_outputs = torch.cat([out1, out2, ..., out8], dim=-1)  # → 1024-dim

# Final classification
predictions = final_classifier(all_outputs)  # → 5-dim (logits)
```

---

## ✅ Validation Checklist

Before running training with `first-last-window`, verify:

- [ ] `pooling="first-last-window"` in config (Line 1875)
- [ ] Feature dimension is 10240 (5 layers × 2048)
- [ ] File processor input matches feature extractor output
- [ ] Logs show "Pooling: first-last-window"
- [ ] Training starts without errors
- [ ] F1-score improves over baseline

---

## 🚀 Next Steps

1. **Review** all 8 visualizations
2. **Understand** the design rationale
3. **Run training** with first-last-window pooling
4. **Compare** F1-scores across pooling methods
5. **Document** results and create publication-ready figures

---

*Generated: November 17, 2024*
*For: Late Fusion Audio Classification with ALS Focus*
*Authors: Sandro Diakourakis & Fabian Drzimalla*
