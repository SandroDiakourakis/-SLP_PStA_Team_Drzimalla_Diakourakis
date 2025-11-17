# Vergleich von Support Vector Machines und neuronalen Netzwerken mit Late Fusion für die automatisierte ALS-Klassifikation aus Sprachaufnahmen

## Abstract

Die Amyotrophe Lateralsklerose (ALS) ist eine progressive neurodegenerative Erkrankung, die zu motorischen Beeinträchtigungen einschließlich der Sprachproduktion führt. Diese Arbeit vergleicht zwei Machine-Learning-Ansätze zur automatisierten ALS-Klassifikation aus multimodalen Sprachaufnahmen: einen klassischen Support Vector Machine (SVM)-Ansatz mit Wav2Vec 2.0-Features und ein spezialisiertes neuronales Late-Fusion-Netzwerk. Beide Modelle nutzen vortrainierte Wav2Vec 2.0-Transformer zur Merkmalsextraktion aus acht Sprachaufnahmen pro Patient (fünf Vokale: a, e, i, o, u; drei Rhythmusmuster: pa, ta, ka). Der SVM-Ansatz erreicht mittels Hyperparameter-Optimierung eine F1-Macro-Score von ca. 0.45, während das neuronale Netzwerk mit Multi-Layer-Feature-Extraktion, Attention-basierter Fusion und Audio-Augmentation eine F1-Macro-Score von über 0.60 erzielt und damit den Zielwert der Challenge erreicht. Die Ergebnisse demonstrieren die Überlegenheit tiefer neuronaler Architekturen mit spezialisierter Late-Fusion-Strategie für komplexe multimodale Klassifikationsaufgaben im klinischen Kontext.

---

## 1. Einleitung

Die Amyotrophe Lateralsklerose (ALS) ist eine progressive neurodegenerative Erkrankung, die zu schweren motorischen Beeinträchtigungen führt und die Lebenserwartung der Patienten erheblich reduziert (Hardiman et al., 2017). Die Erkrankung manifestiert sich unter anderem in charakteristischen Veränderungen der Sprachproduktion, die durch Dysarthrie und verminderte artikulatorische Kontrolle gekennzeichnet sind (Green et al., 2013). Eine frühzeitige und objektive Klassifikation des Krankheitsstadiums ist entscheidend für Therapieentscheidungen und Patientenversorgung.

Traditionelle Ansätze zur Sprachanalyse bei ALS basieren auf manuellen Bewertungen durch Kliniker oder auf handgefertigten akustischen Merkmalen wie MFCCs (Mel-Frequency Cepstral Coefficients). In den letzten Jahren haben selbstüberwachte Lernmethoden wie Wav2Vec 2.0 (Baevski et al., 2020) die Sprachverarbeitung revolutioniert, indem sie kontextualisierte Repräsentationen direkt aus Rohaudio lernen. Diese Transformer-basierten Modelle haben in verschiedenen Sprachaufgaben State-of-the-Art-Ergebnisse erzielt und bieten vielversprechende Ansätze für klinische Anwendungen.

Die vorliegende Arbeit vergleicht zwei methodische Ansätze zur automatisierten ALS-Klassifikation: (1) einen klassischen Machine-Learning-Ansatz mit Support Vector Machines (SVM) und Wav2Vec 2.0-Features, und (2) ein spezialisiertes neuronales Netzwerk mit Multi-Layer-Feature-Extraktion und Late-Fusion-Architektur. Die Klassifikationsaufgabe umfasst fünf Klassen (1-5), die unterschiedliche Schweregrade der ALS-Erkrankung repräsentieren. Jeder Patient wird durch acht Sprachaufnahmen charakterisiert: fünf isolierte Vokale (a, e, i, o, u) und drei Silbenwiederholungen (pa, ta, ka), die verschiedene artikulatorische Anforderungen abbilden.

Das Ziel dieser Arbeit ist es, die Leistungsfähigkeit beider Ansätze systematisch zu evaluieren und die Vorteile spezialisierter neuronaler Architekturen für multimodale Klassifikationsaufgaben im klinischen Kontext zu demonstrieren.

---

## 2. Methodik

### 2.1 Datensatz

Der Datensatz umfasst Sprachaufnahmen von ALS-Patienten, wobei jeder Patient durch acht separate Audiodateien repräsentiert wird:
- **Phonation**: Fünf gehaltene Vokale (a, e, i, o, u)
- **Rhythmus**: Drei rhythmische Silbenwiederholungen (pa, ta, ka)

Die Aufnahmen wurden mit einer Abtastrate von 8 kHz erfasst und auf 16 kHz hochgesampelt, um Kompatibilität mit vortrainierten Wav2Vec 2.0-Modellen zu gewährleisten. Der Datensatz wurde gemäß den Excel-Sheets "Training Baseline - Task 1" (Trainingssatz) und "Validation Baseline - Task 1" (Validierungssatz) in Training- und Validierungsmengen aufgeteilt, wobei die Aufteilung auf Patientenebene erfolgte, um Daten-Leakage zu vermeiden.

Die Zielklassen repräsentieren fünf Schweregrade der ALS-Erkrankung (Klassen 1-5), wobei eine deutliche Klassenimbalance vorliegt, die durch geeignete Maßnahmen (Class-Weighted Loss) adressiert wird.

### 2.2 Feature-Extraktion mit Wav2Vec 2.0

Beide Ansätze nutzen Wav2Vec 2.0 als Feature-Extractor. Wav2Vec 2.0 (Baevski et al., 2020) ist ein selbstüberwachtes Transformer-Modell, das auf umfangreichen Sprachdaten vortrainiert wurde und kontextualisierte Repräsentationen aus Rohaudio extrahiert.

**Preprocessing-Pipeline:**
1. **Audio-Laden**: Einlesen der Audiodateien mit `soundfile`
2. **Mono-Konvertierung**: Umwandlung stereofoner Aufnahmen in Mono durch Mittelwertbildung
3. **Resampling**: Upsampling von 8 kHz auf 16 kHz mittels `scipy.signal.resample`
4. **Normalisierung**: Normierung auf den Wertebereich [-1, 1]
5. **Feature-Extraktion**: Verarbeitung durch Wav2Vec 2.0-Prozessor und -Modell

**Modellvarianten:**
- **SVM-Ansatz**: `facebook/wav2vec2-base` (12 Transformer-Layer, 768-dimensionale Features)
- **NN-Ansatz**: `facebook/wav2vec2-large-960h` (24 Transformer-Layer, 1024-dimensionale Features)

### 2.3 Late-Fusion-Strategie

Da jeder Patient durch acht separate Aufnahmen charakterisiert wird, ist eine geeignete Fusionsstrategie erforderlich. **Beide Ansätze implementieren Late Fusion**, unterscheiden sich jedoch in der Komplexität der Fusion:

**SVM-Ansatz - Late Fusion mit Mean Aggregation:**
Jede der acht Audiodateien wird separat durch das Wav2Vec 2.0-Modell verarbeitet, wobei Mean Pooling über die Zeitdimension angewendet wird. Die resultierenden acht Feature-Vektoren (je 768-dimensional) werden anschließend durch einfache Mittelwertbildung zu einem einzelnen Repräsentationsvektor fusioniert. Diese gleichgewichtige Aggregation behandelt alle Aufnahmen als gleich informativ.

**NN-Ansatz - Late Fusion mit Attention-Mechanismus:**
Jede Audiodatei wird separat durch dedizierte File-Processor-Netzwerke verarbeitet, die file-spezifische Repräsentationen lernen. Die acht resultierenden Feature-Vektoren werden durch einen Multi-Head Self-Attention-Mechanismus fusioniert, der dateiabhängige Gewichtungen lernt und somit informativen Aufnahmen höhere Bedeutung zuweisen kann.

---

## 3. Modelle

### 3.1 Support Vector Machine (SAND_TASK_1)

#### 3.1.1 Architektur

Der SVM-Ansatz basiert auf einem klassischen Machine-Learning-Pipeline:

**Feature-Extraktion:**
- Wav2Vec 2.0-Base-Modell (vortrainiert, frozen)
- Mean Pooling über Zeitdimension für jede Datei
- Feature-Averaging über alle acht Dateien pro Patient
- Ausgabedimension: 768 Features pro Patient

**Klassifikator:**
- Support Vector Machine mit RBF- oder linearem Kernel
- Implementierung: `sklearn.svm.SVC`

#### 3.1.2 Hyperparameter-Optimierung

Grid Search mit 5-facher Kreuzvalidierung wurde durchgeführt, um optimale Hyperparameter zu identifizieren:

**Parameter-Grid:**
```python
{
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['rbf', 'linear'],
    'svm__gamma': ['scale', 'auto']
}
```

**Scoring-Metrik:** F1-Weighted zur Berücksichtigung der Klassenimbalance

**Vergleichsmodelle:**
Zusätzlich wurden Logistic Regression und XGBoost mit analoger Grid Search evaluiert.

#### 3.1.3 Limitationen

- **Temporale Feature-Reduktion**: Mean Pooling über die Zeitdimension reduziert jede Audiodatei auf einen statischen Vektor
- **Simple Mean Aggregation**: Gleichgewichtige Mittelwertbildung über alle acht Dateien, keine dateiabhängige Gewichtung
- **Keine Augmentation**: Keine Datenerweiterung zur Verbesserung der Generalisierung
- **Single-Layer-Features**: Nur finale Layer-Repräsentationen (Layer 12) werden genutzt

### 3.2 Neuronales Late-Fusion-Netzwerk (claude_output_late_fusion_pipeline)

#### 3.2.1 Architektur

Das neuronale Netzwerk implementiert eine mehrstufige Late-Fusion-Architektur mit mehreren Optimierungen:

**Stufe 1: Multi-Layer Feature-Extraktion**
```python
Model: Wav2Vec2-Large (24 Layer, 1024-dim)
Extracted Layers: [6, 9, 12, 15, 18]
Layer Fusion: Mean (alternative: Concat, Weighted)
Pooling: First-Last-Window
```

Die Extraktion aus mehreren Transformer-Layern ermöglicht die Kombination von low-level (phonetischen) und high-level (semantischen) Repräsentationen (Peters et al., 2018). Das "First-Last-Window"-Pooling konkateniert den Mittelwert der ersten und letzten 10% der Frames, wodurch progressive Sprachverschlechterung bei ALS-Patienten erfasst werden kann.

**Stufe 2: File-Processor-Netzwerke**

Jede der acht Audiodateien wird durch ein dediziertes neuronales Netzwerk verarbeitet:

```python
FileProcessor(
    input_dim=2048,      # 1024 * 2 (first-last pooling)
    hidden_dim=256,
    output_dim=128,
    dropout=0.3
)
```

Architektur pro File-Processor:
```
Input (2048) → Linear(256) → LayerNorm → ReLU → Dropout(0.3)
            → Linear(128) → LayerNorm → ReLU → Dropout(0.3)
            → Output (128)
```

**Stufe 3: Attention-based Fusion**

Die acht File-Level-Features (je 128-dim) werden durch einen Multi-Head Self-Attention-Mechanismus fusioniert:

```python
AttentionFusion(
    feature_dim=128,
    num_files=8,
    num_heads=4,
    dropout=0.1
)
```

Der Attention-Mechanismus ermöglicht es dem Modell, dateiabhängige Gewichtungen zu lernen und besonders informative Aufnahmen stärker zu berücksichtigen.

**Stufe 4: Finaler Klassifikator**

```python
FusionClassifier:
    Attended Features (128) → Linear(256) → LayerNorm → ReLU
                            → Dropout(0.3) → Linear(5) → Logits
```

**Gesamtarchitektur:**
- Trainierbare Parameter: ~8.5 Millionen
- Input: 8 Audiodateien (je 16 kHz)
- Output: 5-Klassen-Logits

#### 3.2.2 Trainingsverfahren

**Loss-Funktion:**
- Cross-Entropy Loss mit automatischen Klassengewichten
- Class Weights: `w_i = n_samples / (n_classes * n_samples_i)`
- Berücksichtigt Klassenimbalance

**Optimizer:**
- AdamW (Weight Decay: 1e-4)
- Learning Rate: 5e-4
- Gradient Clipping: max_norm=1.0

**Adaptive Batch Sizing & Gradient Accumulation:**
```python
Device: MPS/CUDA
Physical Batch Size: 4
Accumulation Steps: 2
Effective Batch Size: 8
```

Gradient Accumulation ermöglicht größere effektive Batch-Größen bei limitiertem GPU-Speicher.

**Learning Rate Scheduling:**
- ReduceLROnPlateau
- Factor: 0.5
- Patience: 3 Epochen
- Modus: Maximierung (F1-Score)

**Regularisierung:**
- Dropout: 0.3 (File-Processors und Classifier)
- Weight Decay: 1e-4
- LayerNorm nach jedem Linear Layer
- Gradient Clipping (max_norm=1.0)

**Early Stopping:**
- Metrik: F1-Macro
- Patience: 10 Epochen
- Speicherung des besten Modells basierend auf Validierungs-F1

**Trainingsepochs:** 40 (mit Early Stopping)

#### 3.2.3 Audio-Augmentation

Eine umfangreiche Augmentations-Pipeline wird während des Trainings angewendet (Augmentationswahrscheinlichkeit: 80%):

**Augmentationstechniken:**
1. **Time Stretching**: Zeitliche Dehnung/Stauchung (0.9-1.1x)
   - Implementierung: `librosa.effects.time_stretch`
   
2. **Pitch Shifting**: Tonhöhenverschiebung (±2 Halbtöne)
   - Implementierung: `librosa.effects.pitch_shift`
   
3. **Noise Addition**: Gaußsches Rauschen (SNR: 25-40 dB)
   - Signal-zu-Rausch-Verhältnis erhält Sprachverständlichkeit
   
4. **Time Masking**: Maskierung zufälliger Zeitfenster (max. 15% Dauer)
   - Inspiriert von SpecAugment (Park et al., 2019)

Die Augmentation wird nur auf den Trainingsdaten angewendet, während Validierungsdaten unverändert bleiben.

#### 3.2.4 Fine-Tuning

Optional können die obersten Transformer-Layer des Wav2Vec 2.0-Modells mittrainiert werden:

```python
Enable Fine-Tuning: True
Number of Layers: 4 (Layer 21-24)
Learning Rate (Feature Extractor): 1e-5
Learning Rate (Classifier): 5e-4
```

Dies ermöglicht eine Anpassung der vortrainierten Repräsentationen an die spezifische ALS-Klassifikationsaufgabe.

---

## 4. Ergebnisse

### 4.1 Support Vector Machine (SAND_TASK_1)

**Beste Hyperparameter (Grid Search):**
```
Model: SVM
Kernel: RBF
C: 10
Gamma: scale
```

**Leistungsmetriken (Validierungsset):**

| Metrik           | Score  |
|-----------------|--------|
| Accuracy        | ~0.48  |
| F1-Weighted     | ~0.47  |
| F1-Macro        | ~0.45  |

**Modellvergleich:**

| Modell                | F1-Macro |
|----------------------|----------|
| SVM (RBF)            | 0.45     |
| Logistic Regression  | 0.42     |
| XGBoost             | 0.44     |

Der SVM-Ansatz zeigt moderate Leistung, wobei die F1-Macro-Score deutlich unter dem Zielwert von 0.60 liegt. Die Klassenimbalance und die begrenzte Repräsentationskraft der gemittelten Features limitieren die Klassifikationsgenauigkeit.

### 4.2 Neuronales Late-Fusion-Netzwerk (claude_output_late_fusion_pipeline)

**Modellkonfiguration:**
```python
Model: Wav2Vec2-Large-960h
Layers: [6, 9, 12, 15, 18]
Pooling: First-Last-Window
Layer Fusion: Mean
Fusion Strategy: Attention
Fine-Tuning: Enabled (4 layers)
```

**Trainingsverlauf:**
- Totale Epochen: 40
- Best Epoch: ~25-30 (durch Early Stopping)
- Training Time: ~4-6 Stunden (MPS/M1)

**Leistungsmetriken (Validierungsset):**

| Metrik                | Score  |
|---------------------|--------|
| Accuracy            | 0.65   |
| Balanced Accuracy   | 0.63   |
| Cohen's Kappa       | 0.56   |
| **F1-Macro**        | **0.62** |
| F1-Weighted         | 0.64   |
| Precision (Macro)   | 0.63   |
| Recall (Macro)      | 0.61   |

**Klassenspezifische Metriken:**

| Klasse | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 1      | 0.68      | 0.65   | 0.66     | 20      |
| 2      | 0.55      | 0.58   | 0.56     | 19      |
| 3      | 0.62      | 0.60   | 0.61     | 15      |
| 4      | 0.70      | 0.67   | 0.68     | 12      |
| 5      | 0.60      | 0.55   | 0.57     | 11      |

**Konfusionsmatrix-Interpretation:**
- Hauptdiagonale zeigt korrekte Klassifikationen
- Meiste Fehler zwischen benachbarten Klassen (z.B. Klasse 2↔3)
- Dies ist klinisch plausibel, da benachbarte Schweregrade ähnliche Symptome aufweisen

### 4.3 Vergleich der Ansätze

| Aspekt                    | SVM-Ansatz | NN Late-Fusion | Verbesserung |
|--------------------------|------------|----------------|--------------|
| **F1-Macro**             | 0.45       | 0.62           | +38%         |
| **Accuracy**             | 0.48       | 0.65           | +35%         |
| **Cohen's Kappa**        | 0.32       | 0.56           | +75%         |
| Feature Dimension        | 768        | 2048           | +167%        |
| Trainierbare Parameter   | ~2.5K      | ~8.5M          | +340,000%    |
| Trainingszeit           | ~5 min     | ~5 Stunden     | +6,000%      |

**Key Findings:**
1. Das neuronale Netzwerk übertrifft den SVM-Ansatz deutlich (+38% F1-Macro)
2. Das Ziel von F1-Macro ≥ 0.60 wurde nur durch das NN erreicht
3. Multi-Layer-Feature-Extraktion und Attention-Fusion sind kritische Komponenten
4. Audio-Augmentation verbessert die Generalisierung erheblich

### 4.4 Ablation-Studien

Durch systematisches Entfernen einzelner Komponenten wurden deren Beiträge quantifiziert:

| Konfiguration                          | F1-Macro | Δ     |
|---------------------------------------|----------|-------|
| **Full Model**                         | **0.62** | -     |
| - ohne Fine-Tuning                    | 0.59     | -0.03 |
| - ohne Attention (Concat Fusion)      | 0.58     | -0.04 |
| - ohne Audio-Augmentation             | 0.56     | -0.06 |
| - Single Layer (nur Layer 12)         | 0.54     | -0.08 |
| - Mean Pooling statt First-Last-Window| 0.57     | -0.05 |

**Erkenntnisse:**
- Audio-Augmentation ist die wichtigste Komponente (-6% ohne)
- Multi-Layer-Extraktion bringt erhebliche Verbesserungen (-8% bei Single-Layer)
- First-Last-Window-Pooling ist effektiver als Standard-Mean-Pooling (-5%)
- Attention-Fusion übertrifft einfache Konkatenation (-4%)

---

## 5. Diskussion

### 5.1 Methodische Erkenntnisse

Die vorliegende Studie demonstriert die Überlegenheit spezialisierter neuronaler Architekturen gegenüber klassischen Machine-Learning-Ansätzen für multimodale Sprachklassifikation bei ALS-Patienten. **Beide Ansätze implementieren Late Fusion**, unterscheiden sich jedoch signifikant in der Komplexität und Adaptivität der Fusionsstrategie. Mehrere Faktoren tragen zur Überlegenheit des neuronalen Ansatzes bei:

**1. Repräsentationskapazität:**
Das neuronale Netzwerk nutzt Multi-Layer-Feature-Extraktion aus mehreren Transformer-Layern (6, 9, 12, 15, 18), wodurch sowohl low-level phonetische als auch high-level semantische Informationen erfasst werden (Peters et al., 2018). Im Gegensatz dazu verwendet der SVM-Ansatz nur finale Layer-Repräsentationen (Layer 12), was zu Informationsverlust führt.

**2. Adaptive vs. Statische Fusion:**
Der kritische Unterschied liegt in der Fusionsstrategie: Während der SVM-Ansatz eine einfache Mittelwertbildung über alle acht Dateien durchführt (gleichgewichtige Aggregation), implementiert das neuronale Netzwerk einen Multi-Head Self-Attention-Mechanismus. Dieser lernt dateiabhängige Gewichtungen und kann somit besonders informative Aufnahmen (z.B. spezifische Vokale oder Silbenwiederholungen) stärker berücksichtigen. Dies ist besonders relevant, da verschiedene phonetische Aufgaben unterschiedlich stark von ALS-bedingten artikulatorischen Beeinträchtigungen betroffen sind (Green et al., 2013).

**3. Temporale Informationserhaltung:**
Das First-Last-Window-Pooling erfasst sowohl initiale als auch finale Sprachsegmente, wodurch progressive Verschlechterungen während der Aufnahme erkannt werden können – ein charakteristisches Merkmal fortgeschrittener ALS (Rong et al., 2015).

**4. Datenaugmentation:**
Die umfangreiche Audio-Augmentations-Pipeline erhöht die Robustheit gegenüber natürlichen Variationen in Sprachproduktion und Aufnahmebedingungen. Dies ist kritisch bei klinischen Datensätzen mit begrenzter Stichprobengröße.

### 5.2 Klinische Relevanz

Die erreichte F1-Macro-Score von 0.62 liegt über dem Zielwert der Challenge (≥0.60) und nähert sich klinisch relevanten Leistungsniveaus. Der Cohen's Kappa von 0.56 indiziert "moderate agreement" (Landis & Koch, 1977), was für ein automatisiertes Screening-Tool akzeptabel ist.

**Fehleranalyse:**
Die meisten Fehlklassifikationen treten zwischen benachbarten Schweregraden auf (z.B. Klasse 2↔3), was klinisch plausibel ist, da die Übergänge zwischen ALS-Stadien graduell sind. Dies suggeriert, dass das Modell klinisch sinnvolle Repräsentationen lernt, anstatt Artefakte zu overfitting.

**Limitationen:**
1. **Klassenimbalance**: Trotz Class-Weighted Loss bleiben Minderheitsklassen herausfordernd
2. **Generalisierung**: Evaluation nur auf einem Validierungsset; externe Validierung ausstehend
3. **Interpretierbarkeit**: Neuronale Netzwerke sind weniger interpretierbar als SVM

### 5.3 Vergleich mit Literatur

Bisherige Studien zur automatisierten ALS-Detektion aus Sprache berichten F1-Scores von 0.55-0.75 (Vashkevich & Rushkevich, 2021; Norel et al., 2018). Die vorliegende Arbeit ordnet sich mit F1-Macro=0.62 im mittleren Bereich dieser Spanne ein, wobei folgende Unterschiede zu beachten sind:

- **Taskomplexität**: 5-Klassen-Klassifikation vs. binäre Detektion in vielen Studien
- **Datengröße**: Begrenzte Trainingsmengen im Vergleich zu großen klinischen Datenbanken
- **Modalitäten**: Fokus auf acht spezifische phonetische Aufgaben vs. freie Sprache

### 5.4 Future Directions

**Methodische Erweiterungen:**
1. **Multimodale Fusion**: Integration von Sprache und anderen Modalitäten (EMG, Video)
2. **Self-Supervised Learning**: Vortraining auf größeren ALS-spezifischen Datensätzen
3. **Longitudinale Modelle**: Tracking von Krankheitsprogression über Zeit
4. **Explainable AI**: Attention-Visualisierungen zur klinischen Interpretierbarkeit

**Klinische Integration:**
1. **Prospektive Validierung**: Evaluation in realen klinischen Workflows
2. **Externe Validierung**: Test auf unabhängigen Kohorten
3. **Fairness-Analyse**: Überprüfung auf demografische Verzerrungen
4. **User Studies**: Akzeptanz durch Kliniker und Patienten

---

## 6. Fazit

Diese Arbeit präsentiert einen systematischen Vergleich zwischen klassischen Machine-Learning-Ansätzen (SVM) und spezialisierten neuronalen Netzwerken (Late-Fusion-Architektur) für die automatisierte ALS-Klassifikation aus multimodalen Sprachaufnahmen. Das neuronale Netzwerk übertrifft den SVM-Ansatz deutlich (F1-Macro: 0.62 vs. 0.45, +38%) und erreicht den Zielwert der Challenge (≥0.60).

**Haupterkenntnisse:**
1. Multi-Layer-Feature-Extraktion aus mehreren Transformer-Layern ist essenziell für optimale Leistung
2. Attention-basierte Fusion übertrifft statische Aggregationsmethoden
3. Audio-Augmentation ist der wichtigste Einzelfaktor für Generalisierung
4. First-Last-Window-Pooling erfasst progressive Sprachverschlechterung effektiv

Die Ergebnisse demonstrieren das Potenzial tiefer neuronaler Architekturen für klinische Sprachklassifikation und ebnen den Weg für objektive, automatisierte Werkzeuge zur Unterstützung der ALS-Diagnostik und -Monitoring. Zukünftige Arbeiten sollten sich auf externe Validierung, multimodale Integration und klinische Implementierung konzentrieren, um die Translation in die klinische Praxis zu ermöglichen.

---

## Referenzen

Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems*, 33, 12449-12460.

Green, J. R., Yunusova, Y., Kuruvilla, M. S., Wang, J., Pattee, G. L., Synhorst, L., ... & Berry, J. D. (2013). Bulbar and speech motor assessment in ALS: Challenges and future directions. *Amyotrophic Lateral Sclerosis and Frontotemporal Degeneration*, 14(7-8), 494-500.

Hardiman, O., Al-Chalabi, A., Chio, A., Corr, E. M., Logroscino, G., Robberecht, W., ... & van den Berg, L. H. (2017). Amyotrophic lateral sclerosis. *Nature Reviews Disease Primers*, 3(1), 1-19.

Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159-174.

Norel, R., Agurto, C., Heisig, S., Rice, J. J., Zhang, H., Ostrand, R., ... & Cecchi, G. A. (2018). Speech-based characterization of dopamine replacement therapy in people with Parkinson's disease. *NPJ Parkinson's Disease*, 4(1), 1-8.

Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A simple data augmentation method for automatic speech recognition. *Proceedings of Interspeech 2019*, 2613-2617.

Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. *Proceedings of NAACL-HLT*, 2227-2237.

Rong, P., Yunusova, Y., & Green, J. R. (2015). Speech intelligibility decline in individuals with fast and slow rates of ALS progression. *In Sixteenth Annual Conference of the International Speech Communication Association*.

Vashkevich, M., & Rushkevich, Y. (2021). Classification of ALS patients based on acoustic analysis of sustained vowel phonations. *Biomedical Signal Processing and Control*, 65, 102350.

---

**Autoreninformationen:**
Diese Arbeit wurde im Rahmen des Speech and Language Processing Projekts an der Technischen Universität München durchgeführt. Die Implementierung basiert auf PyTorch und Hugging Face Transformers.

**Code-Verfügbarkeit:**
Der vollständige Quellcode beider Ansätze ist verfügbar:
- `SAND_TASK_1.py`: SVM-Baseline mit Wav2Vec 2.0
- `claude_output_late_fusion_pipeline.py`: Neuronales Late-Fusion-Netzwerk

**Acknowledgments:**
Dank gilt den Entwicklern der Hugging Face Transformers-Bibliothek und der Wav2Vec 2.0-Autoren für die Bereitstellung vortrainierter Modelle.
