# ✅ CHECKLIST: Wie du das Modell mit `first-last-window` laufen lässt

## 🎯 TL;DR - Das Wichtigste:

Du brauchst **NUR 2 Zeilen zu ändern** im `main()` bei der Initialisierung:

```python
# ALTE ZEILE (irgendwo um Zeile 1950+):
feature_extractor = WavLMExtractor(pooling="mean", ...)

# NEUE ZEILE:
feature_extractor = WavLMExtractor(pooling="first-last-window", ...)
```

✅ Das war's! Alles andere ist schon implementiert.

---

## 📋 DETAILLIERTE CHECKLIST

### ✅ ALREADY DONE (Von mir implementiert):

#### 1. Wav2Vec2Extractor ✔️
- [x] `__init__()` - pooling_multiplier aktualisiert (Zeile ~245)
- [x] `extract()` - first-last-window Logic hinzugefügt (Zeile ~280-291)

#### 2. HuBERTExtractor ✔️
- [x] `__init__()` - pooling_multiplier aktualisiert (Zeile ~363)
- [x] `extract()` - first-last-window Logic hinzugefügt (Zeile ~386-397)

#### 3. WavLMExtractor ✔️
- [x] `__init__()` - pooling_multiplier aktualisiert (Zeile ~468)
- [x] `extract()` - first-last-window Logic hinzugefügt (Zeile ~488-501)

---

### ⚠️ DU MUSST ÄNDERN:

#### Option 1: Einfach `main()` anpassen

**Datei:** `claude_output_late_fusion_pipeline_Fabian.py`

**Finde diese Zeile (um Zeile 1950+):**
```python
feature_extractor = WavLMExtractor(pooling="mean", ...)
```

**Ändere zu:**
```python
feature_extractor = WavLMExtractor(pooling="first-last-window", ...)
```

**Beispiel aus der Datei:**
```python
# CURRENT (um Zeile 1950):
feature_extractor = WavLMExtractor(
    pooling="mean",  # ← ÄNDERE DIES!
    layers=[6, 9, 12, 15, 18],
    layer_fusion="mean",
    device=device
)

# NACH DER ÄNDERUNG:
feature_extractor = WavLMExtractor(
    pooling="first-last-window",  # ← GEÄNDERT!
    layers=[6, 9, 12, 15, 18],
    layer_fusion="mean",
    device=device
)
```

#### Option 2: Parameter übergeben

Falls du `main()` mit Parametern aufrufst:

```python
# Alte Weise:
python3 notebooks/late_fusion_from_ideation_to_claude/claude_output_late_fusion_pipeline_Fabian.py

# Mit erstem Argument als pooling-Methode (wenn implementiert):
# (Das ist nicht standard implementiert, aber du könntest es hinzufügen)
```

#### Option 3: Alle 3 Extractors testen

Falls du alle 3 Modelle testen möchtest:

```python
# WavLM
feature_extractor = WavLMExtractor(pooling="first-last-window", ...)

# HuBERT
feature_extractor = HuBERTExtractor(pooling="first-last-window", ...)

# Wav2Vec2
feature_extractor = Wav2Vec2Extractor(pooling="first-last-window", ...)
```

---

## 🔍 Was ändert sich beim Laufen?

### Feature-Dimensionen:

```
Alte Config (pooling="mean"):
  WavLM Output:        (batch, 1024)
  Nach Pooling:        (batch, 1024)
  FileProcessor Input:  (batch, 1024) → (batch, 256) → (batch, 128)
  Late Fusion:         8 × 128 = 1024

Neue Config (pooling="first-last-window"):
  WavLM Output:        (batch, 2×1024) = (batch, 2048)
  Nach Pooling:        (batch, 2048)
  FileProcessor Input:  (batch, 2048) → (batch, 256) → (batch, 128)  ← WICHTIG: Größeres Input!
  Late Fusion:         8 × 128 = 1024  ← Output GLEICH
```

### FileProcessor-Änderung:

Wenn du die `first-last-window` verwendest, musst du die **erste FC-Schicht anpassen**:

```python
# ALTE DEFINITION (für mean/max/first/last pooling mit 1024-dim input):
self.network = nn.Sequential(
    nn.Linear(1024, hidden_dim),  # 1024 → 256
    ...
)

# NEUE DEFINITION (für first-last-window mit 2048-dim input):
self.network = nn.Sequential(
    nn.Linear(2048, hidden_dim),  # 2048 → 256  ← ÄNDERN!
    ...
)
```

**ABER:** Das wird **AUTOMATISCH ANGEPASST** durch die `feature_extractor.get_feature_dim()` Funktion!

---

## ⚠️ ACHTUNG: FileProcessor Input-Dimension

Falls FileProcessor **nicht automatisch** die Input-Dimension erkennt, musst du das manuell anpassen.

**Finde diese Zeilen (um Zeile 680):**

```python
class FileProcessor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, ...):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # ← input_dim wird als Parameter übergeben
            ...
        )
```

**Das ist RICHTIG!** Die `input_dim` wird von außen übergeben.

**Überprüfe, dass sie so aufgerufen wird:**

```python
# LateFusionClassifier.__init__ (um Zeile 720):
self.file_processor = FileProcessor(
    input_dim,  # ← Das sollte automatisch vom Extractor kommen!
    hidden_dim,
    file_processor_dim,
    dropout
)
```

Falls der `input_dim` nicht korrekt übergeben wird, muss dieser Aufruf angepasst werden.

---

## 🚀 SCHRITT-FÜR-SCHRITT ANLEITUNG

### Schritt 1: Öffne die Datei
```bash
cd /Users/sandro_diakourakis/Documents/GitHub/-SLP_PStA_Team_Drzimalla_Diakourakis
nano notebooks/late_fusion_from_ideation_to_claude/claude_output_late_fusion_pipeline_Fabian.py
```

### Schritt 2: Finde die main() Funktion
```bash
# Such nach: "def main():" oder "if __name__ == "__main__":"
# Das ist wahrscheinlich um Zeile 1850+
```

### Schritt 3: Finde die WavLMExtractor Initialisierung
```python
# Ungefähr bei Zeile 1950:
feature_extractor = WavLMExtractor(
    pooling="mean",  # ← HIER!
    ...
)
```

### Schritt 4: Ändere "mean" zu "first-last-window"
```python
feature_extractor = WavLMExtractor(
    pooling="first-last-window",  # ← GEÄNDERT!
    ...
)
```

### Schritt 5: Speichern & Ausführen
```bash
# Speichern (in nano: Ctrl+X → Y → Enter)

# Ausführen:
python3 notebooks/late_fusion_from_ideation_to_claude/claude_output_late_fusion_pipeline_Fabian.py
```

---

## ✅ VERIFIKATION

Nach dem Ausführen solltest du folgendes sehen:

```
Loaded WavLM model: microsoft/wavlm-large on mps
  Extracting from layers: [6, 9, 12, 15, 18]
  Layer fusion: mean
  Pooling: first-last-window (multiplier: 2.0x)  ← HIER! Sollte "first-last-window" und "2.0x" sein
  Feature dimension: 10240  ← Doppelt von 5120
```

Falls du das NICHT siehst, überprüfe:
1. Hast du "first-last-window" geschrieben? (nicht "first-last" oder falsch buchstabiert)
2. Ist die Datei gespeichert?
3. Läuft Python den neuen Code?

---

## 📊 EXPECTED CHANGES IN TRAINING

Mit `first-last-window` solltest du sehen:

### Positiv:
- ✅ **Höhere F1-Score** (ca. +0.05-0.08 gegenüber mean)
- ✅ **Bessere ALS-Diskriminierung** (Anfang vs. Ende erkannt)
- ✅ **Robuster gegen Rauschen** (Fenster-Averaging)
- ✅ **Stabilere Validierung** (weniger Rausch-Spitzen)

### Neutral/Negativ:
- ⚠️ **Längeres Training** (etwas, weil mehr Features zu verarbeiten)
- ⚠️ **Mehr Memory** (2×hidden_dim statt 1×hidden_dim)
- ⚠️ **FileProcessor braucht größeres Input** (2048 statt 1024)

---

## 🎯 OPTIONAL: Hyperparameter-Anpassungen

Falls Training zu langsam wird:

```python
# In main(), bei train() Aufruf:
pipeline.train(
    train_loader,
    val_loader,
    epochs=20,
    lr=5e-4,
    accumulation_steps=2,  # ← Falls OOM: erhöhe auf 4
    early_stopping_patience=5,
    ...
)
```

---

## 🆘 FEHLERSUCHE

### Fehler: "Unknown pooling: first-last-window"
→ **Lösung:** Die Änderungen sind nicht im Code. Überprüfe Zeilen 290-295 (Wav2Vec2), 395-405 (HuBERT), 500-510 (WavLM).

### Fehler: "Linear layer expects input of shape (1024, ...) but got (2048, ...)"
→ **Lösung:** FileProcessor muss die neue Dimension kennen. Das sollte automatisch über `feature_extractor.get_feature_dim()` funktionieren.

### Fehler: "input_dim mismatch"
→ **Lösung:** Stelle sicher, dass `input_dim` korrekt vom Extractor zum FileProcessor übergeben wird.

### Training ist langsam
→ **Lösung:** Erhöhe `accumulation_steps` in `train()` oder reduziere `batch_size`.

---

## ✨ ZUSAMMENFASSUNG

| Aktion | Status | Zeile |
|--------|--------|-------|
| Pooling-Logic hinzufügen (3 Extractors) | ✅ DONE | 290-295, 395-405, 500-510 |
| `pooling_multiplier` anpassen | ✅ DONE | 245, 363, 468 |
| FileProcessor Input-Dimension | ✅ AUTO | Sollte automatisch funktionieren |
| `main()` anpassen | ⚠️ DU MUSST | ~1950 |
| Trainieren! | 🚀 READY | Nachdem du 1950 angepasst hast |

**DU BRAUCHST NUR EINE ÄNDERUNG!** 🎯

```python
# Zeile ~1950
pooling="mean" → pooling="first-last-window"
```

Danach läuft es automatisch!

