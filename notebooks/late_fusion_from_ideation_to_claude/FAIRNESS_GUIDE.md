# Fairness Regularization - Nutzungsanleitung

## Übersicht

Die Fairness-Regularisierung wurde erfolgreich in das Late Fusion Training-Setup integriert. Diese Funktion stellt sicher, dass das Modell über beide Geschlechter (männlich/weiblich) hinweg gleich gut lernt, ohne das Geschlecht explizit als Feature zu nutzen oder vorherzusagen.

## Kernkonzept

**Verlustfunktion:**
```
loss_gesamt = loss_haupt + lambda_fair × |loss_männlich - loss_weiblich|
```

- `loss_haupt`: Standard CrossEntropyLoss für Klassifikation
- `loss_männlich`: Durchschnittlicher Loss für männliche Sprecher in der Batch
- `loss_weiblich`: Durchschnittlicher Loss für weibliche Sprecher in der Batch
- `lambda_fair`: Gewichtungsfaktor (Standard: 0.1)

## Excel-Datei Vorbereitung

Ihre Excel-Datei muss eine **'Sex'**-Spalte enthalten:

| ID    | Age | Sex | Class |
|-------|-----|-----|-------|
| ID001 | 45  | m   | 3     |
| ID002 | 52  | w   | 2     |
| ID003 | 38  | f   | 4     |

**Unterstützte Werte für 'Sex':**
- **Männlich**: `'m'`, `'male'`, `'männlich'`, `'M'`
- **Weiblich**: `'w'`, `'f'`, `'female'`, `'weiblich'`, `'W'`, `'F'`
- **Unbekannt**: Leeres Feld oder `NaN`

## Verwendung im Code

### Standard-Training mit Fairness (lambda_fair=0.1)

```python
pipeline = LateFusionPipeline(feature_extractor, model, device=DEVICE)

pipeline.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=5e-4,
    weight_decay=1e-4,
    lambda_fair=0.1  # Fairness aktiviert (Standard)
)
```

### Training OHNE Fairness-Regularisierung

```python
pipeline.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=5e-4,
    weight_decay=1e-4,
    lambda_fair=0.0  # Fairness deaktiviert
)
```

### Training mit stärkerer Fairness-Regularisierung

```python
pipeline.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    lr=5e-4,
    weight_decay=1e-4,
    lambda_fair=0.5  # Stärkere Fairness-Gewichtung
)
```

## Logging-Ausgabe

Wenn Fairness aktiviert ist (`lambda_fair > 0`), werden während des Trainings zusätzliche Metriken geloggt:

```
Epoch 1/20: Train Loss=1.2345, Train Acc=0.4520, Val Loss=1.3456, Val Acc=0.4302, Val F1=0.4027
  Fairness: loss_male=1.2100, loss_female=1.2590, gap=0.0490
```

**Interpretation:**
- `loss_male`: Durchschnittlicher Verlust für männliche Sprecher
- `loss_female`: Durchschnittlicher Verlust für weibliche Sprecher
- `gap`: Absolute Differenz zwischen den beiden Verlusten
- **Ziel**: `gap` sollte während des Trainings klein werden (idealerweise < 0.1)

## Hyperparameter-Tuning

### Empfohlene lambda_fair-Werte

| lambda_fair | Anwendungsfall |
|-------------|----------------|
| 0.0         | Fairness deaktiviert, Standard-Training |
| 0.05 - 0.1  | Leichte Fairness-Regularisierung (empfohlen für Start) |
| 0.2 - 0.5   | Mittlere Fairness-Regularisierung (bei starker Ungleichheit) |
| 0.5 - 1.0   | Starke Fairness-Regularisierung (Achtung: Kann Hauptaufgabe beeinträchtigen!) |

### Wie finde ich den optimalen lambda_fair-Wert?

1. **Baseline erstellen**: Trainieren Sie mit `lambda_fair=0.0` und notieren Sie F1-Score
2. **Fairness testen**: Trainieren Sie mit verschiedenen lambda_fair-Werten (0.05, 0.1, 0.2, 0.5)
3. **Trade-off beobachten**: 
   - Schauen Sie auf die `gap`-Metrik (sollte sinken)
   - Prüfen Sie, ob Val F1-Score stabil bleibt oder sich verbessert
4. **Best Practice**: Wählen Sie den lambda_fair-Wert, der:
   - Die Fairness-Lücke reduziert (gap < 0.1)
   - Dabei die Validierungs-Performance nicht verschlechtert

## Technische Details

### Interne Kodierung

- Männlich → `0`
- Weiblich → `1`
- Unbekannt → `None` (in Excel) → `-1` (im Tensor)

### Numerische Stabilität

Die Implementierung ist robust:
- ✅ Fairness-Loss wird nur berechnet, wenn beide Geschlechter in der Batch vorhanden sind
- ✅ Samples mit unbekanntem Geschlecht (`sex == -1`) werden bei Fairness-Berechnung übersprungen
- ✅ Keine Division durch Null oder NaN-Werte

### Keine Feature-Leakage

**Wichtig**: Das Geschlecht wird NICHT als Input-Feature verwendet!
- ✅ Geschlecht wird nur für Fairness-Berechnung während des Trainings genutzt
- ✅ Model-Architektur bleibt unverändert
- ✅ Während der Inferenz (Predict) wird Geschlecht ignoriert

## Experimentieren

### Experiment 1: Baseline vs. Fairness

```python
# Experiment A: Ohne Fairness
pipeline_baseline = LateFusionPipeline(feature_extractor, model, 
                                       experiment_name="baseline_no_fairness")
pipeline_baseline.train(..., lambda_fair=0.0)

# Experiment B: Mit Fairness
pipeline_fair = LateFusionPipeline(feature_extractor, model, 
                                   experiment_name="with_fairness_0.1")
pipeline_fair.train(..., lambda_fair=0.1)

# Vergleichen Sie die Ergebnisse in experiments/ Ordner
```

### Experiment 2: Lambda-Fair Grid Search

```python
lambda_values = [0.0, 0.05, 0.1, 0.2, 0.5]

for lambda_val in lambda_values:
    pipeline = LateFusionPipeline(
        feature_extractor, model,
        experiment_name=f"fairness_lambda_{lambda_val}"
    )
    pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        lr=5e-4,
        lambda_fair=lambda_val
    )
```

## Fehlerbehebung

### Problem: "Sex column not found in Excel sheet"

**Lösung**: Fügen Sie eine 'Sex'-Spalte zu Ihrer Excel-Datei hinzu mit Werten `'m'` oder `'w'`

### Problem: Fairness-Metriken werden nicht geloggt

**Mögliche Ursachen**:
1. `lambda_fair=0.0` → Fairness ist deaktiviert
2. Alle Samples haben `sex=None` → Keine Fairness-Berechnung möglich
3. Batches enthalten nur ein Geschlecht → Fairness-Loss wird übersprungen

**Lösung**: Prüfen Sie, ob die 'Sex'-Spalte korrekt befüllt ist und `lambda_fair > 0`

### Problem: Training wird instabil mit hohem lambda_fair

**Symptom**: Loss springt stark, kein Konvergenz

**Lösung**: Reduzieren Sie `lambda_fair` auf 0.1 oder niedriger

## Best Practices

1. ✅ **Immer zuerst Baseline trainieren**: `lambda_fair=0.0` als Vergleich
2. ✅ **Kleine Schritte**: Beginnen Sie mit `lambda_fair=0.1`
3. ✅ **Fairness-Metriken beobachten**: `gap` sollte über Epochen sinken
4. ✅ **Performance nicht opfern**: Val F1 sollte nicht signifikant schlechter werden
5. ✅ **Experiment-Tracking nutzen**: Alle Metriken werden automatisch in `experiments/` gespeichert

## Zusammenfassung

Die Fairness-Regularisierung ist jetzt vollständig in Ihr Training integriert:

- 📁 **Excel**: Fügen Sie 'Sex'-Spalte hinzu
- 🔧 **Training**: Setzen Sie `lambda_fair` Parameter
- 📊 **Monitoring**: Beobachten Sie Fairness-Metriken im Log
- 💾 **Tracking**: Alle Ergebnisse werden in `experiments/` gespeichert

**Nächste Schritte:**
1. Prüfen Sie Ihre Excel-Datei auf 'Sex'-Spalte
2. Trainieren Sie mit `lambda_fair=0.1`
3. Vergleichen Sie mit Baseline (`lambda_fair=0.0`)
4. Optimieren Sie `lambda_fair` basierend auf Ergebnissen
