#!/bin/bash
# KORRIGIERTE STUFE 1: Features NEU extrahieren und trainieren

echo "🔍 PROBLEM-DIAGNOSE"
echo "==================="
echo ""
echo "Ihr vorheriger Versuch hatte Macro F1: 0.225 (keine Verbesserung!)"
echo "Grund: Features wurden wahrscheinlich NICHT neu extrahiert."
echo ""
echo "Lösung: Wir löschen alte Features und extrahieren NEU."
echo ""

source venv/bin/activate

# WICHTIG: Lösche alte Features um Sicherzustellen dass neue extrahiert werden
echo "🗑️  Lösche alte MFCC Features..."
rm -rf features/mfcc/train/*.npy
rm -rf features/mfcc/test/*.npy
rm -f features/mfcc/train/manifest.csv
rm -f features/mfcc/test/manifest.csv
echo "✓ Alte Features gelöscht"
echo ""

echo "📊 Extrahiere NEUE erweiterte MFCC Features..."
echo "   - 20 MFCCs (statt 13)"
echo "   - 80 Mel-Bins (statt 40)"
echo "   - 12 Statistiken (statt 2)"
echo "   → Erwartete Features: ~720 (statt 78)"
echo ""
echo "⏱️  Dauer: ~8-10 Minuten"
echo ""

# Feature Extraction mit improved_features Config
python3 -m sandcli.main --config-name=improved_features command=featurize

# Validiere dass neue Features extrahiert wurden
echo ""
echo "🔍 Validiere neue Features..."
python3 << 'EOF'
import numpy as np
import pandas as pd
from pathlib import Path

train_manifest = Path('features/mfcc/train/manifest.csv')
if train_manifest.exists():
    df = pd.read_csv(train_manifest)
    if len(df) > 0:
        first_feature = np.load(df.iloc[0]['feature_path'])
        print(f"✓ Feature-Dimension: {first_feature.shape[0]}")

        if first_feature.shape[0] > 500:
            print(f"✅ ERFOLG: Neue Features wurden extrahiert!")
            print(f"   Erwartete Verbesserung: Macro F1 0.22 → 0.45-0.55")
        elif first_feature.shape[0] == 78:
            print(f"❌ FEHLER: Alte Features (78) wurden verwendet!")
            print(f"   Die Config wurde nicht richtig angewendet!")
            exit(1)
        else:
            print(f"⚠️  Unerwartete Dimension: {first_feature.shape[0]}")
    else:
        print("❌ Manifest ist leer!")
        exit(1)
else:
    print("❌ Manifest wurde nicht erstellt!")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Feature-Extraktion fehlgeschlagen!"
    exit 1
fi

echo ""
echo "🤖 Trainiere Modell mit Class Weighting..."
python3 -m sandcli.main --config-name=improved_features command=train

echo ""
echo "✅ STUFE 1 abgeschlossen!"
echo ""
echo "📊 Vergleiche Ergebnisse:"
echo "   ALT (baseline): Macro F1 = 0.22"
echo "   Schaue in: runs/improved_features_*/metrics.json"

