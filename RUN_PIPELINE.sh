#!/bin/bash
# FINALER TEST - Führt die komplette Pipeline aus

echo "🚀 SAND Task 1 - Finale Pipeline Execution"
echo "=========================================="
echo ""

# Aktiviere Virtual Environment
source venv/bin/activate

# Schritt 1: Data Preparation (sollte bereits gemacht sein)
echo "📋 Schritt 1: Überprüfe Manifests..."
if [ -f "data/manifests/train_manifest.csv" ] && [ -f "data/manifests/test_manifest.csv" ]; then
    echo "✅ Manifests vorhanden"
else
    echo "⚠️  Manifests fehlen - führe prep und split aus..."
    python3 -m sandcli.main command=prep
    python3 -m sandcli.main command=split
fi

echo ""

# Schritt 2: Feature Extraction (sollte bereits gemacht sein)
echo "🎵 Schritt 2: Überprüfe Features..."
if [ -d "features/mfcc/train" ] && [ -f "features/mfcc/train/manifest.csv" ]; then
    echo "✅ MFCC Features vorhanden"
else
    echo "⚠️  Features fehlen - extrahiere MFCC..."
    python3 -m sandcli.main command=featurize features.type=mfcc
fi

echo ""

# Schritt 3: Model Training
echo "🤖 Schritt 3: Trainiere XGBoost Model..."
echo "Dies kann 2-5 Minuten dauern..."
python3 -m sandcli.main --config-name=baseline_ml command=train

echo ""
echo "=========================================="
echo "✅ Pipeline abgeschlossen!"
echo ""

# Zeige Ergebnisse
echo "📊 Neueste Experiment-Ergebnisse:"
LATEST_EXP=$(ls -t runs/ | grep baseline | head -1)
if [ -n "$LATEST_EXP" ]; then
    echo "Experiment: $LATEST_EXP"
    echo ""

    if [ -f "runs/$LATEST_EXP/metrics.json" ]; then
        echo "Metriken:"
        python3 << 'EOF'
import json
import sys
with open('runs/' + sys.argv[1] + '/metrics.json') as f:
    m = json.load(f)
print(f"  Accuracy: {m['accuracy']:.4f}")
print(f"  Macro F1: {m['macro_f1']:.4f}")
EOF
    else
        echo "⚠️  Training noch nicht abgeschlossen oder fehlgeschlagen"
        echo "Überprüfe: runs/$LATEST_EXP/"
    fi
fi

