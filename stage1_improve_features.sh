#!/bin/bash
# STUFE 1: Verbesserte Features extrahieren und trainieren

echo "🚀 STUFE 1: FEATURE VERBESSERUNG"
echo "================================="
echo ""

source venv/bin/activate

echo "📊 Extrahiere erweiterte MFCC Features..."
echo "   - 20 MFCCs (statt 13)"
echo "   - 80 Mel-Bins (statt 40)"
echo "   - 12 Statistiken (statt 2)"
echo "   → Erwartete Features: ~720 (statt 78)"
echo ""
echo "⏱️  Dauer: ~8-10 Minuten"
echo ""

# Feature Extraction mit verbesserter Config
# Verwende +command statt command= um Struct-Mode zu umgehen
python3 -m sandcli.main --config-name=improved_features +command=featurize

echo ""
echo "🤖 Trainiere Modell mit Class Weighting..."
python3 -m sandcli.main --config-name=improved_features +command=train

echo ""
echo "✅ STUFE 1 abgeschlossen!"
echo ""
echo "Erwartete Verbesserung:"
echo "  Macro F1: 0.22 → 0.45-0.55"
echo "  Accuracy: 0.33 → 0.50-0.60"

