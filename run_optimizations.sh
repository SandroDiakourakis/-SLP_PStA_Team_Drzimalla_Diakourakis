#!/bin/bash
# SAND Task 1 - Automatisierte Optimierungs-Pipeline
# Führt die wichtigsten Optimierungen nacheinander aus

echo "🚀 SAND Task 1 - Optimierungs-Pipeline"
echo "======================================"
echo ""

# Farbcodes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Python Command erkennen
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Fehler: Weder python noch python3 gefunden!${NC}"
    exit 1
fi

echo "Using: $PYTHON_CMD"
echo ""

# Fehler-Counter
FAILED_OPTIMIZATIONS=0

# ============================================================================
# OPTIMIERUNG 1: MFCC + eGeMAPSv02 Features
# ============================================================================
echo -e "${BLUE}[1/3] Feature-Kombination: MFCC + eGeMAPSv02${NC}"
echo "Erwarteter Boost: +10-15% Macro F1"
echo ""

echo "  → Feature-Extraktion läuft..."
if $PYTHON_CMD -m sandcli.main command=featurize features.type=mfcc+egemaps; then
    echo "  → Training läuft..."
    if $PYTHON_CMD -m sandcli.main command=train features.type=mfcc+egemaps; then
        echo -e "${GREEN}✓ Optimierung 1 abgeschlossen${NC}"
    else
        echo -e "${RED}✗ Optimierung 1 Training fehlgeschlagen${NC}"
        FAILED_OPTIMIZATIONS=$((FAILED_OPTIMIZATIONS + 1))
    fi
else
    echo -e "${RED}✗ Optimierung 1 Featurization fehlgeschlagen${NC}"
    FAILED_OPTIMIZATIONS=$((FAILED_OPTIMIZATIONS + 1))
fi
echo ""

# ============================================================================
# OPTIMIERUNG 2: Mehr MFCC-Koeffizienten
# ============================================================================
echo -e "${BLUE}[2/3] Erhöhte MFCC-Koeffizienten (20 → 40)${NC}"
echo "Erwarteter Boost: +3-5% Macro F1"
echo ""

echo "  → Feature-Extraktion läuft..."
if $PYTHON_CMD -m sandcli.main command=featurize features.mfcc.n_mfcc=40; then
    echo "  → Training läuft..."
    if $PYTHON_CMD -m sandcli.main command=train features.mfcc.n_mfcc=40; then
        echo -e "${GREEN}✓ Optimierung 2 abgeschlossen${NC}"
    else
        echo -e "${RED}✗ Optimierung 2 Training fehlgeschlagen${NC}"
        FAILED_OPTIMIZATIONS=$((FAILED_OPTIMIZATIONS + 1))
    fi
else
    echo -e "${RED}✗ Optimierung 2 Featurization fehlgeschlagen${NC}"
    FAILED_OPTIMIZATIONS=$((FAILED_OPTIMIZATIONS + 1))
fi
echo ""

# ============================================================================
# OPTIMIERUNG 3: Multi-Seed Ensemble (5 Seeds)
# ============================================================================
echo -e "${BLUE}[3/3] Multi-Seed Ensemble (5 verschiedene Seeds)${NC}"
echo "Erwarteter Boost: +2-5% Macro F1"
echo ""

SEEDS=(42 123 456 789 1024)
SEED_FAILURES=0
for seed in "${SEEDS[@]}"; do
    echo "  → Training mit Seed $seed..."
    if ! $PYTHON_CMD -m sandcli.main command=train seed=$seed experiment_name="seed_${seed}"; then
        echo -e "${RED}✗ Seed $seed fehlgeschlagen${NC}"
        SEED_FAILURES=$((SEED_FAILURES + 1))
    fi
done

if [ $SEED_FAILURES -eq 0 ]; then
    echo -e "${GREEN}✓ Optimierung 3 abgeschlossen (alle Seeds erfolgreich)${NC}"
elif [ $SEED_FAILURES -lt 5 ]; then
    echo -e "${YELLOW}⚠ Optimierung 3 teilweise erfolgreich ($SEED_FAILURES/$((${#SEEDS[@]})) fehlgeschlagen)${NC}"
    FAILED_OPTIMIZATIONS=$((FAILED_OPTIMIZATIONS + 1))
else
    echo -e "${RED}✗ Optimierung 3 komplett fehlgeschlagen${NC}"
    FAILED_OPTIMIZATIONS=$((FAILED_OPTIMIZATIONS + 1))
fi
echo ""

# ============================================================================
# ZUSAMMENFASSUNG
# ============================================================================
echo "======================================"
if [ $FAILED_OPTIMIZATIONS -eq 0 ]; then
    echo -e "${GREEN}🎉 Alle Optimierungen erfolgreich abgeschlossen!${NC}"
else
    echo -e "${YELLOW}⚠️  $FAILED_OPTIMIZATIONS von 3 Optimierungen fehlgeschlagen${NC}"
fi
echo ""

echo "📊 Ergebnisse vergleichen:"
echo "  ls -t runs/*/metrics.json | head -5 | xargs -I {} sh -c 'echo {}; cat {} | grep macro_f1'"
echo ""
echo "🏆 Beste Runs finden:"
echo "  for file in runs/*/metrics.json; do"
echo "    f1=\$(cat \$file | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"macro_f1\"])')"
