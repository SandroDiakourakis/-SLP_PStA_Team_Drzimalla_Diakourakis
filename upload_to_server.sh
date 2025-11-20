#!/bin/bash
set -e

# ============================================================================
# Upload Script - Final Model Pipeline to Server
# ============================================================================
# Dieses Skript lädt alle notwendigen Dateien auf den Server hoch:
# 1. Validation-Daten (alle Audio-Dateien)
# 2. Excel-Metadaten für Validation
# 3. Python-Skripte (final_model_pipeline.py + claude_output_late_fusion_pipeline.py)
# 4. Shell-Skript zum Starten
# ============================================================================

# Konfiguration
REMOTE_USER_HOST="drzi@tesla.inf.fh-rosenheim.de"
REMOTE_BASE_DIR="~/slp_project"
SSH_KEY="~/.ssh/id_rsa_uni"

# Lokale Pfade
BASE_DIR="/Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis"
VALIDATION_DIR="$BASE_DIR/data/task1/validation"
TEST_EXCEL="$BASE_DIR/data/task1/sand_task1_test.xlsx"
EXPERIMENT_DIR="$BASE_DIR/experiments/experiments/experiment_20251116_132709"
PIPELINE_SCRIPT="$BASE_DIR/final_model_pipeline.py"
LATE_FUSION_SCRIPT="$BASE_DIR/notebooks/late_fusion_from_ideation_to_claude/claude_output_late_fusion_pipeline.py"
RUN_SCRIPT="$BASE_DIR/run_final_model.sh"
VERIFY_SCRIPT="$BASE_DIR/verify_server_setup.sh"

echo "============================================================================"
echo "UPLOAD ZUM SERVER: Final Model Pipeline"
echo "============================================================================"
echo ""
echo "Server: $REMOTE_USER_HOST"
echo "Zielordner: $REMOTE_BASE_DIR"
echo ""

# Schritt 1: Erstelle Remote-Verzeichnisstruktur
echo "[1/6] Erstelle Verzeichnisstruktur auf Server..."
ssh -i "$SSH_KEY" "$REMOTE_USER_HOST" "mkdir -p $REMOTE_BASE_DIR/data/task1/validation"
ssh -i "$SSH_KEY" "$REMOTE_USER_HOST" "mkdir -p $REMOTE_BASE_DIR/notebooks/late_fusion_from_ideation_to_claude"
ssh -i "$SSH_KEY" "$REMOTE_USER_HOST" "mkdir -p $REMOTE_BASE_DIR/experiments/experiments"
echo "✓ Verzeichnisse erstellt"
echo ""

# Schritt 2: Upload Validation-Daten (alle Audio-Dateien)
echo "[2/6] Upload Validation-Daten..."
echo "  Übertrage validation/ Ordner mit allen Audio-Dateien..."
echo "  (Dies kann einige Minuten dauern, je nach Verbindungsgeschwindigkeit)"

# Übertrage den kompletten validation Ordner mit allen Unterordnern
scp -i "$SSH_KEY" -r "$VALIDATION_DIR" "$REMOTE_USER_HOST:$REMOTE_BASE_DIR/data/task1/"

echo "✓ Validation-Daten hochgeladen"
echo ""

# Schritt 3: Upload Excel-Metadaten
echo "[3/6] Upload Excel-Metadaten..."
scp -i "$SSH_KEY" "$TEST_EXCEL" "$REMOTE_USER_HOST:$REMOTE_BASE_DIR/data/task1/"
echo "✓ sand_task1_test.xlsx hochgeladen"
echo ""

# Schritt 4: Upload Experiment-Verzeichnis
echo "[4/6] Upload Experiment-Verzeichnis..."
echo "  - experiment_20251116_132709/ (Config + Checkpoints)"
scp -i "$SSH_KEY" -r "$EXPERIMENT_DIR" "$REMOTE_USER_HOST:$REMOTE_BASE_DIR/experiments/experiments/"
echo "✓ Experiment-Verzeichnis hochgeladen"
echo ""

# Schritt 5: Upload Python-Skripte
echo "[5/6] Upload Python-Skripte..."
echo "  - final_model_pipeline.py"
scp -i "$SSH_KEY" "$PIPELINE_SCRIPT" "$REMOTE_USER_HOST:$REMOTE_BASE_DIR/"

echo "  - claude_output_late_fusion_pipeline.py"
scp -i "$SSH_KEY" "$LATE_FUSION_SCRIPT" "$REMOTE_USER_HOST:$REMOTE_BASE_DIR/notebooks/late_fusion_from_ideation_to_claude/"

echo "✓ Python-Skripte hochgeladen"
echo ""

# Schritt 6: Upload Shell-Skripte und mache sie ausführbar
echo "[6/6] Upload Shell-Skripte..."
scp -i "$SSH_KEY" "$RUN_SCRIPT" "$REMOTE_USER_HOST:$REMOTE_BASE_DIR/"
scp -i "$SSH_KEY" "$VERIFY_SCRIPT" "$REMOTE_USER_HOST:$REMOTE_BASE_DIR/"
ssh -i "$SSH_KEY" "$REMOTE_USER_HOST" "chmod +x $REMOTE_BASE_DIR/run_final_model.sh $REMOTE_BASE_DIR/verify_server_setup.sh"
echo "✓ Shell-Skripte hochgeladen und ausführbar gemacht"
echo ""

# Zusammenfassung
echo "============================================================================"
echo "UPLOAD ABGESCHLOSSEN"
echo "============================================================================"
echo ""
echo "Hochgeladene Dateien:"
echo "  ✓ validation/ Ordner (8 Unterordner mit je 67 Audio-Dateien)"
echo "  ✓ sand_task1_test.xlsx (Test-Metadaten)"
echo "  ✓ experiment_20251116_132709/ (Best Experiment Config + Checkpoints)"
echo "  ✓ final_model_pipeline.py (Haupt-Pipeline)"
echo "  ✓ claude_output_late_fusion_pipeline.py (Late Fusion Pipeline)"
echo "  ✓ run_final_model.sh (Start-Skript)"
echo "  ✓ verify_server_setup.sh (Verifizierungs-Skript)"
echo ""
echo "Verzeichnisstruktur auf Server:"
echo "  $REMOTE_BASE_DIR/"
echo "  ├── data/task1/"
echo "  │   ├── validation/"
echo "  │   │   ├── phonationA/ (67 .wav)"
echo "  │   │   ├── phonationE/ (67 .wav)"
echo "  │   │   ├── phonationI/ (67 .wav)"
echo "  │   │   ├── phonationO/ (67 .wav)"
echo "  │   │   ├── phonationU/ (67 .wav)"
echo "  │   │   ├── rhythmKA/ (67 .wav)"
echo "  │   │   ├── rhythmPA/ (67 .wav)"
echo "  │   │   └── rhythmTA/ (67 .wav)"
echo "  │   └── sand_task1_test.xlsx"
echo "  ├── experiments/experiments/"
echo "  │   └── experiment_20251116_132709/"
echo "  │       ├── config.json"
echo "  │       └── best_model_checkpoint.pth"
echo "  ├── notebooks/late_fusion_from_ideation_to_claude/"
echo "  │   └── claude_output_late_fusion_pipeline.py"
echo "  ├── final_model_pipeline.py"
echo "  ├── run_final_model.sh"
echo "  └── verify_server_setup.sh"
echo ""
echo "============================================================================"
echo "NÄCHSTE SCHRITTE AUF DEM SERVER"
echo "============================================================================"
echo ""
echo "1. Mit Server verbinden:"
echo "   ssh -i $SSH_KEY $REMOTE_USER_HOST"
echo ""
echo "2. Zum Projektordner wechseln:"
echo "   cd $REMOTE_BASE_DIR"
echo ""
echo "3. ZUERST: Setup verifizieren (WICHTIG!):"
echo "   ./verify_server_setup.sh"
echo ""
echo "4. Pipeline starten:"
echo "   ./run_final_model.sh"
echo ""
echo "   ODER direkt mit Python:"
echo "   python3 final_model_pipeline.py"
echo ""
echo "5. Output wird gespeichert in:"
echo "   final_model/final_model_YYYYMMDD_HHMMSS/submission.csv"
echo ""
echo "6. Ergebnis herunterladen (von lokalem Terminal):"
echo "   scp -i $SSH_KEY -r $REMOTE_USER_HOST:$REMOTE_BASE_DIR/final_model ./downloaded_results/"
echo ""
echo "============================================================================"
