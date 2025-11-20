#!/bin/bash
set -e

# ============================================================================
# Download Script - Final Model Results from Server
# ============================================================================
# Dieses Skript lädt die Ergebnisse vom Server herunter:
# - final_model/ Ordner mit allen Checkpoints
# - submission.csv (die Einreichungsdatei)
# - training_history.csv
# - config.json
# ============================================================================

# Konfiguration
REMOTE_USER_HOST="drzi@tesla.inf.fh-rosenheim.de"
REMOTE_BASE_DIR="~/slp_project"
SSH_KEY="~/.ssh/id_rsa_uni"

# Lokaler Zielordner
LOCAL_DOWNLOAD_DIR="/Users/fabian.drzimalla/Master_Projects/SLP_PStA_Team_Drzimalla_Diakourakis/downloaded_results"

echo "============================================================================"
echo "DOWNLOAD VOM SERVER: Final Model Ergebnisse"
echo "============================================================================"
echo ""
echo "Server: $REMOTE_USER_HOST"
echo "Quellordner: $REMOTE_BASE_DIR/final_model"
echo "Zielordner: $LOCAL_DOWNLOAD_DIR"
echo ""

# Erstelle lokalen Download-Ordner
mkdir -p "$LOCAL_DOWNLOAD_DIR"

# Liste verfügbare final_model Ordner auf dem Server
echo "[1/2] Suche nach final_model Ordnern auf dem Server..."
echo ""
ssh -i "$SSH_KEY" "$REMOTE_USER_HOST" "ls -lhd $REMOTE_BASE_DIR/final_model/final_model_* 2>/dev/null || echo 'Keine final_model Ordner gefunden'"
echo ""

# Frage Benutzer welchen Ordner er herunterladen möchte
echo "Möchtest du:"
echo "  1) Den neuesten final_model Ordner herunterladen (automatisch)"
echo "  2) Alle final_model Ordner herunterladen"
echo "  3) Einen bestimmten Ordner auswählen"
echo ""
read -p "Auswahl (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "[2/2] Lade neuesten final_model Ordner herunter..."
        # Finde neuesten Ordner (sortiert nach Datum im Namen)
        LATEST_DIR=$(ssh -i "$SSH_KEY" "$REMOTE_USER_HOST" "ls -d $REMOTE_BASE_DIR/final_model/final_model_* 2>/dev/null | sort -r | head -n 1 || echo ''")
        
        if [ -z "$LATEST_DIR" ]; then
            echo "Fehler: Kein final_model Ordner gefunden!"
            exit 1
        fi
        
        echo "Lade herunter: $LATEST_DIR"
        scp -i "$SSH_KEY" -r "$REMOTE_USER_HOST:$LATEST_DIR" "$LOCAL_DOWNLOAD_DIR/"
        
        # Zeige submission.csv an
        FOLDER_NAME=$(basename "$LATEST_DIR")
        if [ -f "$LOCAL_DOWNLOAD_DIR/$FOLDER_NAME/submission.csv" ]; then
            echo ""
            echo "✓ Download abgeschlossen!"
            echo ""
            echo "Submission-Datei: $LOCAL_DOWNLOAD_DIR/$FOLDER_NAME/submission.csv"
            echo ""
            echo "Erste 10 Zeilen der submission.csv:"
            head -n 11 "$LOCAL_DOWNLOAD_DIR/$FOLDER_NAME/submission.csv"
        fi
        ;;
        
    2)
        echo ""
        echo "[2/2] Lade alle final_model Ordner herunter..."
        scp -i "$SSH_KEY" -r "$REMOTE_USER_HOST:$REMOTE_BASE_DIR/final_model" "$LOCAL_DOWNLOAD_DIR/"
        echo "✓ Download abgeschlossen!"
        ;;
        
    3)
        echo ""
        # Liste alle Ordner
        FOLDERS=$(ssh -i "$SSH_KEY" "$REMOTE_USER_HOST" "ls -d $REMOTE_BASE_DIR/final_model/final_model_* 2>/dev/null || echo ''")
        
        if [ -z "$FOLDERS" ]; then
            echo "Fehler: Keine final_model Ordner gefunden!"
            exit 1
        fi
        
        echo "Verfügbare Ordner:"
        echo "$FOLDERS" | nl
        echo ""
        read -p "Ordner-Nummer: " folder_num
        
        SELECTED_DIR=$(echo "$FOLDERS" | sed -n "${folder_num}p")
        
        if [ -z "$SELECTED_DIR" ]; then
            echo "Ungültige Auswahl!"
            exit 1
        fi
        
        echo ""
        echo "[2/2] Lade herunter: $SELECTED_DIR"
        scp -i "$SSH_KEY" -r "$REMOTE_USER_HOST:$SELECTED_DIR" "$LOCAL_DOWNLOAD_DIR/"
        echo "✓ Download abgeschlossen!"
        ;;
        
    *)
        echo "Ungültige Auswahl!"
        exit 1
        ;;
esac

echo ""
echo "============================================================================"
echo "DOWNLOAD ABGESCHLOSSEN"
echo "============================================================================"
echo ""
echo "Heruntergeladene Dateien befinden sich in:"
echo "  $LOCAL_DOWNLOAD_DIR"
echo ""
echo "Wichtige Dateien:"
echo "  - submission.csv (Einreichungsdatei)"
echo "  - training_history.csv (Trainings-Metriken)"
echo "  - best_model_checkpoint.pth (Best Model)"
echo "  - final_model.pth (Final Model)"
echo "  - config.json (Verwendete Konfiguration)"
echo ""
echo "Um die submission.csv zu finden:"
echo "  ls -lh $LOCAL_DOWNLOAD_DIR/*/submission.csv"
echo ""
echo "============================================================================"
