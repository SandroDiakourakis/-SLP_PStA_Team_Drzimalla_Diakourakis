# 🗑️ Aufräum-Liste für Repository

## Dateien die gelöscht werden können:

### Shell-Skripte (temporär/einmalig)
```bash
rm fix_pipeline.sh
rm test_pipeline.sh  
rm quick_fix.sh
rm QUICK_COMMANDS.sh  # Optional behalten als Referenz
```

### Backup-Dateien
```bash
rm sandcli/train_backup.py
```

### Alte Outputs & Experimente
```bash
rm -rf outputs/2025-10-19/
rm -rf outputs/2025-10-21/18-28-44/
rm -rf outputs/2025-10-21/18-32-36/
rm -rf outputs/2025-10-21/18-34-57/
rm -rf outputs/2025-10-21/18-36-10/
rm -rf outputs/2025-10-21/18-42-34/
rm -rf outputs/2025-10-21/18-43-41/
rm -rf outputs/2025-10-21/18-58-58/
# Nur das neueste behalten!
```

### Fehlgeschlagene Experimente
```bash
rm -rf runs/baseline_ml_xgboost_20251021_184341/
rm -rf runs/baseline_ml_xgboost_20251021_185858/
rm -rf runs/baseline_ml_xgboost_20251021_185948/
rm -rf runs/experiment_YYYYMMDD_HHMMSS/
```

### Leere Feature-Verzeichnisse
```bash
rm -rf features/egemaps/  # Leer
rm -rf features/ssl_embeddings/  # Für Phase 2
```

### Excel-Temp-Dateien
```bash
rm data/task1/~$sand_task_1.xlsx
```

### Python Cache (wird automatisch neu erstellt)
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
```

### Optionale Redundanz
```bash
# rm NEW_README.md  # Nur wenn Sie README.md bevorzugen
# rm branch-history.md  # Ihre persönlichen Notizen
```

---

## Aufräum-Skript (ALL-IN-ONE):

```bash
#!/bin/bash
# cleanup.sh - Räumt unnötige Dateien auf

echo "🧹 Räume Repository auf..."

# Shell-Skripte
rm -f fix_pipeline.sh test_pipeline.sh quick_fix.sh

# Backups
rm -f sandcli/train_backup.py

# Alte Outputs
rm -rf outputs/2025-10-19/
find outputs/2025-10-21/ -type d -mindepth 1 | sort | head -n -1 | xargs rm -rf

# Fehlgeschlagene Experimente (behalte nur das neueste)
find runs/ -name "baseline_ml_xgboost_*" -type d | sort | head -n -2 | xargs rm -rf
rm -rf runs/experiment_YYYYMMDD_HHMMSS/

# Leere Verzeichnisse
rm -rf features/egemaps/
rm -rf features/ssl_embeddings/

# Temp-Dateien
rm -f data/task1/~$sand_task_1.xlsx

# Python Cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

echo "✅ Aufräumen abgeschlossen!"
echo ""
echo "Verbleibende Struktur:"
du -sh features/mfcc runs/baseline_ml_xgboost_* 2>/dev/null
```

---

## Nach dem Aufräumen - Erwartete Größen:

- `features/mfcc/`: ~50-100 MB (Feature Arrays)
- `runs/baseline_ml_xgboost_*/`: ~5-10 MB pro Experiment
- Gesamt-Repository: ~150-200 MB (statt vorher ~300-500 MB)

---

**WICHTIG:** Erstellen Sie ein Backup bevor Sie löschen!
```bash
tar -czf backup_$(date +%Y%m%d).tar.gz \
  fix_pipeline.sh test_pipeline.sh sandcli/train_backup.py \
  outputs/ runs/baseline_ml_xgboost_20251021_*
```

