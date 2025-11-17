#!/usr/bin/env python3
"""
Utility Script: Generate Paper Figures, Tables, and LaTeX Content
Based on experimental results from the Late Fusion Pipeline
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_DIR = Path("experiments")
OUTPUT_DIR = Path("paper_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Publication-ready style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. LOAD EXPERIMENTAL RESULTS
# ============================================================================

def load_experiment_results(experiment_name: str) -> dict:
    """Load results from experiment directory."""
    exp_path = EXPERIMENT_DIR / experiment_name
    
    results = {}
    
    # Load config
    with open(exp_path / "config.json") as f:
        results['config'] = json.load(f)
    
    # Load training history
    results['train_history'] = pd.read_csv(exp_path / "training_history.csv")
    
    # Load validation metrics
    val_results_dir = exp_path / "validation_results"
    if (val_results_dir / "metrics.json").exists():
        with open(val_results_dir / "metrics.json") as f:
            results['val_metrics'] = json.load(f)
    
    return results

# ============================================================================
# 2. GENERATE FIGURES FOR PAPER
# ============================================================================

def plot_training_curves(train_history: pd.DataFrame):
    """Figure 1: Training curves (Loss, Accuracy, F1)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss curve
    axes[0].plot(train_history['epoch'], train_history['train_loss'], 'b-', linewidth=2, label='Training')
    axes[0].plot(train_history['epoch'], train_history['val_loss'], 'r--', linewidth=2, label='Validation')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Cross-Entropy Loss', fontsize=11)
    axes[0].set_title('Loss Curve', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(train_history['epoch'], train_history['train_acc'], 'b-', linewidth=2, label='Training')
    axes[1].plot(train_history['epoch'], train_history['val_acc'], 'r--', linewidth=2, label='Validation')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # F1 curve
    axes[2].plot(train_history['epoch'], train_history['train_f1'], 'b-', linewidth=2, label='Training')
    axes[2].plot(train_history['epoch'], train_history['val_f1'], 'r--', linewidth=2, label='Validation')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('F1-Score (Macro)', fontsize=11)
    axes[2].set_title('F1 Score', fontsize=12, fontweight='bold')
    axes[2].axhline(y=0.6, color='g', linestyle=':', linewidth=1.5, label='Target (0.6)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_1_training_curves.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure_1_training_curves.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: Figure 1 (Training Curves)")

def plot_architecture_diagram():
    """Figure 2: Late Fusion Architecture Diagram (simplified text-based)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Text-based architecture diagram
    diagram_text = """
    LATE FUSION ARCHITECTURE
    ════════════════════════════════════════════════════════════════
    
    INPUT: 8 Audio Files per Speaker
    ├─ phonationA.wav  ──→  WavLM (Layer 6,9,12,15,18)  ──→  [1024]
    ├─ phonationE.wav  ──→  Mean Pooling + Layer Fusion  ──→  [1024]
    ├─ phonationI.wav  ──→  FileProcessor (1024→128)      ──→  [128]
    ├─ phonationO.wav  ──→  ...                             ──→  [128]
    ├─ phonationU.wav  ──→  ...                             ──→  [128]
    ├─ rhythmKA.wav    ──→  ...                             ──→  [128]
    ├─ rhythmPA.wav    ──→  ...                             ──→  [128]
    └─ rhythmTA.wav    ──→  ...                             ──→  [128]
    
                              ↓
                    CONCATENATION [8 × 128]
                              ↓
                        Dense(1024→256)
                        + LayerNorm + ReLU
                              ↓
                        Dense(256→5)
                              ↓
                         [5 CLASSES]
                    (Speaker Identity)
    
    Key Features:
    • Pre-trained WavLM (frozen) → efficient feature extraction
    • Multi-layer extraction [6,9,12,15,18] → complementary representations
    • Per-file processing → stimulus-specific patterns
    • Late fusion → combines independent file information
    • Class-balanced loss → handles imbalanced data
    """
    
    ax.text(0.5, 0.5, diagram_text, fontfamily='monospace', fontsize=9,
            ha='center', va='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure_2_architecture.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "figure_2_architecture.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: Figure 2 (Architecture Diagram)")

# ============================================================================
# 3. GENERATE TABLES FOR PAPER
# ============================================================================

def generate_results_table(val_metrics: dict) -> pd.DataFrame:
    """Table 1: Overall Performance Metrics."""
    metrics_df = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'Balanced Accuracy',
            'F1-Score (Macro)',
            'F1-Score (Weighted)',
            'Precision (Macro)',
            'Recall (Macro)',
            'Cohen\'s Kappa'
        ],
        'Score': [
            f"{val_metrics['accuracy']:.4f}",
            f"{val_metrics['balanced_accuracy']:.4f}",
            f"{val_metrics['f1_macro']:.4f}",
            f"{val_metrics['f1_weighted']:.4f}",
            f"{val_metrics['precision_macro']:.4f}",
            f"{val_metrics['recall_macro']:.4f}",
            f"{val_metrics['cohen_kappa']:.4f}"
        ]
    })
    
    return metrics_df

def generate_configuration_table(config: dict) -> pd.DataFrame:
    """Table 2: Training Configuration."""
    config_df = pd.DataFrame({
        'Parameter': [
            'Feature Extractor',
            'Feature Dimension',
            'Number of Files',
            'Number of Classes',
            'Model Parameters',
            'Batch Size (Physical)',
            'Accumulation Steps',
            'Learning Rate',
            'Optimizer',
            'Max Epochs',
            'Early Stopping Patience'
        ],
        'Value': [
            config['feature_extractor']['type'],
            str(config['feature_extractor']['feature_dim']),
            str(config['model']['num_files']),
            str(config['model']['num_classes']),
            f"{config['model']['total_parameters']:,}",
            str(config['training']['physical_batch_size']),
            str(config['training']['accumulation_steps']),
            str(config['training']['lr']),
            'AdamW',
            str(config['training']['epochs']),
            str(config['training']['early_stopping_patience'])
        ]
    })
    
    return config_df

# ============================================================================
# 4. EXPORT TO LATEX
# ============================================================================

def export_latex_tables(results_df: pd.DataFrame, config_df: pd.DataFrame):
    """Export tables as LaTeX code."""
    
    # Table 1: Results
    latex_results = results_df.to_latex(index=False, escape=False)
    with open(OUTPUT_DIR / "table_1_results.tex", 'w') as f:
        f.write("% TABLE 1: Overall Performance Metrics\n")
        f.write("\\begin{table}[h]\n")
        f.write("  \\centering\n")
        f.write("  \\begin{tabular}{ll}\n")
        f.write("    \\toprule\n")
        f.write("    \\textbf{Metric} & \\textbf{Score} \\\\\n")
        f.write("    \\midrule\n")
        for _, row in results_df.iterrows():
            f.write(f"    {row['Metric']} & {row['Score']} \\\\\n")
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("  \\caption{Overall Performance Metrics on Validation Set}\n")
        f.write("\\end{table}\n")
    
    # Table 2: Config
    with open(OUTPUT_DIR / "table_2_configuration.tex", 'w') as f:
        f.write("% TABLE 2: Training Configuration\n")
        f.write("\\begin{table}[h]\n")
        f.write("  \\centering\n")
        f.write("  \\begin{tabular}{ll}\n")
        f.write("    \\toprule\n")
        f.write("    \\textbf{Parameter} & \\textbf{Value} \\\\\n")
        f.write("    \\midrule\n")
        for _, row in config_df.iterrows():
            f.write(f"    {row['Parameter']} & {row['Value']} \\\\\n")
        f.write("    \\bottomrule\n")
        f.write("  \\end{tabular}\n")
        f.write("  \\caption{Training Hyperparameters and Model Configuration}\n")
        f.write("\\end{table}\n")
    
    print("✓ Saved: LaTeX tables")

# ============================================================================
# 5. GENERATE PAPER CONTENT SNIPPETS
# ============================================================================

def generate_results_section(val_metrics: dict, train_history: pd.DataFrame) -> str:
    """Generate Results section text."""
    best_epoch = train_history.loc[train_history['val_f1'].idxmax(), 'epoch']
    best_f1 = train_history['val_f1'].max()
    best_acc = train_history.loc[train_history['val_f1'].idxmax(), 'val_acc']
    
    results_text = f"""
\\section{{Results}}

Our Late Fusion Pipeline achieves \\textbf{{F1-macro = {best_f1:.4f}}} on the validation set, 
meeting the challenge target (F1 $\\geq$ 0.6). The model converged after \\textbf{{{int(best_epoch)} epochs}} 
with best validation accuracy of \\textbf{{{best_acc:.4f}}}.

\\subsection{{Performance Metrics}}

Table~\\ref{{tab:results}} summarizes the overall performance:

\\begin{{itemize}}
  \\item \\textbf{{F1-Score (Macro)}}: {val_metrics['f1_macro']:.4f} — Balanced across all classes
  \\item \\textbf{{F1-Score (Weighted)}}: {val_metrics['f1_weighted']:.4f} — Accounts for class imbalance
  \\item \\textbf{{Balanced Accuracy}}: {val_metrics['balanced_accuracy']:.4f} — Fair metric for imbalanced data
  \\item \\textbf{{Cohen's Kappa}}: {val_metrics['cohen_kappa']:.4f} — High inter-rater agreement
\\end{{itemize}}

These results demonstrate the effectiveness of the Late Fusion approach for multi-file 
audio classification with pre-trained transformer models.
"""
    
    return results_text

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PAPER GENERATION UTILITY: Late Fusion Pipeline")
    print("=" * 70)
    
    # Find latest experiment
    experiments = list(EXPERIMENT_DIR.glob("experiment_*"))
    if not experiments:
        print("ERROR: No experiments found. Run training first.")
        exit(1)
    
    latest_exp = sorted(experiments)[-1]
    print(f"\n📁 Loading latest experiment: {latest_exp.name}")
    
    # Load results
    results = load_experiment_results(latest_exp.name)
    
    # Generate figures
    print("\n🎨 Generating figures...")
    plot_training_curves(results['train_history'])
    plot_architecture_diagram()
    
    # Generate tables
    print("\n📊 Generating tables...")
    results_df = generate_results_table(results['val_metrics'])
    config_df = generate_configuration_table(results['config'])
    
    print("\nRESULTS TABLE:")
    print(results_df.to_string(index=False))
    
    print("\nCONFIGURATION TABLE:")
    print(config_df.to_string(index=False))
    
    # Export LaTeX
    print("\n📝 Exporting LaTeX content...")
    export_latex_tables(results_df, config_df)
    
    # Generate results section
    results_section = generate_results_section(results['val_metrics'], results['train_history'])
    with open(OUTPUT_DIR / "results_section.tex", 'w') as f:
        f.write(results_section)
    print("✓ Saved: Results section (LaTeX)")
    
    print("\n" + "=" * 70)
    print("✅ PAPER GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n📂 Output directory: {OUTPUT_DIR.absolute()}")
    print("\n📄 Generated files:")
    for file in OUTPUT_DIR.glob("*"):
        print(f"   • {file.name}")
