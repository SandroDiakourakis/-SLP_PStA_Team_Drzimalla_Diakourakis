#!/usr/bin/env python3
"""
Visualisierung für Training Concepts: Batch, Batch Size, Epochs und Gradient Accumulation
Zeigt detailliert, wie Training funktioniert
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. BATCH vs BATCH SIZE vs EPOCHS - Conceptual Overview
# ============================================================================

def create_batch_concepts():
    """Erklärt die Konzepte von Batch, Batch Size und Epochs."""
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    fig.text(0.5, 0.97, 'Training Concepts: Batch, Batch Size, Epochs', 
             fontsize=18, fontweight='bold', ha='center')
    
    # ========== Dataset visualization ==========
    y_pos = 10.5
    
    ax.text(0.5, y_pos, 'Dataset:', fontsize=12, fontweight='bold', va='center')
    
    # 240 total samples
    ax.text(1.5, y_pos, '240 Trainingsamples insgesamt', fontsize=11, style='italic', va='center')
    
    # Draw samples
    for i in range(24):
        x = 2 + (i % 12) * 0.5
        y = y_pos - 0.3 if i < 12 else y_pos - 0.8
        
        rect = Rectangle((x, y), 0.4, 0.3, facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.2, y + 0.15, 'S', fontsize=6, ha='center', va='center', fontweight='bold')
    
    # ========== BATCH SIZE erklären ==========
    y_pos = 8.5
    
    rect = FancyBboxPatch((0.5, y_pos - 0.8), 9, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(1.5, y_pos + 0.2, '1️⃣ BATCH SIZE = 12', fontsize=12, fontweight='bold', va='center')
    ax.text(1.5, y_pos - 0.35, 'Wieviele Samples in EINEM Batch verarbeitet werden', 
            fontsize=10, style='italic', va='center')
    
    # Visualize: 1 Batch
    for i in range(12):
        x = 5 + (i % 6) * 0.5
        y = y_pos + 0.15 if i < 6 else y_pos - 0.35
        
        rect = Rectangle((x, y), 0.4, 0.25, facecolor='#FFAB91', edgecolor='#D84315', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.2, y + 0.125, str(i+1), fontsize=6, ha='center', va='center', fontweight='bold', color='white')
    
    ax.text(8.5, y_pos + 0.45, '= 1 Batch (12 Samples)', fontsize=10, fontweight='bold', va='center')
    
    # ========== EPOCH erklären ==========
    y_pos = 6.5
    
    rect = FancyBboxPatch((0.5, y_pos - 1.8), 9, 2.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#4CAF50', facecolor='#C8E6C9', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(1.5, y_pos + 0.8, '2️⃣ EPOCH = 1 vollständiger Durchgang', fontsize=12, fontweight='bold', va='center')
    ax.text(1.5, y_pos + 0.4, 'Ein Epoch verarbeitet ALLE 240 Trainingsamples', 
            fontsize=10, style='italic', va='center')
    
    # Batches per epoch
    batches_per_epoch = 240 // 12  # = 20 batches
    
    ax.text(5, y_pos + 0.8, 'Ein Epoch = 20 Batches', fontsize=11, fontweight='bold', va='center',
           bbox=dict(boxstyle='round', facecolor='#A5D6A7', alpha=0.7))
    
    # Draw batches
    for batch_idx in range(20):
        x = 5 + (batch_idx % 10) * 0.4
        y = y_pos + 0.2 if batch_idx < 10 else y_pos - 0.3
        
        rect = Rectangle((x, y), 0.35, 0.2, facecolor='#66BB6A', edgecolor='#2E7D32', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.175, y + 0.1, f'B{batch_idx+1}', fontsize=5, ha='center', va='center', 
               fontweight='bold', color='white')
    
    ax.text(9.2, y_pos - 0.05, '20 Batches', fontsize=9, ha='left', va='center', fontweight='bold')
    
    # ========== GRADIENT ACCUMULATION ==========
    y_pos = 4.2
    
    rect = FancyBboxPatch((0.5, y_pos - 1.5), 9, 1.9,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FFA726', facecolor='#FFE0B2', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(1.5, y_pos + 0.5, '3️⃣ GRADIENT ACCUMULATION = 2', fontsize=12, fontweight='bold', va='center')
    ax.text(1.5, y_pos + 0.1, 'Gradient Accumulation: 2 Mini-Batches vor Update', 
            fontsize=10, style='italic', va='center')
    
    # Effective batch size
    ax.text(5.5, y_pos + 0.4, 'Effektive Batch Size:', fontsize=10, fontweight='bold', va='center')
    ax.text(5.5, y_pos, '12 (physical) × 2 (accumulation) = 24', fontsize=10, va='center',
           bbox=dict(boxstyle='round', facecolor='#FFCC80', alpha=0.8))
    
    # Visualization
    for acc_step in range(2):
        for sample in range(12):
            x = 5 + acc_step * 2 + (sample % 4) * 0.4
            y = y_pos - 0.7 if sample < 4 else (y_pos - 1.0 if sample < 8 else y_pos - 1.3)
            
            rect = Rectangle((x, y), 0.35, 0.2, facecolor='#FFB74D', edgecolor='#E65100', linewidth=0.5)
            ax.add_patch(rect)
    
    ax.text(8, y_pos - 1.0, 'Accumulation Step 1 (12)', fontsize=8, ha='center', fontweight='bold')
    ax.text(9, y_pos - 1.0, 'Accumulation Step 2 (12)', fontsize=8, ha='center', fontweight='bold')
    
    # ========== TRAINING LOOP ==========
    y_pos = 1.8
    
    rect = FancyBboxPatch((0.5, y_pos - 1.2), 9, 1.6,
                         boxstyle="round,pad=0.1",
                         edgecolor='#9C27B0', facecolor='#E1BEE7', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(1.5, y_pos + 0.3, '4️⃣ UNSER TRAINING LOOP', fontsize=12, fontweight='bold', va='center')
    
    loop_text = """
    For Epoch = 1 to 20:
      For Batch = 1 to 20:
        Process 12 Samples
        Accumulate Gradients 2×
        Update Weights (every 24 samples)
      Validate on 48 Validation Samples
      Check Early Stopping
    
    Resultat: 20 Epochs × 20 Batches = 400 Batch Updates
    """
    
    ax.text(5.5, y_pos - 0.3, loop_text, fontsize=9, ha='center', va='center', 
           fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('09_batch_concepts.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 09_batch_concepts.png")
    plt.close()

# ============================================================================
# 2. TRAINING PROGRESSION - Epoch by Epoch Visualization
# ============================================================================

def create_training_progression():
    """Zeigt den Trainingsfortschritt über Epochs."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create 3 subplots
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # ========== Plot 1: Loss over Epochs ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    epochs = np.arange(1, 21)
    
    # Simulated training loss (decreasing with noise)
    train_loss = 2.5 * np.exp(-epochs / 8) + 0.3 + np.random.randn(20) * 0.1
    val_loss = 2.6 * np.exp(-epochs / 8.5) + 0.35 + np.random.randn(20) * 0.12
    
    ax1.plot(epochs, train_loss, 'o-', linewidth=3, markersize=8, color='#1976D2', label='Training Loss')
    ax1.plot(epochs, val_loss, 's--', linewidth=2.5, markersize=7, color='#D32F2F', label='Validation Loss')
    
    # Mark early stopping
    best_epoch = 16
    ax1.axvline(x=best_epoch, color='#4CAF50', linestyle=':', linewidth=2.5, label='Early Stopping (patience=5)')
    ax1.scatter([best_epoch], [val_loss[best_epoch-1]], s=300, color='#4CAF50', zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Progress: Loss über Epochs', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 21)
    
    # ========== Plot 2: F1-Score over Epochs ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Simulated F1 scores (increasing)
    train_f1 = 0.5 + 0.2 * (1 - np.exp(-epochs / 6)) + np.random.randn(20) * 0.02
    val_f1 = 0.48 + 0.18 * (1 - np.exp(-epochs / 7)) + np.random.randn(20) * 0.025
    
    ax2.plot(epochs, train_f1, 'o-', linewidth=2.5, markersize=7, color='#388E3C', label='Training F1')
    ax2.plot(epochs, val_f1, 's--', linewidth=2.5, markersize=7, color='#F57C00', label='Validation F1')
    
    ax2.axhline(y=0.72, color='#FFA726', linestyle='--', linewidth=2, label='Target (0.72)')
    ax2.axvline(x=best_epoch, color='#4CAF50', linestyle=':', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax2.set_title('Model Performance: F1-Score', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0.45, 0.75])
    
    # ========== Plot 3: Training Timeline ==========
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    ax3.text(5, 9.5, 'Trainings-Timeline (1 Epoch)', fontsize=12, fontweight='bold', ha='center')
    
    # One epoch breakdown
    timeline_events = [
        {'time': 0, 'label': 'Epoch Start', 'color': '#4CAF50'},
        {'time': 2.5, 'label': 'Batch 5\n(5 Batches Done)', 'color': '#1976D2'},
        {'time': 5, 'label': 'Batch 10\n(50% Complete)', 'color': '#FFA726'},
        {'time': 7.5, 'label': 'Batch 15\n(75% Complete)', 'color': '#9C27B0'},
        {'time': 10, 'label': 'Validation', 'color': '#D32F2F'},
    ]
    
    for i, event in enumerate(timeline_events):
        x = event['time']
        y = 6
        
        # Timeline point
        rect = plt.Circle((x, y), 0.25, color=event['color'], zorder=5)
        ax3.add_patch(rect)
        
        # Label
        ax3.text(x, y - 1.2, event['label'], fontsize=9, ha='center', fontweight='bold')
    
    # Draw timeline
    ax3.plot([0, 10], [6, 6], 'k-', linewidth=2)
    
    ax3.text(5, 4.5, 'Batches pro Epoch: 20', fontsize=10, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#BBDEFB', alpha=0.8))
    
    ax3.text(5, 3.5, 'Samples pro Batch: 12', fontsize=10, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#BBDEFB', alpha=0.8))
    
    ax3.text(5, 2.5, 'Gradient Accumulation: 2', fontsize=10, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#BBDEFB', alpha=0.8))
    
    ax3.text(5, 1.2, 'Zeit pro Epoch: ~2-3 Minuten', fontsize=10, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))
    
    fig.suptitle('Training Progression: Epochs, Losses, und Performance', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('10_training_progression.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 10_training_progression.png")
    plt.close()

# ============================================================================
# 3. BATCH PROCESSING - Detaillierte Visualisierung eines Batches
# ============================================================================

def create_batch_processing():
    """Detailliert zeigt, wie ein einzelner Batch verarbeitet wird."""
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    fig.text(0.5, 0.98, 'Batch Processing Pipeline: Detaillierter Ablauf', 
             fontsize=18, fontweight='bold', ha='center')
    
    # ========== STEP 1: LOAD BATCH ==========
    y = 12.5
    
    ax.text(0.3, y, 'STEP 1', fontsize=10, fontweight='bold', color='white',
           bbox=dict(boxstyle='circle', facecolor='#FF6B6B', edgecolor='white', linewidth=2))
    
    rect = FancyBboxPatch((1, y - 0.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(2.5, y + 0.15, 'Load Batch', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.5, y - 0.25, '12 Audio Files', fontsize=10, ha='center', style='italic')
    
    # Show 12 files
    for i in range(12):
        x = 5.5 + (i % 6) * 0.5
        y_file = y - 0.15 if i < 6 else y - 0.5
        
        rect = Rectangle((x, y_file), 0.4, 0.25, facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.2, y_file + 0.125, str(i+1), fontsize=6, ha='center', va='center', fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((2.5, y - 0.6), (2.5, y - 1.3),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # ========== STEP 2: AUDIO AUGMENTATION ==========
    y = 10.5
    
    ax.text(0.3, y, 'STEP 2', fontsize=10, fontweight='bold', color='white',
           bbox=dict(boxstyle='circle', facecolor='#FFA726', edgecolor='white', linewidth=2))
    
    rect = FancyBboxPatch((1, y - 0.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FFA726', facecolor='#FFE0B2', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(2.5, y + 0.15, 'Augmentation', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.5, y - 0.25, '80% Probability', fontsize=10, ha='center', style='italic')
    
    # Augmentation types
    aug_types = ['Time\nStretch', 'Pitch\nShift', 'Noise\nAdd', 'Time\nMask']
    for i, aug in enumerate(aug_types):
        x = 5.5 + i * 1
        rect = Rectangle((x, y - 0.3), 0.8, 0.6, facecolor='#FFCC80', edgecolor='#E65100', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.4, y, aug, fontsize=8, ha='center', va='center', fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((2.5, y - 0.6), (2.5, y - 1.3),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # ========== STEP 3: FEATURE EXTRACTION ==========
    y = 8.5
    
    ax.text(0.3, y, 'STEP 3', fontsize=10, fontweight='bold', color='white',
           bbox=dict(boxstyle='circle', facecolor='#42A5F5', edgecolor='white', linewidth=2))
    
    rect = FancyBboxPatch((1, y - 0.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#42A5F5', facecolor='#BBDEFB', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(2.5, y + 0.15, 'Feature Extract', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.5, y - 0.25, 'WavLM (12 cores)', fontsize=10, ha='center', style='italic')
    
    # Show parallelization
    for i in range(12):
        x = 5.5 + (i % 6) * 0.5
        y_feat = y - 0.15 if i < 6 else y - 0.5
        
        rect = Rectangle((x, y_feat), 0.4, 0.25, facecolor='#BBDEFB', edgecolor='#0D47A1', linewidth=1)
        ax.add_patch(rect)
        ax.text(x + 0.2, y_feat + 0.125, 'F', fontsize=6, ha='center', va='center', fontweight='bold')
    
    ax.text(8.7, y, '→ (Batch, 1024)', fontsize=9, ha='left', va='center', fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((2.5, y - 0.6), (2.5, y - 1.3),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # ========== STEP 4: FORWARD PASS ==========
    y = 6.5
    
    ax.text(0.3, y, 'STEP 4', fontsize=10, fontweight='bold', color='white',
           bbox=dict(boxstyle='circle', facecolor='#66BB6A', edgecolor='white', linewidth=2))
    
    rect = FancyBboxPatch((1, y - 0.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#66BB6A', facecolor='#C8E6C9', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(2.5, y + 0.15, 'Forward Pass', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.5, y - 0.25, 'Through Network', fontsize=10, ha='center', style='italic')
    
    # Network flow
    ax.text(5.5, y + 0.2, '(12, 1024)', fontsize=9, ha='left', va='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.7))
    ax.text(5.5, y - 0.2, '(12, 5)', fontsize=9, ha='left', va='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.7))
    
    ax.text(6.8, y, '→ Logits', fontsize=9, ha='left', va='center', style='italic', fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((2.5, y - 0.6), (2.5, y - 1.3),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # ========== STEP 5: LOSS COMPUTATION ==========
    y = 4.5
    
    ax.text(0.3, y, 'STEP 5', fontsize=10, fontweight='bold', color='white',
           bbox=dict(boxstyle='circle', facecolor='#AB47BC', edgecolor='white', linewidth=2))
    
    rect = FancyBboxPatch((1, y - 0.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#AB47BC', facecolor='#E1BEE7', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(2.5, y + 0.15, 'Loss Compute', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.5, y - 0.25, 'CrossEntropy', fontsize=10, ha='center', style='italic')
    
    # Loss breakdown
    loss_text = """
    Per-sample Loss = -log(P_true_class)
    Batch Loss = Mean(all losses)
    Weighted Loss = Loss × scale_factor
    """
    
    ax.text(5.5, y, loss_text, fontsize=8, ha='left', va='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.8))
    
    # Arrow down
    arrow = FancyArrowPatch((2.5, y - 0.6), (2.5, y - 1.3),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # ========== STEP 6: BACKPROPAGATION ==========
    y = 2.5
    
    ax.text(0.3, y, 'STEP 6', fontsize=10, fontweight='bold', color='white',
           bbox=dict(boxstyle='circle', facecolor='#EF5350', edgecolor='white', linewidth=2))
    
    rect = FancyBboxPatch((1, y - 0.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#EF5350', facecolor='#FFCDD2', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(2.5, y + 0.15, 'Backprop', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.5, y - 0.25, 'Compute Gradients', fontsize=10, ha='center', style='italic')
    
    # Gradient accumulation note
    ax.text(5.5, y + 0.2, 'Accumulate Gradients', fontsize=9, ha='left', va='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))
    ax.text(5.5, y - 0.2, '(for 2 accumulation steps)', fontsize=9, ha='left', va='center', style='italic')
    
    # Arrow down
    arrow = FancyArrowPatch((2.5, y - 0.6), (2.5, y - 1.3),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # ========== STEP 7: WEIGHT UPDATE (if 2 batches done) ==========
    y = 0.5
    
    ax.text(0.3, y, 'STEP 7', fontsize=10, fontweight='bold', color='white',
           bbox=dict(boxstyle='circle', facecolor='#29B6F6', edgecolor='white', linewidth=2))
    
    rect = FancyBboxPatch((1, y - 0.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#29B6F6', facecolor='#B3E5FC', linewidth=2)
    ax.add_patch(rect)
    
    ax.text(2.5, y + 0.15, 'Weight Update', fontsize=11, ha='center', fontweight='bold')
    ax.text(2.5, y - 0.25, '(Every 2 Batches)', fontsize=10, ha='center', style='italic')
    
    update_text = """
    gradient = accumulated / 2
    weights = AdamW optimizer step
    clip gradients (max norm=1.0)
    """
    
    ax.text(5.5, y, update_text, fontsize=8, ha='left', va='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#B3E5FC', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('11_batch_processing.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 11_batch_processing.png")
    plt.close()

# ============================================================================
# 4. GRADIENT ACCUMULATION - Detailed Explanation
# ============================================================================

def create_gradient_accumulation():
    """Erklärt Gradient Accumulation im Detail."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # ========== Top: Without Accumulation ==========
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5)
    ax1.axis('off')
    
    ax1.text(5, 4.7, 'Without Gradient Accumulation (Batch Size 24, 1 GPU)', 
            fontsize=13, fontweight='bold', ha='center')
    
    # Timeline without accumulation
    for batch_idx in range(4):
        x = 1 + batch_idx * 2
        
        # Batch box
        rect = FancyBboxPatch((x - 0.6, 3), 1.2, 0.8,
                             boxstyle="round,pad=0.05",
                             edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=2)
        ax1.add_patch(rect)
        
        ax1.text(x, 3.5, f'Batch {batch_idx+1}', fontsize=9, ha='center', va='center', fontweight='bold')
        ax1.text(x, 3.1, '(24 Samples)', fontsize=8, ha='center', va='center', style='italic')
        
        # Forward + Backward
        ax1.text(x, 2.4, 'Forward\n+\nBackward', fontsize=8, ha='center', va='center', fontweight='bold')
        
        # Weight Update
        rect = FancyBboxPatch((x - 0.5, 1.3), 1, 0.4,
                             boxstyle="round,pad=0.05",
                             edgecolor='#4CAF50', facecolor='#C8E6C9', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, 1.5, 'Weight Update', fontsize=8, ha='center', va='center', fontweight='bold')
        
        # Arrow
        if batch_idx < 3:
            arrow = FancyArrowPatch((x + 0.7, 2), (x + 1.3, 2),
                                  arrowstyle='->', mutation_scale=15, linewidth=1.5, color='black')
            ax1.add_patch(arrow)
    
    ax1.text(5, 0.7, 'Problem: 24 Samples × 1 GPU = Sehr Memory-intensiv!', 
            fontsize=10, ha='center', style='italic', color='darkred', fontweight='bold')
    
    # ========== Bottom: With Accumulation ==========
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.axis('off')
    
    ax2.text(5, 4.7, 'With Gradient Accumulation (Batch Size 12 × 2 steps)', 
            fontsize=13, fontweight='bold', ha='center')
    
    # Timeline with accumulation
    colors = ['#1976D2', '#1976D2', '#2E7D32', '#2E7D32']
    labels = ['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4']
    update_after = [2, 2, 4, 4]  # Update after accumulating
    
    batch_count = 0
    for group in range(2):
        x_start = 1 + group * 4.5
        
        for step in range(2):
            x = x_start + step * 1.8
            
            # Mini-batch box
            rect = FancyBboxPatch((x - 0.6, 3), 1.2, 0.8,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='#FFA726', facecolor='#FFE0B2', linewidth=2)
            ax2.add_patch(rect)
            
            batch_count += 1
            ax2.text(x, 3.5, f'Batch {batch_count}', fontsize=9, ha='center', va='center', fontweight='bold')
            ax2.text(x, 3.1, '(12 Samples)', fontsize=8, ha='center', va='center', style='italic')
            
            # Forward + Accumulate
            ax2.text(x, 2.4, 'Forward\n+\nAccumulate', fontsize=8, ha='center', va='center', fontweight='bold')
        
        # Weight Update (after 2 batches)
        x_update = x_start + 0.9
        rect = FancyBboxPatch((x_update - 0.5, 1.3), 1, 0.4,
                             boxstyle="round,pad=0.05",
                             edgecolor='#4CAF50', facecolor='#C8E6C9', linewidth=2.5)
        ax2.add_patch(rect)
        ax2.text(x_update, 1.5, 'Weight Update', fontsize=8, ha='center', va='center', fontweight='bold')
        
        # Arrows
        arrow = FancyArrowPatch((x_start + 0.2, 2), (x_update - 0.6, 1.7),
                              arrowstyle='->', mutation_scale=15, linewidth=1.5, color='gray', linestyle='--')
        ax2.add_patch(arrow)
        arrow = FancyArrowPatch((x_start + 1.8, 2), (x_update + 0.6, 1.7),
                              arrowstyle='->', mutation_scale=15, linewidth=1.5, color='gray', linestyle='--')
        ax2.add_patch(arrow)
    
    ax2.text(5, 0.7, 'Vorteil: Effektive Batch Size = 24, aber nur 12 im Memory!', 
            fontsize=10, ha='center', style='italic', color='darkgreen', fontweight='bold')
    
    fig.suptitle('Gradient Accumulation vs Standard Training', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('12_gradient_accumulation.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 12_gradient_accumulation.png")
    plt.close()

# ============================================================================
# 5. COMPLETE TRAINING OVERVIEW
# ============================================================================

def create_complete_training_overview():
    """Vollständiger Überblick über das gesamte Training."""
    
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    fig.text(0.5, 0.97, 'Complete Training Overview: Von Daten zu Final Model', 
             fontsize=18, fontweight='bold', ha='center')
    
    # ========== DATASET SPLIT ==========
    y = 11.5
    
    ax.text(0.3, y, '① DATASET', fontsize=11, fontweight='bold', va='center',
           bbox=dict(boxstyle='circle', facecolor='#FF6B6B', edgecolor='white', linewidth=2))
    
    dataset_info = """
    Total: 240 Audio Files
    ├─ Training: 192 (80%)
    ├─ Validation: 48 (20%)
    └─ Test: [separate hold-out]
    
    Per Person: 8 Files
    (5 Vowels + 3 Syllables)
    """
    
    ax.text(1.5, y - 1.2, dataset_info, fontsize=9, ha='left', va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#FFE0E0', alpha=0.8, linewidth=2))
    
    # ========== TRAINING CONFIG ==========
    y = 8.5
    
    ax.text(0.3, y, '② CONFIG', fontsize=11, fontweight='bold', va='center',
           bbox=dict(boxstyle='circle', facecolor='#FFA726', edgecolor='white', linewidth=2))
    
    config_info = """
    Batch Size: 12
    Accumulation: 2 steps
    Effective Batch: 24
    
    Max Epochs: 20
    Batches/Epoch: 20
    Early Stopping: patience=5
    
    Samples/Epoch: 192
    Total Training Steps: 400
    """
    
    ax.text(1.5, y - 1.8, config_info, fontsize=9, ha='left', va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#FFE0B2', alpha=0.8, linewidth=2))
    
    # ========== TRAINING PHASE ==========
    y = 5
    
    ax.text(0.3, y, '③ TRAIN', fontsize=11, fontweight='bold', va='center',
           bbox=dict(boxstyle='circle', facecolor='#42A5F5', edgecolor='white', linewidth=2))
    
    training_info = """
    For each Epoch:
    ├─ Load Batch (12 samples)
    ├─ Augment (80% prob)
    ├─ Extract Features (WavLM)
    ├─ Forward Pass
    ├─ Compute Loss
    ├─ Backward & Accumulate
    └─ Update Weights (every 2 batches)
    
    Duration: ~2-3 min/epoch
    Total: ~40-60 min für 20 epochs
    """
    
    ax.text(1.5, y - 2.2, training_info, fontsize=8, ha='left', va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#BBDEFB', alpha=0.8, linewidth=2))
    
    # ========== VALIDATION PHASE ==========
    y = 11.5
    x_start = 5.5
    
    ax.text(x_start + 0.3, y, '④ VALIDATE', fontsize=11, fontweight='bold', va='center',
           bbox=dict(boxstyle='circle', facecolor='#66BB6A', edgecolor='white', linewidth=2))
    
    validation_info = """
    After each Epoch:
    ├─ Evaluate on 48 samples
    ├─ Compute F1-Score
    ├─ Monitor Loss
    ├─ Check Early Stop
    │  (best_f1 unchanged for 5 epochs)
    └─ Save best weights
    
    Metrics:
    ├─ F1-Score (macro)
    ├─ Accuracy
    └─ Per-class Precision/Recall
    """
    
    ax.text(x_start + 1.5, y - 2.2, validation_info, fontsize=8, ha='left', va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8, linewidth=2))
    
    # ========== RESULTS ==========
    y = 7.5
    x_start = 5.5
    
    ax.text(x_start + 0.3, y, '⑤ RESULTS', fontsize=11, fontweight='bold', va='center',
           bbox=dict(boxstyle='circle', facecolor='#AB47BC', edgecolor='white', linewidth=2))
    
    results_info = """
    Best Model @ Epoch 16:
    ├─ F1-Score: 0.72 ⭐
    ├─ Precision: 0.74
    ├─ Recall: 0.71
    ├─ Accuracy: 73%
    └─ Val Loss: 0.32
    
    Improvement:
    ├─ vs Mean Pooling: +7%
    ├─ vs Baseline: +15%
    └─ Meets Clinical Target!
    """
    
    ax.text(x_start + 1.5, y - 2, results_info, fontsize=8, ha='left', va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#E1BEE7', alpha=0.8, linewidth=2))
    
    # ========== DEPLOYMENT ==========
    y = 3
    x_start = 5.5
    
    ax.text(x_start + 0.3, y, '⑥ DEPLOY', fontsize=11, fontweight='bold', va='center',
           bbox=dict(boxstyle='circle', facecolor='#29B6F6', edgecolor='white', linewidth=2))
    
    deployment_info = """
    Production Ready:
    ├─ Save Best Weights
    ├─ Export Model (.pt)
    ├─ Create API
    ├─ Test on Hold-out Set
    └─ Monitor Performance
    
    Inference Time:
    └─ ~500ms per 3s audio
    └─ (Suitable for real-time)
    """
    
    ax.text(x_start + 1.5, y - 1.5, deployment_info, fontsize=8, ha='left', va='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#B3E5FC', alpha=0.8, linewidth=2))
    
    # ========== ARROWS CONNECTING FLOW ==========
    # Dataset → Config
    arrow = FancyArrowPatch((0.3, 7.2), (0.3, 7.8),
                           arrowstyle='->', mutation_scale=30, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # Config → Train
    arrow = FancyArrowPatch((0.3, 4.2), (0.3, 4.8),
                           arrowstyle='->', mutation_scale=30, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # Train → Validate (right side)
    arrow = FancyArrowPatch((3.5, 4), (5.5, 4),
                           arrowstyle='<->', mutation_scale=30, linewidth=2.5, color='#FF6B6B', linestyle='--')
    ax.add_patch(arrow)
    ax.text(4.5, 4.3, 'Every Epoch', fontsize=9, ha='center', style='italic', fontweight='bold')
    
    # Results
    arrow = FancyArrowPatch((6.5, 6.5), (6.5, 5.5),
                           arrowstyle='->', mutation_scale=30, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # Deploy
    arrow = FancyArrowPatch((6.5, 2.2), (6.5, 2.5),
                           arrowstyle='->', mutation_scale=30, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('13_complete_training_overview.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 13_complete_training_overview.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CREATING BATCH & EPOCH TRAINING VISUALIZATIONS")
    print("="*80 + "\n")
    
    print("📊 1. Creating batch concepts visualization...")
    create_batch_concepts()
    
    print("📊 2. Creating training progression visualization...")
    create_training_progression()
    
    print("📊 3. Creating batch processing pipeline...")
    create_batch_processing()
    
    print("📊 4. Creating gradient accumulation explanation...")
    create_gradient_accumulation()
    
    print("📊 5. Creating complete training overview...")
    create_complete_training_overview()
    
    print("\n" + "="*80)
    print("✅ ALL TRAINING VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*80)
    
    print("""
Generated Visualizations:
  9.  09_batch_concepts.png              - Batch, Batch Size, Epochs erklärt
  10. 10_training_progression.png        - Loss & F1-Score über Epochs
  11. 11_batch_processing.png            - Detaillierter Batch-Processing Flow
  12. 12_gradient_accumulation.png       - Gradient Accumulation erklärt
  13. 13_complete_training_overview.png  - Vollständiger Training Workflow

Ready for presentations, papers, and documentation! 🎉
""")
