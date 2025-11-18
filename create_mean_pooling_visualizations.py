#!/usr/bin/env python3
"""
Mean Pooling Visualisierungen
Detaillierte Darstellung von Mean Pooling, Vergleiche und Ablauf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. MEAN POOLING CONCEPT & MATHEMATICS
# ============================================================================

def create_mean_pooling_concept():
    """Erklärt das Konzept von Mean Pooling mathematisch und visuell."""
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    fig.text(0.5, 0.97, 'Mean Pooling: Konzept & Mathematik', 
             fontsize=18, fontweight='bold', ha='center')
    
    # ========== DEFINITION ==========
    y = 10.8
    
    rect = FancyBboxPatch((0.5, y - 0.6), 9, 1.1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#1976D2', facecolor='#BBDEFB', linewidth=2.5)
    ax.add_patch(rect)
    
    ax.text(1, y + 0.25, 'Definition:', fontsize=12, fontweight='bold', va='center')
    ax.text(2.2, y + 0.25, 'Mean Pooling nimmt den DURCHSCHNITT aller Frames', 
           fontsize=11, va='center', style='italic')
    
    ax.text(1, y - 0.25, 'Mathematik:', fontsize=12, fontweight='bold', va='center')
    ax.text(2.2, y - 0.25, r'$F_{pooled} = \frac{1}{T} \sum_{t=1}^{T} F_t$', 
           fontsize=12, va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== EXAMPLE: SEQUENCE OF FRAMES ==========
    y = 9
    
    ax.text(0.5, y, 'Beispiel: Audio-Sequenz', fontsize=11, fontweight='bold', va='top')
    
    # Simulate voice feature sequence (energy over time)
    time_steps = np.linspace(0, 3, 12)
    # Normal speaker: relatively stable energy
    energy = 0.85 + 0.1 * np.sin(2 * np.pi * time_steps / 3) + np.random.randn(12) * 0.05
    
    # Draw frames
    frame_y = 8
    for i, (t, e) in enumerate(zip(time_steps, energy)):
        x = 1 + i * 0.7
        
        # Frame box (height represents energy)
        height = e * 0.8
        rect = Rectangle((x, frame_y - height), 0.6, height, 
                        facecolor='#42A5F5', edgecolor='#0D47A1', linewidth=1.5, alpha=0.7)
        ax.add_patch(rect)
        
        # Frame number
        ax.text(x + 0.3, frame_y - height - 0.3, f'F{i+1}', 
               fontsize=7, ha='center', fontweight='bold')
        
        # Energy value
        ax.text(x + 0.3, frame_y + 0.3, f'{e:.2f}', 
               fontsize=7, ha='center', style='italic', color='darkblue')
    
    ax.text(1, frame_y + 0.8, 'Zeitliche Sequenz (12 Frames)', fontsize=10, fontweight='bold')
    ax.text(9.5, frame_y, '← Features\nüber Zeit', fontsize=9, ha='center', style='italic')
    
    # ========== MEAN CALCULATION STEP BY STEP ==========
    y = 6.8
    
    ax.text(0.5, y, 'Berechnung Schritt-für-Schritt:', fontsize=11, fontweight='bold')
    
    y -= 0.6
    
    # Step 1: Sum
    sum_text = f"Schritt 1: SUMME aller Frames"
    ax.text(1, y, sum_text, fontsize=10, fontweight='bold', color='darkgreen')
    
    sum_values = f"Sum = {energy[0]:.2f} + {energy[1]:.2f} + ... + {energy[-1]:.2f} = {np.sum(energy):.2f}"
    ax.text(1.2, y - 0.4, sum_values, fontsize=9, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.7))
    
    y -= 1.2
    
    # Step 2: Count
    count_text = f"Schritt 2: ANZAHL der Frames"
    ax.text(1, y, count_text, fontsize=10, fontweight='bold', color='darkgreen')
    
    count_values = f"Count = {len(energy)} Frames"
    ax.text(1.2, y - 0.4, count_values, fontsize=9, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.7))
    
    y -= 1.2
    
    # Step 3: Divide
    mean_value = np.mean(energy)
    divide_text = f"Schritt 3: DURCHSCHNITT = Sum ÷ Count"
    ax.text(1, y, divide_text, fontsize=10, fontweight='bold', color='darkgreen')
    
    divide_values = f"Mean = {np.sum(energy):.2f} ÷ {len(energy)} = {mean_value:.3f}"
    ax.text(1.2, y - 0.4, divide_values, fontsize=9, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#FFF59D', alpha=0.8, linewidth=2))
    
    # ========== RESULT ==========
    y = 2.5
    
    rect = FancyBboxPatch((1, y - 0.6), 4, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=2.5)
    ax.add_patch(rect)
    
    ax.text(3, y + 0.3, 'RESULT: Pooled Feature', fontsize=11, fontweight='bold', ha='center', color='darkred')
    ax.text(3, y - 0.2, f'{mean_value:.4f} (1-dimensional)', fontsize=10, ha='center', 
           fontweight='bold', color='darkred')
    
    # ========== RIGHT SIDE: DIMENSIONALITY ==========
    x_start = 5.5
    
    ax.text(x_start, 10, 'Dimensionalität:', fontsize=11, fontweight='bold')
    
    # Input dimensions
    y = 9.3
    rect = FancyBboxPatch((x_start, y - 0.4), 4, 0.8,
                         boxstyle="round,pad=0.05",
                         edgecolor='#42A5F5', facecolor='#BBDEFB', linewidth=2)
    ax.add_patch(rect)
    ax.text(x_start + 2, y, 'Input: (T=300, D=1024)', fontsize=10, ha='center', fontweight='bold')
    ax.text(x_start + 2, y - 0.8, 'T = Anzahl Frames', fontsize=8, ha='center', style='italic')
    ax.text(x_start + 2, y - 1.2, 'D = Feature Dimension', fontsize=8, ha='center', style='italic')
    
    # Arrow down
    arrow = FancyArrowPatch((x_start + 2, y - 0.5), (x_start + 2, y - 1.6),
                           arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow)
    
    # Mean pooling operation
    y = 7.5
    ax.text(x_start + 2, y + 0.3, 'Mean Pooling', fontsize=10, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))
    ax.text(x_start + 2, y - 0.3, r'$\frac{1}{300}\sum_{t=1}^{300} F_t$', fontsize=10, ha='center')
    
    # Arrow down
    arrow = FancyArrowPatch((x_start + 2, y - 0.8), (x_start + 2, y - 1.5),
                           arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
    ax.add_patch(arrow)
    
    # Output dimensions
    y = 5.8
    rect = FancyBboxPatch((x_start, y - 0.4), 4, 0.8,
                         boxstyle="round,pad=0.05",
                         edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=2)
    ax.add_patch(rect)
    ax.text(x_start + 2, y, 'Output: (D=1024)', fontsize=10, ha='center', fontweight='bold')
    ax.text(x_start + 2, y - 0.8, '300 Frames → 1 Value', fontsize=8, ha='center', style='italic')
    
    # ========== PROPERTY TABLE ==========
    y = 4.2
    
    ax.text(x_start, y, 'Eigenschaften:', fontsize=11, fontweight='bold')
    
    properties = [
        ('Robustheit', '⭐⭐⭐ Sehr robust'),
        ('Tempo-Info', '❌ Verliert Timing'),
        ('Rechenzeit', '⭐⭐⭐ Schnell'),
        ('Dimension', '1024-dim'),
    ]
    
    for i, (prop, value) in enumerate(properties):
        y_prop = y - 0.6 - i * 0.5
        ax.text(x_start + 0.2, y_prop, f'{prop}:', fontsize=9, fontweight='bold')
        ax.text(x_start + 2.2, y_prop, value, fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('14_mean_pooling_concept.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 14_mean_pooling_concept.png")
    plt.close()

# ============================================================================
# 2. MEAN POOLING EXAMPLE: NORMAL vs ALS
# ============================================================================

def create_mean_pooling_normal_vs_als():
    """Zeigt Mean Pooling bei normalen Sprechern vs ALS-Patienten."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    fig.suptitle('Mean Pooling: Normal vs ALS Patient', fontsize=16, fontweight='bold', y=0.98)
    
    # ========== NORMAL SPEAKER ==========
    
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    ax1.text(5, 9.5, 'Normal Speaker', fontsize=13, fontweight='bold', ha='center')
    
    # Time series
    t = np.linspace(0, 3, 100)
    normal_energy = 0.85 + 0.08 * np.sin(2 * np.pi * t / 3) + np.random.randn(100) * 0.04
    
    # Plot
    time_steps = np.linspace(0, 3, 100)
    for i in range(0, 100, 5):
        x_pos = 1 + (i / 100) * 8
        height = normal_energy[i] * 2
        
        rect = Rectangle((x_pos - 0.15, 5 - height), 0.3, height, 
                        facecolor='#4CAF50', edgecolor='#2E7D32', linewidth=0.5, alpha=0.7)
        ax1.add_patch(rect)
    
    # Mean line
    mean_normal = np.mean(normal_energy)
    ax1.axhline(y=5 - mean_normal * 2, color='#2E7D32', linestyle='--', linewidth=2.5, 
               label=f'Mean = {mean_normal:.3f}')
    
    ax1.set_ylim([2, 8])
    ax1.text(5, 1.8, 'Energie über Zeit (konsistent)', fontsize=10, ha='center', style='italic')
    ax1.legend(fontsize=10, loc='upper right')
    
    # Stats box
    stats_text = f"""
    Energie-Statistik:
    Min: {np.min(normal_energy):.3f}
    Max: {np.max(normal_energy):.3f}
    Mean: {np.mean(normal_energy):.3f}
    Std: {np.std(normal_energy):.3f}
    
    → Pooled Feature: {np.mean(normal_energy):.4f}
    → 1024-dimensional
    """
    
    ax1.text(5, 0.5, stats_text, fontsize=8, ha='center', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#C8E6C9', alpha=0.8))
    
    # ========== ALS PATIENT ==========
    
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    ax2.text(5, 9.5, 'ALS Patient', fontsize=13, fontweight='bold', ha='center')
    
    # Time series with degradation
    als_energy = 1.0 - 0.35 * (t / 3) + np.random.randn(100) * 0.08
    
    # Plot
    for i in range(0, 100, 5):
        x_pos = 1 + (i / 100) * 8
        height = als_energy[i] * 2
        
        # Color gradient from green to red
        progress = i / 100
        color = plt.cm.RdYlGn_r(progress)  # Red at end, Green at start
        
        rect = Rectangle((x_pos - 0.15, 5 - height), 0.3, height, 
                        facecolor=color, edgecolor='darkred', linewidth=0.5, alpha=0.7)
        ax2.add_patch(rect)
    
    # Mean line
    mean_als = np.mean(als_energy)
    ax2.axhline(y=5 - mean_als * 2, color='#D32F2F', linestyle='--', linewidth=2.5, 
               label=f'Mean = {mean_als:.3f}')
    
    ax2.set_ylim([2, 8])
    ax2.text(5, 1.8, 'Energie mit Degradation', fontsize=10, ha='center', style='italic')
    ax2.legend(fontsize=10, loc='upper right')
    
    # Stats box
    stats_text = f"""
    Energie-Statistik:
    Min: {np.min(als_energy):.3f}
    Max: {np.max(als_energy):.3f}
    Mean: {np.mean(als_energy):.3f}
    Std: {np.std(als_energy):.3f}
    
    → Pooled Feature: {np.mean(als_energy):.4f}
    → 1024-dimensional
    
    ⚠ Problem: Mean verliert
    die DEGRADATION Information!
    """
    
    ax2.text(5, 0.5, stats_text, fontsize=8, ha='center', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('15_mean_pooling_normal_vs_als.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 15_mean_pooling_normal_vs_als.png")
    plt.close()

# ============================================================================
# 3. MEAN POOLING LIMITATIONS & PROBLEMS
# ============================================================================

def create_mean_pooling_limitations():
    """Zeigt die Limitationen von Mean Pooling."""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Mean Pooling: Limitationen und Probleme', fontsize=16, fontweight='bold', y=0.98)
    
    # ========== PROBLEM 1: TEMPORAL INFORMATION LOSS ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    t = np.linspace(0, 3, 100)
    
    # Two very different patterns, same mean!
    pattern1 = 0.5 + 0.3 * np.sin(2 * np.pi * t / 3)  # Smooth
    pattern2 = np.concatenate([np.ones(50) * 0.8, np.ones(50) * 0.2])  # Degradation
    
    mean_val = np.mean(pattern1)
    
    ax1.plot(t, pattern1, 'g-', linewidth=2.5, label=f'Smooth (Mean={mean_val:.3f})')
    ax1.plot(t, pattern2, 'r--', linewidth=2.5, label=f'Degradation (Mean={np.mean(pattern2):.3f})')
    ax1.fill_between(t, pattern1, alpha=0.2, color='green')
    ax1.fill_between(t, pattern2, alpha=0.2, color='red')
    
    ax1.set_title('Problem 1: Temporal Information Loss', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    ax1.text(1.5, 0.05, 'Same Mean ≠ Same Pattern!', fontsize=10, ha='center', 
            style='italic', color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))
    
    # ========== PROBLEM 2: LOSES ASYMMETRY ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Normal speaker (consistent)
    normal = 0.85 + np.random.randn(100) * 0.02
    # ALS patient (degraded)
    als = np.linspace(1.0, 0.5, 100) + np.random.randn(100) * 0.05
    
    ax2.boxplot([normal, als], labels=['Normal', 'ALS'], patch_artist=True,
               boxprops=dict(facecolor='#C8E6C9'),
               medianprops=dict(color='darkgreen', linewidth=2),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5))
    
    ax2.axhline(y=np.mean(normal), color='green', linestyle='--', linewidth=2, alpha=0.7, label='Normal Mean')
    ax2.axhline(y=np.mean(als), color='red', linestyle='--', linewidth=2, alpha=0.7, label='ALS Mean')
    
    ax2.set_title('Problem 2: Loses Asymmetry Info', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Energy')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')
    
    ax2.text(1.5, 0.45, 'Means are VERY similar!\nBut patterns are different!', 
            fontsize=9, ha='center', style='italic', color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))
    
    # ========== PROBLEM 3: NOISE SENSITIVITY ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Clean signal
    clean = 0.8 + np.zeros(100)
    # Signal with noise spikes
    noisy = 0.8 + np.random.randn(100) * 0.1
    noisy[25] = 1.5  # Outlier
    noisy[75] = 0.2  # Outlier
    
    ax3.scatter(range(100), clean, alpha=0.5, s=30, color='green', label='Clean')
    ax3.scatter(range(100), noisy, alpha=0.5, s=30, color='red', label='With Noise/Outliers')
    
    ax3.axhline(y=np.mean(clean), color='green', linestyle='-', linewidth=2.5, label=f'Clean Mean: {np.mean(clean):.3f}')
    ax3.axhline(y=np.mean(noisy), color='red', linestyle='--', linewidth=2.5, label=f'Noisy Mean: {np.mean(noisy):.3f}')
    
    ax3.set_title('Problem 3: Outlier Sensitivity', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Energy')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    ax3.text(50, -0.1, 'Outliers shift the mean!', fontsize=10, ha='center', 
            style='italic', color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))
    
    # ========== PROBLEM 4: NO TEMPORAL GRADIENT CAPTURE ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Different degradation patterns
    slow_degrade = 1.0 - 0.1 * (np.linspace(0, 1, 100) ** 1)  # Linear
    fast_degrade = 1.0 - 0.1 * (np.linspace(0, 1, 100) ** 2)  # Quadratic
    
    ax4.plot(np.linspace(0, 3, 100), slow_degrade, 'o-', linewidth=2, markersize=4, 
            label=f'Slow Degrade (Mean={np.mean(slow_degrade):.3f})')
    ax4.plot(np.linspace(0, 3, 100), fast_degrade, 's-', linewidth=2, markersize=4, 
            label=f'Fast Degrade (Mean={np.mean(fast_degrade):.3f})')
    
    ax4.set_title('Problem 4: Loses Degradation Rate', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Energy')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    ax4.text(1.5, 0.82, 'Different degradation rates\nbut similar means!', 
            fontsize=10, ha='center', style='italic', color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.8))
    
    plt.savefig('16_mean_pooling_limitations.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 16_mean_pooling_limitations.png")
    plt.close()

# ============================================================================
# 4. MEAN POOLING vs OTHER METHODS - DETAILED COMPARISON
# ============================================================================

def create_mean_pooling_comparison():
    """Vergleicht Mean Pooling mit anderen Pooling-Methoden."""
    
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    fig.text(0.5, 0.97, 'Mean Pooling vs Alternative Pooling Methods', 
             fontsize=18, fontweight='bold', ha='center')
    
    # Create sample audio patterns
    t = np.linspace(0, 3, 300)
    
    # Normal speaker
    normal = 0.85 + 0.1 * np.sin(2 * np.pi * t / 3) + np.random.randn(300) * 0.03
    
    # ALS patient (degraded)
    als = 1.0 - 0.4 * (t / 3) + np.random.randn(300) * 0.08
    
    # ========== COMPARISON TABLE ==========
    y_base = 11.5
    
    methods = [
        {
            'name': 'Mean Pooling',
            'formula': r'$\frac{1}{T}\sum_{t=1}^{T} F_t$',
            'pros': ['Robust', 'Simple', 'Fast'],
            'cons': ['Loses timing', 'No asymmetry', 'Loses gradients'],
            'dim': '1024',
            'color': '#E3F2FD',
            'score_normal': 0.85,
            'score_als': 0.70,
            'y': y_base - 2,
        },
        {
            'name': 'Max Pooling',
            'formula': r'$\max(F_1, ..., F_T)$',
            'pros': ['Captures peaks', 'Simple'],
            'cons': ['Too spiky', 'Noise', 'Single frame'],
            'dim': '1024',
            'color': '#FFF3E0',
            'score_normal': 0.82,
            'score_als': 0.62,
            'y': y_base - 5.5,
        },
        {
            'name': 'First-Last Pooling',
            'formula': r'$[F_1 \parallel F_T]$',
            'pros': ['Measures gap', 'Temporal info'],
            'cons': ['Spike sensitive', 'Two frames only'],
            'dim': '2048',
            'color': '#F3E5F5',
            'score_normal': 0.88,
            'score_als': 0.82,
            'y': y_base - 9,
        },
    ]
    
    for method in methods:
        y = method['y']
        
        # Header
        rect = FancyBboxPatch((0.5, y + 0.5), 9, 0.7,
                             boxstyle="round,pad=0.05",
                             edgecolor='#424242', facecolor=method['color'], linewidth=2)
        ax.add_patch(rect)
        
        ax.text(1, y + 0.85, method['name'], fontsize=11, fontweight='bold', va='center')
        ax.text(3.5, y + 0.85, f"Formula: {method['formula']}", fontsize=9, va='center')
        ax.text(7, y + 0.85, f"Output: {method['dim']}-dim", fontsize=9, va='center', fontweight='bold')
        
        # Details
        y -= 0.1
        
        # Pros
        ax.text(0.7, y - 0.3, '✓ Pros:', fontsize=9, fontweight='bold', color='green')
        for i, pro in enumerate(method['pros']):
            ax.text(1.2, y - 0.3 - (i+1)*0.25, f'• {pro}', fontsize=8, color='green')
        
        # Cons
        ax.text(4, y - 0.3, '✗ Cons:', fontsize=9, fontweight='bold', color='red')
        for i, con in enumerate(method['cons']):
            ax.text(4.5, y - 0.3 - (i+1)*0.25, f'• {con}', fontsize=8, color='red')
        
        # Score bars
        ax.text(7.5, y - 0.3, 'Discrimination:', fontsize=9, fontweight='bold')
        
        # Normal score
        bar_width = 1.5
        rect_normal = Rectangle((7.5, y - 1.1), bar_width * method['score_normal'], 0.25,
                               facecolor='#4CAF50', edgecolor='#2E7D32', linewidth=1)
        ax.add_patch(rect_normal)
        ax.text(7.5 + bar_width * 0.5, y - 0.95, f"N: {method['score_normal']:.2f}", 
               fontsize=8, ha='center', fontweight='bold')
        
        # ALS score
        rect_als = Rectangle((7.5, y - 1.5), bar_width * method['score_als'], 0.25,
                            facecolor='#F44336', edgecolor='#D32F2F', linewidth=1)
        ax.add_patch(rect_als)
        ax.text(7.5 + bar_width * 0.5, y - 1.35, f"ALS: {method['score_als']:.2f}", 
               fontsize=8, ha='center', fontweight='bold')
    
    # ========== BOTTOM SUMMARY ==========
    y = 0.8
    
    rect = FancyBboxPatch((0.5, y - 0.7), 9, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=2.5)
    ax.add_patch(rect)
    
    ax.text(1, y + 0.3, '📌 Key Insight:', fontsize=10, fontweight='bold', color='darkred')
    
    summary = """
    Mean Pooling ist GUT für allgemeine Robustheit, aber SCHLECHT für ALS-Diagnose!
    Es verliert die temporale Degradationsinformation, die für die Diagnose essentiell ist.
    
    First-Last-Window bietet den besten Kompromiss: Robustheit PLUS Degradation-Erfassung!
    """
    
    ax.text(1.2, y - 0.2, summary, fontsize=9, style='italic', va='top')
    
    plt.tight_layout()
    plt.savefig('17_mean_pooling_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 17_mean_pooling_comparison.png")
    plt.close()

# ============================================================================
# 5. MEAN POOLING DIMENSIONALITY FLOW
# ============================================================================

def create_mean_pooling_dimensionality():
    """Zeigt den Dimensionalitätsfluss durch Mean Pooling."""
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    fig.text(0.5, 0.97, 'Mean Pooling: Dimensionalität und Datenfluss', 
             fontsize=18, fontweight='bold', ha='center')
    
    # ========== STEP BY STEP FLOW ==========
    y = 11
    
    steps = [
        {
            'label': 'Audio Input',
            'shape': '16 kHz, 3 sec',
            'dim': '(48000,)',
            'color': '#FF6B6B',
        },
        {
            'label': 'WavLM Extraction',
            'shape': '300 Time Steps',
            'dim': '(300, 1024)',
            'color': '#42A5F5',
        },
        {
            'label': 'Mean Pooling',
            'shape': 'Average over Time',
            'dim': '→ (1024,)',
            'color': '#FFA726',
        },
        {
            'label': 'Per-File FC',
            'shape': '1024 → 128',
            'dim': '(128,)',
            'color': '#9C27B0',
        },
        {
            'label': 'Concatenate ×8',
            'shape': '8 Files merged',
            'dim': '(1024,)',
            'color': '#4CAF50',
        },
        {
            'label': 'Final Classifier',
            'shape': '1024 → 5 classes',
            'dim': '(5,)',
            'color': '#D32F2F',
        },
    ]
    
    x_positions = np.linspace(1, 9, len(steps))
    
    for i, (x, step) in enumerate(zip(x_positions, steps)):
        # Draw box
        rect = FancyBboxPatch((x - 0.4, y - 0.8), 0.8, 1.4,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=step['color'], linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        # Label
        ax.text(x, y + 0.5, step['label'], fontsize=9, ha='center', fontweight='bold')
        
        # Shape info
        ax.text(x, y + 0.05, step['shape'], fontsize=8, ha='center', style='italic')
        
        # Dimension
        ax.text(x, y - 0.6, step['dim'], fontsize=9, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Arrow to next step
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((x + 0.5, y), (x_positions[i+1] - 0.5, y),
                                  arrowstyle='->', mutation_scale=20, linewidth=2.5, color='black')
            ax.add_patch(arrow)
    
    # ========== DETAIL: MEAN POOLING OPERATION ==========
    y = 7.5
    
    ax.text(2.5, y, 'DETAIL: Mean Pooling Operation', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8, linewidth=2))
    
    # Visual representation
    y -= 1
    
    # Input frames
    ax.text(1, y, 'Input:', fontsize=10, fontweight='bold')
    ax.text(1.3, y, '300 Frames × 1024 Features', fontsize=9, style='italic')
    
    for i in range(10):
        x_frame = 1.5 + i * 0.35
        rect = Rectangle((x_frame, y - 0.6), 0.3, 0.4, facecolor='#BBDEFB', 
                        edgecolor='#0D47A1', linewidth=1)
        ax.add_patch(rect)
        ax.text(x_frame + 0.15, y - 0.85, f'F{i+1}', fontsize=6, ha='center', fontweight='bold')
    
    ax.text(5.5, y - 0.4, '... (300 Frames total)', fontsize=8, style='italic')
    
    # Operation
    y -= 1.3
    
    ax.text(1, y, 'Operation:', fontsize=10, fontweight='bold')
    
    operation_text = r'''
    For each of 1024 feature dimensions:
    
    pooled[d] = (f[0,d] + f[1,d] + ... + f[299,d]) / 300
    
    where f[t,d] = feature at frame t, dimension d
    '''
    
    ax.text(1.3, y - 0.5, operation_text, fontsize=8, fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))
    
    # Output
    y -= 2.2
    
    ax.text(1, y, 'Output:', fontsize=10, fontweight='bold')
    ax.text(1.3, y, '1 × 1024 Vector (single average representation)', fontsize=9, style='italic')
    
    for i in range(10):
        x_feat = 1.5 + i * 0.35
        rect = Rectangle((x_feat, y - 0.6), 0.3, 0.4, facecolor='#FFE0B2',
                        edgecolor='#E65100', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x_feat + 0.15, y - 0.85, f'P{i+1}', fontsize=6, ha='center', fontweight='bold')
    
    ax.text(5.5, y - 0.4, '... (1024 dimensions)', fontsize=8, style='italic')
    
    # ========== KEY POINTS ==========
    y = 0.8
    
    ax.text(6, y + 0.8, '🔑 Key Points:', fontsize=11, fontweight='bold')
    
    points = [
        '✓ Komplett ZEITINFORMATION zerstört',
        '✓ Alle 300 Frames in 1 Vektor',
        '✓ Kein Bezug zu WANN Degradation passiert',
        '✗ Daher schlecht für ALS-Diagnose',
        '✓ Aber: Gut für allgemeine Sprechererkennung',
    ]
    
    for i, point in enumerate(points):
        ax.text(6.2, y + 0.8 - i * 0.35, point, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('18_mean_pooling_dimensionality.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 18_mean_pooling_dimensionality.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CREATING MEAN POOLING VISUALIZATIONS")
    print("="*80 + "\n")
    
    print("📊 1. Creating mean pooling concept visualization...")
    create_mean_pooling_concept()
    
    print("📊 2. Creating normal vs ALS comparison...")
    create_mean_pooling_normal_vs_als()
    
    print("📊 3. Creating mean pooling limitations...")
    create_mean_pooling_limitations()
    
    print("📊 4. Creating pooling methods comparison...")
    create_mean_pooling_comparison()
    
    print("📊 5. Creating dimensionality flow visualization...")
    create_mean_pooling_dimensionality()
    
    print("\n" + "="*80)
    print("✅ ALL MEAN POOLING VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*80)
    
    print("""
Generated Mean Pooling Visualizations:
  14. 14_mean_pooling_concept.png          - Konzept & Mathematik
  15. 15_mean_pooling_normal_vs_als.png    - Normal vs ALS Beispiele
  16. 16_mean_pooling_limitations.png      - 4 Hauptprobleme
  17. 17_mean_pooling_comparison.png       - Vergleich mit anderen Methoden
  18. 18_mean_pooling_dimensionality.png   - Dimensionalität & Datenfluss

Ready for presentations, papers, and documentation! 🎉
""")
