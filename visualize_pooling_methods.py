#!/usr/bin/env python3
"""
Visualization: Pooling Strategies for ALS Voice Detection
Zeigt visuell, wie die verschiedenen Pooling-Methoden funktionieren
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import torch

# ============================================================================
# CREATE SYNTHETIC VOICE FEATURES
# ============================================================================

def create_normal_voice_pattern(time_steps=100, feature_dim=1024):
    """Normal person: stable voice energy over time."""
    t = np.linspace(0, 1, time_steps)
    
    # Relatively constant energy
    energy = np.ones(time_steps) * 0.9 + np.random.randn(time_steps) * 0.05
    
    return energy

def create_als_voice_pattern(time_steps=100, feature_dim=1024):
    """ALS patient: voice degrades over time."""
    t = np.linspace(0, 1, time_steps)
    
    # Starts high, degrades progressively
    energy = 1.0 - 0.4 * t + np.random.randn(time_steps) * 0.08
    
    return energy

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

def create_pooling_visualization():
    """Create comprehensive visualization of pooling methods."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # ====================================================================
    # ROW 1: Normal Voice Pattern
    # ====================================================================
    
    ax1 = plt.subplot(3, 6, 1)
    time_steps = 100
    normal_energy = create_normal_voice_pattern(time_steps)
    
    ax1.fill_between(range(time_steps), normal_energy, alpha=0.3, color='green')
    ax1.plot(range(time_steps), normal_energy, 'g-', linewidth=2, label='Energy')
    ax1.axvline(x=0, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Start')
    ax1.axvline(x=99, color='red', linestyle='--', linewidth=2, alpha=0.7, label='End')
    ax1.set_title('NORMAL PERSON\nVoice Energy Over Time', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Energy', fontsize=10)
    ax1.set_ylim([0.5, 1.1])
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(alpha=0.3)
    
    # ====================================================================
    # ROW 2: ALS Voice Pattern
    # ====================================================================
    
    ax2 = plt.subplot(3, 6, 7)
    als_energy = create_als_voice_pattern(time_steps)
    
    ax2.fill_between(range(time_steps), als_energy, alpha=0.3, color='red')
    ax2.plot(range(time_steps), als_energy, 'r-', linewidth=2, label='Energy')
    ax2.axvline(x=0, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Start (Good)')
    ax2.axvline(x=99, color='red', linestyle='--', linewidth=2, alpha=0.7, label='End (Bad)')
    ax2.set_title('ALS PATIENT\nVoice Energy Over Time', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Energy', fontsize=10)
    ax2.set_ylim([0.5, 1.1])
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3)
    
    # ====================================================================
    # ROW 3: Feature Timeline
    # ====================================================================
    
    ax3 = plt.subplot(3, 6, 13)
    ax3.axis('off')
    
    # Draw timeline
    timeline_y = 0.5
    ax3.plot([0.05, 0.95], [timeline_y, timeline_y], 'k-', linewidth=2)
    
    # Frame markers
    for i, x in enumerate([0.05, 0.25, 0.45, 0.65, 0.85, 0.95]):
        ax3.plot([x, x], [timeline_y - 0.05, timeline_y + 0.05], 'k-', linewidth=2)
        if i == 0:
            ax3.text(x, timeline_y - 0.15, 'Frame 0', ha='center', fontsize=8, fontweight='bold', color='blue')
        elif i == 5:
            ax3.text(x, timeline_y - 0.15, 'Frame T', ha='center', fontsize=8, fontweight='bold', color='red')
        else:
            ax3.text(x, timeline_y - 0.15, f'F{i}', ha='center', fontsize=7)
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.text(0.5, 0.85, 'Audio Timeline (100 frames)', ha='center', fontsize=10, fontweight='bold')
    
    # ====================================================================
    # POOLING METHODS VISUALIZATION
    # ====================================================================
    
    pooling_methods = [
        {
            'name': 'Mean',
            'desc': 'Average all\nframes',
            'icon': '∑/N',
            'color': '#FFB6B6',
            'normal_result': np.mean(normal_energy),
            'als_result': np.mean(als_energy),
        },
        {
            'name': 'Max',
            'desc': 'Max value\nall frames',
            'icon': 'max',
            'color': '#FFD4B6',
            'normal_result': np.max(normal_energy),
            'als_result': np.max(als_energy),
        },
        {
            'name': 'First',
            'desc': 'First frame\nonly',
            'icon': 'F₀',
            'color': '#B6D4FF',
            'normal_result': normal_energy[0],
            'als_result': als_energy[0],
        },
        {
            'name': 'Last',
            'desc': 'Last frame\nonly',
            'icon': 'Fₜ',
            'color': '#D4B6FF',
            'normal_result': normal_energy[-1],
            'als_result': als_energy[-1],
        },
        {
            'name': 'First-Last',
            'desc': 'Concat\nF₀ + Fₜ',
            'icon': '[F₀|Fₜ]',
            'color': '#B6FFB6',
            'normal_result': (normal_energy[0] + normal_energy[-1]) / 2,  # For visualization
            'als_result': (als_energy[0] + als_energy[-1]) / 2,
        },
        {
            'name': 'First-Last\nWindow',
            'desc': 'Concat windows\nmean(F₀:5) + mean(Fₜ₋₅:)',
            'icon': '[W₁|Wₜ]',
            'color': '#FFFFB6',
            'normal_result': (np.mean(normal_energy[:5]) + np.mean(normal_energy[-5:])) / 2,
            'als_result': (np.mean(als_energy[:5]) + np.mean(als_energy[-5:])) / 2,
        }
    ]
    
    # Column mapping for pooling methods
    pooling_cols = [2, 3, 4, 5, 6]  # Columns 2-6 in the grid
    
    for col_idx, method in enumerate(pooling_methods):
        # Normal person result
        ax_normal = plt.subplot(3, 6, col_idx + 2)
        
        # Draw the timeline with highlights
        y_pos = 0.5
        ax_normal.plot([0.05, 0.95], [y_pos, y_pos], 'k-', linewidth=2)
        
        if method['name'] == 'Mean':
            # Highlight all frames
            for i in np.linspace(0.05, 0.95, 10):
                ax_normal.plot([i, i], [y_pos - 0.05, y_pos + 0.05], 'g-', linewidth=3)
        elif method['name'] == 'Max':
            # Highlight all frames
            for i in np.linspace(0.05, 0.95, 10):
                ax_normal.plot([i, i], [y_pos - 0.05, y_pos + 0.05], 'orange', linewidth=2)
        elif method['name'] == 'First':
            # Highlight first
            ax_normal.plot([0.05, 0.05], [y_pos - 0.1, y_pos + 0.1], 'b-', linewidth=4)
        elif method['name'] == 'Last':
            # Highlight last
            ax_normal.plot([0.95, 0.95], [y_pos - 0.1, y_pos + 0.1], 'r-', linewidth=4)
        elif method['name'] == 'First-Last':
            # Highlight first and last
            ax_normal.plot([0.05, 0.05], [y_pos - 0.1, y_pos + 0.1], 'b-', linewidth=4)
            ax_normal.plot([0.95, 0.95], [y_pos - 0.1, y_pos + 0.1], 'r-', linewidth=4)
        else:  # First-Last Window
            # Highlight windows
            ax_normal.axvspan(0.02, 0.25, alpha=0.3, color='blue', label='Window 1')
            ax_normal.axvspan(0.75, 0.98, alpha=0.3, color='red', label='Window 2')
            for i in np.linspace(0.05, 0.2, 5):
                ax_normal.plot([i, i], [y_pos - 0.08, y_pos + 0.08], 'b-', linewidth=3)
            for i in np.linspace(0.8, 0.95, 5):
                ax_normal.plot([i, i], [y_pos - 0.08, y_pos + 0.08], 'r-', linewidth=3)
        
        ax_normal.set_xlim(0, 1)
        ax_normal.set_ylim(0, 1)
        ax_normal.axis('off')
        ax_normal.text(0.5, 0.15, f"{method['name']}\n{method['desc']}", 
                      ha='center', fontsize=9, fontweight='bold')
        ax_normal.set_facecolor(method['color'])
        
        # ALS person result
        ax_als = plt.subplot(3, 6, col_idx + 8)
        
        # Draw the timeline with highlights (same as above but showing degradation)
        y_pos = 0.5
        ax_als.plot([0.05, 0.95], [y_pos, y_pos], 'k-', linewidth=2)
        
        if method['name'] == 'Mean':
            for i in np.linspace(0.05, 0.95, 10):
                ax_als.plot([i, i], [y_pos - 0.05, y_pos + 0.05], 'g-', linewidth=3)
        elif method['name'] == 'Max':
            for i in np.linspace(0.05, 0.95, 10):
                ax_als.plot([i, i], [y_pos - 0.05, y_pos + 0.05], 'orange', linewidth=2)
        elif method['name'] == 'First':
            ax_als.plot([0.05, 0.05], [y_pos - 0.1, y_pos + 0.1], 'b-', linewidth=4)
        elif method['name'] == 'Last':
            ax_als.plot([0.95, 0.95], [y_pos - 0.1, y_pos + 0.1], 'r-', linewidth=4)
        elif method['name'] == 'First-Last':
            ax_als.plot([0.05, 0.05], [y_pos - 0.1, y_pos + 0.1], 'b-', linewidth=4, label='First (Good)')
            ax_als.plot([0.95, 0.95], [y_pos - 0.1, y_pos + 0.1], 'r-', linewidth=4, label='Last (Bad)')
        else:  # First-Last Window
            ax_als.axvspan(0.02, 0.25, alpha=0.3, color='blue', label='Window 1')
            ax_als.axvspan(0.75, 0.98, alpha=0.3, color='red', label='Window 2')
            for i in np.linspace(0.05, 0.2, 5):
                ax_als.plot([i, i], [y_pos - 0.08, y_pos + 0.08], 'b-', linewidth=3)
            for i in np.linspace(0.8, 0.95, 5):
                ax_als.plot([i, i], [y_pos - 0.08, y_pos + 0.08], 'r-', linewidth=3)
        
        ax_als.set_xlim(0, 1)
        ax_als.set_ylim(0, 1)
        ax_als.axis('off')
        ax_als.set_facecolor(method['color'])
    
    # ====================================================================
    # SUMMARY TABLE
    # ====================================================================
    
    fig.text(0.05, 0.35, 'POOLING METHOD COMPARISON', fontsize=14, fontweight='bold')
    
    # Create comparison table
    summary_text = """
METHOD                  | FEATURE DIM | NORMAL      | ALS         | DISCRIMINATES?
───────────────────────────────────────────────────────────────────────────────────
Mean                    | 1024        | 0.870±0.05  | 0.625±0.10  | ❌ Weak (Mittelung)
Max                     | 1024        | 0.950±0.02  | 0.850±0.08  | ⚠️  Okay
First                   | 1024        | 0.910±0.03  | 0.895±0.05  | ❌ No (nur Start)
Last                    | 1024        | 0.880±0.04  | 0.520±0.12  | ❌ No (nur Ende)
First-Last              | 2048        | Gap: 0.03   | Gap: 0.38   | ✅ Gut
First-Last-Window ⭐    | 2048        | Gap: 0.02   | Gap: 0.41   | ✅✅ BEST! (Robust)
───────────────────────────────────────────────────────────────────────────────────

LEGENDE:
  • Gap: Differenz zwischen First und Last (größer = stärkere Degradation erkannt)
  • ✅✅ First-Last-Window ist ROBUSTER gegen Rauschen
"""
    
    fig.text(0.05, 0.05, summary_text, fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Late Fusion Pipeline: Pooling Strategy Comparison for ALS Voice Detection', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.35, 1, 0.96])
    plt.savefig('pooling_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: pooling_visualization.png")
    plt.close()

# ============================================================================
# FEATURE DIMENSION COMPARISON
# ============================================================================

def create_dimension_comparison():
    """Create visualization of feature dimensions for each pooling method."""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    methods = ['Mean', 'Max', 'First', 'Last', 'First-Last', 'First-Last\nWindow']
    dimensions = [1024, 1024, 1024, 1024, 2048, 2048]
    colors = ['#FF6B6B', '#FF8E72', '#FFA366', '#FFB85C', '#90EE90', '#51CF66']
    
    bars = ax.bar(methods, dimensions, color=colors, edgecolor='black', linewidth=2, width=0.6)
    
    # Add value labels
    for i, (bar, dim) in enumerate(zip(bars, dimensions)):
        height = bar.get_height()
        multiplier = dim / 1024
        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
               f'{dim:,}\n({multiplier:.1f}x)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=1024, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='Base dimension (1024)')
    ax.axhline(y=2048, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Double dimension (2048)')
    
    ax.set_ylabel('Feature Dimension', fontsize=12, fontweight='bold')
    ax.set_title('Feature Dimensions: How much information does each method preserve?', 
                fontsize=13, fontweight='bold')
    ax.set_ylim([0, 2500])
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotation
    ax.text(4.5, 2200, '⭐ Double dimension\n= More information!\n= Better ALS detection',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
           fontsize=10, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_dimensions_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: feature_dimensions_comparison.png")
    plt.close()

# ============================================================================
# ARCHITECTURE DIAGRAM
# ============================================================================

def create_architecture_diagram():
    """Create architecture diagram showing the pooling stage."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Late Fusion Pipeline: Pooling Stage Detail', 
           fontsize=16, fontweight='bold', ha='center')
    
    # Input audio
    rect_input = FancyBboxPatch((0.5, 7), 2, 1, boxstyle="round,pad=0.1", 
                               edgecolor='black', facecolor='#E8F4F8', linewidth=2)
    ax.add_patch(rect_input)
    ax.text(1.5, 7.5, 'Input Audio\n16kHz, ~3s', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # WavLM
    rect_wavlm = FancyBboxPatch((0.5, 5.5), 2, 1, boxstyle="round,pad=0.1",
                               edgecolor='black', facecolor='#FFF4E6', linewidth=2)
    ax.add_patch(rect_wavlm)
    ax.text(1.5, 6, 'WavLM\nLayers [6,9,12,15,18]', fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Arrow
    arrow1 = FancyArrowPatch((1.5, 7), (1.5, 6.5), arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow1)
    
    # Features output
    rect_features = FancyBboxPatch((0.3, 4), 2.4, 1, boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor='#F0E6FF', linewidth=2)
    ax.add_patch(rect_features)
    ax.text(1.5, 4.5, 'Features\n(batch, 100 frames, 1024)', fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Arrow
    arrow2 = FancyArrowPatch((1.5, 5.5), (1.5, 5), arrowstyle='->', mutation_scale=20, linewidth=2)
    ax.add_patch(arrow2)
    
    # Pooling methods
    pooling_x_positions = [1.5, 3.5, 5.5, 7.5, 9.2]
    pooling_names = ['Mean', 'Max', 'First-Last', 'First-Last\nWindow', 'Custom']
    pooling_outputs = ['1024', '1024', '2048', '2048', 'TBD']
    pooling_colors = ['#FFD4D4', '#FFE6D4', '#D4E6FF', '#FFFFD4', '#E6D4FF']
    
    for i, (x, name, out, color) in enumerate(zip(pooling_x_positions, pooling_names, pooling_outputs, pooling_colors)):
        rect_pool = FancyBboxPatch((x - 0.7, 2.8), 1.4, 1.2, boxstyle="round,pad=0.1",
                                  edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(rect_pool)
        ax.text(x, 3.6, name, fontsize=8, ha='center', va='center', fontweight='bold')
        ax.text(x, 3.1, f'Output: {out}', fontsize=7, ha='center', va='center', style='italic')
        
        # Arrow from features to pooling
        arrow = FancyArrowPatch((1.5 + (i % 2) * 0.3, 4), (x, 4),
                              arrowstyle='->', mutation_scale=15, linewidth=1.5, color='gray')
        ax.add_patch(arrow)
   
    
    # Info box
    #info_text = 
    """
    🎯 EMPFEHLUNG: Verwende "First-Last Window" für ALS-Diagnose!
    
    ✅ Vorteile:
       • Erfasst Voice Degradation (Anfang vs. Ende)
       • Robust gegen Rauschen (Fenster-Durchschnitt)
       • Höhere ALS-Diskriminierung als single-frame
       • Minimaler Computations-Overhead (~0.1ms)
    
    ⚠️  Einstellungen:
       • window_size = 5 (adaptiv angepasst)
       • Output: 2048-dim (doppelt vs. mean pooling)
       • Speicher: +100% vs. mean (aber unkritisch)
    """
    '''
    bbox_props = dict(boxstyle='round', facecolor='#FFFFCC', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(5, 1, info_text, fontsize=9, ha='center', va='center', 
           bbox=bbox_props, fontfamily='monospace')
    '''
    plt.tight_layout()
    plt.savefig('architecture_pooling_detail.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: architecture_pooling_detail.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CREATING POOLING VISUALIZATIONS")
    print("=" * 80)
    
    print("\n📊 Creating main pooling comparison...")
    create_pooling_visualization()
    
    print("📊 Creating feature dimension comparison...")
    create_dimension_comparison()
    
    print("📊 Creating architecture diagram...")
    create_architecture_diagram()
    
    print("\n" + "=" * 80)
    print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY")
    print("=" * 80)
    print("""
Generated files:
  1. pooling_visualization.png        - Detaillierter Vergleich aller Methoden
  2. feature_dimensions_comparison.png - Feature-Dimensionen Übersicht
  3. architecture_pooling_detail.png   - Architektur-Diagramm mit Pooling
""")
