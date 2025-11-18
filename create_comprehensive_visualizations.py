#!/usr/bin/env python3
"""
Comprehensive Visualization Suite for Late Fusion Audio Classification Pipeline
Erstellt professionelle Visualisierungen für Präsentationen und Dokumentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. MAIN PIPELINE ARCHITECTURE
# ============================================================================

def create_pipeline_architecture():
    """Complete pipeline architecture with all components."""
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.97, 'Late Fusion Audio Classification Pipeline', 
             fontsize=18, fontweight='bold', ha='center')
    fig.text(0.5, 0.94, 'Speaker Identification with Multi-File Audio Input', 
             fontsize=12, ha='center', style='italic', color='gray')
    
    # ========== STAGE 1: INPUT ==========
    y_pos = 10.5
    
    # 8 Audio Files
    files = ['phonation_A', 'phonation_E', 'phonation_I', 'phonation_O', 'phonation_U', 'rhythm_KA', 'rhythm_PA', 'rhythm_TA']
    
    for i, fname in enumerate(files):
        x_pos = 0.5 + i * 1.15
        rect = FancyBboxPatch((x_pos - 0.4, y_pos - 0.3), 0.8, 0.6,
                             boxstyle="round,pad=0.05", 
                             edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos, y_pos, fname.replace('_', '\n'), fontsize=7, ha='center', va='center', fontweight='bold')
    
    ax.text(-0.3, y_pos, 'INPUT (8 files/person)', fontsize=10, fontweight='bold', va='center')
    
    # Arrow down
    arrow1 = FancyArrowPatch((5, y_pos - 0.5), (5, y_pos - 1.2),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow1)
    
    # ========== STAGE 2: FEATURE EXTRACTION ==========
    y_pos = 8.8
    
    # Three extractors in parallel
    extractors = [
        {'name': 'Wav2Vec2', 'color': '#B3E5FC', 'abbr': 'W2V'},
        {'name': 'HuBERT', 'color': '#C8E6C9', 'abbr': 'HBT'},
        {'name': 'WavLM', 'color': '#F8BBD0', 'abbr': 'WLM'}
    ]
    
    x_positions = [1.5, 5, 8.5]
    for i, (x, ext) in enumerate(zip(x_positions, extractors)):
        # Main box
        rect = FancyBboxPatch((x - 0.8, y_pos - 0.5), 1.6, 1,
                             boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor=ext['color'], linewidth=2.5)
        ax.add_patch(rect)
        
        ax.text(x, y_pos + 0.25, ext['name'], fontsize=11, ha='center', fontweight='bold')
        ax.text(x, y_pos - 0.15, 'Pre-trained\nTransformer', fontsize=8, ha='center', style='italic')
        
        # Arrow from input files to extractor
        arrow = FancyArrowPatch((5, 10.2), (x, y_pos + 0.5),
                              arrowstyle='->', mutation_scale=20, linewidth=2, 
                              color='gray', alpha=0.6, linestyle='--')
        ax.add_patch(arrow)
    
    ax.text(-0.3, y_pos, 'FEATURE\nEXTRACTION', fontsize=10, fontweight='bold', va='center')
    
    # Arrow down
    arrow = FancyArrowPatch((5, y_pos - 0.6), (5, y_pos - 1.3),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow)
    
    # ========== STAGE 3: POOLING STRATEGIES ==========
    y_pos = 6.8
    
    poolings = [
        {'name': 'Mean', 'dim': '1024', 'color': '#FFCCBC'},
        {'name': 'First-Last', 'dim': '2048', 'color': '#C5E1A5'},
        {'name': 'First-Last\nWindow', 'dim': '2048', 'color': '#FFF59D', 'best': True}
    ]
    
    x_start = 1.2
    for i, pool in enumerate(poolings):
        x = x_start + i * 2.6
        
        rect = FancyBboxPatch((x - 0.7, y_pos - 0.4), 1.4, 0.8,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor=pool['color'], 
                             linewidth=2.5 if pool.get('best') else 1.5)
        ax.add_patch(rect)
        
        ax.text(x, y_pos + 0.15, pool['name'], fontsize=9, ha='center', fontweight='bold')
        ax.text(x, y_pos - 0.15, f"→ {pool['dim']}-dim", fontsize=8, ha='center', style='italic')
        
        if pool.get('best'):
            # Star annotation
            ax.text(x + 0.85, y_pos + 0.35, '⭐', fontsize=16, ha='center')
    
    ax.text(-0.3, y_pos, 'POOLING\nSTRATEGY', fontsize=10, fontweight='bold', va='center')
    
    # Arrow down
    arrow = FancyArrowPatch((5, y_pos - 0.5), (5, y_pos - 1.2),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow)
    
    # ========== STAGE 4: PER-FILE PROCESSING ==========
    y_pos = 4.8
    
    # Show 8 file processors in a grid
    ax.text(-0.3, y_pos, 'PER-FILE\nPROCESSOR\n(×8)', fontsize=10, fontweight='bold', va='center')
    
    for i in range(8):
        x = 0.5 + i * 1.15
        
        rect = FancyBboxPatch((x - 0.35, y_pos - 0.3), 0.7, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='#9C27B0', facecolor='#E1BEE7', linewidth=2)
        ax.add_patch(rect)
        
        ax.text(x, y_pos, 'FC\n128', fontsize=7, ha='center', va='center', fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((5, y_pos - 0.4), (5, y_pos - 1.1),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow)
    
    # ========== STAGE 5: LATE FUSION ==========
    y_pos = 2.8
    
    ax.text(-0.3, y_pos, 'LATE\nFUSION', fontsize=10, fontweight='bold', va='center')
    
    # Concatenation
    rect = FancyBboxPatch((2.5, y_pos - 0.4), 1.5, 0.8,
                         boxstyle="round,pad=0.1",
                         edgecolor='#00796B', facecolor='#B2DFDB', linewidth=2.5)
    ax.add_patch(rect)
    
    ax.text(3.25, y_pos + 0.15, 'Concatenate', fontsize=10, ha='center', fontweight='bold')
    ax.text(3.25, y_pos - 0.15, '8×128 → 1024', fontsize=8, ha='center', style='italic')
    
    # Arrow
    arrow = FancyArrowPatch((3.25, y_pos - 0.5), (3.25, y_pos - 1.1),
                           arrowstyle='->', mutation_scale=30, linewidth=3, color='black')
    ax.add_patch(arrow)
    
    # ========== STAGE 6: CLASSIFICATION ==========
    y_pos = 1.0
    
    ax.text(-0.3, y_pos, 'OUTPUT', fontsize=10, fontweight='bold', va='center')
    
    rect = FancyBboxPatch((2.2, y_pos - 0.5), 2.1, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#D32F2F', facecolor='#FFCDD2', linewidth=2.5)
    ax.add_patch(rect)
    
    ax.text(3.25, y_pos + 0.2, 'Final Classifier', fontsize=10, ha='center', fontweight='bold')
    ax.text(3.25, y_pos - 0.25, '5 Classes\n(Speaker IDs)', fontsize=8, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('01_pipeline_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 01_pipeline_architecture.png")
    plt.close()

# ============================================================================
# 2. FEATURE EXTRACTOR COMPARISON
# ============================================================================

def create_feature_extractor_comparison():
    """Compare WavLM, Wav2Vec2, and HuBERT."""
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    models = [
        {
            'name': 'Wav2Vec2',
            'org': 'Meta AI',
            'size': 'Large',
            'layers': 24,
            'hidden': 1024,
            'pretrain': 'Contrastive Learning',
            'strength': ['General audio', 'Fast inference'],
            'weakness': ['Less ALS-specific']
        },
        {
            'name': 'HuBERT',
            'org': 'Meta AI',
            'size': 'Large',
            'layers': 24,
            'hidden': 1024,
            'pretrain': 'Masked Prediction',
            'strength': ['Speech-specific', 'Robust'],
            'weakness': ['Slower training']
        },
        {
            'name': 'WavLM',
            'org': 'Microsoft',
            'size': 'Large',
            'layers': 24,
            'hidden': 1024,
            'pretrain': 'Multi-task Learning',
            'strength': ['Best for speech', 'Clinical potential'],
            'weakness': ['Newer model']
        }
    ]
    
    colors = ['#B3E5FC', '#C8E6C9', '#F8BBD0']
    
    for ax, model, color in zip(axes, models, colors):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Title
        ax.text(5, 9.5, model['name'], fontsize=14, ha='center', fontweight='bold')
        ax.text(5, 9, f"by {model['org']}", fontsize=10, ha='center', style='italic', color='gray')
        
        # Main box
        rect = FancyBboxPatch((0.5, 4), 9, 4.5,
                             boxstyle="round,pad=0.2",
                             edgecolor='black', facecolor=color, linewidth=2, alpha=0.3)
        ax.add_patch(rect)
        
        # Details
        y = 8.2
        details = [
            f"Size: {model['size']}",
            f"Layers: {model['layers']}",
            f"Hidden Dim: {model['hidden']}",
            f"Pretraining: {model['pretrain']}"
        ]
        
        for detail in details:
            ax.text(1, y, detail, fontsize=10, fontweight='bold')
            y -= 0.6
        
        # Strengths
        ax.text(1, y - 0.4, "✓ Strengths:", fontsize=10, fontweight='bold', color='green')
        y -= 0.8
        for s in model['strength']:
            ax.text(1.3, y, f"• {s}", fontsize=9)
            y -= 0.5
        
        # Weaknesses
        ax.text(1, y - 0.3, "⚠ Considerations:", fontsize=10, fontweight='bold', color='orange')
        y -= 0.8
        for w in model['weakness']:
            ax.text(1.3, y, f"• {w}", fontsize=9)
            y -= 0.5
    
    fig.suptitle('Pre-trained Audio Feature Extractor Models', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('02_feature_extractor_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 02_feature_extractor_comparison.png")
    plt.close()

# ============================================================================
# 3. MULTI-LAYER FEATURE EXTRACTION
# ============================================================================

def create_multi_layer_extraction():
    """Visualize multi-layer feature extraction."""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    fig.text(0.5, 0.97, 'Multi-Layer Feature Extraction from WavLM', 
             fontsize=16, fontweight='bold', ha='center')
    
    # Input audio
    y = 10.5
    rect = FancyBboxPatch((3.5, y - 0.4), 3, 0.8,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, y, 'Audio Input\n16kHz, ~3 seconds', fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((5, y - 0.5), (5, y - 1.2),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='black')
    ax.add_patch(arrow)
    
    # WavLM architecture - 24 layers
    y = 9
    ax.text(0.3, y, 'WavLM\n24 Layers', fontsize=10, fontweight='bold', va='center')
    
    # Draw layers
    layer_height = 0.35
    layer_positions = []
    
    for i in range(24):
        y_layer = y - i * layer_height
        
        # Highlight selected layers
        if (i + 1) in [6, 9, 12, 15, 18]:
            color = '#C5E1A5'
            linewidth = 2.5
            label = f'Layer {i+1} ✓'
        else:
            color = '#E0E0E0'
            linewidth = 1
            label = f'Layer {i+1}'
        
        rect = FancyBboxPatch((1.5, y_layer - 0.15), 1.5, 0.3,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor=color, linewidth=linewidth)
        ax.add_patch(rect)
        
        ax.text(2.25, y_layer, label, fontsize=7, ha='center', va='center', fontweight='bold')
        
        # Store positions of selected layers
        if (i + 1) in [6, 9, 12, 15, 18]:
            layer_positions.append((2.25, y_layer, i+1))
    
    # Annotations for layer fusion
    ax.text(4.5, 9, 'Selected\nLayers:', fontsize=9, fontweight='bold', ha='center')
    ax.text(4.5, 8.5, '[6, 9, 12, 15, 18]', fontsize=9, ha='center', 
           bbox=dict(boxstyle='round', facecolor='#C5E1A5', alpha=0.7))
    
    # Arrow to layer fusion
    arrow = FancyArrowPatch((3.2, 6), (4.5, 5.8),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#FF9800')
    ax.add_patch(arrow)
    
    # Layer fusion box
    rect = FancyBboxPatch((4, 4.5), 2.5, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FF9800', facecolor='#FFE0B2', linewidth=2.5)
    ax.add_patch(rect)
    
    ax.text(5.25, 5.3, 'Layer Fusion', fontsize=11, ha='center', fontweight='bold')
    ax.text(5.25, 4.9, 'Strategies:', fontsize=9, ha='center', style='italic')
    
    # Fusion strategies
    strategies = ['Mean', 'Concat', 'Weighted']
    x_pos = [2, 5.25, 8.5]
    
    for x, strat in zip(x_pos, strategies):
        rect = FancyBboxPatch((x - 0.6, 3.2), 1.2, 0.6,
                             boxstyle="round,pad=0.05",
                             edgecolor='black', facecolor='#E8D5F5', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 3.5, strat, fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Arrow from layer fusion
        arrow = FancyArrowPatch((5.25, 4.5), (x, 3.85),
                               arrowstyle='->', mutation_scale=15, linewidth=1.5, color='gray')
        ax.add_patch(arrow)
    
    # Output
    y = 2
    rect = FancyBboxPatch((3.5, y - 0.4), 3, 0.8,
                         boxstyle="round,pad=0.1",
                         edgecolor='#4CAF50', facecolor='#C8E6C9', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, y, 'Fused Features\n1024-dim (or more)', fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Legend
    legend_text = """
    Why Multi-Layer Extraction?
    • Layer 6: Low-level acoustic features
    • Layer 9-12: Mid-level phonetic features  
    • Layer 15-18: High-level linguistic features
    • Fusion: Combine multiple perspectives for richer representation
    """
    
    ax.text(5, 0.5, legend_text, fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('03_multi_layer_extraction.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 03_multi_layer_extraction.png")
    plt.close()

# ============================================================================
# 4. POOLING STRATEGIES DETAILED
# ============================================================================

def create_pooling_strategies_detailed():
    """Detailed visualization of all pooling strategies."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create a 2x3 grid
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Simulate voice features over time
    time_steps = np.linspace(0, 3, 100)
    
    # Normal person
    normal_energy = 0.9 + 0.05 * np.random.randn(100)
    
    # ALS patient
    als_energy = 1.0 - 0.4 * (time_steps / 3) + 0.08 * np.random.randn(100)
    
    # Pool methods
    pools = [
        {'name': 'Mean', 'func': lambda x: np.mean(x)},
        {'name': 'Max', 'func': lambda x: np.max(x)},
        {'name': 'First', 'func': lambda x: x[0]},
        {'name': 'Last', 'func': lambda x: x[-1]},
        {'name': 'First-Last', 'func': lambda x: (x[0] + x[-1]) / 2},
        {'name': 'First-Last\nWindow', 'func': lambda x: (np.mean(x[:5]) + np.mean(x[-5:])) / 2},
    ]
    
    for idx, pool_info in enumerate(pools):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        # Plot features
        ax.plot(time_steps, normal_energy, 'g-', linewidth=2, label='Normal', alpha=0.7)
        ax.plot(time_steps, als_energy, 'r-', linewidth=2, label='ALS', alpha=0.7)
        ax.fill_between(time_steps, normal_energy, alpha=0.2, color='green')
        ax.fill_between(time_steps, als_energy, alpha=0.2, color='red')
        
        # Highlight pooling regions
        if pool_info['name'] == 'Mean':
            ax.fill_between(time_steps, 0, 1.5, alpha=0.1, color='blue', label='Pooling region')
        elif pool_info['name'] == 'Max':
            ax.fill_between(time_steps, 0, 1.5, alpha=0.1, color='orange', label='Pooling region')
        elif pool_info['name'] == 'First':
            ax.axvline(x=time_steps[0], color='blue', linestyle='--', linewidth=2.5, label='Selected frame')
        elif pool_info['name'] == 'Last':
            ax.axvline(x=time_steps[-1], color='red', linestyle='--', linewidth=2.5, label='Selected frame')
        elif pool_info['name'] == 'First-Last':
            ax.axvline(x=time_steps[0], color='blue', linestyle='--', linewidth=2.5, label='F_first')
            ax.axvline(x=time_steps[-1], color='red', linestyle='--', linewidth=2.5, label='F_last')
        else:  # Window
            ax.axvspan(time_steps[0], time_steps[4], alpha=0.2, color='blue', label='Window 1')
            ax.axvspan(time_steps[-5], time_steps[-1], alpha=0.2, color='red', label='Window 2')
        
        ax.set_title(pool_info['name'], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Energy', fontsize=10)
        ax.set_ylim([0.4, 1.4])
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    
    fig.suptitle('Pooling Strategies: How Each Captures Voice Features', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('04_pooling_strategies_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 04_pooling_strategies_detailed.png")
    plt.close()

# ============================================================================
# 5. LATE FUSION vs EARLY FUSION
# ============================================================================

def create_fusion_comparison():
    """Compare late fusion vs early fusion."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, title, fusion_type in zip([ax1, ax2], 
                                      ['Early Fusion', 'Late Fusion (Ours)'],
                                      ['early', 'late']):
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        ax.text(5, 11.5, title, fontsize=14, fontweight='bold', ha='center')
        
        if fusion_type == 'early':
            # Early Fusion: Concatenate raw features first
            
            # Input
            y = 10.5
            for i in range(8):
                x = 0.5 + i * 1.15
                rect = FancyBboxPatch((x - 0.35, y - 0.25), 0.7, 0.5,
                                     boxstyle="round,pad=0.05",
                                     edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(x, y, f'F{i}', fontsize=8, ha='center', va='center', fontweight='bold')
            
            # Arrow
            arrow = FancyArrowPatch((5, y - 0.3), (5, y - 1.0),
                                  arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
            ax.add_patch(arrow)
            
            # Early concatenation
            y = 8.8
            rect = FancyBboxPatch((2, y - 0.4), 6, 0.8,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#FF9800', facecolor='#FFE0B2', linewidth=2)
            ax.add_patch(rect)
            ax.text(5, y, 'Concatenate All Features\n8×1024 = 8192-dim', 
                   fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Arrow
            arrow = FancyArrowPatch((5, y - 0.5), (5, y - 1.1),
                                  arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
            ax.add_patch(arrow)
            
            # Single classifier
            y = 7.2
            rect = FancyBboxPatch((2.5, y - 0.4), 5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#D32F2F', facecolor='#FFCDD2', linewidth=2)
            ax.add_patch(rect)
            ax.text(5, y, 'Single Large Classifier\n8192 → 5 classes', 
                   fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Disadvantages
            y = 5.5
            disadv = [
                "❌ Very high dimensionality (8192)",
                "❌ Difficult to learn fine stimulus differences",
                "❌ Cross-stimulus interference",
                "❌ Less interpretable (black box)",
                "❌ Slower training & inference"
            ]
            
            for i, d in enumerate(disadv):
                ax.text(0.5, y - i*0.6, d, fontsize=9, fontweight='bold', color='darkred')
        
        else:  # Late Fusion
            
            # Input
            y = 10.5
            for i in range(8):
                x = 0.5 + i * 1.15
                rect = FancyBboxPatch((x - 0.35, y - 0.25), 0.7, 0.5,
                                     boxstyle="round,pad=0.05",
                                     edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(x, y, f'F{i}', fontsize=8, ha='center', va='center', fontweight='bold')
            
            # Arrow
            arrow = FancyArrowPatch((5, y - 0.3), (5, y - 1.0),
                                  arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
            ax.add_patch(arrow)
            
            # Individual processing
            y = 8.8
            ax.text(0.2, y, '× 8', fontsize=10, fontweight='bold', va='center')
            
            for i in range(8):
                x = 0.5 + i * 1.15
                rect = FancyBboxPatch((x - 0.35, y - 0.25), 0.7, 0.5,
                                     boxstyle="round,pad=0.05",
                                     edgecolor='#9C27B0', facecolor='#E1BEE7', linewidth=1.5)
                ax.add_patch(rect)
                ax.text(x, y, '128', fontsize=8, ha='center', va='center', fontweight='bold')
            
            # Arrow
            arrow = FancyArrowPatch((5, y - 0.3), (5, y - 1.0),
                                  arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
            ax.add_patch(arrow)
            
            # Late concatenation
            y = 7.2
            rect = FancyBboxPatch((2, y - 0.4), 6, 0.8,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#00796B', facecolor='#B2DFDB', linewidth=2)
            ax.add_patch(rect)
            ax.text(5, y, 'Concatenate Processed Features\n8×128 = 1024-dim', 
                   fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Arrow
            arrow = FancyArrowPatch((5, y - 0.5), (5, y - 1.1),
                                  arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
            ax.add_patch(arrow)
            
            # Final classifier
            y = 5.5
            rect = FancyBboxPatch((2.5, y - 0.4), 5, 0.8,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='#4CAF50', facecolor='#C8E6C9', linewidth=2)
            ax.add_patch(rect)
            ax.text(5, y, 'Final Classifier\n1024 → 5 classes', 
                   fontsize=10, ha='center', va='center', fontweight='bold')
            
            # Advantages
            y = 3.8
            adv = [
                "✅ Manageable dimensionality (1024)",
                "✅ Each stimulus processed independently",
                "✅ Learn stimulus-specific patterns",
                "✅ More interpretable (per-file features)",
                "✅ Faster & more efficient training"
            ]
            
            for i, a in enumerate(adv):
                ax.text(0.5, y - i*0.6, a, fontsize=9, fontweight='bold', color='darkgreen')
    
    fig.suptitle('Fusion Strategy Comparison: Early vs Late', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('05_fusion_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 05_fusion_comparison.png")
    plt.close()

# ============================================================================
# 6. AUDIO AUGMENTATION TECHNIQUES
# ============================================================================

def create_audio_augmentation():
    """Visualize audio augmentation techniques."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Create sample audio signal
    t = np.linspace(0, 1, 500)
    original = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 10 * t)
    
    # 1. Time Stretching
    ax = axes[0]
    stretched = np.sin(2 * np.pi * 5 * (t * 0.95)) + 0.3 * np.sin(2 * np.pi * 10 * (t * 0.95))
    
    ax.plot(t, original, 'b-', linewidth=2, label='Original')
    ax.plot(t, stretched, 'r--', linewidth=2, label='Stretched (0.95×)')
    ax.set_title('1. Time Stretching\nRange: 0.9× to 1.1×', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Pitch Shifting
    ax = axes[1]
    pitched = original * np.sin(2 * np.pi * 0.5 * t)  # Simplified pitch shift
    
    ax.plot(t, original, 'b-', linewidth=2, label='Original')
    ax.plot(t, pitched, 'g--', linewidth=2, label='Pitch shifted (±2 semitones)')
    ax.set_title('2. Pitch Shifting\nRange: ±2 semitones', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Noise Addition
    ax = axes[2]
    noise = np.random.normal(0, 0.15, len(original))
    noisy = original + noise
    
    ax.plot(t, original, 'b-', linewidth=2, label='Original', alpha=0.7)
    ax.plot(t, noisy, 'orange', linewidth=1.5, label='With noise (SNR 25-40 dB)', alpha=0.7)
    ax.fill_between(t, original, noisy, alpha=0.2, color='orange')
    ax.set_title('3. Noise Addition\nSNR: 25-40 dB', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Time Masking
    ax = axes[3]
    masked = original.copy()
    mask_start = int(len(masked) * 0.3)
    mask_end = int(len(masked) * 0.5)
    masked[mask_start:mask_end] = 0
    
    ax.plot(t, original, 'b-', linewidth=2, label='Original')
    ax.plot(t, masked, 'm--', linewidth=2, label='Time masked region')
    ax.axvspan(t[mask_start], t[mask_end], alpha=0.2, color='red', label='Masked (15% max)')
    ax.set_title('4. Time Masking\nMax 15% of duration', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.suptitle('Audio Data Augmentation Techniques for Training Robustness', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('06_audio_augmentation.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 06_audio_augmentation.png")
    plt.close()

# ============================================================================
# 7. TRAINING FLOW & OPTIMIZATION
# ============================================================================

def create_training_flow():
    """Visualize the complete training flow."""
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    fig.text(0.5, 0.97, 'Training Pipeline with Advanced Optimizations', 
             fontsize=16, fontweight='bold', ha='center')
    
    # Step 1: Data Loading
    y = 11
    rect = FancyBboxPatch((1, y - 0.5), 4, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, y, 'Step 1: Data Loading\n+ Augmentation (80%)', 
           fontsize=10, ha='center', va='center', fontweight='bold')
    
    arrow = FancyArrowPatch((3, y - 0.6), (3, y - 1.3),
                           arrowstyle='->', mutation_scale=25, linewidth=2)
    ax.add_patch(arrow)
    
    # Step 2: Batch Creation with Gradient Accumulation
    y = 9
    rect = FancyBboxPatch((1, y - 0.6), 4, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#FFA726', facecolor='#FFE0B2', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, y + 0.15, 'Step 2: Gradient Accumulation', fontsize=10, ha='center', fontweight='bold')
    ax.text(3, y - 0.35, 'Physical: 12 | Accumulation: 2\nEffective: 24', 
           fontsize=9, ha='center', va='center', style='italic')
    
    arrow = FancyArrowPatch((3, y - 0.7), (3, y - 1.4),
                           arrowstyle='->', mutation_scale=25, linewidth=2)
    ax.add_patch(arrow)
    
    # Step 3: Forward & Loss
    y = 7
    rect = FancyBboxPatch((1, y - 0.6), 4, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#66BB6A', facecolor='#C8E6C9', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, y + 0.15, 'Step 3: Forward Pass & Loss', fontsize=10, ha='center', fontweight='bold')
    ax.text(3, y - 0.35, 'Class-Balanced CrossEntropy\n+ Fairness Loss (optional)', 
           fontsize=9, ha='center', va='center', style='italic')
    
    arrow = FancyArrowPatch((3, y - 0.7), (3, y - 1.4),
                           arrowstyle='->', mutation_scale=25, linewidth=2)
    ax.add_patch(arrow)
    
    # Step 4: Backprop
    y = 5
    rect = FancyBboxPatch((1, y - 0.5), 4, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='#42A5F5', facecolor='#BBDEFB', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, y, 'Step 4: Backpropagation\n+ Gradient Clipping', 
           fontsize=10, ha='center', va='center', fontweight='bold')
    
    arrow = FancyArrowPatch((3, y - 0.6), (3, y - 1.3),
                           arrowstyle='->', mutation_scale=25, linewidth=2)
    ax.add_patch(arrow)
    
    # Step 5: Validation & Early Stopping
    y = 3
    rect = FancyBboxPatch((1, y - 0.6), 4, 1.2,
                         boxstyle="round,pad=0.1",
                         edgecolor='#AB47BC', facecolor='#E1BEE7', linewidth=2)
    ax.add_patch(rect)
    ax.text(3, y + 0.15, 'Step 5: Validation & Metrics', fontsize=10, ha='center', fontweight='bold')
    ax.text(3, y - 0.35, 'F1-Score, Accuracy, Loss\nEarly Stopping (patience=5)', 
           fontsize=9, ha='center', va='center', style='italic')
    
    # Right side: Optimizations
    x = 6.5
    
    # Optimization 1
    rect = FancyBboxPatch((x - 0.3, 10.2), 3, 0.8,
                         boxstyle="round,pad=0.05",
                         edgecolor='#FF6B6B', facecolor='#FFE0E0', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 1.2, 10.6, '#1: Batch Sizing\n& Accumulation', fontsize=9, ha='center', fontweight='bold')
    
    # Optimization 2
    rect = FancyBboxPatch((x - 0.3, 9), 3, 0.8,
                         boxstyle="round,pad=0.05",
                         edgecolor='#FFA726', facecolor='#FFE0B2', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 1.2, 9.4, '#2: Audio\nAugmentation', fontsize=9, ha='center', fontweight='bold')
    
    # Optimization 3
    rect = FancyBboxPatch((x - 0.3, 7.8), 3, 0.8,
                         boxstyle="round,pad=0.05",
                         edgecolor='#66BB6A', facecolor='#C8E6C9', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 1.2, 8.2, '#3: Fairness\nRegularization', fontsize=9, ha='center', fontweight='bold')
    
    # Optimization 4
    rect = FancyBboxPatch((x - 0.3, 6.6), 3, 0.8,
                         boxstyle="round,pad=0.05",
                         edgecolor='#42A5F5', facecolor='#BBDEFB', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + 1.2, 7, '#4: Multi-Layer\nExtraction', fontsize=9, ha='center', fontweight='bold')
    
    # Hyperparameters
    rect = FancyBboxPatch((x - 0.3, 4.5), 3, 1.8,
                         boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor='#FFF9C4', linewidth=2)
    ax.add_patch(rect)
    
    hypertext = """Key Hyperparameters:
Learning Rate: 5e-4
Weight Decay: 1e-4
Epochs: 20
Batch Size: 12
Accumulation: 2"""
    
    ax.text(x + 1.2, 5.4, hypertext, fontsize=8, ha='center', va='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('07_training_flow.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 07_training_flow.png")
    plt.close()

# ============================================================================
# 8. ALS VOICE CHARACTERISTICS
# ============================================================================

def create_als_voice_characteristics():
    """Visualize ALS voice degradation patterns."""
    
    fig = plt.figure(figsize=(15, 10))
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Energy degradation
    ax = fig.add_subplot(gs[0, 0])
    t = np.linspace(0, 3, 1000)
    
    normal_energy = 0.9 + 0.05 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(1000) * 0.02
    als_energy = 1.0 - 0.3 * (t / 3) + 0.08 * np.random.randn(1000)
    
    ax.plot(t, normal_energy, 'g-', linewidth=2.5, label='Normal Speaker')
    ax.plot(t, als_energy, 'r-', linewidth=2.5, label='ALS Patient')
    ax.fill_between(t, normal_energy, alpha=0.2, color='green')
    ax.fill_between(t, als_energy, alpha=0.2, color='red')
    
    ax.set_title('1. Voice Energy Over Time', fontsize=12, fontweight='bold')
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Normalized Energy')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 2. Jitter (pitch variation)
    ax = fig.add_subplot(gs[0, 1])
    
    normal_jitter = np.random.randn(30) * 5
    als_jitter = np.random.randn(30) * 25  # Much higher variation
    
    ax.scatter(np.arange(30), normal_jitter, s=50, alpha=0.6, color='green', label='Normal (σ ≈ 5%)')
    ax.scatter(np.arange(30), als_jitter, s=50, alpha=0.6, color='red', label='ALS (σ ≈ 25%)')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('2. Pitch Jitter (Variation)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Phonation Frame')
    ax.set_ylabel('Pitch Deviation (%)')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    ax.set_ylim([-60, 60])
    
    # 3. Tremor (low frequency oscillation)
    ax = fig.add_subplot(gs[1, 0])
    
    t_tremor = np.linspace(0, 3, 1000)
    normal_signal = np.sin(2 * np.pi * 100 * t_tremor) + 0.05 * np.sin(2 * np.pi * 5 * t_tremor)
    als_signal = np.sin(2 * np.pi * 100 * t_tremor) + 0.3 * np.sin(2 * np.pi * 5 * t_tremor)
    
    ax.plot(t_tremor[:500], normal_signal[:500], 'g-', linewidth=1.5, label='Normal (low tremor)')
    ax.plot(t_tremor[:500], als_signal[:500], 'r-', linewidth=1.5, label='ALS (high tremor)')
    
    ax.set_title('3. Voice Tremor (Oscillation)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # 4. Diagnostic potential of pooling
    ax = fig.add_subplot(gs[1, 1])
    
    methods = ['Mean', 'First-Last', 'First-Last\nWindow']
    normal_gaps = [0.03, 0.08, 0.06]
    als_gaps = [0.05, 0.45, 0.48]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, normal_gaps, width, label='Normal', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, als_gaps, width, label='ALS', color='red', alpha=0.7)
    
    ax.set_title('4. First-Last Degradation Gap\n(Discriminative Power)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Gap')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle('ALS Voice Characteristics & Detection Strategy', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('08_als_voice_characteristics.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: 08_als_voice_characteristics.png")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE VISUALIZATION SUITE")
    print("="*80 + "\n")
    
    print("📊 1. Creating pipeline architecture diagram...")
    create_pipeline_architecture()
    
    print("📊 2. Creating feature extractor comparison...")
    create_feature_extractor_comparison()
    
    print("📊 3. Creating multi-layer extraction visualization...")
    create_multi_layer_extraction()
    
    print("📊 4. Creating detailed pooling strategies...")
    create_pooling_strategies_detailed()
    
    print("📊 5. Creating early vs late fusion comparison...")
    create_fusion_comparison()
    
    print("📊 6. Creating audio augmentation techniques...")
    create_audio_augmentation()
    
    print("📊 7. Creating training flow diagram...")
    create_training_flow()
    
    print("📊 8. Creating ALS voice characteristics...")
    create_als_voice_characteristics()
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
    print("="*80)
    
    print("""
Generated Visualizations:
  1. 01_pipeline_architecture.png         - Complete pipeline overview
  2. 02_feature_extractor_comparison.png  - WavLM vs Wav2Vec2 vs HuBERT
  3. 03_multi_layer_extraction.png        - Multi-layer feature extraction
  4. 04_pooling_strategies_detailed.png   - All pooling methods compared
  5. 05_fusion_comparison.png             - Early vs Late Fusion
  6. 06_audio_augmentation.png            - Augmentation techniques
  7. 07_training_flow.png                 - Training pipeline with optimizations
  8. 08_als_voice_characteristics.png     - ALS voice patterns & detection

Ready for presentations, papers, and documentation! 🎉
""")
