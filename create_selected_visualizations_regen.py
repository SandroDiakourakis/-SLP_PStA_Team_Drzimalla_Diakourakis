#!/usr/bin/env python3
"""
Regeneration Script for Selected Visualizations
- Recreates: 01,03,04 (mean & FLW),05,07
- Avoids special glyphs and constrained_layout issues
- Saves PNG and PDF for each image (publication-friendly)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

def add_box(ax, xy, w, h, text, facecolor, edgecolor='black', fontsize=10, linewidth=1.8):
    rect = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle='round,pad=0.05',
                         edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)
    ax.add_patch(rect)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

# 01 Pipeline
def create_pipeline_architecture():
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.suptitle('Late Fusion Audio Classification Pipeline', fontsize=20, fontweight='bold')

    # Input
    y = 8.9
    files = ['A', 'E', 'I', 'O', 'U', 'KA', 'PA', 'TA']
    x_start = 0.6
    gap = 1.4
    box_w, box_h = 0.95, 0.6
    for i, f in enumerate(files):
        x = x_start + i * gap
        add_box(ax, (x, y), box_w, box_h, f, facecolor='#FFECEC', linewidth=1.6)
    ax.text(0.05, y + box_h/2, 'INPUT (8 files)', fontsize=12, fontweight='bold')

    # Feature extractors
    y = 6.8
    extractors = [('Wav2Vec2', '#BEE9FF'), ('HuBERT', '#DFF7DF'), ('WavLM', '#FFDCE6')]
    x_positions = [2.0, 6.8, 11.6]
    for x, (name, color) in zip(x_positions, extractors):
        add_box(ax, (x - 0.9, y - 0.3), 1.8, 0.9, name + '\nPretrained', facecolor=color, linewidth=1.8)
    for x in x_positions:
        arrow = FancyArrowPatch((6.4, 8.2), (x, y + 0.3), arrowstyle='->', mutation_scale=18, linewidth=1.6, color='gray')
        ax.add_patch(arrow)

    # Pooling strategies (only showing Mean and FLW in pipeline but keep concise)
    y = 4.9
    poolings = [('Mean', '1024', '#FFF2CC'), ('First-Last-Window', '2048', '#E8FFD6')]
    px = [4.0, 8.8]
    for x, (name, dim, color) in zip(px, poolings):
        add_box(ax, (x - 1.0, y - 0.25), 2.0, 0.75, f"{name}\n→ {dim}-dim", facecolor=color, linewidth=1.8)
    ax.text(0.05, y + 0.25, 'POOLING STRATEGY', fontsize=12, fontweight='bold')

    # Per-file processors
    y = 3.3
    for i in range(8):
        x = x_start + i * gap
        add_box(ax, (x, y - 0.25), 0.95, 0.6, 'FC\n128', facecolor='#F3E8FF', linewidth=1.4)

    # Fusion and classifier
    add_box(ax, (5.6, 1.8), 3.6, 0.9, 'Concatenate\n8×128 → 1024', facecolor='#DFF3EF', linewidth=1.9)
    add_box(ax, (5.6, 0.4), 3.6, 1.0, 'Final Classifier\n5 classes', facecolor='#FFECEC', linewidth=1.9)

    for ys, ye in [(8.1, 7.0), (6.0, 5.4), (4.6, 4.0), (3.0, 2.6)]:
        arrow = FancyArrowPatch((6.4, ys), (6.4, ye), arrowstyle='->', mutation_scale=18, linewidth=1.4, color='black')
        ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig('01_pipeline_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('01_pipeline_architecture.pdf', dpi=300, bbox_inches='tight')
    print('✓ Saved: 01_pipeline_architecture.png and .pdf')
    plt.close()

# 03 Multi-layer extraction (remove special checkmarks)
def create_multi_layer_extraction():
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.suptitle('Multi-Layer Feature Extraction from WavLM', fontsize=18, fontweight='bold')

    top_y = 8.8
    layer_h = 0.32
    left_x = 1.4
    selected = [6, 9, 12, 15, 18]
    for i in range(24):
        y = top_y - i * (layer_h + 0.03)
        if (i + 1) in selected:
            face = '#C6F6D5'
            lw = 1.8
            label = f'Layer {i+1} (selected)'
        else:
            face = '#EFEFEF'
            lw = 0.9
            label = f'Layer {i+1}'
        add_box(ax, (left_x, y - layer_h/2), 2.6, layer_h, label, facecolor=face, linewidth=lw)

    # Fusion box and strategies
    add_box(ax, (5.8, 4.4), 4.2, 1.0, 'Layer Fusion', facecolor='#FFF3D9', linewidth=1.9)
    strategies = ['Mean', 'Concat', 'Weighted']
    sx = [6.6, 8.6, 10.6]
    for x, s in zip(sx, strategies):
        add_box(ax, (x - 0.9, 2.8), 1.8, 0.7, s, facecolor='#EDE7F6', linewidth=1.2)

    add_box(ax, (5.8, 1.2), 4.2, 0.9, 'Fused Features\n1024-dim', facecolor='#E8F8F5', linewidth=1.9)

    legend_text = (
        "Layer roles:\n"
        "- Layer 6: Low-level acoustic\n"
        "- Layer 9-12: Mid-level phonetic\n"
        "- Layer 15-18: High-level linguistic\n"
        "Fusion: combine multiple perspectives"
    )
    ax.text(10.2, 0.6, legend_text, fontsize=10, ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='#FFF7E6', alpha=0.95))

    plt.tight_layout()
    plt.savefig('03_multi_layer_extraction.png', dpi=300, bbox_inches='tight')
    plt.savefig('03_multi_layer_extraction.pdf', dpi=300, bbox_inches='tight')
    print('✓ Saved: 03_multi_layer_extraction.png and .pdf')
    plt.close()

# 04 Pooling strategies (Mean and First-Last-Window)
def create_pooling_strategies_two():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    time = np.linspace(0, 3, 400)
    rng = np.random.RandomState(42)
    normal = 0.9 + 0.03 * np.sin(2 * np.pi * 0.5 * time)
    als = 1.0 - 0.35 * (time / 3) + 0.05 * rng.normal(size=len(time))

    # Mean
    ax = axes[0]
    ax.plot(time, normal, color='#2E8B57', label='Normal', linewidth=2)
    ax.plot(time, als, color='#C62828', label='ALS', linewidth=2)
    ax.fill_between(time, normal, alpha=0.12, color='#2E8B57')
    ax.fill_between(time, als, alpha=0.10, color='#C62828')
    mean_norm = normal.mean()
    mean_als = als.mean()
    ax.hlines([mean_norm, mean_als], xmin=0, xmax=3, colors=['#2E8B57', '#C62828'], linestyles='--')
    ax.set_title('Mean Pooling', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.legend()
    ax.grid(alpha=0.2)

    # First-Last-Window
    ax = axes[1]
    ax.plot(time, normal, color='#2E8B57', label='Normal', linewidth=2)
    ax.plot(time, als, color='#C62828', label='ALS', linewidth=2)
    w = 20
    win1_norm = normal[:w].mean(); win2_norm = normal[-w:].mean()
    win1_als = als[:w].mean(); win2_als = als[-w:].mean()
    ax.axvspan(time[0], time[w-1], alpha=0.18, color='#4FC3F7')
    ax.axvspan(time[-w], time[-1], alpha=0.18, color='#FFB74D')
    ax.text(0.02, 0.95, f'Win1_norm={win1_norm:.2f}\nWin2_norm={win2_norm:.2f}', transform=ax.transAxes, fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.85))
    ax.text(0.60, 0.95, f'Win1_als={win1_als:.2f}\nWin2_als={win2_als:.2f}', transform=ax.transAxes, fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.85))
    ax.set_title('First-Last-Window Pooling', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(alpha=0.2)

    fig.suptitle('Selected Pooling Strategies (Mean vs First-Last-Window)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('04_pooling_strategies_detailed.png', dpi=300, bbox_inches='tight')
    plt.savefig('04_pooling_strategies_detailed.pdf', dpi=300, bbox_inches='tight')
    print('✓ Saved: 04_pooling_strategies_detailed.png and .pdf')
    plt.close()

# 05 Fusion comparison
def create_fusion_comparison():
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(121)
    ax1.axis('off')
    ax1.set_title('Early Fusion', fontsize=14, fontweight='bold')
    y = 4.6
    for i in range(8):
        x = 0.4 + i * 0.9
        add_box(ax1, (x, y), 0.75, 0.5, f'F{i}', facecolor='#FFECEC', linewidth=1.2)
    add_box(ax1, (1.6, 3.0), 6.0, 0.9, 'Concatenate All\n8×1024 = 8192-dim', facecolor='#FFF3D9', linewidth=1.6)
    add_box(ax1, (2.2, 1.6), 5.0, 0.9, 'Single Large Classifier\n8192 → 5', facecolor='#FFEBEE', linewidth=1.6)

    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    ax2.set_title('Late Fusion (Ours)', fontsize=14, fontweight='bold')
    y = 4.6
    for i in range(8):
        x = 0.4 + i * 0.9
        add_box(ax2, (x, y), 0.75, 0.5, '128', facecolor='#F3E8FF', linewidth=1.2)
    add_box(ax2, (1.6, 3.0), 6.0, 0.9, 'Concatenate Processed\n8×128 = 1024-dim', facecolor='#E8F8F5', linewidth=1.6)
    add_box(ax2, (2.2, 1.6), 5.0, 0.9, 'Final Classifier\n1024 → 5', facecolor='#E8F5FF', linewidth=1.6)

    plt.tight_layout()
    plt.savefig('05_fusion_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('05_fusion_comparison.pdf', dpi=300, bbox_inches='tight')
    print('✓ Saved: 05_fusion_comparison.png and .pdf')
    plt.close()

# 07 Training flow
def create_training_flow():
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.suptitle('Training Pipeline with Optimizations', fontsize=16, fontweight='bold')

    steps = [
        ('Data Loading\n+ Augmentation', '#FFECEC'),
        ('Gradient Accumulation\nPhysical:12 Accum:2 → Eff:24', '#FFF3D9'),
        ('Forward Pass & Loss\nClass-balanced CE', '#E8F8F5'),
        ('Backprop & Clip\nOptimizer: AdamW', '#E8E9FF'),
        ('Validation & Early Stopping\nMetrics: F1, Acc', '#F3E8FF')
    ]

    y = 8.8
    for title, color in steps:
        add_box(ax, (1.0, y - 0.4), 12.0, 0.9, title, facecolor=color, linewidth=1.6)
        arrow = FancyArrowPatch((6.0, y - 0.4), (6.0, y - 1.0), arrowstyle='->', mutation_scale=18, linewidth=1.4)
        ax.add_patch(arrow)
        y -= 1.5

    add_box(ax, (13.2, 6.2), 2.6, 2.4, 'Hyperparams\nLR:5e-4\nWD:1e-4\nEpochs:20\nBatch:12\nAccum:2', facecolor='#FFF7E6', linewidth=1.2)

    plt.tight_layout()
    plt.savefig('07_training_flow.png', dpi=300, bbox_inches='tight')
    plt.savefig('07_training_flow.pdf', dpi=300, bbox_inches='tight')
    print('✓ Saved: 07_training_flow.png and .pdf')
    plt.close()

if __name__ == '__main__':
    print('\nREGENERATING SELECTED VISUALIZATIONS (PNG + PDF)')
    create_pipeline_architecture()
    create_multi_layer_extraction()
    create_pooling_strategies_two()
    create_fusion_comparison()
    create_training_flow()
    print('\nDONE')
