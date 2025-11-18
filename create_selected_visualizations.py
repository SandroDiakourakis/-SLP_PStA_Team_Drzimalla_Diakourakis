#!/usr/bin/env python3
"""
Focused Visualization Script
Generates only the requested, non-overlapping, publication-quality figures:
- 01_pipeline_architecture.png
- 03_multi_layer_extraction.png
- 04_pooling_strategies_detailed.png (only Mean and First-Last-Window)
- 05_fusion_comparison.png
- 07_training_flow.png

This script intentionally duplicates and simplifies parts of the comprehensive script,
but improves layout (constrained_layout and explicit spacing) to avoid overlaps.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.constrained_layout.use': True,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

# Helper to add rounded box with text
def add_box(ax, xy, w, h, text, facecolor, edgecolor='black', fontsize=10, linewidth=1.8):
    rect = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle='round,pad=0.05',
                         edgecolor=edgecolor, facecolor=facecolor, linewidth=linewidth)
    ax.add_patch(rect)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

# 01 Pipeline (improved spacing)
def create_pipeline_architecture():
    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.axis('off')

    fig.suptitle('Late Fusion Audio Classification Pipeline', fontsize=18, fontweight='bold')

    # Input row (8 files)
    y = 8.9
    files = ['A', 'E', 'I', 'O', 'U', 'KA', 'PA', 'TA']
    x_start = 0.6
    gap = 1.4
    box_w, box_h = 0.95, 0.6

    for i, f in enumerate(files):
        x = x_start + i * gap
        add_box(ax, (x, y), box_w, box_h, f, facecolor='#FFECEC', linewidth=1.6)

    ax.text(0.05, y + box_h/2, 'INPUT (8 files)', fontsize=11, fontweight='bold')

    # Feature extractors (three columns) placed lower with clear separation
    y = 6.8
    extractors = [('Wav2Vec2', '#BEE9FF'), ('HuBERT', '#DFF7DF'), ('WavLM', '#FFDCE6')]
    x_positions = [2.0, 6.8, 11.6]
    for x, (name, color) in zip(x_positions, extractors):
        add_box(ax, (x - 0.9, y - 0.3), 1.8, 0.9, name + '\nPretrained', facecolor=color, linewidth=1.8)

    # Arrows from input cluster to extractors (fan-out) - start from center above
    for x in x_positions:
        arrow = FancyArrowPatch((6.4, 8.2), (x, y + 0.3), arrowstyle='->', mutation_scale=18, linewidth=1.6, color='gray')
        ax.add_patch(arrow)

    # Pooling strategies (compact row)
    y = 4.9
    poolings = [('Mean', '1024', '#FFF2CC'), ('First-Last-Window', '2048', '#E8FFD6')]
    px = [4.0, 8.8]
    for x, (name, dim, color) in zip(px, poolings):
        add_box(ax, (x - 1.0, y - 0.25), 2.0, 0.75, f"{name}\n→ {dim}-dim", facecolor=color, linewidth=1.8)

    ax.text(0.05, y + 0.25, 'POOLING STRATEGY', fontsize=11, fontweight='bold')

    # Per-file processor row
    y = 3.3
    for i in range(8):
        x = x_start + i * gap
        add_box(ax, (x, y - 0.25), 0.95, 0.6, 'FC\n128', facecolor='#F3E8FF', linewidth=1.4)

    # Late fusion box and classifier
    add_box(ax, (5.6, 1.8), 3.6, 0.9, 'Concatenate\n8×128 → 1024', facecolor='#DFF3EF', linewidth=1.9)
    add_box(ax, (5.6, 0.4), 3.6, 1.0, 'Final Classifier\n5 classes', facecolor='#FFECEC', linewidth=1.9)

    # Vertical arrows with spacing
    for ys, ye in [(8.1, 7.0), (6.0, 5.4), (4.6, 4.0), (3.0, 2.6)]:
        arrow = FancyArrowPatch((6.4, ys), (6.4, ye), arrowstyle='->', mutation_scale=18, linewidth=1.4, color='black')
        ax.add_patch(arrow)

    plt.savefig('01_pipeline_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('✓ Saved: 01_pipeline_architecture.png')
    plt.close()

# 03 Multi-layer extraction (clear vertical stack with spacing and legend)
def create_multi_layer_extraction():
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.suptitle('Multi-Layer Feature Extraction from WavLM', fontsize=16, fontweight='bold')

    top_y = 8.8
    layer_h = 0.28
    left_x = 1.4

    selected = [6, 9, 12, 15, 18]

    for i in range(24):
        y = top_y - i * (layer_h + 0.02)
        if (i + 1) in selected:
            face = '#C6F6D5'
            lw = 1.8
            label = f'Layer {i+1} ✓'
        else:
            face = '#EFEFEF'
            lw = 0.9
            label = f'Layer {i+1}'
        add_box(ax, (left_x, y - layer_h/2), 2.4, layer_h, label, facecolor=face, linewidth=lw)

    # Fusion box and strategies spaced horizontally
    add_box(ax, (5.6, 4.4), 4.0, 1.0, 'Layer Fusion', facecolor='#FFF3D9', linewidth=1.9)
    strategies = ['Mean', 'Concat', 'Weighted']
    sx = [6.4, 8.4, 10.4]
    for x, s in zip(sx, strategies):
        add_box(ax, (x - 0.9, 2.8), 1.8, 0.7, s, facecolor='#EDE7F6', linewidth=1.2)

    # Output box
    add_box(ax, (5.6, 1.2), 4.0, 0.9, 'Fused Features\n1024-dim', facecolor='#E8F8F5', linewidth=1.9)

    # Legend block
    legend_text = (
        "Layer roles:\n"
        "• Layer 6: Low-level acoustic\n"
        "• Layer 9-12: Mid-level phonetic\n"
        "• Layer 15-18: High-level linguistic\n"
        "Fusion: combine multiple perspectives"
    )
    ax.text(9.2, 0.6, legend_text, fontsize=9, ha='left', va='bottom', bbox=dict(boxstyle='round', facecolor='#FFF7E6', alpha=0.9))

    plt.savefig('03_multi_layer_extraction.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('✓ Saved: 03_multi_layer_extraction.png')
    plt.close()

# 04 pooling strategies (only Mean and First-Last-Window)
def create_pooling_strategies_two():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    time = np.linspace(0, 3, 200)
    normal = 0.9 + 0.03 * np.sin(2 * np.pi * 0.5 * time)
    als = 1.0 - 0.35 * (time / 3) + 0.05 * np.random.randn(len(time))

    # Mean
    ax = axes[0]
    ax.plot(time, normal, color='#2E8B57', label='Normal', linewidth=2)
    ax.plot(time, als, color='#C62828', label='ALS', linewidth=2)
    ax.fill_between(time, normal, alpha=0.15, color='#2E8B57')
    ax.fill_between(time, als, alpha=0.12, color='#C62828')
    mean_norm = normal.mean()
    mean_als = als.mean()
    ax.hlines([mean_norm, mean_als], xmin=0, xmax=3, colors=['#2E8B57', '#C62828'], linestyles='--')
    ax.set_title('Mean Pooling', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.legend()
    ax.grid(alpha=0.25)

    # First-Last-Window
    ax = axes[1]
    ax.plot(time, normal, color='#2E8B57', label='Normal', linewidth=2)
    ax.plot(time, als, color='#C62828', label='ALS', linewidth=2)
    ax.fill_between(time, normal, alpha=0.15, color='#2E8B57')
    ax.fill_between(time, als, alpha=0.12, color='#C62828')
    # windows
    w = 10
    win1_norm = normal[:w].mean(); win2_norm = normal[-w:].mean()
    win1_als = als[:w].mean(); win2_als = als[-w:].mean()
    ax.axvspan(time[0], time[w-1], alpha=0.18, color='#4FC3F7')
    ax.axvspan(time[-w], time[-1], alpha=0.18, color='#FFB74D')
    ax.text(0.25, 0.95, f'Win1_norm={win1_norm:.2f}\nWin2_norm={win2_norm:.2f}', transform=ax.transAxes, fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.7))
    ax.text(0.65, 0.95, f'Win1_als={win1_als:.2f}\nWin2_als={win2_als:.2f}', transform=ax.transAxes, fontsize=9, va='top', bbox=dict(facecolor='white', alpha=0.7))
    ax.set_title('First-Last-Window Pooling', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.legend()
    ax.grid(alpha=0.25)

    fig.suptitle('Selected Pooling Strategies (Baseline vs. FLW)', fontsize=15, fontweight='bold')
    plt.savefig('04_pooling_strategies_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('✓ Saved: 04_pooling_strategies_detailed.png')
    plt.close()

# 05 fusion comparison (early vs late improved spacing)
def create_fusion_comparison():
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    # Early
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.set_title('Early Fusion', fontsize=14, fontweight='bold')
    # inputs
    y = 4.6
    for i in range(8):
        x = 0.4 + i * 0.9
        add_box(ax, (x, y), 0.75, 0.5, f'F{i}', facecolor='#FFECEC', linewidth=1.2)
    add_box(ax, (1.6, 3.0), 6.0, 0.9, 'Concatenate All\n8×1024 = 8192-dim', facecolor='#FFF3D9', linewidth=1.6)
    add_box(ax, (2.2, 1.6), 5.0, 0.9, 'Single Large Classifier\n8192 → 5', facecolor='#FFEBEE', linewidth=1.6)

    # Late
    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    ax.set_title('Late Fusion (Ours)', fontsize=14, fontweight='bold')
    y = 4.6
    for i in range(8):
        x = 0.4 + i * 0.9
        add_box(ax, (x, y), 0.75, 0.5, '128', facecolor='#F3E8FF', linewidth=1.2)
    add_box(ax, (1.6, 3.0), 6.0, 0.9, 'Concatenate Processed\n8×128 = 1024-dim', facecolor='#E8F8F5', linewidth=1.6)
    add_box(ax, (2.2, 1.6), 5.0, 0.9, 'Final Classifier\n1024 → 5', facecolor='#E8F5FF', linewidth=1.6)

    fig.suptitle('Fusion Strategy Comparison', fontsize=16, fontweight='bold')
    plt.savefig('05_fusion_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('✓ Saved: 05_fusion_comparison.png')
    plt.close()

# 07 training flow (improved distances)
def create_training_flow():
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
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
        add_box(ax, (1.2, y - 0.4), 9.6, 0.9, title, facecolor=color, linewidth=1.6)
        # arrows
        arrow = FancyArrowPatch((6.0, y - 0.4), (6.0, y - 1.0), arrowstyle='->', mutation_scale=18, linewidth=1.4)
        ax.add_patch(arrow)
        y -= 1.5

    # Right side hyperparams block
    add_box(ax, (11.0, 6.2), 2.2, 2.4, 'Hyperparams\nLR:5e-4\nWD:1e-4\nEpochs:20\nBatch:12\nAccum:2', facecolor='#FFF7E6', linewidth=1.2)

    plt.savefig('07_training_flow.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('✓ Saved: 07_training_flow.png')
    plt.close()

# MAIN
if __name__ == '__main__':
    print('\n' + '='*70)
    print('CREATING SELECTED VISUALIZATIONS (01,03,04,05,07)')
    print('='*70 + '\n')

    create_pipeline_architecture()
    create_multi_layer_extraction()
    create_pooling_strategies_two()
    create_fusion_comparison()
    create_training_flow()

    print('\nAll requested images created.\n')
