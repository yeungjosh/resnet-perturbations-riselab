#!/usr/bin/env python3
"""
ResNet Perturbation Analysis - Visualization Demo
==================================================

This script creates comprehensive visualizations showcasing the project's
methodology and results. It demonstrates:

1. How different perturbations affect input images
2. Example robustness curves comparing optimizers
3. Visual explanation of the analysis pipeline

Usage:
    python3 visualization_demo.py

Outputs:
    - demo_perturbations.png: Shows images at different noise levels
    - demo_robustness_curves.png: Simulated optimizer comparison
    - demo_architecture.png: ResNet architecture visualization
    - demo_complete.png: All-in-one comprehensive visualization

Author: Mahoney Group, RISELab
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# 1. PERTURBATION VISUALIZATION
# ============================================================================

def create_sample_image(size=28):
    """Create a simple synthetic handwritten-digit-like image."""
    img = np.zeros((size, size))

    # Create a "7"-like shape
    img[5:8, 8:20] = 0.9  # Top horizontal line
    img[5:22, 17:20] = 0.9  # Diagonal/vertical line

    # Add some noise to make it look more realistic
    img += np.random.normal(0, 0.05, (size, size))
    img = np.clip(img, 0, 1)

    return img


def add_gaussian_noise(img, noise_level):
    """Add Gaussian white noise to an image."""
    noise = np.random.randn(*img.shape) * noise_level
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 1)


def add_salt_pepper_noise(img, noise_level):
    """Add salt & pepper noise to an image."""
    noisy_img = img.copy()
    n_pixels = int(noise_level * img.size)

    # Salt (white pixels)
    coords = [np.random.randint(0, i, n_pixels // 2) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 1.0

    # Pepper (black pixels)
    coords = [np.random.randint(0, i, n_pixels // 2) for i in img.shape]
    noisy_img[coords[0], coords[1]] = 0.0

    return noisy_img


def visualize_perturbations():
    """Create visualization showing different perturbation types and levels."""
    np.random.seed(42)

    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    fig.suptitle('Image Perturbations: How Noise Affects Input Data',
                 fontsize=16, fontweight='bold', y=0.98)

    # Original image
    original = create_sample_image()

    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

    # Row 1: Original progression
    for idx, noise_level in enumerate(noise_levels):
        axes[0, idx].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(f'Original', fontsize=10)
        axes[0, idx].axis('off')
        if idx > 0:
            axes[0, idx].set_title(f'λ={noise_level}', fontsize=10)

    # Row 2: Gaussian white noise
    for idx, noise_level in enumerate(noise_levels):
        noisy = add_gaussian_noise(original, noise_level)
        axes[1, idx].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        if idx == 0:
            axes[1, idx].set_ylabel('Gaussian\nNoise', fontsize=12, fontweight='bold')
        axes[1, idx].axis('off')

    # Row 3: Salt & Pepper noise
    for idx, noise_level in enumerate(noise_levels):
        noisy = add_salt_pepper_noise(original, noise_level)
        axes[2, idx].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        if idx == 0:
            axes[2, idx].set_ylabel('Salt & Pepper\nNoise', fontsize=12, fontweight='bold')
        axes[2, idx].axis('off')

    plt.tight_layout()
    plt.savefig('demo_perturbations.png', dpi=300, bbox_inches='tight')
    print("✓ Created demo_perturbations.png")
    return fig


# ============================================================================
# 2. ROBUSTNESS CURVES VISUALIZATION
# ============================================================================

def generate_robustness_curve(optimizer_name, noise_levels, base_acc=0.95,
                               robustness=1.0, seed=None):
    """
    Generate a simulated robustness curve.

    Args:
        optimizer_name: Name of the optimizer
        noise_levels: Array of noise levels
        base_acc: Base accuracy at noise=0
        robustness: Robustness factor (higher = more robust)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)

    # Create decay curve with some randomness
    decay_rate = 3.0 / robustness
    accuracies = base_acc * np.exp(-decay_rate * noise_levels)

    # Add asymptotic behavior (model doesn't go to 0% accuracy)
    asymptote = 0.1 + 0.05 * robustness
    accuracies = asymptote + (accuracies - asymptote * np.exp(-decay_rate * noise_levels))

    # Add some realistic noise
    noise = np.random.normal(0, 0.01, len(noise_levels))
    accuracies += noise
    accuracies = np.clip(accuracies, 0.05, 1.0)

    return accuracies


def visualize_robustness_curves():
    """Create visualization comparing optimizer robustness."""
    np.random.seed(42)

    noise_levels = np.array([0.0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Optimizer Robustness Comparison: Accuracy Under Perturbation',
                 fontsize=16, fontweight='bold')

    # Define optimizer characteristics
    optimizers = {
        'SGD': {'base_acc': 0.985, 'robustness': 1.3, 'marker': 'o', 'linestyle': '-'},
        'Adadelta': {'base_acc': 0.980, 'robustness': 1.1, 'marker': 's', 'linestyle': '--'},
        'Adam': {'base_acc': 0.982, 'robustness': 0.9, 'marker': '^', 'linestyle': '-.'},
        'Adahessian': {'base_acc': 0.979, 'robustness': 1.0, 'marker': 'd', 'linestyle': ':'},
    }

    colors = sns.color_palette("husl", len(optimizers))

    # Plot 1: Gaussian Noise
    for idx, (opt_name, params) in enumerate(optimizers.items()):
        accuracies = generate_robustness_curve(
            opt_name, noise_levels,
            params['base_acc'], params['robustness'],
            seed=42+idx
        )

        ax1.plot(noise_levels, accuracies,
                marker=params['marker'],
                linestyle=params['linestyle'],
                linewidth=2.5,
                markersize=8,
                label=opt_name,
                color=colors[idx],
                alpha=0.8)

    ax1.set_xlabel('Noise Level (λ)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Gaussian White Noise Robustness', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax1.text(0.15, 0.52, 'Critical Threshold (50%)', fontsize=9, color='red', alpha=0.7)

    # Plot 2: Salt & Pepper Noise
    for idx, (opt_name, params) in enumerate(optimizers.items()):
        # S&P noise typically more challenging
        accuracies = generate_robustness_curve(
            opt_name, noise_levels,
            params['base_acc'] - 0.01,
            params['robustness'] * 0.85,
            seed=100+idx
        )

        ax2.plot(noise_levels, accuracies,
                marker=params['marker'],
                linestyle=params['linestyle'],
                linewidth=2.5,
                markersize=8,
                label=opt_name,
                color=colors[idx],
                alpha=0.8)

    ax2.set_xlabel('Noise Level (λ)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Salt & Pepper Noise Robustness', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)

    plt.tight_layout()
    plt.savefig('demo_robustness_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Created demo_robustness_curves.png")
    return fig


# ============================================================================
# 3. ARCHITECTURE VISUALIZATION
# ============================================================================

def visualize_architecture():
    """Create a visual representation of the ResNet architecture."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    fig.suptitle('ResNet Architecture for CIFAR-10',
                 fontsize=18, fontweight='bold', y=0.96)

    # Color scheme
    colors = {
        'input': '#E8F4F8',
        'conv': '#B3E5FC',
        'resblock': '#81D4FA',
        'pool': '#4FC3F7',
        'fc': '#29B6F6',
        'skip': '#FFB74D'
    }

    y_pos = 11
    x_center = 5

    # Input
    ax.add_patch(Rectangle((x_center-0.5, y_pos-0.4), 1, 0.6,
                           facecolor=colors['input'], edgecolor='black', linewidth=2))
    ax.text(x_center, y_pos-0.1, 'Input Image\n3×32×32',
           ha='center', va='center', fontsize=10, fontweight='bold')
    y_pos -= 1.0

    # Arrow down
    ax.arrow(x_center, y_pos+0.5, 0, -0.3, head_width=0.15,
            head_length=0.1, fc='black', ec='black')
    y_pos -= 0.5

    # Initial Conv
    ax.add_patch(Rectangle((x_center-0.6, y_pos-0.4), 1.2, 0.6,
                           facecolor=colors['conv'], edgecolor='black', linewidth=2))
    ax.text(x_center, y_pos-0.1, 'Conv 3×3\n+ BatchNorm + ReLU',
           ha='center', va='center', fontsize=9, fontweight='bold')
    y_pos -= 1.0

    # ResNet Blocks
    block_configs = [
        ('Layer 1: 3 blocks\n16 channels', colors['resblock']),
        ('Layer 2: 3 blocks\n32 channels', colors['resblock']),
        ('Layer 3: 3 blocks\n64 channels', colors['resblock'])
    ]

    for block_text, color in block_configs:
        # Arrow
        ax.arrow(x_center, y_pos+0.5, 0, -0.3, head_width=0.15,
                head_length=0.1, fc='black', ec='black')
        y_pos -= 0.5

        # Residual block with skip connection
        # Main path
        main_box = Rectangle((x_center-0.7, y_pos-0.8), 1.4, 1.2,
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(main_box)
        ax.text(x_center, y_pos-0.2, block_text,
               ha='center', va='center', fontsize=9, fontweight='bold')

        # Skip connection (curved arrow)
        skip_arrow = FancyArrowPatch((x_center+1.2, y_pos+0.2), (x_center+1.2, y_pos-1.0),
                                    connectionstyle="arc3,rad=.5",
                                    arrowstyle='->', mutation_scale=20,
                                    linewidth=2.5, color=colors['skip'])
        ax.add_patch(skip_arrow)
        ax.text(x_center+1.8, y_pos-0.4, 'Skip', fontsize=8,
               color=colors['skip'], fontweight='bold')

        y_pos -= 1.5

    # Global Average Pooling
    ax.arrow(x_center, y_pos+0.5, 0, -0.3, head_width=0.15,
            head_length=0.1, fc='black', ec='black')
    y_pos -= 0.5

    ax.add_patch(Rectangle((x_center-0.6, y_pos-0.4), 1.2, 0.6,
                           facecolor=colors['pool'], edgecolor='black', linewidth=2))
    ax.text(x_center, y_pos-0.1, 'Global Avg Pool',
           ha='center', va='center', fontsize=9, fontweight='bold')
    y_pos -= 1.0

    # Fully Connected
    ax.arrow(x_center, y_pos+0.5, 0, -0.3, head_width=0.15,
            head_length=0.1, fc='black', ec='black')
    y_pos -= 0.5

    ax.add_patch(Rectangle((x_center-0.6, y_pos-0.4), 1.2, 0.6,
                           facecolor=colors['fc'], edgecolor='black', linewidth=2))
    ax.text(x_center, y_pos-0.1, 'Fully Connected\n10 classes',
           ha='center', va='center', fontsize=9, fontweight='bold')
    y_pos -= 1.0

    # Output
    ax.arrow(x_center, y_pos+0.5, 0, -0.3, head_width=0.15,
            head_length=0.1, fc='black', ec='black')
    y_pos -= 0.5

    ax.add_patch(Rectangle((x_center-0.5, y_pos-0.3), 1, 0.4,
                           facecolor='#C8E6C9', edgecolor='black', linewidth=2))
    ax.text(x_center, y_pos-0.1, 'Output',
           ha='center', va='center', fontsize=10, fontweight='bold')

    # Add legend
    legend_y = 10.5
    legend_x = 0.5
    ax.text(legend_x, legend_y+0.5, 'Total: 20 layers (ResNet-20)',
           fontsize=10, fontweight='bold')
    ax.text(legend_x, legend_y, '• 1 initial conv layer', fontsize=8)
    ax.text(legend_x, legend_y-0.3, '• 3×3×2 = 18 residual layers', fontsize=8)
    ax.text(legend_x, legend_y-0.6, '• 1 fully connected layer', fontsize=8)

    plt.tight_layout()
    plt.savefig('demo_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Created demo_architecture.png")
    return fig


# ============================================================================
# 4. COMPLETE WORKFLOW VISUALIZATION
# ============================================================================

def visualize_complete_workflow():
    """Create a comprehensive visualization of the entire pipeline."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('ResNet Perturbation Analysis: Complete Workflow',
                 fontsize=20, fontweight='bold', y=0.98)

    # 1. Top-left: Training data
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Step 1: Training Data', fontsize=12, fontweight='bold')
    for i in range(5):
        sample = create_sample_image() + np.random.normal(0, 0.1, (28, 28))
        sample = np.clip(sample, 0, 1)
        ax1.imshow(sample, cmap='gray', extent=[i*30, (i+1)*30, 0, 30])
    ax1.text(75, -10, 'MNIST / CIFAR-10\n50,000 training images',
            ha='center', fontsize=9)
    ax1.axis('off')

    # 2. Top-middle: Training process
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Step 2: Train with Different Optimizers', fontsize=12, fontweight='bold')
    ax2.text(0.5, 0.8, 'Optimizers:', ha='center', fontsize=11, fontweight='bold',
            transform=ax2.transAxes)
    optimizers_text = ['SGD', 'Adam', 'Adadelta', 'Adahessian']
    for i, opt in enumerate(optimizers_text):
        ax2.text(0.5, 0.6 - i*0.15, f'• {opt}', ha='center', fontsize=10,
                transform=ax2.transAxes)
    ax2.text(0.5, 0.05, '110 epochs each', ha='center', fontsize=9,
            style='italic', transform=ax2.transAxes)
    ax2.axis('off')

    # 3. Top-right: Trained models
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Step 3: Trained Models', fontsize=12, fontweight='bold')
    ax3.text(0.5, 0.5, '4 ResNet Models\n(one per optimizer)',
            ha='center', va='center', fontsize=11, fontweight='bold',
            transform=ax3.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax3.axis('off')

    # 4. Middle-left: Test images
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('Step 4: Test Images', fontsize=12, fontweight='bold')
    original = create_sample_image()
    ax4.imshow(original, cmap='gray')
    ax4.text(14, -3, 'Clean test image', ha='center', fontsize=9)
    ax4.axis('off')

    # 5. Middle-middle: Add perturbations
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('Step 5: Apply Perturbations', fontsize=12, fontweight='bold')
    noise_levels_demo = [0.0, 0.1, 0.2]
    for i, noise in enumerate(noise_levels_demo):
        noisy = add_gaussian_noise(original, noise)
        ax5.imshow(noisy, cmap='gray', extent=[i*30, (i+1)*30, 0, 30])
    ax5.text(45, -5, 'Noise levels: 0.0 → 0.25', ha='center', fontsize=9)
    ax5.axis('off')

    # 6. Middle-right: Evaluate
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('Step 6: Evaluate Each Model', fontsize=12, fontweight='bold')
    ax6.text(0.5, 0.5, 'Compute accuracy\non perturbed images\n\n' +
            'For each:\n• Optimizer\n• Noise type\n• Noise level',
            ha='center', va='center', fontsize=10,
            transform=ax6.transAxes)
    ax6.axis('off')

    # 7. Bottom: Robustness curves (spanning all columns)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.set_title('Step 7: Analyze Robustness', fontsize=12, fontweight='bold')

    np.random.seed(42)
    noise_levels = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])

    optimizers = {
        'SGD': {'base_acc': 0.98, 'robustness': 1.3},
        'Adadelta': {'base_acc': 0.97, 'robustness': 1.1},
        'Adam': {'base_acc': 0.975, 'robustness': 0.9},
    }

    colors = sns.color_palette("husl", len(optimizers))

    for idx, (opt_name, params) in enumerate(optimizers.items()):
        accuracies = generate_robustness_curve(
            opt_name, noise_levels,
            params['base_acc'], params['robustness'],
            seed=42+idx
        )
        ax7.plot(noise_levels, accuracies, 'o-', linewidth=3,
                markersize=10, label=opt_name, color=colors[idx], alpha=0.8)

    ax7.set_xlabel('Noise Level', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=11, loc='upper right')
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim([0, 1.05])

    # Add annotation
    ax7.annotate('More robust optimizer\n(flatter decline)',
                xy=(0.15, 0.85), xytext=(0.12, 0.6),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                fontsize=10, color='green', fontweight='bold')

    plt.savefig('demo_complete_workflow.png', dpi=300, bbox_inches='tight')
    print("✓ Created demo_complete_workflow.png")
    return fig


# ============================================================================
# 5. METHODOLOGY SUMMARY VISUALIZATION
# ============================================================================

def visualize_methodology_summary():
    """Create a summary visualization of key concepts."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Key Concepts in Optimizer Robustness Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Loss Landscape
    ax1 = axes[0, 0]
    ax1.set_title('Loss Landscape Visualization', fontsize=13, fontweight='bold')

    x = np.linspace(-3, 3, 100)

    # Sharp minimum (Adam-like)
    sharp = 0.5 + 2.5 * (x - 0.5)**2
    # Flat minimum (SGD-like)
    flat = 0.5 + 0.8 * (x + 0.5)**2

    ax1.plot(x, sharp, 'r-', linewidth=3, label='Sharp Minimum\n(Less Robust)', alpha=0.7)
    ax1.plot(x, flat, 'g-', linewidth=3, label='Flat Minimum\n(More Robust)', alpha=0.7)

    ax1.scatter([0.5], [0.5], color='red', s=200, zorder=5, marker='*')
    ax1.scatter([-0.5], [0.5], color='green', s=200, zorder=5, marker='*')

    ax1.set_xlabel('Parameter Space', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss Value', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 8])

    # 2. Optimizer Characteristics
    ax2 = axes[0, 1]
    ax2.set_title('Optimizer Characteristics', fontsize=13, fontweight='bold')
    ax2.axis('off')

    optimizer_info = [
        ('SGD', 'Simple gradient descent', 'Flat minima', 'Most robust'),
        ('Adam', 'Adaptive moments', 'Fast convergence', 'Less robust'),
        ('Adadelta', 'Adaptive LR', 'No manual tuning', 'Medium robust'),
        ('Adahessian', 'Second-order', 'Curvature info', 'Variable'),
    ]

    y_start = 0.9
    for i, (name, desc, prop, robust) in enumerate(optimizer_info):
        y_pos = y_start - i * 0.22

        # Color code by robustness
        color = {'Most robust': 'green', 'Medium robust': 'orange',
                'Less robust': 'red', 'Variable': 'gray'}[robust]

        ax2.text(0.05, y_pos, name, fontsize=12, fontweight='bold',
                transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        ax2.text(0.25, y_pos, f'{desc}\n{prop}', fontsize=9,
                transform=ax2.transAxes)
        ax2.text(0.75, y_pos, robust, fontsize=10, fontweight='bold',
                color=color, transform=ax2.transAxes)

    # 3. Perturbation Types
    ax3 = axes[1, 0]
    ax3.set_title('Perturbation Types Comparison', fontsize=13, fontweight='bold')

    np.random.seed(42)
    sample = create_sample_image(size=50)

    # Show original and two noise types
    images = [
        (sample, 'Original'),
        (add_gaussian_noise(sample, 0.15), 'Gaussian Noise\nλ=0.15'),
        (add_salt_pepper_noise(sample, 0.15), 'Salt & Pepper\nλ=0.15')
    ]

    for i, (img, title) in enumerate(images):
        extent = [i*55, (i+1)*55, 0, 50]
        ax3.imshow(img, cmap='gray', extent=extent, vmin=0, vmax=1)
        ax3.text(i*55 + 27.5, -8, title, ha='center', fontsize=10,
                fontweight='bold')

    ax3.axis('off')
    ax3.set_xlim([0, 165])
    ax3.set_ylim([-15, 50])

    # 4. Key Metrics
    ax4 = axes[1, 1]
    ax4.set_title('Robustness Metrics', fontsize=13, fontweight='bold')
    ax4.axis('off')

    metrics_info = [
        ('Area Under Curve (AUC)',
         'Overall robustness across all noise levels'),
        ('Critical Noise Level',
         'Noise threshold where accuracy drops below 50%'),
        ('Degradation Rate',
         'Slope of accuracy decline (steeper = less robust)'),
        ('Relative Ranking',
         'Which optimizer maintains highest accuracy'),
    ]

    y_start = 0.85
    for i, (metric, description) in enumerate(metrics_info):
        y_pos = y_start - i * 0.20
        ax4.text(0.05, y_pos, f'{i+1}. {metric}', fontsize=11,
                fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.1, y_pos - 0.08, description, fontsize=9,
                style='italic', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig('demo_methodology_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Created demo_methodology_summary.png")
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all visualizations."""
    print("\n" + "="*70)
    print("ResNet Perturbation Analysis - Visualization Demo")
    print("="*70 + "\n")

    print("Generating visualizations...\n")

    # Create all visualizations
    visualize_perturbations()
    visualize_robustness_curves()
    visualize_architecture()
    visualize_complete_workflow()
    visualize_methodology_summary()

    print("\n" + "="*70)
    print("All visualizations created successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. demo_perturbations.png - Perturbation effects on images")
    print("  2. demo_robustness_curves.png - Optimizer comparison curves")
    print("  3. demo_architecture.png - ResNet architecture diagram")
    print("  4. demo_complete_workflow.png - End-to-end pipeline")
    print("  5. demo_methodology_summary.png - Key concepts summary")
    print("\nThese visualizations showcase the project methodology and results.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
