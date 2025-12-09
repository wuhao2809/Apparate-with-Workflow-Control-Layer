"""
Visualization script for Prioritization and Adaptive Batching demonstration.

Creates two plots:
1. SLO Violation Rate Comparison (Prioritization)
2. Queue Length Over Time (Adaptive Batching)
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Try to import seaborn (optional)
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300


def load_results(results_dir):
    """Load test results from JSON file."""
    results_file = os.path.join(results_dir, "test_results.json")
    queue_data_file = os.path.join(results_dir, "queue_data.json")
    
    if not os.path.exists(results_file):
        # Try to find latest results
        parent_dir = os.path.dirname(results_dir) if os.path.dirname(results_dir) else "workflow_results"
        pattern = os.path.join(parent_dir, "prioritization_batching_test_*", "test_results.json")
        matches = glob.glob(pattern)
        if matches:
            results_file = max(matches, key=os.path.getctime)
            results_dir = os.path.dirname(results_file)
            queue_data_file = os.path.join(results_dir, "queue_data.json")
        else:
            raise FileNotFoundError(f"Could not find results file: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    queue_data = None
    if os.path.exists(queue_data_file):
        with open(queue_data_file, 'r') as f:
            queue_data = json.load(f)
    
    return results, queue_data, results_dir


def plot_slo_violations(results, output_dir):
    """
    Plot 1: SLO Violation Rate Comparison
    Shows how prioritization reduces SLO violations, especially for tight SLOs.
    """
    prio_results = results['prioritization']
    baseline_metrics = prio_results['baseline']['metrics']
    prio_metrics = prio_results['prioritization']['metrics']
    
    # Extract violation rates by SLO
    slos = sorted(baseline_metrics['violations_by_slo'].keys())
    baseline_rates = [baseline_metrics['violations_by_slo'][slo] for slo in slos]
    prio_rates = [prio_metrics['violations_by_slo'][slo] for slo in slos]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(slos))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline (FIFO)', 
                    color='#6C757D', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, prio_rates, width, label='With Prioritization', 
                    color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only label if significant
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('SLO (ms)', fontweight='bold')
    ax.set_ylabel('SLO Violation Rate (%)', fontweight='bold')
    ax.set_title('Prioritization: SLO Violation Rate Reduction', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{slo}ms' for slo in slos])
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add improvement annotation for each SLO category
    improvement_text = []
    for slo in slos:
        baseline_rate = baseline_rates[slos.index(slo)]
        prio_rate = prio_rates[slos.index(slo)]
        improvement = baseline_rate - prio_rate
        if improvement > 1.0:  # Only show significant improvements
            improvement_text.append(f'{slo}ms: {improvement:.1f}%↓')
    
    if improvement_text:
        # Position improvement annotation lower to avoid overlap with legend and bar labels
        ax.text(0.98, 0.75, 'Improvements:\n' + '\n'.join(improvement_text),
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "prioritization_slo_violations.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.close()


def plot_queue_length_over_time(queue_data, output_dir):
    """
    Plot 2: Queue Length Over Time
    Shows how adaptive batching keeps queue stable under variable load.
    """
    if queue_data is None:
        print("Warning: No queue data found. Skipping queue length plot.")
        return
    
    fixed_queue = queue_data['fixed_batch']['queue_lengths']
    adaptive_queue = queue_data['adaptive_batch']['queue_lengths']
    
    # Use actual time steps if available, otherwise estimate
    if 'time_steps' in queue_data['fixed_batch'] and queue_data['fixed_batch']['time_steps']:
        time_steps = np.array(queue_data['fixed_batch']['time_steps']) / 1000.0  # Convert ms to seconds
    else:
        # Fallback: estimate time steps
        time_steps = np.arange(len(fixed_queue)) * 0.05  # 50ms intervals
    
    # Downsample for plotting if too many points
    max_points = 2000
    if len(time_steps) > max_points:
        step = len(time_steps) // max_points
        time_steps = time_steps[::step]
        fixed_queue = fixed_queue[::step]
        adaptive_queue = adaptive_queue[::step]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(time_steps, fixed_queue, label='Fixed Batch Size', 
            color='#6C757D', linewidth=2, alpha=0.8)
    ax.plot(time_steps, adaptive_queue, label='Adaptive Batching', 
            color='#A23B72', linewidth=2, alpha=0.8)
    
    # Add shaded regions for load phases
    # Low load: 0-20s, High load: 20-40s, Low load: 40-60s
    ax.axvspan(0, 20, alpha=0.1, color='green', label='Low Load (QPS 30)')
    ax.axvspan(20, 40, alpha=0.1, color='red', label='High Load (QPS 150)')
    ax.axvspan(40, 60, alpha=0.1, color='green')
    
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('Queue Length (requests)', fontweight='bold')
    ax.set_title('Adaptive Batching: Queue Stability Under Variable Load', fontweight='bold', pad=15)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add statistics text - moved down to avoid overlap with legend
    fixed_avg = np.mean(fixed_queue)
    fixed_max = np.max(fixed_queue)
    adaptive_avg = np.mean(adaptive_queue)
    adaptive_max = np.max(adaptive_queue)
    
    stats_text = f'Fixed Batch: Avg={fixed_avg:.1f}, Max={fixed_max}\n'
    stats_text += f'Adaptive: Avg={adaptive_avg:.1f}, Max={adaptive_max}'
    
    ax.text(0.02, 0.75, stats_text,
            transform=ax.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, fontweight='bold', family='monospace')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "adaptive_batching_queue_length.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.close()


def create_combined_summary_plot(results, queue_data, output_dir):
    """Create a combined summary plot with both visualizations side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: SLO Violations
    prio_results = results['prioritization']
    baseline_metrics = prio_results['baseline']['metrics']
    prio_metrics = prio_results['prioritization']['metrics']
    
    slos = sorted(baseline_metrics['violations_by_slo'].keys())
    baseline_rates = [baseline_metrics['violations_by_slo'][slo] for slo in slos]
    prio_rates = [prio_metrics['violations_by_slo'][slo] for slo in slos]
    
    x = np.arange(len(slos))
    width = 0.35
    
    ax1.bar(x - width/2, baseline_rates, width, label='Baseline (FIFO)', 
            color='#6C757D', alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.bar(x + width/2, prio_rates, width, label='With Prioritization', 
            color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax1.set_xlabel('SLO (ms)', fontweight='bold')
    ax1.set_ylabel('SLO Violation Rate (%)', fontweight='bold')
    ax1.set_title('(a) Prioritization: SLO Violation Reduction', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{slo}ms' for slo in slos])
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)
    
    # Plot 2: Queue Length
    if queue_data:
        fixed_queue = queue_data['fixed_batch']['queue_lengths']
        adaptive_queue = queue_data['adaptive_batch']['queue_lengths']
        time_steps = np.arange(len(fixed_queue)) / 1000.0
        
        # Downsample
        if len(time_steps) > 5000:
            step = len(time_steps) // 5000
            time_steps = time_steps[::step]
            fixed_queue = fixed_queue[::step]
            adaptive_queue = adaptive_queue[::step]
        
        ax2.plot(time_steps, fixed_queue, label='Fixed Batch Size', 
                color='#6C757D', linewidth=2, alpha=0.8)
        ax2.plot(time_steps, adaptive_queue, label='Adaptive Batching', 
                color='#A23B72', linewidth=2, alpha=0.8)
        
        ax2.axvspan(0, 20, alpha=0.1, color='green')
        ax2.axvspan(20, 40, alpha=0.1, color='red')
        ax2.axvspan(40, 60, alpha=0.1, color='green')
        
        ax2.set_xlabel('Time (seconds)', fontweight='bold')
        ax2.set_ylabel('Queue Length (requests)', fontweight='bold')
        ax2.set_title('(b) Adaptive Batching: Queue Stability', fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "prioritization_batching_summary.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize prioritization and adaptive batching results')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory containing test results (default: find latest)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as results_dir)')
    
    args = parser.parse_args()
    
    # Load results
    try:
        results, queue_data, results_dir = load_results(args.results_dir or "workflow_results")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run test_prioritization_batching.py first to generate results.")
        return
    
    # Determine output directory
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Generate plots
    plot_slo_violations(results, output_dir)
    plot_queue_length_over_time(queue_data, output_dir)
    create_combined_summary_plot(results, queue_data, output_dir)
    
    print("\n" + "="*70)
    print("Visualization complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

