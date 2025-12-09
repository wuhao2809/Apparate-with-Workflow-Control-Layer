"""
Visualize component-level analysis results.

Creates:
- Bar charts: Latency improvement by feature combination
- Component contribution analysis
- Ablation study charts
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from collections import defaultdict

# Try to import seaborn (optional)
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False

# Color scheme
COLORS = {
    'baseline': '#6C757D',
    'prioritization': '#2E86AB',
    'adaptive_batching': '#A23B72',
    'feedback': '#F18F01',
    'all_features': '#28A745'
}

plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12


def load_component_results(results_dir):
    """Load component analysis results from JSON files."""
    results = []
    
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    json_files = [f for f in json_files if "summary" not in os.path.basename(f).lower() and "analysis" not in os.path.basename(f).lower()]
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Extract config name from JSON data (more reliable than filename)
                config_name = data.get("config_name", "")
                if not config_name:
                    # Fallback: extract from filename
                    filename = os.path.basename(json_file)
                    # Pattern: config_name_dataset_arch_...json
                    # Find the first part before the dataset/arch pattern
                    if "_imdb_" in filename or "_distilbert-base_" in filename:
                        config_name = filename.split("_imdb_")[0].split("_distilbert-base_")[0]
                    else:
                        parts = filename.replace('.json', '').split('_')
                        config_name = parts[0]
                
                metrics = data.get("metrics", {})
                if not metrics or metrics.get("latency_improvement") is None:
                    print(f"Skipping {os.path.basename(json_file)}: no valid metrics")
                    continue  # Skip files without metrics
                
                result = {
                    "config_name": config_name,
                    "latency_improvement": metrics.get("latency_improvement", 0),
                    "accuracy": metrics.get("overall_accuracy", 0),
                    "exit_rate": metrics.get("exit_rate", 0),
                    "returncode": data.get("returncode", 1)
                }
                
                # Parse feature flags from JSON data (more reliable)
                result["prioritization"] = data.get("enable_prioritization", False) or "prioritization" in config_name
                result["adaptive_batching"] = data.get("enable_adaptive_batching", False) or "batching" in config_name
                result["feedback"] = data.get("enable_feedback", False) or "feedback" in config_name
                result["all_features"] = config_name == "all_features"
                
                results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def normalize_config_name(name):
    """Normalize config name to match standard format."""
    name = name.lower().replace("_", "+").replace(" ", "+")
    if "prioritization" in name and "batching" in name and "feedback" in name:
        return "all_features"
    elif "prioritization" in name and "batching" in name:
        return "prioritization+batching"
    elif "prioritization" in name and "feedback" in name:
        return "prioritization+feedback"
    elif "batching" in name and "feedback" in name:
        return "batching+feedback"
    elif "prioritization" in name:
        return "prioritization_only"
    elif "batching" in name or "adaptive" in name:
        return "adaptive_batching_only"
    elif "feedback" in name:
        return "feedback_only"
    elif "baseline" in name:
        return "baseline"
    return name


def create_component_comparison_plot(results, output_dir="workflow_results/plots"):
    """Create bar chart comparing all component combinations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successful results
    successful = [r for r in results if r.get("returncode") == 0]
    if not successful:
        print("No successful experiments found!")
        return
    
    # Sort by config name for consistent ordering
    config_order = [
        "baseline",
        "prioritization_only",
        "adaptive_batching_only",
        "feedback_only",
        "prioritization+batching",
        "prioritization+feedback",
        "batching+feedback",
        "all_features"
    ]
    
    # Normalize config names in results
    for r in successful:
        r["normalized_name"] = normalize_config_name(r["config_name"])
    
    # Get data in order
    config_names = []
    latency_improvements = []
    accuracies = []
    exit_rates = []
    
    for config_name in config_order:
        matching = [r for r in successful if r["normalized_name"] == config_name]
        if matching:
            r = matching[0]
            # Pretty names for display (using abbreviations to avoid overlap)
            pretty_names = {
                "baseline": "Baseline",
                "prioritization_only": "Prio",
                "adaptive_batching_only": "Batch",
                "feedback_only": "Feedback",
                "prioritization+batching": "Prio+Batch",
                "prioritization+feedback": "Prio+FB",
                "batching+feedback": "Batch+FB",
                "all_features": "All"
            }
            config_names.append(pretty_names.get(config_name, config_name.replace("_", " ").title()))
            latency_improvements.append(r["latency_improvement"])
            accuracies.append(r["accuracy"])
            exit_rates.append(r["exit_rate"] * 100)
    
    # Create figure with better spacing
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle('Component-Level Analysis: Feature Contribution', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    x = np.arange(len(config_names))
    width = 0.7
    
    # Color mapping with better logic
    def get_color(config_name):
        config_lower = config_name.lower().replace("\n", " ").replace("+", " ")
        if "baseline" in config_lower:
            return COLORS['baseline']
        elif "all" in config_lower or ("prioritization" in config_lower and "batching" in config_lower and "feedback" in config_lower):
            return COLORS['all_features']
        elif "prioritization" in config_lower and "batching" in config_lower:
            return '#9B59B6'  # Purple for prioritization + batching
        elif "prioritization" in config_lower and "feedback" in config_lower:
            return '#E67E22'  # Orange for prioritization + feedback
        elif "batching" in config_lower and "feedback" in config_lower:
            return '#D35400'  # Dark orange for batching + feedback
        elif "prioritization" in config_lower:
            return COLORS['prioritization']
        elif "batching" in config_lower or "adaptive" in config_lower:
            return COLORS['adaptive_batching']
        elif "feedback" in config_lower:
            return COLORS['feedback']
        else:
            return '#95A5A6'  # Gray for unknown
    
    colors = [get_color(cn) for cn in config_names]
    
    # 1. Latency Improvement
    ax = axes[0]
    bars = ax.bar(x, latency_improvements, width, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=1.5, zorder=3)
    ax.set_xlabel('Configuration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Latency Improvement (%)', fontweight='bold', fontsize=12)
    ax.set_title('Latency Improvement by Feature Combination', fontweight='bold', pad=12, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1, zorder=0)
    ax.set_ylim(bottom=0, top=max(latency_improvements) * 1.15 if latency_improvements else 20)
    
    # Add value labels with better positioning
    for i, (bar, val) in enumerate(zip(bars, latency_improvements)):
        height = bar.get_height()
        # Highlight significant improvements
        if val > 12:
            label_color = 'darkgreen'
            fontweight = 'bold'
        elif val < 8:
            label_color = 'darkred'
            fontweight = 'normal'
        else:
            label_color = 'black'
            fontweight = 'normal'
        ax.text(bar.get_x() + bar.get_width()/2., height + max(latency_improvements) * 0.02,
               f'{val:.2f}%', ha='center', va='bottom', fontweight=fontweight, 
               fontsize=10, color=label_color)
        
        # Add baseline reference line
        if i == 0:
            baseline_val = val
            ax.axhline(y=baseline_val, color='gray', linestyle=':', linewidth=2, 
                      alpha=0.7, zorder=1, label='Baseline')
    
    # Add legend for baseline
    if len(bars) > 0:
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # 2. Accuracy
    ax = axes[1]
    bars = ax.bar(x, accuracies, width, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=1.5, zorder=3)
    ax.set_xlabel('Configuration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Accuracy by Feature Combination', fontweight='bold', pad=12, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1, zorder=0)
    ax.set_ylim(bottom=min(accuracies) * 0.998 if accuracies else 97, 
                top=max(accuracies) * 1.002 if accuracies else 99)
    
    # Add value labels
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max(accuracies) - min(accuracies)) * 0.01,
               f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Exit Rate
    ax = axes[2]
    bars = ax.bar(x, exit_rates, width, color=colors, alpha=0.85, 
                  edgecolor='black', linewidth=1.5, zorder=3)
    ax.set_xlabel('Configuration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Early Exit Rate (%)', fontweight='bold', fontsize=12)
    ax.set_title('Early Exit Rate by Feature Combination', fontweight='bold', pad=12, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, rotation=45, ha='right', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1, zorder=0)
    ax.set_ylim(bottom=0, top=max(exit_rates) * 1.15 if exit_rates else 30)
    
    # Add value labels
    for bar, val in zip(bars, exit_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(exit_rates) * 0.02,
               f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for rotated labels
    output_path = os.path.join(output_dir, "component_comparison.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved component comparison plot: {output_path}")


def create_ablation_study_plot(results, output_dir="workflow_results/plots"):
    """Create step-by-step improvement chart (ablation study)."""
    os.makedirs(output_dir, exist_ok=True)
    
    successful = [r for r in results if r.get("returncode") == 0]
    if not successful:
        return
    
    # Normalize config names
    for r in successful:
        r["normalized_name"] = normalize_config_name(r["config_name"])
    
    # Define ablation steps (adding one feature at a time)
    ablation_steps = [
        {"name": "Baseline", "normalized": "baseline"},
        {"name": "+ Prioritization", "normalized": "prioritization_only"},
        {"name": "+ Adaptive Batching", "normalized": "prioritization+batching"},
        {"name": "+ Feedback", "normalized": "all_features"},
    ]
    
    # Find matching results
    step_data = []
    for step in ablation_steps:
        matching = [r for r in successful if r["normalized_name"] == step["normalized"]]
        if matching:
            step_data.append({
                "name": step["name"],
                "latency_improvement": matching[0]["latency_improvement"],
                "accuracy": matching[0]["accuracy"]
            })
    
    if len(step_data) < 2:
        print("Not enough data for ablation study")
        return
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ablation Study: Incremental Feature Addition', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    step_names = [s["name"] for s in step_data]
    latency_improvements = [s["latency_improvement"] for s in step_data]
    accuracies = [s["accuracy"] for s in step_data]
    
    x = np.arange(len(step_names))
    
    # Latency improvement
    ax = axes[0]
    bars = ax.bar(x, latency_improvements, color=COLORS['all_features'], alpha=0.9, 
                  edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Latency Improvement (%)', fontweight='bold')
    ax.set_title('Latency Improvement: Step-by-Step', fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(step_names, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    ax.set_ylim(bottom=0)
    
    # Add value labels and improvement arrows
    for i, (bar, prev_val) in enumerate(zip(bars, [0] + latency_improvements[:-1])):
        height = bar.get_height()
        improvement = height - prev_val
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
               f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        if i > 0 and improvement > 0:
            ax.annotate(f'+{improvement:.2f}%', 
                       xy=(bar.get_x() + bar.get_width()/2., height),
                       xytext=(bar.get_x() + bar.get_width()/2., prev_val + improvement/2),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=9, fontweight='bold', color='green', ha='center')
    
    # Accuracy
    ax = axes[1]
    bars = ax.bar(x, accuracies, color=COLORS['adaptive_batching'], alpha=0.9, 
                  edgecolor='black', linewidth=1.5, width=0.6)
    ax.set_xlabel('Configuration', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Accuracy: Step-by-Step', fontweight='bold', pad=12, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(step_names, rotation=0, ha='center', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "ablation_study.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved ablation study plot: {output_path}")


def create_component_contribution_plot(results, output_dir="workflow_results/plots"):
    """Create plot showing individual component contributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    successful = [r for r in results if r.get("returncode") == 0]
    if not successful:
        return
    
    # Normalize config names
    for r in successful:
        r["normalized_name"] = normalize_config_name(r["config_name"])
    
    # Find baseline
    baseline = next((r for r in successful if r["normalized_name"] == "baseline"), None)
    if not baseline:
        print("Baseline not found!")
        return
    
    baseline_latency = baseline["latency_improvement"]
    
    # Calculate individual contributions
    contributions = {}
    
    # Prioritization only
    prio_only = next((r for r in successful if r["normalized_name"] == "prioritization_only"), None)
    if prio_only:
        contributions["Prioritization"] = prio_only["latency_improvement"] - baseline_latency
    
    # Adaptive batching only
    batch_only = next((r for r in successful if r["normalized_name"] == "adaptive_batching_only"), None)
    if batch_only:
        contributions["Adaptive Batching"] = batch_only["latency_improvement"] - baseline_latency
    
    # Feedback only
    feedback_only = next((r for r in successful if r["normalized_name"] == "feedback_only"), None)
    if feedback_only:
        contributions["Feedback Control"] = feedback_only["latency_improvement"] - baseline_latency
    
    # All features
    all_features = next((r for r in successful if r["normalized_name"] == "all_features"), None)
    if all_features:
        contributions["All Features"] = all_features["latency_improvement"] - baseline_latency
    
    if not contributions:
        print("Not enough data for contribution analysis")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = list(contributions.keys())
    values = list(contributions.values())
    colors = [COLORS.get(c.lower().replace(" ", "_"), '#6C757D') for c in components]
    
    bars = ax.bar(components, values, color=colors, alpha=0.9, 
                  edgecolor='black', linewidth=1.5, width=0.6)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Component', fontweight='bold', fontsize=12)
    ax.set_ylabel('Latency Improvement Contribution (%)', fontweight='bold', fontsize=12)
    ax.set_title('Individual Component Contributions\n(Relative to Baseline)', 
                 fontweight='bold', fontsize=13, pad=15)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., val + (0.2 if val > 0 else -0.2),
               f'{val:+.2f}%', ha='center', va='bottom' if val > 0 else 'top', 
               fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "component_contributions.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved component contributions plot: {output_path}")


def main():
    """Main function to generate component analysis visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize component analysis results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing component analysis results")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots (default: same as results_dir/plots)")
    args = parser.parse_args()
    
    print("="*60)
    print("Component Analysis Visualization")
    print("="*60)
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.results_dir, "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir}")
    
    # Load results
    print(f"\nLoading results from {args.results_dir}...")
    results = load_component_results(args.results_dir)
    print(f"Loaded {len(results)} experiment results")
    
    successful = [r for r in results if r.get("returncode") == 0]
    print(f"Successful experiments: {len(successful)}")
    
    if len(successful) == 0:
        print("Error: No successful experiments found!")
        return
    
    # Generate plots
    print("\n[1/3] Creating component comparison plot...")
    create_component_comparison_plot(results, output_dir)
    
    print("\n[2/3] Creating ablation study plot...")
    create_ablation_study_plot(results, output_dir)
    
    print("\n[3/3] Creating component contributions plot...")
    create_component_contribution_plot(results, output_dir)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*60)
    
    # Print summary
    print("\nGenerated plots:")
    plot_files = glob.glob(os.path.join(output_dir, "component*.pdf"))
    for plot_file in sorted(plot_files):
        print(f"  - {os.path.basename(plot_file)}")


if __name__ == "__main__":
    main()

