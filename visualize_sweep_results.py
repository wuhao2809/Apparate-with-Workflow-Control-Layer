"""
Visualize QPS and SLO sweep results.

Creates:
- Heatmaps: QPS vs SLO showing latency improvement
- Line plots: Trends across QPS and SLO levels
- Bar charts: Baseline vs Optimized comparisons
- Summary plots: Key metrics visualization
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import glob
from datetime import datetime

# Try to import seaborn (optional)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Set style for better-looking plots
if HAS_SEABORN:
    try:
        sns.set_style("whitegrid")
        sns.set_palette("Set2")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
    except:
        plt.style.use('default')
else:
    plt.style.use('default')

# Define color scheme
COLORS = {
    'baseline': '#6C757D',      # Gray
    'optimized': '#28A745',     # Green
    'latency': '#2E86AB',       # Blue
    'accuracy': '#A23B72',       # Purple
    'exit_rate': '#F18F01',     # Orange
    'improvement': '#28A745',    # Green
    'degradation': '#DC3545'     # Red
}

# Set default font sizes
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14


def parse_log_file(log_path):
    """Parse log file to extract metrics (from analyze_workflow_results.py)."""
    if not os.path.exists(log_path):
        return None
    
    metrics = {
        "overall_accuracy": None,
        "latency_improvement": None,
        "exit_rate": None,
        "overall_ramp_accuracy": None
    }
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "Serving with complete" in line:
                    parts = line.split(",")
                    for part in parts:
                        if "overall accuracy" in part:
                            try:
                                metrics["overall_accuracy"] = float(
                                    part.split("overall accuracy")[1].split("%")[0].strip()
                                )
                            except:
                                pass
                        if "overall serving latency improvement" in part:
                            try:
                                metrics["latency_improvement"] = float(
                                    part.split("overall serving latency improvement")[1]
                                    .split("%")[0].strip()
                                )
                            except:
                                pass
                        if "overall exit rate" in part:
                            try:
                                import ast
                                import re
                                exit_rate_str = part.split("overall exit rate")[1].strip()
                                start_idx = line.find("overall exit rate")
                                if start_idx != -1:
                                    exit_rate_substr = line[start_idx + len("overall exit rate"):].strip()
                                    brace_count = 0
                                    end_idx = -1
                                    for i, char in enumerate(exit_rate_substr):
                                        if char == '{':
                                            brace_count += 1
                                        elif char == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                end_idx = i + 1
                                                break
                                    if end_idx != -1:
                                        exit_rate_str = exit_rate_substr[:end_idx]
                                        exit_rate_str_clean = re.sub(r'np\.int64\((\d+)\)', r'\1', exit_rate_str)
                                        exit_rate_dict = ast.literal_eval(exit_rate_str_clean)
                                        if isinstance(exit_rate_dict, dict):
                                            ramp_ids = sorted(exit_rate_dict.keys())
                                            final_exit_id = ramp_ids[-1]
                                            total_early_exit_rate = sum(
                                                exit_rate_dict[rid] for rid in ramp_ids if rid != final_exit_id
                                            )
                                            metrics["exit_rate"] = total_early_exit_rate
                            except Exception as e:
                                pass
                        if "overall ramp accuracy" in part:
                            try:
                                metrics["overall_ramp_accuracy"] = float(
                                    part.split("overall ramp accuracy")[1].split("%")[0].strip()
                                )
                            except:
                                pass
                    break
    except Exception as e:
        pass
    
    return metrics


def load_sweep_results(results_dir="workflow_results/exp-1208"):
    """Load sweep results and extract metrics from logs."""
    results = []
    
    # Load JSON result files
    json_files = glob.glob(os.path.join(results_dir, "*_*.json"))
    json_files = [f for f in json_files if "summary" not in f and "analysis" not in f]
    
    print(f"Found {len(json_files)} JSON result files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Extract parameters from filename
                filename = os.path.basename(json_file)
                parts = filename.replace('.json', '').split('_')
                
                if len(parts) >= 5:
                    config = parts[0]
                    dataset = parts[1]
                    arch = '_'.join(parts[2:-2])
                    qps = int(parts[-2].replace('qps', ''))
                    slo = int(parts[-1].replace('slo', ''))
                    
                    # Get metrics from JSON file first (preferred - per-experiment metrics)
                    json_metrics = data.get("metrics", {})
                    
                    # Fallback to log file if metrics not in JSON (for old experiment results)
                    if not json_metrics or all(v is None for v in json_metrics.values()):
                        log_suffix = "baseline" if config == "baseline" else "workflow"
                        log_file = f"logs/output_{arch}_{dataset}_{log_suffix}.log"
                        log_metrics = parse_log_file(log_file)
                        
                        # If not found, try in results directory
                        if log_metrics is None or all(v is None for v in log_metrics.values()):
                            log_file_alt = os.path.join(results_dir, f"output_{arch}_{dataset}_{log_suffix}.log")
                            log_metrics = parse_log_file(log_file_alt)
                        
                        # If still not found, try parent directory
                        if log_metrics is None or all(v is None for v in log_metrics.values()):
                            parent_dir = os.path.dirname(results_dir)
                            log_file_alt2 = os.path.join(parent_dir, "logs", f"output_{arch}_{dataset}_{log_suffix}.log")
                            log_metrics = parse_log_file(log_file_alt2)
                        
                        # Use log metrics as fallback
                        if log_metrics:
                            json_metrics = log_metrics
                    
                    result = {
                        "config": config,
                        "dataset": dataset,
                        "arch": arch,
                        "qps": qps,
                        "slo": slo,
                        "returncode": data.get("returncode", 1),
                        "elapsed_time": data.get("elapsed_time", 0),
                        "latency_improvement": json_metrics.get("latency_improvement", 0) if json_metrics else 0,
                        "accuracy": json_metrics.get("overall_accuracy", 0) if json_metrics else 0,
                        "exit_rate": json_metrics.get("exit_rate", 0) if json_metrics else 0,
                    }
                    
                    results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return results


def create_qps_sweep_plot(results, output_dir="workflow_results/plots"):
    """Create beautiful side-by-side comparison plots for QPS sweep."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by dataset/arch, separate baseline and optimized
    baseline_data = defaultdict(lambda: {"qps": [], "latency_improvement": [], "accuracy": [], "exit_rate": []})
    optimized_data = defaultdict(lambda: {"qps": [], "latency_improvement": [], "accuracy": [], "exit_rate": []})
    
    for result in results:
        if result.get("returncode") != 0:
            continue
        
        # Only QPS sweep (SLO fixed at 60)
        if result.get("slo") == 60:
            config = result.get("config")
            dataset = result.get("dataset")
            arch = result.get("arch")
            key = f"{dataset}_{arch}"
            
            data_dict = baseline_data if config == "baseline" else optimized_data
            data_dict[key]["qps"].append(result.get("qps"))
            data_dict[key]["latency_improvement"].append(result.get("latency_improvement", 0))
            data_dict[key]["accuracy"].append(result.get("accuracy", 0))
            data_dict[key]["exit_rate"].append(result.get("exit_rate", 0))
    
    # Create comparison plots for each dataset/arch
    all_keys = set(list(baseline_data.keys()) + list(optimized_data.keys()))
    
    for key in all_keys:
        if key not in baseline_data or key not in optimized_data:
            continue
        
        # Sort by QPS
        def sort_data(data_dict):
            sorted_indices = np.argsort(data_dict["qps"])
            return {
                "qps": np.array([data_dict["qps"][i] for i in sorted_indices]),
                "latency": np.array([data_dict["latency_improvement"][i] for i in sorted_indices]),
                "accuracy": np.array([data_dict["accuracy"][i] for i in sorted_indices]),
                "exit_rate": np.array([data_dict["exit_rate"][i] for i in sorted_indices]) * 100
            }
        
        baseline_sorted = sort_data(baseline_data[key])
        optimized_sorted = sort_data(optimized_data[key])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle(f'QPS Sweep Analysis: {key.replace("_", " ").title()} (SLO=60ms)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Latency Improvement Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(baseline_sorted["qps"], baseline_sorted["latency"], 
                marker='o', linewidth=3, markersize=12, label='Baseline', 
                color=COLORS['baseline'], markerfacecolor='white', markeredgewidth=2)
        ax1.plot(optimized_sorted["qps"], optimized_sorted["latency"], 
                marker='s', linewidth=3, markersize=12, label='Optimized', 
                color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
        ax1.set_xlabel('QPS', fontweight='bold')
        ax1.set_ylabel('Latency Improvement (%)', fontweight='bold')
        ax1.set_title('Latency Improvement', fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax1.set_xticks(baseline_sorted["qps"])
        
        # 2. Accuracy Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(baseline_sorted["qps"], baseline_sorted["accuracy"], 
                marker='o', linewidth=3, markersize=12, label='Baseline', 
                color=COLORS['baseline'], markerfacecolor='white', markeredgewidth=2)
        ax2.plot(optimized_sorted["qps"], optimized_sorted["accuracy"], 
                marker='s', linewidth=3, markersize=12, label='Optimized', 
                color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
        ax2.set_xlabel('QPS', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Accuracy', fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax2.set_xticks(baseline_sorted["qps"])
        
        # 3. Exit Rate Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(baseline_sorted["qps"], baseline_sorted["exit_rate"], 
                marker='o', linewidth=3, markersize=12, label='Baseline', 
                color=COLORS['baseline'], markerfacecolor='white', markeredgewidth=2)
        ax3.plot(optimized_sorted["qps"], optimized_sorted["exit_rate"], 
                marker='s', linewidth=3, markersize=12, label='Optimized', 
                color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
        ax3.set_xlabel('QPS', fontweight='bold')
        ax3.set_ylabel('Early Exit Rate (%)', fontweight='bold')
        ax3.set_title('Early Exit Rate', fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax3.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax3.set_xticks(baseline_sorted["qps"])
        
        # 4. Improvement Delta (spans 2 columns)
        ax4 = fig.add_subplot(gs[1, :])
        qps_levels = baseline_sorted["qps"]
        deltas = optimized_sorted["latency"] - baseline_sorted["latency"]
        colors = [COLORS['improvement'] if d > 0 else COLORS['degradation'] for d in deltas]
        bars = ax4.bar(qps_levels, deltas, color=colors, alpha=0.8, width=25, 
                      edgecolor='black', linewidth=1.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax4.set_xlabel('QPS', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Latency Improvement Delta (%)', fontweight='bold', fontsize=12)
        ax4.set_title('Workflow Control Benefit: Optimized - Baseline', 
                     fontweight='bold', fontsize=13, pad=15)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
        ax4.set_xticks(qps_levels)
        
        # Add value labels on bars
        for qps, delta in zip(qps_levels, deltas):
            ax4.text(qps, delta + (0.3 if delta > 0 else -0.3), f'{delta:+.2f}%',
                    ha='center', va='bottom' if delta > 0 else 'top', 
                    fontsize=11, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        output_path = os.path.join(output_dir, f"qps_sweep_{key}.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved QPS sweep plot: {output_path}")


def create_slo_sweep_plot(results, output_dir="workflow_results/plots"):
    """Create beautiful side-by-side comparison plots for SLO sweep."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by dataset/arch, separate baseline and optimized
    baseline_data = defaultdict(lambda: {"slo": [], "latency_improvement": [], "accuracy": [], "exit_rate": []})
    optimized_data = defaultdict(lambda: {"slo": [], "latency_improvement": [], "accuracy": [], "exit_rate": []})
    
    for result in results:
        if result.get("returncode") != 0:
            continue
        
        # Only SLO sweep (QPS fixed at 90, or 60 for backward compatibility)
        qps = result.get("qps")
        if qps == 90 or qps == 60:
            config = result.get("config")
            dataset = result.get("dataset")
            arch = result.get("arch")
            key = f"{dataset}_{arch}"
            
            data_dict = baseline_data if config == "baseline" else optimized_data
            data_dict[key]["slo"].append(result.get("slo"))
            data_dict[key]["latency_improvement"].append(result.get("latency_improvement", 0))
            data_dict[key]["accuracy"].append(result.get("accuracy", 0))
            data_dict[key]["exit_rate"].append(result.get("exit_rate", 0))
    
    # Create comparison plots for each dataset/arch
    all_keys = set(list(baseline_data.keys()) + list(optimized_data.keys()))
    
    for key in all_keys:
        if key not in baseline_data or key not in optimized_data:
            continue
        
        # Sort by SLO
        def sort_data(data_dict):
            sorted_indices = np.argsort(data_dict["slo"])
            return {
                "slo": np.array([data_dict["slo"][i] for i in sorted_indices]),
                "latency": np.array([data_dict["latency_improvement"][i] for i in sorted_indices]),
                "accuracy": np.array([data_dict["accuracy"][i] for i in sorted_indices]),
                "exit_rate": np.array([data_dict["exit_rate"][i] for i in sorted_indices]) * 100
            }
        
        baseline_sorted = sort_data(baseline_data[key])
        optimized_sorted = sort_data(optimized_data[key])
        
        # Get QPS from results for title
        qps_val = None
        for result in results:
            if result.get("dataset") == key.split("_")[0] and result.get("arch") == "_".join(key.split("_")[1:]):
                qps_val = result.get("qps")
                break
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        qps_str = f"QPS={qps_val}" if qps_val else "QPS=90"
        fig.suptitle(f'SLO Sweep Analysis: {key.replace("_", " ").title()} ({qps_str})', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # 1. Latency Improvement Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(baseline_sorted["slo"], baseline_sorted["latency"], 
                marker='o', linewidth=3, markersize=12, label='Baseline', 
                color=COLORS['baseline'], markerfacecolor='white', markeredgewidth=2)
        ax1.plot(optimized_sorted["slo"], optimized_sorted["latency"], 
                marker='s', linewidth=3, markersize=12, label='Optimized', 
                color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
        ax1.set_xlabel('SLO (ms)', fontweight='bold')
        ax1.set_ylabel('Latency Improvement (%)', fontweight='bold')
        ax1.set_title('Latency Improvement', fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax1.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax1.set_xticks(baseline_sorted["slo"])
        
        # 2. Accuracy Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(baseline_sorted["slo"], baseline_sorted["accuracy"], 
                marker='o', linewidth=3, markersize=12, label='Baseline', 
                color=COLORS['baseline'], markerfacecolor='white', markeredgewidth=2)
        ax2.plot(optimized_sorted["slo"], optimized_sorted["accuracy"], 
                marker='s', linewidth=3, markersize=12, label='Optimized', 
                color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
        ax2.set_xlabel('SLO (ms)', fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontweight='bold')
        ax2.set_title('Accuracy', fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax2.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax2.set_xticks(baseline_sorted["slo"])
        
        # 3. Exit Rate Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(baseline_sorted["slo"], baseline_sorted["exit_rate"], 
                marker='o', linewidth=3, markersize=12, label='Baseline', 
                color=COLORS['baseline'], markerfacecolor='white', markeredgewidth=2)
        ax3.plot(optimized_sorted["slo"], optimized_sorted["exit_rate"], 
                marker='s', linewidth=3, markersize=12, label='Optimized', 
                color=COLORS['optimized'], markerfacecolor='white', markeredgewidth=2)
        ax3.set_xlabel('SLO (ms)', fontweight='bold')
        ax3.set_ylabel('Early Exit Rate (%)', fontweight='bold')
        ax3.set_title('Early Exit Rate', fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=1)
        ax3.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax3.set_xticks(baseline_sorted["slo"])
        
        # 4. Improvement Delta (spans 2 columns)
        ax4 = fig.add_subplot(gs[1, :])
        slo_levels = baseline_sorted["slo"]
        deltas = optimized_sorted["latency"] - baseline_sorted["latency"]
        colors = [COLORS['improvement'] if d > 0 else COLORS['degradation'] for d in deltas]
        bars = ax4.bar(slo_levels, deltas, color=colors, alpha=0.8, width=20, 
                      edgecolor='black', linewidth=1.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax4.set_xlabel('SLO (ms)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Latency Improvement Delta (%)', fontweight='bold', fontsize=12)
        ax4.set_title('Workflow Control Benefit: Optimized - Baseline', 
                     fontweight='bold', fontsize=13, pad=15)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
        ax4.set_xticks(slo_levels)
        
        # Add value labels on bars
        for slo, delta in zip(slo_levels, deltas):
            ax4.text(slo, delta + (0.3 if delta > 0 else -0.3), f'{delta:+.2f}%',
                    ha='center', va='bottom' if delta > 0 else 'top', 
                    fontsize=11, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        output_path = os.path.join(output_dir, f"slo_sweep_{key}.pdf")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved SLO sweep plot: {output_path}")


def create_baseline_vs_optimized_plot(results, output_dir="workflow_results/plots"):
    """Create beautiful side-by-side bar charts comparing baseline vs optimized."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Separate QPS and SLO sweep results
    qps_sweep_results = [r for r in results if r.get("slo") == 60 and r.get("returncode") == 0]
    slo_sweep_results = [r for r in results if (r.get("qps") == 90 or r.get("qps") == 60) and r.get("returncode") == 0]
    
    # Create combined figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Baseline vs Optimized Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    # QPS Sweep Comparison
    if qps_sweep_results:
        qps_levels = sorted(set([r.get("qps") for r in qps_sweep_results]))
        baseline_qps = {qps: [] for qps in qps_levels}
        optimized_qps = {qps: [] for qps in qps_levels}
        
        for result in qps_sweep_results:
            qps = result.get("qps")
            if result.get("config") == "baseline":
                baseline_qps[qps].append(result.get("latency_improvement", 0))
            else:
                optimized_qps[qps].append(result.get("latency_improvement", 0))
        
        ax = axes[0]
        x = np.arange(len(qps_levels))
        width = 0.35
        
        baseline_vals = [np.mean(baseline_qps[q]) if baseline_qps[q] else 0 for q in qps_levels]
        optimized_vals = [np.mean(optimized_qps[q]) if optimized_qps[q] else 0 for q in qps_levels]
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
                      alpha=0.9, color=COLORS['baseline'], edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized', 
                      alpha=0.9, color=COLORS['optimized'], edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                       f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('QPS', fontweight='bold', fontsize=12)
        ax.set_ylabel('Latency Improvement (%)', fontweight='bold', fontsize=12)
        ax.set_title('QPS Sweep (SLO=60ms)', fontweight='bold', fontsize=13, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(qps_levels)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
        ax.set_ylim(bottom=0)
    
    # SLO Sweep Comparison
    if slo_sweep_results:
        slo_levels = sorted(set([r.get("slo") for r in slo_sweep_results]))
        baseline_slo = {slo: [] for slo in slo_levels}
        optimized_slo = {slo: [] for slo in slo_levels}
        
        for result in slo_sweep_results:
            slo = result.get("slo")
            if result.get("config") == "baseline":
                baseline_slo[slo].append(result.get("latency_improvement", 0))
            else:
                optimized_slo[slo].append(result.get("latency_improvement", 0))
        
        ax = axes[1]
        x = np.arange(len(slo_levels))
        width = 0.35
        
        baseline_vals = [np.mean(baseline_slo[s]) if baseline_slo[s] else 0 for s in slo_levels]
        optimized_vals = [np.mean(optimized_slo[s]) if optimized_slo[s] else 0 for s in slo_levels]
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', 
                      alpha=0.9, color=COLORS['baseline'], edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized', 
                      alpha=0.9, color=COLORS['optimized'], edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                       f'{height:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('SLO (ms)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Latency Improvement (%)', fontweight='bold', fontsize=12)
        qps_val = slo_sweep_results[0].get("qps") if slo_sweep_results else 90
        ax.set_title(f'SLO Sweep (QPS={qps_val})', fontweight='bold', fontsize=13, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(slo_levels)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "baseline_vs_optimized_comparison.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved baseline vs optimized comparison: {output_path}")


def create_improvement_delta_plot(results, output_dir="workflow_results/plots"):
    """Create plot showing improvement delta (optimized - baseline)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by QPS/SLO
    qps_deltas = defaultdict(list)
    slo_deltas = defaultdict(list)
    
    # Process QPS sweep
    qps_results = [r for r in results if r.get("slo") == 60 and r.get("returncode") == 0]
    baseline_qps = {}
    optimized_qps = {}
    
    for result in qps_results:
        qps = result.get("qps")
        if result.get("config") == "baseline":
            baseline_qps[qps] = result.get("latency_improvement", 0)
        else:
            optimized_qps[qps] = result.get("latency_improvement", 0)
    
    # Calculate deltas for QPS
    for qps in set(list(baseline_qps.keys()) + list(optimized_qps.keys())):
        if qps in baseline_qps and qps in optimized_qps:
            delta = optimized_qps[qps] - baseline_qps[qps]
            qps_deltas[qps] = delta
    
    # Process SLO sweep
    slo_results = [r for r in results if (r.get("qps") == 90 or r.get("qps") == 60) and r.get("returncode") == 0]
    baseline_slo = {}
    optimized_slo = {}
    
    for result in slo_results:
        slo = result.get("slo")
        if result.get("config") == "baseline":
            baseline_slo[slo] = result.get("latency_improvement", 0)
        else:
            optimized_slo[slo] = result.get("latency_improvement", 0)
    
    # Calculate deltas for SLO
    for slo in set(list(baseline_slo.keys()) + list(optimized_slo.keys())):
        if slo in baseline_slo and slo in optimized_slo:
            delta = optimized_slo[slo] - baseline_slo[slo]
            slo_deltas[slo] = delta
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Workflow Control Improvement Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # QPS improvement delta
    if qps_deltas:
        ax = axes[0]
        qps_levels = sorted(qps_deltas.keys())
        deltas = [qps_deltas[q] for q in qps_levels]
        colors = [COLORS['improvement'] if d > 0 else COLORS['degradation'] for d in deltas]
        bars = ax.bar(qps_levels, deltas, color=colors, alpha=0.9, width=25, 
                     edgecolor='black', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.set_xlabel('QPS', fontweight='bold', fontsize=12)
        ax.set_ylabel('Improvement Delta (%)', fontweight='bold', fontsize=12)
        ax.set_title('QPS Sweep (SLO=60ms)\nOptimized - Baseline', 
                    fontweight='bold', fontsize=13, pad=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
        for qps, delta in zip(qps_levels, deltas):
            ax.text(qps, delta + (0.5 if delta > 0 else -0.5), f'{delta:+.2f}%',
                   ha='center', va='bottom' if delta > 0 else 'top', 
                   fontsize=11, fontweight='bold')
        ax.set_ylim(bottom=min(0, min(deltas) - 1), top=max(deltas) + 1)
    
    # SLO improvement delta
    if slo_deltas:
        ax = axes[1]
        slo_levels = sorted(slo_deltas.keys())
        deltas = [slo_deltas[s] for s in slo_levels]
        colors = [COLORS['improvement'] if d > 0 else COLORS['degradation'] for d in deltas]
        bars = ax.bar(slo_levels, deltas, color=colors, alpha=0.9, width=20, 
                     edgecolor='black', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.set_xlabel('SLO (ms)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Improvement Delta (%)', fontweight='bold', fontsize=12)
        qps_val = slo_results[0].get("qps") if slo_results else 90
        ax.set_title(f'SLO Sweep (QPS={qps_val})\nOptimized - Baseline', 
                    fontweight='bold', fontsize=13, pad=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
        for slo, delta in zip(slo_levels, deltas):
            ax.text(slo, delta + (0.5 if delta > 0 else -0.5), f'{delta:+.2f}%',
                   ha='center', va='bottom' if delta > 0 else 'top', 
                   fontsize=11, fontweight='bold')
        ax.set_ylim(bottom=min(0, min(deltas) - 1), top=max(deltas) + 1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "improvement_delta.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved improvement delta plot: {output_path}")


def create_mixed_experiments_plot(results, output_dir="workflow_results/plots"):
    """Create visualization for mixed QPS/SLO experiments."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter mixed experiments (not in QPS sweep or SLO sweep)
    mixed_results = []
    qps_sweep_slo = 60
    slo_sweep_qps = {90, 60}  # Support both 90 and 60 for backward compatibility
    
    for result in results:
        if result.get("returncode") != 0:
            continue
        qps = result.get("qps")
        slo = result.get("slo")
        # Mixed if not in standard sweeps
        if slo != qps_sweep_slo and qps not in slo_sweep_qps:
            mixed_results.append(result)
    
    if not mixed_results:
        print("  No mixed experiments found, skipping...")
        return
    
    # Group by config
    baseline_mixed = []
    optimized_mixed = []
    
    for result in mixed_results:
        if result.get("config") == "baseline":
            baseline_mixed.append(result)
        else:
            optimized_mixed.append(result)
    
    # Create heatmap-style visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Mixed Experiments: QPS × SLO Combinations', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (data, config, ax) in enumerate([(baseline_mixed, "Baseline", axes[0]), 
                                               (optimized_mixed, "Optimized", axes[1])]):
        if not data:
            continue
        
        # Create matrix for heatmap
        qps_levels = sorted(set([r.get("qps") for r in data]))
        slo_levels = sorted(set([r.get("slo") for r in data]))
        
        matrix = np.zeros((len(slo_levels), len(qps_levels)))
        for result in data:
            qps = result.get("qps")
            slo = result.get("slo")
            qps_idx = qps_levels.index(qps)
            slo_idx = slo_levels.index(slo)
            matrix[slo_idx, qps_idx] = result.get("latency_improvement", 0)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(20, matrix.max()))
        
        # Set ticks
        ax.set_xticks(np.arange(len(qps_levels)))
        ax.set_yticks(np.arange(len(slo_levels)))
        ax.set_xticklabels(qps_levels)
        ax.set_yticklabels(slo_levels)
        
        # Add text annotations
        for i in range(len(slo_levels)):
            for j in range(len(qps_levels)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}%',
                             ha="center", va="center", color="black", fontweight='bold', fontsize=10)
        
        ax.set_xlabel('QPS', fontweight='bold', fontsize=12)
        ax.set_ylabel('SLO (ms)', fontweight='bold', fontsize=12)
        ax.set_title(f'{config} Configuration', fontweight='bold', fontsize=13, pad=10)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Latency Improvement (%)', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "mixed_experiments_heatmap.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved mixed experiments plot: {output_path}")


def find_latest_results_folder(base_dir="workflow_results"):
    """Find the most recent experiment results folder."""
    if not os.path.exists(base_dir):
        return None
    
    # Look for folders matching exp-*, qps_sweep-*, slo_sweep-* patterns
    folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and (
            item.startswith("exp-") or 
            item.startswith("qps_sweep-") or 
            item.startswith("slo_sweep-")
        ):
            folders.append((item_path, os.path.getmtime(item_path)))
    
    if not folders:
        return None
    
    # Return the most recently modified folder
    folders.sort(key=lambda x: x[1], reverse=True)
    return folders[0][0]


def main():
    """Main function to generate all visualizations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize QPS/SLO sweep results")
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Directory containing experiment results (default: auto-detect latest)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots (default: auto-create timestamped folder)")
    args = parser.parse_args()
    
    print("="*60)
    print("Generating Visualizations for QPS/SLO Sweep Results")
    print("="*60)
    
    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # Auto-detect latest results folder
        results_dir = find_latest_results_folder()
        if not results_dir:
            print("Error: No results folder found in workflow_results/")
            print("Please specify --results_dir or run experiments first")
            return
        print(f"Auto-detected results folder: {results_dir}")
    
    # Create timestamped output directory for plots
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"workflow_results/plots_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to: {output_dir}")
    
    # Load results
    print(f"\n[1/5] Loading experiment results from {results_dir}...")
    results = load_sweep_results(results_dir)
    print(f"Loaded {len(results)} experiment results")
    
    successful = [r for r in results if r.get("returncode") == 0]
    print(f"Successful experiments: {len(successful)}")
    
    if len(successful) == 0:
        print("Error: No successful experiments found!")
        return
    
    # Generate plots
    print("\n[2/5] Creating QPS sweep plots...")
    create_qps_sweep_plot(results, output_dir)
    
    print("\n[3/5] Creating SLO sweep plots...")
    create_slo_sweep_plot(results, output_dir)
    
    print("\n[4/5] Creating comparison plots...")
    create_baseline_vs_optimized_plot(results, output_dir)
    create_improvement_delta_plot(results, output_dir)
    
    print("\n[5/5] Creating mixed experiments visualization...")
    create_mixed_experiments_plot(results, output_dir)
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print(f"Plots saved to: {output_dir}")
    print("="*60)
    
    # Print summary
    print("\nGenerated plots:")
    plot_files = glob.glob(os.path.join(output_dir, "*.pdf"))
    for plot_file in sorted(plot_files):
        print(f"  - {os.path.basename(plot_file)}")


if __name__ == "__main__":
    main()
