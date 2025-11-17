"""
Analyze workflow control layer experiment results.

Generates metrics, visualizations, and comparison reports.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import glob


def load_latency_data(pickle_path):
    """Load latency data from pickle file."""
    if not os.path.exists(pickle_path):
        return None
    try:
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return None


def parse_log_file(log_path):
    """Parse log file to extract metrics."""
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
            # Look for the final summary line
            for line in reversed(lines):
                if "Serving with complete" in line:
                    # Parse: "overall accuracy X%, overall serving latency improvement Y%"
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
                                metrics["exit_rate"] = float(
                                    part.split("overall exit rate")[1].split(",")[0].strip()
                                )
                            except:
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
        print(f"Error parsing log {log_path}: {e}")
    
    return metrics


def calculate_latency_metrics(latencies):
    """Calculate latency statistics."""
    if latencies is None or len(latencies) == 0:
        return None
    
    latencies_array = np.array(latencies)
    
    return {
        "median": np.median(latencies_array),
        "mean": np.mean(latencies_array),
        "p95": np.percentile(latencies_array, 95),
        "p99": np.percentile(latencies_array, 99),
        "min": np.min(latencies_array),
        "max": np.max(latencies_array),
        "std": np.std(latencies_array)
    }


def plot_latency_cdf(baseline_latencies, optimized_latencies, output_path, title=""):
    """Plot latency CDF comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if baseline_latencies and len(baseline_latencies) > 0:
        sorted_baseline = np.sort(baseline_latencies)
        y_baseline = np.arange(1, len(sorted_baseline) + 1) / len(sorted_baseline)
        ax.plot(sorted_baseline, y_baseline * 100, label='Baseline', linewidth=2)
    
    if optimized_latencies and len(optimized_latencies) > 0:
        sorted_optimized = np.sort(optimized_latencies)
        y_optimized = np.arange(1, len(sorted_optimized) + 1) / len(sorted_optimized)
        ax.plot(sorted_optimized, y_optimized * 100, label='Optimized (Workflow)', linewidth=2)
    
    ax.set_xlabel('Latency (ms)', fontsize=12)
    ax.set_ylabel('CDF (%)', fontsize=12)
    ax.set_title(title or 'Latency CDF Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CDF plot: {output_path}")


def plot_metric_comparison(metrics_dict, output_path):
    """Plot metric comparison bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    datasets = list(metrics_dict.keys())
    
    # Latency improvement
    ax = axes[0, 0]
    baseline_improvements = [metrics_dict[d].get("baseline", {}).get("latency_improvement", 0) or 0
                            for d in datasets]
    optimized_improvements = [metrics_dict[d].get("optimized", {}).get("latency_improvement", 0) or 0
                             for d in datasets]
    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width/2, baseline_improvements, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, optimized_improvements, width, label='Optimized', alpha=0.8)
    ax.set_ylabel('Latency Improvement (%)', fontsize=11)
    ax.set_title('Latency Improvement Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Accuracy
    ax = axes[0, 1]
    baseline_acc = [metrics_dict[d].get("baseline", {}).get("overall_accuracy", 0) or 0
                    for d in datasets]
    optimized_acc = [metrics_dict[d].get("optimized", {}).get("overall_accuracy", 0) or 0
                     for d in datasets]
    ax.bar(x - width/2, baseline_acc, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, optimized_acc, width, label='Optimized', alpha=0.8)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # P95 Latency
    ax = axes[1, 0]
    baseline_p95 = [metrics_dict[d].get("baseline", {}).get("p95_latency", 0) or 0
                   for d in datasets]
    optimized_p95 = [metrics_dict[d].get("optimized", {}).get("p95_latency", 0) or 0
                    for d in datasets]
    ax.bar(x - width/2, baseline_p95, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, optimized_p95, width, label='Optimized', alpha=0.8)
    ax.set_ylabel('P95 Latency (ms)', fontsize=11)
    ax.set_title('Tail Latency (P95) Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Exit Rate
    ax = axes[1, 1]
    baseline_exit = [metrics_dict[d].get("baseline", {}).get("exit_rate", 0) or 0
                    for d in datasets]
    optimized_exit = [metrics_dict[d].get("optimized", {}).get("exit_rate", 0) or 0
                      for d in datasets]
    ax.bar(x - width/2, baseline_exit, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, optimized_exit, width, label='Optimized', alpha=0.8)
    ax.set_ylabel('Exit Rate', fontsize=11)
    ax.set_title('Early Exit Rate Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metric comparison: {output_path}")


def analyze_results():
    """Analyze all experiment results."""
    
    results_dir = "workflow_results"
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load experiment summary
    summary_path = os.path.join(results_dir, "experiment_summary.json")
    if not os.path.exists(summary_path):
        print(f"Error: Experiment summary not found at {summary_path}")
        print("Please run run_workflow_experiments.py first")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Organize results by dataset
    metrics_by_dataset = defaultdict(lambda: {"baseline": {}, "optimized": {}})
    
    # Process each experiment
    for exp in summary["experiments"]:
        if not exp["success"]:
            continue
        
        dataset = exp["dataset"]
        arch = exp["arch"]
        config = exp["config"]
        qps = exp["qps"]
        slo = exp["slo"]
        
        # Load latency data - use config-specific suffix
        workflow_suffix = "_workflow" if config == "optimized" else "_baseline"
        # Try both CV and NLP formats
        latency_file = None
        # Try CV format first (with slo and qps)
        cv_file = f"../apparate_latency/{arch}_{dataset}_{int(slo)}_fixed_{int(qps)}{workflow_suffix}.pickle"
        # Try NLP format (azure)
        nlp_file = f"../apparate_latency/{arch}_{dataset}_azure{workflow_suffix}.pickle"
        
        if os.path.exists(cv_file):
            latency_file = cv_file
        elif os.path.exists(nlp_file):
            latency_file = nlp_file
        else:
            # Fallback: try without suffix (old format)
            latency_file = f"../apparate_latency/{arch}_{dataset}_azure.pickle"
        
        latencies = load_latency_data(latency_file)
        
        # Load log metrics - use config-specific suffix
        log_file = f"logs/output_{arch}_{dataset}{workflow_suffix}.log"
        log_metrics = parse_log_file(log_file)
        
        # Calculate latency stats
        latency_stats = calculate_latency_metrics(latencies)
        
        # Store metrics
        metrics = {
            "latency_improvement": log_metrics.get("latency_improvement", 0) if log_metrics else 0,
            "overall_accuracy": log_metrics.get("overall_accuracy", 0) if log_metrics else 0,
            "exit_rate": log_metrics.get("exit_rate", 0) if log_metrics else 0,
            "overall_ramp_accuracy": log_metrics.get("overall_ramp_accuracy", 0) if log_metrics else 0,
        }
        
        if latency_stats:
            metrics.update({
                "median_latency": latency_stats["median"],
                "mean_latency": latency_stats["mean"],
                "p95_latency": latency_stats["p95"],
                "p99_latency": latency_stats["p99"]
            })
        
        # Convert latencies to list if it's a numpy array
        latencies_list = []
        if latencies is not None:
            if isinstance(latencies, np.ndarray):
                latencies_list = latencies.tolist()
            elif isinstance(latencies, list):
                latencies_list = latencies
            else:
                # Try to convert to list
                try:
                    latencies_list = list(latencies)
                except:
                    latencies_list = []
        
        metrics_by_dataset[f"{dataset}_{arch}"][config] = {
            **metrics,
            "latencies": latencies_list
        }
    
    # Generate plots
    for dataset_key, metrics in metrics_by_dataset.items():
        baseline_latencies = metrics["baseline"].get("latencies", [])
        optimized_latencies = metrics["optimized"].get("latencies", [])
        
        if baseline_latencies or optimized_latencies:
            plot_path = os.path.join(plots_dir, f"{dataset_key}_latency_cdf.pdf")
            plot_latency_cdf(
                baseline_latencies,
                optimized_latencies,
                plot_path,
                title=f"Latency CDF: {dataset_key}"
            )
    
    # Overall comparison
    if metrics_by_dataset:
        comparison_path = os.path.join(plots_dir, "metric_comparison.pdf")
        plot_metric_comparison(metrics_by_dataset, comparison_path)
    
    # Generate report
    report_path = os.path.join(results_dir, "analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(metrics_by_dataset, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Analysis Summary")
    print("="*60)
    
    for dataset_key, metrics in metrics_by_dataset.items():
        print(f"\n{dataset_key}:")
        baseline = metrics.get("baseline", {})
        optimized = metrics.get("optimized", {})
        
        if baseline and optimized:
            latency_improvement_diff = (
                optimized.get("latency_improvement", 0) - 
                baseline.get("latency_improvement", 0)
            )
            p95_reduction = (
                (baseline.get("p95_latency", 0) - optimized.get("p95_latency", 0)) /
                baseline.get("p95_latency", 1) * 100
            ) if baseline.get("p95_latency", 0) > 0 else 0
            
            print(f"  Latency improvement: {baseline.get('latency_improvement', 0):.2f}% → "
                  f"{optimized.get('latency_improvement', 0):.2f}% "
                  f"(+{latency_improvement_diff:.2f}%)")
            print(f"  P95 latency: {baseline.get('p95_latency', 0):.2f}ms → "
                  f"{optimized.get('p95_latency', 0):.2f}ms "
                  f"({p95_reduction:.2f}% reduction)")
            print(f"  Accuracy: {baseline.get('overall_accuracy', 0):.2f}% → "
                  f"{optimized.get('overall_accuracy', 0):.2f}%")
    
    print(f"\nPlots saved to: {plots_dir}")
    print(f"Report saved to: {report_path}")
    print("="*60)


if __name__ == "__main__":
    analyze_results()

