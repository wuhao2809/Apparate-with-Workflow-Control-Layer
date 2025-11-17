"""
Experiment runner for workflow control layer evaluation.

Runs baseline (without workflow) and optimized (with workflow) experiments
for CV and NLP workloads suitable for Mac M1 (simulation mode).
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path


def run_experiment(config_name, dataset, arch, enable_workflow=False, 
                  qps=30, slo=45, output_dir="workflow_results"):
    """
    Run a single experiment configuration.
    
    Args:
        config_name: Name for this configuration (e.g., "baseline", "optimized")
        dataset: Dataset name
        arch: Architecture name
        enable_workflow: Whether to enable workflow control layer
        qps: Queries per second
        slo: SLO in ms
        output_dir: Directory to store results
    """
    print(f"\n{'='*60}")
    print(f"Running: {config_name} - {dataset} - {arch}")
    print(f"Workflow enabled: {enable_workflow}")
    print(f"QPS: {qps}, SLO: {slo}ms")
    print(f"{'='*60}\n")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Build command - use venv Python explicitly
    python_cmd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "bin", "python")
    if not os.path.exists(python_cmd):
        python_cmd = "python"  # Fallback to system python
    
    cmd_parts = [
        python_cmd, "controller.py",
        f"--dataset", dataset,
        f"--arch", arch,
        f"--batch_size", "8",
        f"--profile_dir", "profile_pickles_bs",
        f"--batching_scheme", "clockwork",
        f"--slo", str(slo),
        f"--qps", str(qps),
        f"--simulation_pickle_path", f"../simulation_pickles/{dataset}_{arch}.pickle",
        f"--bootstrap_pickle_path", f"../bootstrap_pickles/bootstrap_{dataset}_{arch}.pickle",
    ]
    
    # Batch decision path format differs for CV vs NLP
    # CV: {arch}_{slo_multiplier}_fixed_{qps}.pickle
    # NLP: {arch}_{slo_multiplier}_fixed.pickle or {arch}_azure.pickle
    if "bert" in arch.lower() or "gpt" in arch.lower():
        batch_decision_path = f"../batch_decisions/{arch}_azure.pickle"  # NLP uses azure trace
    else:
        batch_decision_path = f"../batch_decisions/{arch}_1_fixed_{int(qps) if qps <= 30 else 30}.pickle"
    
    cmd_parts.extend([
        f"--batch_decision_path", batch_decision_path
    ])
    
    if enable_workflow:
        cmd_parts.extend([
            "--enable_workflow",
            "--enable_prioritization",
            "--enable_adaptive_batching",
            "--enable_feedback"
        ])
    
    # Use subprocess list format to avoid shell quoting issues
    # Run experiment
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd_parts,
            shell=False,  # Use list format instead of shell=True
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        elapsed_time = time.time() - start_time
        
        # Save results
        result_file = os.path.join(
            output_dir,
            f"{config_name}_{dataset}_{arch}_qps{qps}_slo{slo}.json"
        )
        
        result_data = {
            "config_name": config_name,
            "dataset": dataset,
            "arch": arch,
            "enable_workflow": enable_workflow,
            "qps": qps,
            "slo": slo,
            "elapsed_time": elapsed_time,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
            "stderr": result.stderr[-1000:] if result.stderr else ""
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        if result.returncode == 0:
            print(f"✓ Experiment completed in {elapsed_time:.1f}s")
        else:
            print(f"✗ Experiment failed with return code {result.returncode}")
            print(f"Error: {result.stderr[-500:]}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"✗ Experiment timed out after 1 hour")
        return False
    except Exception as e:
        print(f"✗ Experiment failed with error: {e}")
        return False


def run_workflow_experiments():
    """Run all workflow control experiments."""
    
    # Experiment configurations
    # Note: Using available datasets from downloaded data files
    experiments = [
        # CV experiments - Using available urban dataset with ResNet18
        {
            "dataset": "auburn",  # Available in simulation_pickles
            "arch": "resnet18",   # Matches auburn_resnet18.pickle
            "qps": 30,
            "slo": 45,
            "configs": ["baseline", "optimized"]
        },
        {
            "dataset": "auburn",
            "arch": "resnet18",
            "qps": 60,
            "slo": 45,
            "configs": ["baseline", "optimized"]
        },
        
        # NLP experiments - Using available IMDB dataset with DistilBERT
        {
            "dataset": "imdb",  # Available in simulation_pickles
            "arch": "distilbert-base",  # Matches imdb_distilbert-base.pickle
            "qps": 30,
            "slo": 50,
            "configs": ["baseline", "optimized"]
        },
    ]
    
    results_summary = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "experiments": []
    }
    
    print("="*60)
    print("Workflow Control Layer Experiments")
    print("="*60)
    print(f"Total experiments: {sum(len(exp['configs']) for exp in experiments)}")
    print("="*60)
    
    for exp in experiments:
        for config in exp["configs"]:
            enable_workflow = (config == "optimized")
            
            results_summary["total"] += 1
            success = run_experiment(
                config_name=config,
                dataset=exp["dataset"],
                arch=exp["arch"],
                enable_workflow=enable_workflow,
                qps=exp["qps"],
                slo=exp["slo"]
            )
            
            if success:
                results_summary["passed"] += 1
            else:
                results_summary["failed"] += 1
            
            results_summary["experiments"].append({
                "config": config,
                "dataset": exp["dataset"],
                "arch": exp["arch"],
                "qps": exp["qps"],
                "slo": exp["slo"],
                "success": success
            })
            
            # Small delay between experiments
            time.sleep(2)
    
    # Save summary
    summary_file = "workflow_results/experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    print(f"Total: {results_summary['total']}")
    print(f"Passed: {results_summary['passed']}")
    print(f"Failed: {results_summary['failed']}")
    print(f"Summary saved to: {summary_file}")
    print("="*60)
    
    return results_summary


if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("controller.py"):
        print("Error: Must run from apparate directory")
        print("Usage: cd apparate && python run_workflow_experiments.py")
        sys.exit(1)
    
    # Run experiments
    results = run_workflow_experiments()
    
    if results["failed"] > 0:
        sys.exit(1)
    else:
        print("\n✓ All experiments completed successfully!")
        sys.exit(0)

