"""
Enhanced Component Analysis Script

This script tests all three workflow components with proper conditions:
1. Prioritization: Uses mixed SLOs to show effect
2. Adaptive Batching: Implemented in batch decision generation
3. Feedback Control: Already works (adjusts thresholds dynamically)

Key differences from standard component analysis:
- Uses mixed SLO distribution (30ms, 60ms, 120ms) for prioritization testing
- Implements adaptive batching in batch decision generation
- Ensures all components can demonstrate their effects
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path

# Import batch_systems and utils if available
try:
    import batch_systems
    import utils
except ImportError:
    batch_systems = None
    utils = None
    print("Warning: batch_systems or utils not available. Batch decision generation will be limited.")


def ensure_batch_decision_with_mixed_slo(arch, dataset, qps, slo_distribution, 
                                         batching_scheme="clockwork",
                                         profile_dir="../profile_pickles_bs",
                                         batch_decisions_dir="../batch_decisions",
                                         total_num_requests=70000,
                                         enable_prioritization=True,
                                         enable_adaptive_batching=False):
    """
    Generate batch decision with mixed SLOs for proper prioritization testing.
    
    Args:
        arch: Architecture name
        dataset: Dataset name
        qps: Queries per second
        slo_distribution: Dict mapping SLO values to probabilities
                         e.g., {30: 0.3, 60: 0.4, 120: 0.3}
        batching_scheme: Batching scheme
        profile_dir: Profile directory
        batch_decisions_dir: Batch decisions directory
        total_num_requests: Total number of requests
        enable_prioritization: Whether to enable prioritization
        enable_adaptive_batching: Whether to enable adaptive batching
        
    Returns:
        Path to batch decision file
    """
    if batch_systems is None or utils is None:
        raise ImportError("Cannot generate batch decisions: batch_systems or utils not available.")
    
    # Create feature suffix for filename
    feature_suffix = ""
    if not enable_prioritization and not enable_adaptive_batching:
        feature_suffix = "_baseline"
    elif enable_prioritization and not enable_adaptive_batching:
        feature_suffix = "_prio"
    elif not enable_prioritization and enable_adaptive_batching:
        feature_suffix = "_batch"
    elif enable_prioritization and enable_adaptive_batching:
        feature_suffix = "_prio_batch"
    
    # Create SLO distribution string for filename
    slo_str = "_".join([f"{k}_{int(v*100)}" for k, v in sorted(slo_distribution.items())])
    
    # Determine filename
    if "bert" in arch.lower() or "gpt" in arch.lower():
        filename = f"{batching_scheme}_{arch}_mixed_{slo_str}_qps{round(qps)}{feature_suffix}.pickle"
    else:
        filename = f"{batching_scheme}_{arch}_mixed_{slo_str}_qps{int(qps)}{feature_suffix}.pickle"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(batch_decisions_dir):
        batch_decisions_dir = os.path.join(os.path.dirname(script_dir), batch_decisions_dir.lstrip("../"))
    
    batch_decision_path = os.path.join(batch_decisions_dir, filename)
    
    # Check if file exists
    if os.path.exists(batch_decision_path):
        print(f"[LOG] Using existing batch decision: {batch_decision_path}")
        rel_path = os.path.relpath(batch_decision_path, script_dir)
        return rel_path if not rel_path.startswith("..") else batch_decision_path
    
    # Generate batch decision with mixed SLOs
    print(f"[LOG] Generating batch decision with mixed SLOs...")
    print(f"[LOG]   SLO distribution: {slo_distribution}")
    print(f"[LOG]   Prioritization: {enable_prioritization}")
    print(f"[LOG]   Adaptive Batching: {enable_adaptive_batching}")
    print(f"[LOG] This may take a few minutes...")
    
    # Get model serving time
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(profile_dir):
            if not profile_dir.startswith(".."):
                project_root = os.path.dirname(script_dir)
                profile_dir = os.path.join(project_root, profile_dir)
            else:
                profile_dir = os.path.join(script_dir, profile_dir)
        
        if not os.path.exists(profile_dir):
            raise RuntimeError(f"Profile directory does not exist: {profile_dir}")
        
        latency_calc_list = []
        model_serving_time = batch_systems.get_model_serving_time(arch, profile_dir, latency_calc_list)
    except Exception as e:
        raise RuntimeError(f"Failed to load model serving time for {arch}: {e}")
    
    # Create requests with mixed SLOs
    all_requests, avg_qps, interarrival_time = batch_systems.create_mixed_slo_requests(
        num_requests=total_num_requests,
        slo_distribution=slo_distribution,
        qps=qps,
        poisson_arrival=True,
        seed=2023
    )
    
    print(f"[LOG] Created {len(all_requests)} requests with mixed SLOs")
    
    # Generate batch decision with feature flags
    print(f"[LOG] Generating batch decision...")
    max_batch_size, batch_timeout_ms, max_enqueued_batches, \
        batch_decision, per_request_stats, total_num_requests, total_time = \
        batch_systems.get_batch_decision(
            batching_scheme, all_requests, model_serving_time,
            slo=0.0,  # Not used when requests have individual SLOs
            max_batch_size=8, batch_timeout_ms=60, max_enqueued_batches=2,
            enable_prioritization=enable_prioritization,
            enable_adaptive_batching=enable_adaptive_batching
        )
    
    # Save batch decision
    os.makedirs(batch_decisions_dir, exist_ok=True)
    batch_info = {
        "batching_decision": batch_decision,
        "per_request_stats": per_request_stats,
        "total_num_requests": total_num_requests,
        "batching_scheme": batching_scheme,
        "arch": arch,
        "slo": slo_distribution,  # Store distribution instead of single SLO
        "dataset": dataset,
        "avg_qps": avg_qps,
        "end_time": total_time,
        "max_batch_size": max_batch_size,
        "batch_timeout_ms": batch_timeout_ms,
        "max_enqueued_batches": max_enqueued_batches,
        "enable_prioritization": enable_prioritization,
        "enable_adaptive_batching": enable_adaptive_batching,
    }
    
    with open(batch_decision_path, "wb") as f:
        import pickle
        pickle.dump(batch_info, f)
    
    print(f"[LOG] Batch decision saved to: {batch_decision_path}")
    rel_path = os.path.relpath(batch_decision_path, script_dir)
    return rel_path if not rel_path.startswith("..") else batch_decision_path


def run_experiment_enhanced(config_name, dataset, arch, enable_workflow=False,
                            qps=90, slo_distribution=None,
                            output_dir="workflow_results",
                            timeout=3600,
                            enable_prioritization=True,
                            enable_adaptive_batching=True,
                            enable_feedback=True):
    """
    Run a single experiment with mixed SLOs.
    
    Args:
        config_name: Configuration name
        dataset: Dataset name
        arch: Architecture name
        enable_workflow: Whether to enable workflow
        qps: Queries per second
        slo_distribution: SLO distribution dict (e.g., {30: 0.3, 60: 0.4, 120: 0.3})
        output_dir: Output directory
        timeout: Timeout in seconds
        enable_prioritization: Enable prioritization
        enable_adaptive_batching: Enable adaptive batching
        enable_feedback: Enable feedback control
    """
    import sys as sys_module
    
    if slo_distribution is None:
        # Default mixed SLO distribution
        slo_distribution = {30: 0.3, 60: 0.4, 120: 0.3}
    
    print(f"\n{'='*60}")
    print(f"Running: {config_name} - {dataset} - {arch}")
    print(f"Workflow enabled: {enable_workflow}")
    if enable_workflow:
        print(f"  Prioritization: {enable_prioritization}")
        print(f"  Adaptive Batching: {enable_adaptive_batching}")
        print(f"  Feedback: {enable_feedback}")
    print(f"QPS: {qps}, SLO Distribution: {slo_distribution}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    python_cmd = sys_module.executable
    
    # Use average SLO for the --slo parameter (controller expects a single value)
    avg_slo = sum(k * v for k, v in slo_distribution.items())
    
    cmd_parts = [
        python_cmd, "controller.py",
        f"--dataset", dataset,
        f"--arch", arch,
        f"--batch_size", "8",
        f"--profile_dir", "profile_pickles_bs",
        f"--batching_scheme", "clockwork",
        f"--slo", str(int(avg_slo)),  # Use average for compatibility
        f"--qps", str(qps),
        f"--simulation_pickle_path", f"../simulation_pickles/{dataset}_{arch}.pickle",
        f"--bootstrap_pickle_path", f"../bootstrap_pickles/bootstrap_{dataset}_{arch}.pickle",
    ]
    
    # Generate or get batch decision with mixed SLOs
    try:
        batch_decision_path = ensure_batch_decision_with_mixed_slo(
            arch=arch,
            dataset=dataset,
            qps=qps,
            slo_distribution=slo_distribution,
            batching_scheme="clockwork",
            profile_dir="../profile_pickles_bs",
            batch_decisions_dir="../batch_decisions",
            enable_prioritization=enable_prioritization if enable_workflow else False,
            enable_adaptive_batching=enable_adaptive_batching if enable_workflow else False
        )
        if os.path.isabs(batch_decision_path):
            batch_decision_path = os.path.relpath(batch_decision_path,
                                                 os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        print(f"[LOG] Warning: Could not generate batch decision: {e}")
        # Fallback - use a default path (won't work with mixed SLOs, but won't crash)
        batch_decision_path = f"../batch_decisions/{arch}_azure.pickle"
    
    cmd_parts.extend([
        f"--batch_decision_path", batch_decision_path
    ])
    
    if enable_workflow:
        cmd_parts.append("--enable_workflow")
        if not enable_prioritization:
            cmd_parts.append("--disable_prioritization")
        if not enable_adaptive_batching:
            cmd_parts.append("--disable_adaptive_batching")
        if not enable_feedback:
            cmd_parts.append("--disable_feedback")
    
    # Run experiment
    print(f"[LOG] Executing command: {' '.join(cmd_parts[:5])}... (truncated)")
    start_time = time.time()
    try:
        print(f"[LOG] Starting subprocess at {time.strftime('%H:%M:%S', time.localtime())}...")
        result = subprocess.run(
            cmd_parts,
            shell=False,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed_time = time.time() - start_time
        print(f"[LOG] Subprocess completed in {elapsed_time:.1f}s with return code {result.returncode}")
        
        # Parse metrics (reuse existing parsing logic)
        from run_comprehensive_experiments import parse_metrics_from_stdout
        metrics = parse_metrics_from_stdout(result.stdout) if result.stdout else {}
        
        if not metrics or all(v is None for v in metrics.values()):
            # Try log file
            import shutil
            log_suffix = "baseline" if config_name == "baseline" else "workflow"
            log_file = f"logs/output_{arch}_{dataset}_{log_suffix}.log"
            unique_log_file = os.path.join(
                output_dir,
                f"log_{config_name}_{dataset}_{arch}_mixed_qps{qps}.log"
            )
            time.sleep(0.5)
            if os.path.exists(log_file):
                try:
                    shutil.copy2(log_file, unique_log_file)
                    with open(unique_log_file, 'r') as f:
                        log_lines = f.readlines()
                    for line in reversed(log_lines):
                        if "Serving with complete" in line:
                            log_metrics = parse_metrics_from_stdout(line)
                            if log_metrics and any(v is not None for v in log_metrics.values()):
                                metrics = log_metrics
                            break
                except Exception as e:
                    print(f"[LOG] Warning: Could not parse log file: {e}")
        
        # Save results
        slo_str = "_".join([f"{k}_{int(v*100)}" for k, v in sorted(slo_distribution.items())])
        result_file = os.path.join(
            output_dir,
            f"{config_name}_{dataset}_{arch}_mixed_{slo_str}_qps{qps}.json"
        )
        print(f"[LOG] Saving results to {result_file}...")
        
        result_data = {
            "config_name": config_name,
            "dataset": dataset,
            "arch": arch,
            "enable_workflow": enable_workflow,
            "qps": qps,
            "slo_distribution": slo_distribution,
            "elapsed_time": elapsed_time,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "metrics": metrics,
            "enable_prioritization": enable_prioritization,
            "enable_adaptive_batching": enable_adaptive_batching,
            "enable_feedback": enable_feedback
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"[LOG] ✓ Results saved")
        
        if result.returncode == 0:
            print(f"\n[LOG] ✓ Experiment completed: {config_name}")
            print(f"[LOG]   Latency improvement: {metrics.get('latency_improvement', 'N/A')}%")
            print(f"[LOG]   Accuracy: {metrics.get('overall_accuracy', 'N/A')}%")
            print(f"[LOG]   Exit rate: {metrics.get('exit_rate', 'N/A')}")
        else:
            print(f"[LOG] ✗ Experiment failed with return code {result.returncode}")
        
        return result.returncode == 0, elapsed_time
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"[LOG] ✗ Experiment timed out after {timeout}s: {config_name}")
        return False, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[LOG] ✗ Experiment failed: {config_name} - {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed_time


def run_enhanced_component_analysis(dataset, arch, qps=90,
                                   slo_distribution=None,
                                   output_dir="workflow_results",
                                   timeout=3600,
                                   create_subfolder=True):
    """
    Run enhanced component analysis with mixed SLOs.
    
    This ensures all three components can show their effects:
    - Prioritization: Uses mixed SLOs (different urgency levels)
    - Adaptive Batching: Implemented in batch decision generation
    - Feedback Control: Adjusts thresholds dynamically
    """
    from run_comprehensive_experiments import create_results_folder
    
    if slo_distribution is None:
        # Default mixed SLO distribution for prioritization testing
        slo_distribution = {30: 0.3, 60: 0.4, 120: 0.3}
    
    if create_subfolder:
        results_dir = create_results_folder(output_dir, prefix="component_analysis_enhanced")
    else:
        results_dir = output_dir
        os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"Enhanced Component Analysis: {dataset} - {arch}")
    print(f"QPS: {qps}")
    print(f"SLO Distribution: {slo_distribution}")
    print(f"Results directory: {results_dir}")
    print("="*60)
    
    # Define all feature combinations
    feature_configs = [
        {
            "name": "baseline",
            "enable_workflow": False,
            "prioritization": False,
            "adaptive_batching": False,
            "feedback": False
        },
        {
            "name": "prioritization_only",
            "enable_workflow": True,
            "prioritization": True,
            "adaptive_batching": False,
            "feedback": False
        },
        {
            "name": "adaptive_batching_only",
            "enable_workflow": True,
            "prioritization": False,
            "adaptive_batching": True,
            "feedback": False
        },
        {
            "name": "feedback_only",
            "enable_workflow": True,
            "prioritization": False,
            "adaptive_batching": False,
            "feedback": True
        },
        {
            "name": "prioritization+batching",
            "enable_workflow": True,
            "prioritization": True,
            "adaptive_batching": True,
            "feedback": False
        },
        {
            "name": "prioritization+feedback",
            "enable_workflow": True,
            "prioritization": True,
            "adaptive_batching": False,
            "feedback": True
        },
        {
            "name": "batching+feedback",
            "enable_workflow": True,
            "prioritization": False,
            "adaptive_batching": True,
            "feedback": True
        },
        {
            "name": "all_features",
            "enable_workflow": True,
            "prioritization": True,
            "adaptive_batching": True,
            "feedback": True
        },
    ]
    
    results = []
    total_configs = len(feature_configs)
    start_time_total = time.time()
    
    print(f"\n[LOG] Starting enhanced component analysis with {total_configs} configurations")
    print(f"[LOG] Using mixed SLOs: {slo_distribution}")
    print(f"[LOG] Estimated time: ~{total_configs * 40 / 60:.1f} minutes (if cached)")
    print(f"[LOG] Timeout per experiment: {timeout}s\n")
    
    for idx, config in enumerate(feature_configs, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{total_configs}] Testing: {config['name']}")
        print(f"  Workflow enabled: {config['enable_workflow']}")
        print(f"  Prioritization: {config['prioritization']}")
        print(f"  Adaptive Batching: {config['adaptive_batching']}")
        print(f"  Feedback: {config['feedback']}")
        print(f"{'='*60}")
        
        exp_start_time = time.time()
        print(f"[LOG] Starting experiment at {time.strftime('%H:%M:%S', time.localtime())}")
        
        try:
            success, elapsed = run_experiment_enhanced(
                config_name=config['name'],
                dataset=dataset,
                arch=arch,
                enable_workflow=config['enable_workflow'],
                qps=qps,
                slo_distribution=slo_distribution,
                output_dir=results_dir,
                timeout=timeout,
                enable_prioritization=config['prioritization'],
                enable_adaptive_batching=config['adaptive_batching'],
                enable_feedback=config['feedback']
            )
            
            exp_elapsed = time.time() - exp_start_time
            print(f"[LOG] Experiment completed in {exp_elapsed:.1f}s")
            print(f"[LOG] Success: {success}")
            
            if success:
                print(f"[LOG] ✓ Configuration '{config['name']}' completed successfully")
            else:
                print(f"[LOG] ✗ Configuration '{config['name']}' failed or timed out")
            
        except Exception as e:
            print(f"[LOG] ✗ ERROR in experiment: {e}")
            import traceback
            traceback.print_exc()
            success = False
            elapsed = time.time() - exp_start_time
        
        results.append({
            "config_name": config['name'],
            "prioritization": config['prioritization'],
            "adaptive_batching": config['adaptive_batching'],
            "feedback": config['feedback'],
            "success": success,
            "elapsed_time": elapsed
        })
        
        # Progress summary
        completed = sum(1 for r in results if r["success"])
        failed = sum(1 for r in results if not r["success"])
        elapsed_total = time.time() - start_time_total
        remaining = total_configs - idx
        avg_time_per_exp = elapsed_total / idx if idx > 0 else 0
        estimated_remaining = avg_time_per_exp * remaining
        
        print(f"\n[LOG] Progress: {idx}/{total_configs} completed ({completed} successful, {failed} failed)")
        print(f"[LOG] Elapsed: {elapsed_total/60:.1f} minutes")
        if remaining > 0:
            print(f"[LOG] Estimated remaining: {estimated_remaining/60:.1f} minutes")
        
        time.sleep(1)  # Small delay between experiments
    
    # Save summary
    total_elapsed = time.time() - start_time_total
    summary = {
        "test_type": "enhanced_component_analysis",
        "dataset": dataset,
        "arch": arch,
        "qps": qps,
        "slo_distribution": slo_distribution,
        "results_directory": results_dir,
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_time": sum(r["elapsed_time"] for r in results),
        "wall_clock_time": total_elapsed,
        "results": results
    }
    
    summary_file = os.path.join(results_dir, "enhanced_component_analysis_summary.json")
    print(f"\n[LOG] Saving summary to {summary_file}...")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[LOG] Summary saved successfully")
    
    print("\n" + "="*60)
    print("Enhanced Component Analysis Summary")
    print("="*60)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total experiment time: {summary['total_time']:.1f}s ({summary['total_time']/60:.1f} minutes)")
    print(f"Wall clock time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"Summary saved to: {summary_file}")
    
    # List successful configurations
    if summary['successful'] > 0:
        print(f"\n[LOG] Successful configurations:")
        for r in results:
            if r["success"]:
                print(f"  ✓ {r['config_name']}")
    
    # List failed configurations
    if summary['failed'] > 0:
        print(f"\n[LOG] Failed configurations:")
        for r in results:
            if not r["success"]:
                print(f"  ✗ {r['config_name']}")
    
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run enhanced component analysis with mixed SLOs")
    parser.add_argument("--dataset", default="imdb",
                       help="Dataset name")
    parser.add_argument("--arch", default="distilbert-base",
                       help="Architecture name")
    parser.add_argument("--qps", type=int, default=90,
                       help="QPS level")
    parser.add_argument("--slo_distribution", type=str, default="30:0.3,60:0.4,120:0.3",
                       help="SLO distribution as 'slo1:prob1,slo2:prob2,...' (e.g., '30:0.3,60:0.4,120:0.3')")
    parser.add_argument("--output_dir", default="workflow_results",
                       help="Output directory")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout per experiment in seconds")
    parser.add_argument("--no_subfolder", action="store_true",
                       help="Don't create timestamped subfolder")
    
    args = parser.parse_args()
    
    # Parse SLO distribution
    slo_distribution = {}
    for pair in args.slo_distribution.split(','):
        slo_str, prob_str = pair.split(':')
        slo_distribution[int(slo_str)] = float(prob_str)
    
    # Normalize probabilities
    total_prob = sum(slo_distribution.values())
    slo_distribution = {k: v / total_prob for k, v in slo_distribution.items()}
    
    summary = run_enhanced_component_analysis(
        dataset=args.dataset,
        arch=args.arch,
        qps=args.qps,
        slo_distribution=slo_distribution,
        output_dir=args.output_dir,
        timeout=args.timeout,
        create_subfolder=not args.no_subfolder
    )
    
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()

