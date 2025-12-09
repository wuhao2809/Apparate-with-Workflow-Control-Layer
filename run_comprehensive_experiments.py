"""
Comprehensive experiment runner with QPS/SLO sweeps and mixed SLO/QPS support.

Supports:
- QPS sweeps
- SLO sweeps  
- Mixed SLO/QPS testing
- Component-level testing
"""

import os
import sys
import subprocess
import time
import json
import argparse
import re
import ast
import pickle
import numpy as np
from pathlib import Path
from itertools import product
from datetime import datetime

# Import batch_systems functions for generating batch decisions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import batch_systems
    import utils
except ImportError:
    # If imports fail, we'll handle it in the function
    batch_systems = None
    utils = None


def create_results_folder(base_dir="workflow_results", prefix="exp"):
    """
    Create a timestamped folder inside base_dir for storing experiment results.
    
    Returns:
        Path to the created folder (e.g., "workflow_results/exp-20251208_143022")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder = os.path.join(base_dir, f"{prefix}-{timestamp}")
    os.makedirs(results_folder, exist_ok=True)
    print(f"Results will be saved to: {results_folder}")
    return results_folder


def ensure_batch_decision_exists(arch, dataset, qps, slo, batching_scheme="clockwork", 
                                  profile_dir="profile_pickles_bs", 
                                  batch_decisions_dir="../batch_decisions",
                                  total_num_requests=70000,
                                  enable_prioritization=True, enable_adaptive_batching=False):
    """
    Ensure a batch decision file exists for the given parameters.
    If it doesn't exist, generate it on-the-fly.
    
    Args:
        arch: Architecture name (e.g., "distilbert-base")
        dataset: Dataset name (e.g., "imdb")
        qps: Queries per second
        slo: SLO in ms
        batching_scheme: Batching scheme ("clockwork" or "tf_serve")
        profile_dir: Directory containing profile pickles
        batch_decisions_dir: Directory to store/load batch decisions
        total_num_requests: Total number of requests to generate
        
    Returns:
        Path to the batch decision file
        
    Raises:
        ImportError: If batch_systems or utils cannot be imported
        RuntimeError: If batch decision generation fails
    """
    if batch_systems is None or utils is None:
        raise ImportError("Cannot generate batch decisions: batch_systems or utils not available. "
                         "Make sure you're running in the correct Python environment with all dependencies.")
    
    # Determine filename based on architecture and feature flags
    # Include feature flags in filename to generate different batch decisions for different feature combinations
    feature_suffix = ""
    if not enable_prioritization and not enable_adaptive_batching:
        feature_suffix = "_baseline"
    elif enable_prioritization and not enable_adaptive_batching:
        feature_suffix = "_prio"
    elif not enable_prioritization and enable_adaptive_batching:
        feature_suffix = "_batch"
    elif enable_prioritization and enable_adaptive_batching:
        feature_suffix = "_prio_batch"
    
    if "bert" in arch.lower() or "gpt" in arch.lower():
        # For NLP models, use azure trace format but with SLO
        filename_suffix = f"azure_{round(qps)}{feature_suffix}"
        filename = f"{batching_scheme}_{arch}_{slo}_{filename_suffix}.pickle"
    else:
        # For CV models, use fixed arrival rate format
        filename_suffix = f"fixed_{int(qps)}{feature_suffix}"
        filename = f"{batching_scheme}_{arch}_{slo}_{filename_suffix}.pickle"
    
    # Ensure batch_decisions_dir is an absolute path or relative to the apparate directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(batch_decisions_dir):
        # Make it relative to the parent of apparate directory (where batch_decisions typically lives)
        batch_decisions_dir = os.path.join(os.path.dirname(script_dir), batch_decisions_dir.lstrip("../"))
    
    batch_decision_path = os.path.join(batch_decisions_dir, filename)
    
    # Check if file exists
    if os.path.exists(batch_decision_path):
        print(f"Using existing batch decision: {batch_decision_path}")
        # Return relative path for use in command
        rel_path = os.path.relpath(batch_decision_path, script_dir)
        return rel_path if not rel_path.startswith("..") else batch_decision_path
    
    # Generate batch decision on-the-fly
    print(f"Generating batch decision for {arch} with QPS={qps}, SLO={slo}ms...")
    print(f"This may take a few minutes...")
    
    # Get model serving time from profile
    # Ensure profile_dir is an absolute path or relative to the project root
    try:
        # Get the directory where this script is located (apparate directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Profile directory is at project root, not in apparate directory
        if not os.path.isabs(profile_dir):
            # If profile_dir doesn't start with .., it's relative to apparate, need to go up one level
            if not profile_dir.startswith(".."):
                # Profile is at project root, so go up from apparate
                project_root = os.path.dirname(script_dir)
                profile_dir = os.path.join(project_root, profile_dir)
            else:
                # Already relative path, make it absolute
                profile_dir = os.path.join(script_dir, profile_dir)
        
        # Verify profile directory exists
        if not os.path.exists(profile_dir):
            raise RuntimeError(f"Profile directory does not exist: {profile_dir}")
        
        # latency_calc_list is modified inside get_model_serving_time, so we pass an empty list
        latency_calc_list = []
        model_serving_time = batch_systems.get_model_serving_time(arch, profile_dir, latency_calc_list)
    except Exception as e:
        raise RuntimeError(f"Failed to load model serving time for {arch}: {e}")
    
    # Create requests with specified SLO and QPS
    fixed_arrival_rate = True
    poisson_arrival = True
    all_requests, avg_qps, interarrival_time = batch_systems.create_request(
        fixed_arrival_rate, poisson_arrival, slo, qps
    )
    
    # Generate batch decision with feature flags
    print(f"Generating batch decision with {len(all_requests)} requests...")
    print(f"  Prioritization: {enable_prioritization}")
    print(f"  Adaptive Batching: {enable_adaptive_batching}")
    max_batch_size, batch_timeout_ms, max_enqueued_batches, \
        batch_decision, per_request_stats, total_num_requests, total_time = \
        batch_systems.get_batch_decision(
            batching_scheme, all_requests, model_serving_time, 
            slo=slo, max_batch_size=8, batch_timeout_ms=60, max_enqueued_batches=2,
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
        "slo": slo,
        "dataset": dataset,
        "avg_qps": avg_qps,
        "end_time": total_time,
        "max_batch_size": max_batch_size,
        "batch_timeout_ms": batch_timeout_ms,
        "max_enqueued_batches": max_enqueued_batches,
        "enable_prioritization": enable_prioritization,  # Store feature flags
        "enable_adaptive_batching": enable_adaptive_batching,
    }
    
    with open(batch_decision_path, "wb") as f:
        pickle.dump(batch_info, f)
    
    print(f"Batch decision saved to: {batch_decision_path}")
    # Return relative path for use in command
    rel_path = os.path.relpath(batch_decision_path, script_dir)
    return rel_path if not rel_path.startswith("..") else batch_decision_path


def parse_metrics_from_stdout(stdout):
    """Parse metrics from experiment stdout."""
    metrics = {
        "overall_accuracy": None,
        "latency_improvement": None,
        "exit_rate": None,
        "overall_ramp_accuracy": None
    }
    
    if not stdout:
        return metrics
    
    # Look for the final summary line
    lines = stdout.split('\n')
    for line in reversed(lines):
        if "Serving with complete" in line:
            # Parse: "overall accuracy X%, overall serving latency improvement Y%"
            # Note: We can't just split by comma because dictionaries contain commas
            
            # Parse overall accuracy
            if "overall accuracy" in line:
                try:
                    acc_match = re.search(r'overall accuracy\s+([\d.]+)%', line)
                    if acc_match:
                        metrics["overall_accuracy"] = float(acc_match.group(1))
                except:
                    pass
            
            # Parse latency improvement
            if "overall serving latency improvement" in line:
                try:
                    lat_match = re.search(r'overall serving latency improvement\s+([\d.+-]+)%', line)
                    if lat_match:
                        metrics["latency_improvement"] = float(lat_match.group(1))
                except:
                    pass
            
            # Parse exit rate (dictionary)
            if "overall exit rate" in line:
                try:
                    exit_rate_start = line.find("overall exit rate")
                    if exit_rate_start != -1:
                        exit_rate_substr = line[exit_rate_start + len("overall exit rate"):].strip()
                        # Find the dictionary boundaries
                        brace_count = 0
                        start_idx = exit_rate_substr.find('{')
                        if start_idx != -1:
                            for i in range(start_idx, len(exit_rate_substr)):
                                if exit_rate_substr[i] == '{':
                                    brace_count += 1
                                elif exit_rate_substr[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        exit_rate_str = exit_rate_substr[start_idx:i+1]
                                        exit_rate_str_clean = re.sub(r'np\.int64\((\d+)\)', r'\1', exit_rate_str)
                                        exit_rate_dict = ast.literal_eval(exit_rate_str_clean)
                                        if isinstance(exit_rate_dict, dict):
                                            ramp_ids = sorted(exit_rate_dict.keys())
                                            final_exit_id = ramp_ids[-1]
                                            total_early_exit_rate = sum(
                                                exit_rate_dict[rid] for rid in ramp_ids if rid != final_exit_id
                                            )
                                            metrics["exit_rate"] = total_early_exit_rate
                                        break
                except Exception as e:
                    pass
            
            # Parse ramp accuracy (dictionary) - this comes at the end
            if "overall ramp accuracy" in line:
                try:
                    ramp_acc_start = line.find("overall ramp accuracy")
                    if ramp_acc_start != -1:
                        ramp_acc_substr = line[ramp_acc_start + len("overall ramp accuracy"):].strip()
                        # Find the dictionary boundaries
                        brace_count = 0
                        start_idx = ramp_acc_substr.find('{')
                        if start_idx != -1:
                            for i in range(start_idx, len(ramp_acc_substr)):
                                if ramp_acc_substr[i] == '{':
                                    brace_count += 1
                                elif ramp_acc_substr[i] == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        ramp_acc_str = ramp_acc_substr[start_idx:i+1]
                                        ramp_acc_str_clean = re.sub(r'np\.int64\((\d+)\)', r'\1', ramp_acc_str)
                                        # Store as cleaned string (can be converted to dict later if needed)
                                        metrics["overall_ramp_accuracy"] = ramp_acc_str_clean
                                        break
                except Exception as e:
                    pass
            break
    
    return metrics


def run_experiment(config_name, dataset, arch, enable_workflow=False, 
                  qps=30, slo=45, output_dir="workflow_results",
                  num_requests=None, timeout=3600,
                  enable_prioritization=True, enable_adaptive_batching=True, enable_feedback=True):
    """
    Run a single experiment configuration.
    
    Args:
        config_name: Name for this configuration (e.g., "baseline", "optimized")
        dataset: Dataset name
        arch: Architecture name
        enable_workflow: Whether to enable workflow control layer
        qps: Queries per second
        slo: SLO in ms (or dict for mixed SLO)
        output_dir: Directory to store results
        num_requests: Number of requests (None = use default)
        timeout: Timeout in seconds
    """
    print(f"\n{'='*60}")
    print(f"Running: {config_name} - {dataset} - {arch}")
    print(f"Workflow enabled: {enable_workflow}")
    if enable_workflow:
        print(f"  Prioritization: {enable_prioritization}")
        print(f"  Adaptive Batching: {enable_adaptive_batching}")
        print(f"  Feedback: {enable_feedback}")
    print(f"QPS: {qps}, SLO: {slo}ms")
    print(f"{'='*60}\n")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Build command - use the current Python interpreter (which has PyTorch)
    # This ensures we use the same Python that's running this script
    import sys
    python_cmd = sys.executable  # Use current Python interpreter
    
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
    
    # Generate or get feature-specific batch decision path
    try:
        batch_decision_path = ensure_batch_decision_exists(
            arch=arch,
            dataset=dataset,
            qps=qps,
            slo=slo,
            batching_scheme="clockwork",
            profile_dir="../profile_pickles_bs",  # Profile is at project root
            batch_decisions_dir="../batch_decisions",
            enable_prioritization=enable_prioritization if enable_workflow else False,
            enable_adaptive_batching=enable_adaptive_batching if enable_workflow else False
        )
        # Convert to relative path if needed
        if os.path.isabs(batch_decision_path):
            # Make it relative to the apparate directory
            batch_decision_path = os.path.relpath(batch_decision_path, 
                                                 os.path.dirname(os.path.abspath(__file__)))
    except Exception as e:
        print(f"Warning: Could not generate batch decision: {e}")
        print(f"Falling back to old batch decision path format...")
        # Fallback to old format (without SLO)
        if "bert" in arch.lower() or "gpt" in arch.lower():
            batch_decision_path = f"../batch_decisions/{arch}_azure.pickle"
        else:
            batch_decision_path = f"../batch_decisions/{arch}_1_fixed_{int(qps) if qps <= 30 else 30}.pickle"
    
    cmd_parts.extend([
        f"--batch_decision_path", batch_decision_path
    ])
    
    if enable_workflow:
        cmd_parts.append("--enable_workflow")
        # Add individual feature flags based on parameters
        # Note: controller.py uses --disable_* flags (default is enabled when workflow is on)
        if not enable_prioritization:
            cmd_parts.append("--disable_prioritization")
        
        if not enable_adaptive_batching:
            cmd_parts.append("--disable_adaptive_batching")
        
        if not enable_feedback:
            cmd_parts.append("--disable_feedback")
    
    # Run experiment
    print(f"[LOG] Executing command: {' '.join(cmd_parts[:5])}... (truncated)")
    print(f"[LOG] Working directory: {os.path.dirname(os.path.abspath(__file__))}")
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
        
        # Parse metrics from full stdout (not truncated)
        print(f"[LOG] Parsing metrics from stdout (length: {len(result.stdout) if result.stdout else 0} chars)...")
        metrics = parse_metrics_from_stdout(result.stdout) if result.stdout else {}
        
        if metrics and any(v is not None for v in metrics.values()):
            print(f"[LOG] ✓ Metrics found in stdout")
        else:
            print(f"[LOG] No metrics in stdout, trying log file...")
        
        # If metrics not found in stdout, try parsing from log file
        # The "Serving with complete" message is logged to a file, not stdout
        # IMPORTANT: The log file is opened with mode="w+" which OVERWRITES it each time
        # Solution: Copy the log file to a unique location immediately after experiment completes
        if not metrics or all(v is None for v in metrics.values()):
            log_suffix = "baseline" if config_name == "baseline" else "workflow"
            log_file = f"logs/output_{arch}_{dataset}_{log_suffix}.log"
            
            # Create a unique copy of the log file for this experiment
            # This preserves the metrics before the next experiment overwrites it
            import shutil
            unique_log_file = os.path.join(
                output_dir,
                f"log_{config_name}_{dataset}_{arch}_qps{qps}_slo{slo}.log"
            )
            
            # Small delay to ensure log file is flushed to disk
            time.sleep(0.5)
            
            # Copy the log file immediately after experiment (before next one overwrites it)
            if os.path.exists(log_file):
                try:
                    shutil.copy2(log_file, unique_log_file)
                    
                    # Now read from the unique copy
                    with open(unique_log_file, 'r') as f:
                        log_lines = f.readlines()
                    
                    # Find the last "Serving with complete" line
                    for line in reversed(log_lines):
                        if "Serving with complete" in line:
                            log_metrics = parse_metrics_from_stdout(line)
                            if log_metrics and any(v is not None for v in log_metrics.values()):
                                metrics = log_metrics
                            break
                except Exception as e:
                    print(f"Warning: Could not copy/parse log file {log_file}: {e}")
        
        # Save results
        result_file = os.path.join(
            output_dir,
            f"{config_name}_{dataset}_{arch}_qps{qps}_slo{slo}.json"
        )
        print(f"[LOG] Saving results to {result_file}...")
        
        result_data = {
            "config_name": config_name,
            "dataset": dataset,
            "arch": arch,
            "enable_workflow": enable_workflow,
            "qps": qps,
            "slo": slo,
            "elapsed_time": elapsed_time,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "metrics": metrics  # Add parsed metrics
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"[LOG] ✓ Results saved to {result_file}")
        
        if result.returncode == 0:
            print(f"\n[LOG] ✓ Experiment completed: {config_name}")
            print(f"[LOG]   Latency improvement: {metrics.get('latency_improvement', 'N/A')}%")
            print(f"[LOG]   Accuracy: {metrics.get('overall_accuracy', 'N/A')}%")
            print(f"[LOG]   Exit rate: {metrics.get('exit_rate', 'N/A')}")
        else:
            print(f"[LOG] ✗ Experiment failed with return code {result.returncode}")
            print(f"[LOG] Error: {result.stderr[-500:] if result.stderr else 'No stderr'}")
        
        return result.returncode == 0, elapsed_time
        
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"[LOG] ✗ Experiment timed out after {timeout}s: {config_name}")
        return False, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[LOG] ✗ Experiment failed: {config_name} - {e}")
        import traceback
        print(f"[LOG] Error traceback:")
        traceback.print_exc()
        return False, elapsed_time


def run_qps_sweep(dataset, arch, slo, qps_levels, configs=["baseline", "optimized"], 
                  output_dir="workflow_results", timeout=3600, create_subfolder=True):
    """Run QPS sweep experiments."""
    # Create timestamped subfolder if requested
    if create_subfolder:
        results_dir = create_results_folder(output_dir, prefix="qps_sweep")
    else:
        results_dir = output_dir
        os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"QPS Sweep: {dataset} - {arch} - SLO: {slo}ms")
    print(f"QPS levels: {qps_levels}")
    print(f"Results directory: {results_dir}")
    print("="*60)
    
    results = []
    for qps, config in product(qps_levels, configs):
        enable_workflow = (config == "optimized")
        success, elapsed = run_experiment(
            config_name=config,
            dataset=dataset,
            arch=arch,
            enable_workflow=enable_workflow,
            qps=qps,
            slo=slo,
            output_dir=results_dir,  # Use the timestamped folder
            timeout=timeout
        )
        results.append({
            "qps": qps,
            "slo": slo,
            "config": config,
            "success": success,
            "elapsed_time": elapsed
        })
        time.sleep(1)  # Small delay between experiments
    
    return results


def run_slo_sweep(dataset, arch, qps, slo_levels, configs=["baseline", "optimized"],
                  output_dir="workflow_results", timeout=3600, create_subfolder=True):
    """Run SLO sweep experiments."""
    # Create timestamped subfolder if requested
    if create_subfolder:
        results_dir = create_results_folder(output_dir, prefix="slo_sweep")
    else:
        results_dir = output_dir
        os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"SLO Sweep: {dataset} - {arch} - QPS: {qps}")
    print(f"SLO levels: {slo_levels}")
    print(f"Results directory: {results_dir}")
    print("="*60)
    
    results = []
    for slo, config in product(slo_levels, configs):
        enable_workflow = (config == "optimized")
        success, elapsed = run_experiment(
            config_name=config,
            dataset=dataset,
            arch=arch,
            enable_workflow=enable_workflow,
            qps=qps,
            slo=slo,
            output_dir=results_dir,  # Use the timestamped folder
            timeout=timeout
        )
        results.append({
            "qps": qps,
            "slo": slo,
            "config": config,
            "success": success,
            "elapsed_time": elapsed
        })
        time.sleep(1)
    
    return results


def run_mixed_slo_test(dataset, arch, qps, slo_distribution, configs=["baseline", "optimized"],
                       output_dir="workflow_results", timeout=3600):
    """
    Run mixed SLO test.
    
    Args:
        slo_distribution: Dict mapping SLO values to probabilities
                         e.g., {30: 0.3, 45: 0.4, 60: 0.3}
    """
    print("\n" + "="*60)
    print(f"Mixed SLO Test: {dataset} - {arch} - QPS: {qps}")
    print(f"SLO distribution: {slo_distribution}")
    print("="*60)
    
    # For mixed SLO, we use the average SLO for the experiment name
    # The actual mixed SLOs are handled in batch_systems.py
    avg_slo = sum(slo * prob for slo, prob in slo_distribution.items())
    slo_str = "mixed_" + "_".join(f"{k}_{v}" for k, v in sorted(slo_distribution.items()))
    
    results = []
    for config in configs:
        enable_workflow = (config == "optimized")
        # Note: Mixed SLO requires modification to controller.py to accept slo_distribution
        # For now, we'll use average SLO
        success, elapsed = run_experiment(
            config_name=config,
            dataset=dataset,
            arch=arch,
            enable_workflow=enable_workflow,
            qps=qps,
            slo=int(avg_slo),  # Use average for now
            output_dir=output_dir,
            timeout=timeout
        )
        results.append({
            "qps": qps,
            "slo": slo_str,
            "slo_distribution": slo_distribution,
            "config": config,
            "success": success,
            "elapsed_time": elapsed
        })
        time.sleep(1)
    
    return results


def run_component_analysis(dataset, arch, qps=90, slo=60, 
                          output_dir="workflow_results", timeout=3600, create_subfolder=True):
    """
    Run component-level analysis to test individual workflow features.
    
    Tests all combinations of:
    - Prioritization
    - Adaptive Batching
    - Feedback Control
    
    Args:
        dataset: Dataset name
        arch: Architecture name
        qps: QPS level to test
        slo: SLO level to test
        output_dir: Output directory
        timeout: Timeout per experiment
        create_subfolder: If True, create timestamped subfolder
    """
    import time as time_module
    
    print("\n" + "="*60)
    print(f"Component-Level Analysis: {dataset} - {arch}")
    print(f"QPS: {qps}, SLO: {slo}ms")
    print("="*60)
    
    if create_subfolder:
        print(f"[LOG] Creating results folder...")
        results_dir = create_results_folder(output_dir, prefix="component_analysis")
        print(f"[LOG] Results directory: {results_dir}")
    else:
        results_dir = output_dir
        os.makedirs(results_dir, exist_ok=True)
        print(f"[LOG] Using output directory: {results_dir}")
    
    print(f"[LOG] Results will be saved to: {results_dir}")
    
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
    start_time_total = time_module.time()
    
    print(f"\n[LOG] Starting component analysis with {total_configs} configurations")
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
        
        exp_start_time = time_module.time()
        print(f"[LOG] Starting experiment at {time_module.strftime('%H:%M:%S', time_module.localtime())}")
        
        try:
            success, elapsed = run_experiment(
                config_name=config['name'],
                dataset=dataset,
                arch=arch,
                enable_workflow=config['enable_workflow'],
                qps=qps,
                slo=slo,
                output_dir=results_dir,
                timeout=timeout,
                enable_prioritization=config['prioritization'],
                enable_adaptive_batching=config['adaptive_batching'],
                enable_feedback=config['feedback']
            )
            
            exp_elapsed = time_module.time() - exp_start_time
            print(f"[LOG] Experiment completed in {exp_elapsed:.1f}s")
            print(f"[LOG] Success: {success}")
            
            if success:
                print(f"[LOG] ✓ Configuration '{config['name']}' completed successfully")
            else:
                print(f"[LOG] ✗ Configuration '{config['name']}' failed or timed out")
            
        except Exception as e:
            print(f"[LOG] ✗ ERROR in experiment: {e}")
            success = False
            elapsed = time_module.time() - exp_start_time
        
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
        elapsed_total = time_module.time() - start_time_total
        remaining = total_configs - idx
        avg_time_per_exp = elapsed_total / idx if idx > 0 else 0
        estimated_remaining = avg_time_per_exp * remaining
        
        print(f"\n[LOG] Progress: {idx}/{total_configs} completed ({completed} successful, {failed} failed)")
        print(f"[LOG] Elapsed: {elapsed_total/60:.1f} minutes")
        if remaining > 0:
            print(f"[LOG] Estimated remaining: {estimated_remaining/60:.1f} minutes")
        
        time_module.sleep(1)  # Small delay between experiments
    
    # Save summary
    total_elapsed = time_module.time() - start_time_total
    summary = {
        "test_type": "component_analysis",
        "dataset": dataset,
        "arch": arch,
        "qps": qps,
        "slo": slo,
        "results_directory": results_dir,
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_time": sum(r["elapsed_time"] for r in results),
        "wall_clock_time": total_elapsed,
        "results": results
    }
    
    summary_file = os.path.join(results_dir, "component_analysis_summary.json")
    print(f"\n[LOG] Saving summary to {summary_file}...")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[LOG] Summary saved successfully")
    
    print("\n" + "="*60)
    print("Component Analysis Summary")
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


def run_mixed_experiments(dataset, arch, combinations, configs=["baseline", "optimized"],
                          output_dir="workflow_results", timeout=3600):
    """
    Run mixed experiments with specific QPS/SLO combinations.
    
    Args:
        dataset: Dataset name
        arch: Architecture name
        combinations: List of (qps, slo) tuples
        configs: List of configs to test
        output_dir: Output directory
        timeout: Timeout per experiment
    """
    print("\n" + "="*60)
    print(f"Mixed Experiments: {dataset} - {arch}")
    print(f"Combinations: {combinations}")
    print(f"Results directory: {output_dir}")
    print("="*60)
    
    results = []
    for (qps, slo), config in product(combinations, configs):
        enable_workflow = (config == "optimized")
        success, elapsed = run_experiment(
            config_name=config,
            dataset=dataset,
            arch=arch,
            enable_workflow=enable_workflow,
            qps=qps,
            slo=slo,
            output_dir=output_dir,
            timeout=timeout
        )
        results.append({
            "qps": qps,
            "slo": slo,
            "config": config,
            "success": success,
            "elapsed_time": elapsed
        })
        time.sleep(1)
    
    return results


def run_small_validation_test(output_dir="workflow_results", timeout=7200, create_subfolder=True):
    """
    Run a comprehensive test that should finish within 1 hour.
    
    Tests:
    - QPS sweep: 3 levels (30, 90, 150) - 6 experiments (2 configs x 3 QPS)
    - SLO sweep: 3 levels (25, 60, 120) - 6 experiments (2 configs x 3 SLO)
    - Mixed: 3 combinations - 6 experiments (2 configs x 3 combinations)
    - Total: 18 experiments
    
    Runtime estimate:
    - First run: ~38 minutes (batch decision generation + experiments)
    - Cached: ~11 minutes (experiments only)
    
    Args:
        output_dir: Base directory for results (default: "workflow_results")
        timeout: Timeout per experiment in seconds
        create_subfolder: If True, create a timestamped subfolder inside output_dir
    """
    # Create timestamped subfolder if requested
    if create_subfolder:
        results_dir = create_results_folder(output_dir, prefix="exp")
    else:
        results_dir = output_dir
        os.makedirs(results_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EXPERIMENT SUITE")
    print("Expected duration: < 1 hour (first run: ~38 min, cached: ~11 min)")
    print(f"Results directory: {results_dir}")
    print("="*60)
    
    # Use smaller dataset/model for faster testing
    dataset = "imdb"
    arch = "distilbert-base"
    
    all_results = []
    
    # 1. QPS Sweep (3 levels, 2 configs = 6 experiments)
    print("\n[1/3] QPS Sweep (3 levels)")
    print("  Testing QPS: [30, 90, 150] with SLO: 60ms")
    print("  Configs: baseline, optimized")
    print("  Expected: 6 experiments")
    qps_results = run_qps_sweep(
        dataset=dataset,
        arch=arch,
        slo=60,
        qps_levels=[30, 90, 150],  # Larger gaps: 3x multiplier
        configs=["baseline", "optimized"],
        output_dir=results_dir,  # Use the timestamped folder
        timeout=timeout,
        create_subfolder=False  # Already in timestamped folder
    )
    all_results.extend(qps_results)
    
    # 2. SLO Sweep (3 levels, 2 configs = 6 experiments)
    print("\n[2/3] SLO Sweep (3 levels)")
    print("  Testing SLO: [25, 60, 120]ms with QPS: 90")
    print("  Configs: baseline, optimized")
    print("  Expected: 6 experiments")
    slo_results = run_slo_sweep(
        dataset=dataset,
        arch=arch,
        qps=90,  # Use middle QPS value
        slo_levels=[25, 60, 120],  # Larger gaps: 2.4x-2x multiplier
        configs=["baseline", "optimized"],
        output_dir=results_dir,  # Use the timestamped folder
        timeout=timeout,
        create_subfolder=False  # Already in timestamped folder
    )
    all_results.extend(slo_results)
    
    # 3. Mixed Experiments (key combinations, 2 configs = 6 experiments)
    print("\n[3/3] Mixed Experiments (key combinations)")
    print("  Testing combinations: (QPS=30, SLO=25), (QPS=150, SLO=120), (QPS=150, SLO=25)")
    print("  Configs: baseline, optimized")
    print("  Expected: 6 experiments")
    mixed_results = run_mixed_experiments(
        dataset=dataset,
        arch=arch,
        combinations=[
            (30, 25),   # Low QPS + Tight SLO
            (150, 120), # High QPS + Loose SLO
            (150, 25),  # High QPS + Tight SLO (extreme stress)
        ],
        configs=["baseline", "optimized"],
        output_dir=results_dir,
        timeout=timeout
    )
    all_results.extend(mixed_results)
    
    
    # Save summary
    summary = {
        "test_type": "small_validation",
        "results_directory": results_dir,
        "total_experiments": len(all_results),
        "successful": sum(1 for r in all_results if r["success"]),
        "failed": sum(1 for r in all_results if not r["success"]),
        "total_time": sum(r["elapsed_time"] for r in all_results),
        "results": all_results
    }
    
    summary_file = os.path.join(results_dir, "validation_test_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("Validation Test Summary")
    print("="*60)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Total time: {summary['total_time']:.1f}s ({summary['total_time']/3600:.2f} hours)")
    print(f"Summary saved to: {summary_file}")
    print("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive workflow experiments")
    parser.add_argument("--mode", choices=["qps_sweep", "slo_sweep", "mixed_slo", "validation", "component_analysis", "all"],
                       default="validation",
                       help="Experiment mode")
    parser.add_argument("--dataset", default="imdb",
                       help="Dataset name")
    parser.add_argument("--arch", default="distilbert-base",
                       help="Architecture name")
    parser.add_argument("--qps", type=int, default=30,
                       help="QPS (for SLO sweep or mixed SLO)")
    parser.add_argument("--slo", type=int, default=45,
                       help="SLO in ms (for QPS sweep)")
    parser.add_argument("--qps_levels", nargs="+", type=int,
                       default=[30, 90, 150],  # Larger gaps: 3x multiplier
                       help="QPS levels for sweep")
    parser.add_argument("--slo_levels", nargs="+", type=int,
                       default=[25, 60, 120],  # Larger gaps: 2.4x-2x multiplier
                       help="SLO levels for sweep")
    parser.add_argument("--output_dir", default="workflow_results",
                       help="Base output directory (results will be saved in timestamped subfolder)")
    parser.add_argument("--no_subfolder", action="store_true",
                       help="Don't create timestamped subfolder (save directly to output_dir)")
    parser.add_argument("--timeout", type=int, default=3600,
                       help="Timeout per experiment in seconds")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists("controller.py"):
        print("Error: Must run from apparate directory")
        print("Usage: cd apparate && python run_comprehensive_experiments.py")
        sys.exit(1)
    
    create_subfolder = not args.no_subfolder
    
    if args.mode == "validation":
        summary = run_small_validation_test(
            output_dir=args.output_dir,
            timeout=args.timeout,
            create_subfolder=create_subfolder
        )
        sys.exit(0 if summary["failed"] == 0 else 1)
    elif args.mode == "qps_sweep":
        results = run_qps_sweep(
            dataset=args.dataset,
            arch=args.arch,
            slo=args.slo,
            qps_levels=args.qps_levels,
            output_dir=args.output_dir,
            timeout=args.timeout,
            create_subfolder=create_subfolder
        )
    elif args.mode == "slo_sweep":
        results = run_slo_sweep(
            dataset=args.dataset,
            arch=args.arch,
            qps=args.qps,
            slo_levels=args.slo_levels,
            output_dir=args.output_dir,
            timeout=args.timeout,
            create_subfolder=create_subfolder
        )
    elif args.mode == "mixed_slo":
        # Default mixed SLO distribution
        slo_dist = {30: 0.3, 45: 0.4, 60: 0.3}
        results = run_mixed_slo_test(
            dataset=args.dataset,
            arch=args.arch,
            qps=args.qps,
            slo_distribution=slo_dist,
            output_dir=args.output_dir,
            timeout=args.timeout
        )
    elif args.mode == "component_analysis":
        summary = run_component_analysis(
            dataset=args.dataset,
            arch=args.arch,
            qps=args.qps,
            slo=args.slo,
            output_dir=args.output_dir,
            timeout=args.timeout,
            create_subfolder=create_subfolder
        )
        sys.exit(0 if summary["failed"] == 0 else 1)
    else:
        print("Mode 'all' not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()

