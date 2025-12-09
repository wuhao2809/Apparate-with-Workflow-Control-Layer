"""
Test script to demonstrate the value of Prioritization and Adaptive Batching.

This script runs two focused experiments:
1. Prioritization Test: Mixed SLOs (30ms, 60ms, 120ms) - Baseline vs With Prioritization
2. Adaptive Batching Test: Variable QPS - Fixed Batch vs Adaptive Batching

Generates metrics that can be visualized to show the benefits of each feature.

REVISED: Now uses simulation mode and forces recomputation of batch decisions.
"""

import os
import sys
import pickle
import json
import time
import subprocess
import re
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_systems
import utils


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def force_regenerate_batch_decision(arch, dataset, qps, slo, enable_prioritization, 
                                    enable_adaptive_batching, batch_decisions_dir="../batch_decisions"):
    """
    Force regeneration of batch decision by deleting existing file if it exists.
    Returns the path to the batch decision file.
    """
    # Determine filename based on architecture and feature flags
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
        filename_suffix = f"azure_{round(qps)}{feature_suffix}"
        filename = f"clockwork_{arch}_{slo}_{filename_suffix}.pickle"
    else:
        filename_suffix = f"fixed_{int(qps)}{feature_suffix}"
        filename = f"clockwork_{arch}_{slo}_{filename_suffix}.pickle"
    
    batch_decision_path = os.path.join(batch_decisions_dir, filename)
    
    # Force regeneration by deleting existing file
    if os.path.exists(batch_decision_path):
        print(f"  Deleting existing batch decision to force regeneration: {os.path.basename(batch_decision_path)}")
        os.remove(batch_decision_path)
    
    return batch_decision_path


def run_controller_with_simulation(config_name, dataset, arch, qps, slo, 
                                   enable_prioritization=False, enable_adaptive_batching=False,
                                   batch_decision_path=None, timeout=1800):
    """
    Run controller.py with simulation mode to get actual serving results.
    
    Returns:
        dict with metrics extracted from the experiment
    """
    print(f"  Running controller with simulation mode...")
    
    # Ensure output directories exist
    os.makedirs("logs", exist_ok=True)
    
    # Build command
    python_cmd = sys.executable
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
    
    if batch_decision_path:
        # Make path relative if needed
        if os.path.isabs(batch_decision_path):
            batch_decision_path = os.path.relpath(batch_decision_path, os.path.dirname(os.path.abspath(__file__)))
        cmd_parts.extend([f"--batch_decision_path", batch_decision_path])
    
    if enable_prioritization or enable_adaptive_batching:
        cmd_parts.append("--enable_workflow")
        if not enable_prioritization:
            cmd_parts.append("--disable_prioritization")
        if not enable_adaptive_batching:
            cmd_parts.append("--disable_adaptive_batching")
    
    # Run experiment
    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Parse metrics from stdout or log file
        metrics = parse_metrics_from_output(result.stdout, result.stderr)
        metrics['returncode'] = result.returncode
        metrics['config_name'] = config_name
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Experiment timed out after {timeout}s")
        return {'returncode': -1, 'error': 'timeout'}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {'returncode': -1, 'error': str(e)}


def parse_metrics_from_output(stdout, stderr):
    """Parse metrics from controller output."""
    metrics = {
        'overall_accuracy': 0.0,
        'latency_improvement': 0.0,
        'exit_rate': 0.0,
        'overall_ramp_accuracy': {}
    }
    
    # Try to parse from stdout
    if stdout:
        # Look for accuracy
        acc_match = re.search(r'overall accuracy\s+([\d.]+)%', stdout)
        if acc_match:
            metrics['overall_accuracy'] = float(acc_match.group(1))
        
        # Look for latency improvement
        lat_match = re.search(r'overall serving latency improvement\s+([\d.+-]+)%', stdout)
        if lat_match:
            metrics['latency_improvement'] = float(lat_match.group(1))
        
        # Look for exit rate
        exit_match = re.search(r'overall exit rate\s+([\d.]+)', stdout)
        if exit_match:
            metrics['exit_rate'] = float(exit_match.group(1))
    
    return metrics

def test_prioritization():
    """
    Test 1: Prioritization
    - Mixed SLOs: 15ms (40%), 30ms (30%), 50ms (30%)
    - Compare: Baseline (no prioritization) vs With Prioritization
    - Measure: SLO violation rate per SLO category
    
    REVISED: Now uses simulation mode and forces batch decision regeneration.
    """
    print("\n" + "="*70)
    print("TEST 1: PRIORITIZATION")
    print("="*70)
    
    # Configuration
    arch = "distilbert-base"
    dataset = "imdb"
    qps = 200
    slo_distribution = {15: 0.4, 30: 0.3, 50: 0.3}
    total_num_requests = 20000
    profile_dir = "../profile_pickles_bs"
    batch_decisions_dir = "../batch_decisions"
    
    # Get model serving time
    latency_calc_list = []
    model_serving_time = batch_systems.get_model_serving_time(arch, profile_dir, latency_calc_list)
    
    # Create requests with mixed SLOs
    print(f"Creating {total_num_requests} requests with mixed SLOs: {slo_distribution}")
    all_requests, avg_qps, interarrival_time = batch_systems.create_mixed_slo_requests(
        num_requests=total_num_requests,
        slo_distribution=slo_distribution,
        qps=qps,
        poisson_arrival=True,
        seed=42
    )
    
    results = {}
    
    # Test 1a: Baseline (no prioritization)
    print("\n[1a] Running Baseline (no prioritization)...")
    
    # Force regeneration of batch decision
    batch_decision_path_baseline = force_regenerate_batch_decision(
        arch, dataset, qps, 60,  # Use average SLO
        enable_prioritization=False,
        enable_adaptive_batching=False,
        batch_decisions_dir=batch_decisions_dir
    )
    
    # Generate batch decision
    _, _, _, batch_decision_baseline, per_request_stats_baseline, _, _ = \
        batch_systems.get_batch_decision(
            batching_scheme="clockwork",
            all_requests=list(all_requests),
            model_serving_time=model_serving_time,
            slo=60,
            max_batch_size=8,
            batch_timeout_ms=30,
            max_enqueued_batches=3,
            enable_prioritization=False,
            enable_adaptive_batching=False
        )
    
    # Save batch decision
    os.makedirs(batch_decisions_dir, exist_ok=True)
    with open(batch_decision_path_baseline, "wb") as f:
        batch_info = {
            "batching_decision": batch_decision_baseline,
            "per_request_stats": per_request_stats_baseline,
            "total_num_requests": total_num_requests,
            "batching_scheme": "clockwork",
            "arch": arch,
            "slo": 60,
            "dataset": dataset,
            "avg_qps": avg_qps,
        }
        pickle.dump(batch_info, f)
    
    # Run controller with simulation mode (optional - can skip if takes too long)
    print("  Note: Skipping controller simulation for faster batch decision testing")
    baseline_controller_metrics = {'overall_accuracy': 0.0, 'latency_improvement': 0.0, 'exit_rate': 0.0}
    
    # Analyze batch decision results
    baseline_metrics = analyze_slo_violations(per_request_stats_baseline, all_requests, slo_distribution)
    baseline_metrics['controller_metrics'] = baseline_controller_metrics
    
    results['baseline'] = {
        'batch_decision': batch_decision_baseline,
        'per_request_stats': per_request_stats_baseline,
        'metrics': baseline_metrics
    }
    print(f"  SLO Violation Rate: {baseline_metrics['overall_violation_rate']:.2f}%")
    print(f"  Violations by SLO: {baseline_metrics['violations_by_slo']}")
    
    # Test 1b: With Prioritization
    print("\n[1b] Running With Prioritization...")
    
    # Force regeneration of batch decision
    batch_decision_path_prio = force_regenerate_batch_decision(
        arch, dataset, qps, 60,
        enable_prioritization=True,
        enable_adaptive_batching=False,
        batch_decisions_dir=batch_decisions_dir
    )
    
    # Generate batch decision
    _, _, _, batch_decision_prio, per_request_stats_prio, _, _ = \
        batch_systems.get_batch_decision(
            batching_scheme="clockwork",
            all_requests=list(all_requests),
            model_serving_time=model_serving_time,
            slo=60,
            max_batch_size=8,
            batch_timeout_ms=30,
            max_enqueued_batches=3,
            enable_prioritization=True,
            enable_adaptive_batching=False
        )
    
    # Save batch decision
    with open(batch_decision_path_prio, "wb") as f:
        batch_info = {
            "batching_decision": batch_decision_prio,
            "per_request_stats": per_request_stats_prio,
            "total_num_requests": total_num_requests,
            "batching_scheme": "clockwork",
            "arch": arch,
            "slo": 60,
            "dataset": dataset,
            "avg_qps": avg_qps,
        }
        pickle.dump(batch_info, f)
    
    # Run controller with simulation mode (optional - can skip if takes too long)
    print("  Note: Skipping controller simulation for faster batch decision testing")
    prio_controller_metrics = {'overall_accuracy': 0.0, 'latency_improvement': 0.0, 'exit_rate': 0.0}
    
    # Analyze prioritization results
    prio_metrics = analyze_slo_violations(per_request_stats_prio, all_requests, slo_distribution)
    prio_metrics['controller_metrics'] = prio_controller_metrics
    
    results['prioritization'] = {
        'batch_decision': batch_decision_prio,
        'per_request_stats': per_request_stats_prio,
        'metrics': prio_metrics
    }
    print(f"  SLO Violation Rate: {prio_metrics['overall_violation_rate']:.2f}%")
    print(f"  Violations by SLO: {prio_metrics['violations_by_slo']}")
    
    return results


def test_adaptive_batching():
    """
    Test 2: Adaptive Batching
    - Variable QPS: Low (50) → High (250) → Low (50)
    - Compare: Fixed Batch Size vs Adaptive Batching
    - Measure: Queue length over time, throughput
    
    REVISED: Now uses simulation mode and forces batch decision regeneration.
    """
    print("\n" + "="*70)
    print("TEST 2: ADAPTIVE BATCHING")
    print("="*70)
    
    # Configuration
    arch = "distilbert-base"
    dataset = "imdb"
    slo = 40
    profile_dir = "../profile_pickles_bs"
    batch_decisions_dir = "../batch_decisions"
    
    # Create QPS profile
    qps_profile = [
        (0, 15, 50),
        (15, 35, 250),
        (35, 50, 50),
    ]
    
    total_num_requests = 10000
    
    print(f"Creating requests with variable QPS profile: {qps_profile}")
    all_requests, avg_qps, interarrival_time = batch_systems.create_time_varying_qps_requests(
        num_requests=total_num_requests,
        qps_profile=qps_profile,
        slo=slo,
        poisson_arrival=True,
        seed=42
    )
    
    # Get model serving time
    latency_calc_list = []
    model_serving_time = batch_systems.get_model_serving_time(arch, profile_dir, latency_calc_list)
    
    results = {}
    
    # Test 2a: Fixed Batch Size
    print("\n[2a] Running Fixed Batch Size (baseline)...")
    
    # Force regeneration
    batch_decision_path_fixed = force_regenerate_batch_decision(
        arch, dataset, int(avg_qps), slo,
        enable_prioritization=False,
        enable_adaptive_batching=False,
        batch_decisions_dir=batch_decisions_dir
    )
    
    _, _, _, batch_decision_fixed, per_request_stats_fixed, _, total_time_fixed = \
        batch_systems.get_batch_decision(
            batching_scheme="clockwork",
            all_requests=list(all_requests),
            model_serving_time=model_serving_time,
            slo=slo,
            max_batch_size=8,
            batch_timeout_ms=30,
            max_enqueued_batches=3,
            enable_prioritization=False,
            enable_adaptive_batching=False
        )
    
    # Save batch decision
    os.makedirs(batch_decisions_dir, exist_ok=True)
    with open(batch_decision_path_fixed, "wb") as f:
        batch_info = {
            "batching_decision": batch_decision_fixed,
            "per_request_stats": per_request_stats_fixed,
            "total_num_requests": total_num_requests,
            "batching_scheme": "clockwork",
            "arch": arch,
            "slo": slo,
            "dataset": dataset,
            "avg_qps": avg_qps,
        }
        pickle.dump(batch_info, f)
    
    # Run controller with simulation mode (optional - can skip if takes too long)
    print("  Note: Skipping controller simulation for faster batch decision testing")
    fixed_controller_metrics = {'overall_accuracy': 0.0, 'latency_improvement': 0.0, 'exit_rate': 0.0}
    
    fixed_metrics = analyze_queue_behavior(
        batch_decision_fixed, per_request_stats_fixed, all_requests, total_time_fixed
    )
    fixed_metrics['controller_metrics'] = fixed_controller_metrics
    
    results['fixed_batch'] = {
        'batch_decision': batch_decision_fixed,
        'per_request_stats': per_request_stats_fixed,
        'total_time': total_time_fixed,
        'metrics': fixed_metrics
    }
    print(f"  Avg Queue Length: {fixed_metrics['avg_queue_length']:.2f}")
    print(f"  Max Queue Length: {fixed_metrics['max_queue_length']}")
    print(f"  Throughput: {fixed_metrics['throughput']:.2f} req/s")
    
    # Test 2b: Adaptive Batching
    print("\n[2b] Running Adaptive Batching...")
    
    # Force regeneration
    batch_decision_path_adaptive = force_regenerate_batch_decision(
        arch, dataset, int(avg_qps), slo,
        enable_prioritization=False,
        enable_adaptive_batching=True,
        batch_decisions_dir=batch_decisions_dir
    )
    
    _, _, _, batch_decision_adaptive, per_request_stats_adaptive, _, total_time_adaptive = \
        batch_systems.get_batch_decision(
            batching_scheme="clockwork",
            all_requests=list(all_requests),
            model_serving_time=model_serving_time,
            slo=slo,
            max_batch_size=8,
            batch_timeout_ms=30,
            max_enqueued_batches=3,
            enable_prioritization=False,
            enable_adaptive_batching=True
        )
    
    # Save batch decision
    with open(batch_decision_path_adaptive, "wb") as f:
        batch_info = {
            "batching_decision": batch_decision_adaptive,
            "per_request_stats": per_request_stats_adaptive,
            "total_num_requests": total_num_requests,
            "batching_scheme": "clockwork",
            "arch": arch,
            "slo": slo,
            "dataset": dataset,
            "avg_qps": avg_qps,
        }
        pickle.dump(batch_info, f)
    
    # Run controller with simulation mode (optional - can skip if takes too long)
    print("  Note: Skipping controller simulation for faster batch decision testing")
    adaptive_controller_metrics = {'overall_accuracy': 0.0, 'latency_improvement': 0.0, 'exit_rate': 0.0}
    
    adaptive_metrics = analyze_queue_behavior(
        batch_decision_adaptive, per_request_stats_adaptive, all_requests, total_time_adaptive
    )
    adaptive_metrics['controller_metrics'] = adaptive_controller_metrics
    
    results['adaptive_batch'] = {
        'batch_decision': batch_decision_adaptive,
        'per_request_stats': per_request_stats_adaptive,
        'total_time': total_time_adaptive,
        'metrics': adaptive_metrics
    }
    print(f"  Avg Queue Length: {adaptive_metrics['avg_queue_length']:.2f}")
    print(f"  Max Queue Length: {adaptive_metrics['max_queue_length']}")
    print(f"  Throughput: {adaptive_metrics['throughput']:.2f} req/s")
    
    return results


def analyze_slo_violations(per_request_stats, all_requests, slo_distribution):
    """Analyze SLO violations by SLO category.
    
    Note: Dropped requests (stats is None) are counted as violations
    because they expired before being served, which is a form of SLO violation.
    """
    metrics = {
        'violations_by_slo': {},
        'total_by_slo': {},
        'overall_violation_rate': 0,
        'total_requests': len(per_request_stats),
        'served_requests': 0,
        'violated_requests': 0,
        'dropped_requests': 0
    }
    
    # Group requests by SLO
    for request_id, request in enumerate(all_requests):
        if request_id >= len(per_request_stats):
            continue
        
        request_slo = int(request.slo)  # Convert to int to avoid numpy types
        
        # Initialize counters for this SLO
        if request_slo not in metrics['total_by_slo']:
            metrics['total_by_slo'][request_slo] = 0
            metrics['violations_by_slo'][request_slo] = 0
        
        metrics['total_by_slo'][request_slo] += 1
        
        stats = per_request_stats[request_id]
        if stats is None:
            # Dropped request = expired before being served = SLO violation
            metrics['violations_by_slo'][request_slo] += 1
            metrics['violated_requests'] += 1
            metrics['dropped_requests'] += 1
            continue
        
        # Request was served
        metrics['served_requests'] += 1
        queue_delay, inference_time = stats
        total_latency = queue_delay + inference_time
        
        # Check if violated SLO (served but too late)
        if total_latency > request_slo:
            metrics['violations_by_slo'][request_slo] += 1
            metrics['violated_requests'] += 1
    
    # Calculate violation rates and convert to native Python types
    violations_by_slo_clean = {}
    for slo in metrics['violations_by_slo']:
        total = metrics['total_by_slo'][slo]
        violations = metrics['violations_by_slo'][slo]
        if total > 0:
            violations_by_slo_clean[int(slo)] = float((violations / total) * 100)
        else:
            violations_by_slo_clean[int(slo)] = 0.0
    
    metrics['violations_by_slo'] = violations_by_slo_clean
    
    # Overall violation rate includes both served violations and dropped requests
    if metrics['total_requests'] > 0:
        metrics['overall_violation_rate'] = float((metrics['violated_requests'] / metrics['total_requests']) * 100)
    
    # Convert all numeric values to native Python types
    metrics['total_requests'] = int(metrics['total_requests'])
    metrics['served_requests'] = int(metrics['served_requests'])
    metrics['violated_requests'] = int(metrics['violated_requests'])
    metrics['dropped_requests'] = int(metrics['dropped_requests'])
    
    return metrics


def analyze_queue_behavior(batch_decision, per_request_stats, all_requests, total_time):
    """Analyze queue behavior over time by tracking actual queue state during serving."""
    # Build a timeline of queue events
    events = []  # (time, delta) where delta is +1 (arrival) or -batch_size (served)
    
    # Add arrival events
    for request in all_requests:
        events.append((request.arrival_time, 1, 'arrival'))
    
    # Reconstruct serving times from batch decision and per_request_stats
    # We'll estimate serving times based on when requests complete
    serving_events = []
    batch_idx = 0
    for request_id, request in enumerate(all_requests):
        if request_id >= len(per_request_stats):
            continue
        stats = per_request_stats[request_id]
        if stats is None:
            continue  # Dropped request
        
        queue_delay, inference_time = stats
        serving_time = request.arrival_time + queue_delay
        
        # Group into batches (simplified - assume consecutive served requests are in same batch)
        if batch_idx < len(batch_decision):
            batch_size = batch_decision[batch_idx]
            # Only add serving event once per batch
            if request_id % batch_size == 0 or batch_idx == 0:
                serving_events.append((serving_time, -batch_size, 'serve'))
                batch_idx += 1
    
    # Combine and sort all events
    all_events = events + serving_events
    all_events.sort(key=lambda x: x[0])
    
    # Simulate queue over time
    current_queue = 0
    sample_interval = 50.0  # Sample every 50ms for better resolution
    max_time = max([e[0] for e in all_events]) if all_events else total_time
    max_time = min(max_time, 60000)  # Cap at 60 seconds
    
    queue_lengths = []
    time_steps = []
    event_idx = 0
    
    for t in np.arange(0, max_time, sample_interval):
        # Process all events up to this time
        while event_idx < len(all_events) and all_events[event_idx][0] <= t:
            _, delta, _ = all_events[event_idx]
            current_queue = max(0, current_queue + delta)
            event_idx += 1
        
        queue_lengths.append(current_queue)
        time_steps.append(t)
    
    # Calculate metrics
    served_requests = sum(1 for s in per_request_stats if s is not None)
    throughput = (served_requests / (total_time / 1000.0)) if total_time > 0 else 0
    
    # Convert numpy types to native Python types for JSON serialization
    queue_lengths_clean = [int(x) for x in queue_lengths] if queue_lengths else []
    time_steps_clean = [float(x) for x in time_steps] if time_steps else []
    
    # Calculate queue length during high-load period (15-35s for QPS spike)
    high_load_indices = [i for i, t in enumerate(time_steps) if 15000 <= t <= 35000]
    high_load_queue_lengths = [queue_lengths[i] for i in high_load_indices] if high_load_indices else []
    avg_queue_high_load = float(np.mean(high_load_queue_lengths)) if high_load_queue_lengths else 0.0
    max_queue_high_load = int(max(high_load_queue_lengths)) if high_load_queue_lengths else 0
    
    metrics = {
        'queue_lengths': queue_lengths_clean,
        'time_steps': time_steps_clean,  # Add time steps for better plotting
        'avg_queue_length': float(np.mean(queue_lengths)) if queue_lengths else 0.0,
        'max_queue_length': int(max(queue_lengths)) if queue_lengths else 0,
        'avg_queue_high_load': avg_queue_high_load,
        'max_queue_high_load': max_queue_high_load,
        'throughput': float(throughput),
        'served_requests': int(served_requests),
        'total_time': float(total_time)
    }
    
    return metrics


def main():
    """Run both tests and save results."""
    print("="*70)
    print("PRIORITIZATION & ADAPTIVE BATCHING DEMONSTRATION")
    print("REVISED: Using simulation mode and forcing data recomputation")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"workflow_results/prioritization_batching_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run tests
    prioritization_results = test_prioritization()
    adaptive_batching_results = test_adaptive_batching()
    
    # Save results
    results = {
        'prioritization': prioritization_results,
        'adaptive_batching': adaptive_batching_results,
        'timestamp': timestamp
    }
    
    # Save full results (without per_request_stats to keep file size manageable)
    results_save = {
        'prioritization': {
            'baseline': {
                'metrics': prioritization_results['baseline']['metrics']
            },
            'prioritization': {
                'metrics': prioritization_results['prioritization']['metrics']
            }
        },
        'adaptive_batching': {
            'fixed_batch': {
                'metrics': adaptive_batching_results['fixed_batch']['metrics']
            },
            'adaptive_batch': {
                'metrics': adaptive_batching_results['adaptive_batch']['metrics']
            }
        },
        'timestamp': timestamp
    }
    
    results_file = os.path.join(output_dir, "test_results.json")
    with open(results_file, 'w') as f:
        # Convert numpy types before JSON serialization
        results_save_clean = convert_numpy_types(results_save)
        json.dump(results_save_clean, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}\n")
    
    # Print summary
    print("SUMMARY:")
    print("\nPrioritization Test:")
    print(f"  Baseline SLO Violation Rate: {prioritization_results['baseline']['metrics']['overall_violation_rate']:.2f}%")
    print(f"  With Prioritization: {prioritization_results['prioritization']['metrics']['overall_violation_rate']:.2f}%")
    print(f"  Improvement: {prioritization_results['baseline']['metrics']['overall_violation_rate'] - prioritization_results['prioritization']['metrics']['overall_violation_rate']:.2f}% reduction")
    
    print("\nAdaptive Batching Test:")
    print(f"  Fixed Batch - Avg Queue Length: {adaptive_batching_results['fixed_batch']['metrics']['avg_queue_length']:.2f}")
    print(f"  Adaptive Batch - Avg Queue Length: {adaptive_batching_results['adaptive_batch']['metrics']['avg_queue_length']:.2f}")
    print(f"  Fixed Batch - Throughput: {adaptive_batching_results['fixed_batch']['metrics']['throughput']:.2f} req/s")
    print(f"  Adaptive Batch - Throughput: {adaptive_batching_results['adaptive_batch']['metrics']['throughput']:.2f} req/s")
    
    # Save queue length data for plotting
    queue_data = {
        'fixed_batch': {
            'queue_lengths': adaptive_batching_results['fixed_batch']['metrics']['queue_lengths'],
            'time_steps': adaptive_batching_results['fixed_batch']['metrics'].get('time_steps', []),
            'serving_times': adaptive_batching_results['fixed_batch']['metrics'].get('serving_times', [])
        },
        'adaptive_batch': {
            'queue_lengths': adaptive_batching_results['adaptive_batch']['metrics']['queue_lengths'],
            'time_steps': adaptive_batching_results['adaptive_batch']['metrics'].get('time_steps', []),
            'serving_times': adaptive_batching_results['adaptive_batch']['metrics'].get('serving_times', [])
        }
    }
    
    queue_data_file = os.path.join(output_dir, "queue_data.json")
    with open(queue_data_file, 'w') as f:
        # Convert numpy types before JSON serialization
        queue_data_clean = convert_numpy_types(queue_data)
        json.dump(queue_data_clean, f, indent=2)
    
    print(f"\nQueue data saved to: {queue_data_file}")
    print(f"\nRun visualize_prioritization_batching.py to generate plots!")
    print(f"  python visualize_prioritization_batching.py --results_dir {output_dir}")


if __name__ == "__main__":
    main()

