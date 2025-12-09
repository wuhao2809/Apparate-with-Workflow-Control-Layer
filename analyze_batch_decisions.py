"""
Analyze batch decisions to understand why 50ms requests starve.
"""
import pickle
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import batch_systems

def analyze_batch_decision(batch_decision, per_request_stats, all_requests, slo_distribution):
    """Analyze when requests are served and why violations occur."""
    
    # Group requests by SLO
    requests_by_slo = {}
    for req in all_requests:
        slo = req.slo
        if slo not in requests_by_slo:
            requests_by_slo[slo] = []
        requests_by_slo[slo].append(req)
    
    print("\n" + "="*70)
    print("BATCH DECISION ANALYSIS")
    print("="*70)
    
    # Analyze serving times
    for slo in sorted(slo_distribution.keys()):
        requests = requests_by_slo[slo]
        served = []
        dropped = []
        violations = []
        
        for req in requests:
            stats = per_request_stats[req.request_id]
            if stats is None:
                dropped.append(req)
            else:
                queue_delay, inference_time = stats
                total_latency = queue_delay + inference_time
                served.append((req, queue_delay, inference_time, total_latency))
                if total_latency >= req.slo:
                    violations.append((req, total_latency))
        
        print(f"\n{slo}ms SLO Requests:")
        print(f"  Total: {len(requests)}")
        print(f"  Served: {len(served)} ({len(served)/len(requests)*100:.1f}%)")
        print(f"  Dropped: {len(dropped)} ({len(dropped)/len(requests)*100:.1f}%)")
        print(f"  Violations: {len(violations)} ({len(violations)/len(requests)*100:.1f}%)")
        
        if served:
            # Analyze serving times
            serving_times = [s[1] for s in served]  # queue_delay
            avg_queue_delay = sum(serving_times) / len(serving_times)
            max_queue_delay = max(serving_times)
            min_queue_delay = min(serving_times)
            
            print(f"  Queue Delay Stats:")
            print(f"    Avg: {avg_queue_delay:.2f}ms")
            print(f"    Min: {min_queue_delay:.2f}ms")
            print(f"    Max: {max_queue_delay:.2f}ms")
            
            # Check when requests are served relative to their deadline
            served_before_deadline = sum(1 for s in served if s[3] < s[0].slo)
            print(f"  Served Before Deadline: {served_before_deadline}/{len(served)} ({served_before_deadline/len(served)*100:.1f}%)")
            
            # Analyze violations
            if violations:
                print(f"  Violations Analysis:")
                violation_delays = [v[1] - v[0].slo for v in violations]  # How much over deadline
                avg_over = sum(violation_delays) / len(violation_delays)
                print(f"    Avg Over Deadline: {avg_over:.2f}ms")
                
                # Check if violations are due to long queue delay
                violation_queue_delays = [s[1] for s in served if s[3] >= s[0].slo]
                if violation_queue_delays:
                    avg_violation_queue_delay = sum(violation_queue_delays) / len(violation_queue_delays)
                    print(f"    Avg Queue Delay for Violations: {avg_violation_queue_delay:.2f}ms")
        
        # Sample some requests to see their timeline
        if len(served) > 0:
            print(f"  Sample Served Requests (first 5):")
            for i, (req, queue_delay, inference_time, total_latency) in enumerate(served[:5]):
                deadline_met = "✓" if total_latency < req.slo else "✗"
                print(f"    Req {req.request_id}: arrival={req.arrival_time:.1f}ms, "
                      f"queue_delay={queue_delay:.1f}ms, total={total_latency:.1f}ms/{req.slo}ms {deadline_met}")
        
        if len(dropped) > 0:
            print(f"  Sample Dropped Requests (first 5):")
            for i, req in enumerate(dropped[:5]):
                print(f"    Req {req.request_id}: arrival={req.arrival_time:.1f}ms, deadline={req.deadline:.1f}ms")


if __name__ == "__main__":
    # Re-run the test to get fresh data
    print("Running prioritization test to generate batch decisions...")
    
    # Configuration
    total_num_requests = 20000
    slo_distribution = {15: 0.4, 30: 0.3, 50: 0.3}
    qps = 200
    
    # Create requests
    all_requests, avg_qps, interarrival_time = batch_systems.create_mixed_slo_requests(
        num_requests=total_num_requests,
        slo_distribution=slo_distribution,
        qps=qps,
        poisson_arrival=True,
        seed=42
    )
    
    # Get model serving time
    latency_calc_list = []
    model_serving_time = batch_systems.get_model_serving_time(
        arch="distilbert-base",
        profile_dir="../profile_pickles_bs",
        latency_calc_list=latency_calc_list
    )
    
    print("\n" + "="*70)
    print("BASELINE (FIFO)")
    print("="*70)
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
    
    analyze_batch_decision(batch_decision_baseline, per_request_stats_baseline, 
                          all_requests, slo_distribution)
    
    print("\n" + "="*70)
    print("WITH PRIORITIZATION")
    print("="*70)
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
    
    analyze_batch_decision(batch_decision_prio, per_request_stats_prio, 
                          all_requests, slo_distribution)
    
    # Compare specific requests
    print("\n" + "="*70)
    print("COMPARISON: Same Requests in Both Configs")
    print("="*70)
    
    # Find 50ms requests that are served in baseline but dropped/violated in prioritization
    baseline_50ms_served = [req for req in all_requests 
                           if req.slo == 50 and per_request_stats_baseline[req.request_id] is not None]
    prio_50ms_served = [req for req in all_requests 
                       if req.slo == 50 and per_request_stats_prio[req.request_id] is not None]
    
    baseline_50ms_ids = {req.request_id for req in baseline_50ms_served}
    prio_50ms_ids = {req.request_id for req in prio_50ms_served}
    
    served_in_baseline_not_prio = baseline_50ms_ids - prio_50ms_ids
    served_in_prio_not_baseline = prio_50ms_ids - baseline_50ms_ids
    
    print(f"\n50ms Requests:")
    print(f"  Served in Baseline: {len(baseline_50ms_ids)}")
    print(f"  Served in Prioritization: {len(prio_50ms_ids)}")
    print(f"  Served in Baseline but NOT in Prioritization: {len(served_in_baseline_not_prio)}")
    print(f"  Served in Prioritization but NOT in Baseline: {len(served_in_prio_not_baseline)}")
    
    if served_in_baseline_not_prio:
        print(f"\n  Example: Request served in baseline but not in prioritization:")
        example_id = list(served_in_baseline_not_prio)[0]
        req = all_requests[example_id]
        baseline_stats = per_request_stats_baseline[example_id]
        prio_stats = per_request_stats_prio[example_id]
        
        print(f"    Request {example_id}: SLO={req.slo}ms, arrival={req.arrival_time:.1f}ms, deadline={req.deadline:.1f}ms")
        if baseline_stats:
            queue_delay, inference_time = baseline_stats
            total = queue_delay + inference_time
            print(f"    Baseline: queue_delay={queue_delay:.1f}ms, total={total:.1f}ms (served at {req.arrival_time + queue_delay:.1f}ms)")
        if prio_stats is None:
            print(f"    Prioritization: DROPPED")
        else:
            queue_delay, inference_time = prio_stats
            total = queue_delay + inference_time
            print(f"    Prioritization: queue_delay={queue_delay:.1f}ms, total={total:.1f}ms (served at {req.arrival_time + queue_delay:.1f}ms)")

