# Apparate with Workflow Control Layer

This project extends [Apparate](https://arxiv.org/abs/2312.05385) - a system for optimizing ML serving using early exits - with a **workflow control layer** that improves latency-throughput trade-offs through intelligent request management.

## Overview

### Original Apparate

Apparate is a system that uses early exits in neural networks to balance latency and throughput in ML serving. It allows models to exit early when confident enough, reducing inference time while maintaining accuracy.

### Workflow Control Layer (This Project)

This project adds a workflow control layer on top of Apparate with three key optimizations:

1. **SLO-Aware Request Prioritization**: Prioritizes requests based on deadline urgency
2. **Confidence-Aware Adaptive Batching**: Dynamically adjusts batch sizes based on predicted early exit confidence
3. **Queue-Aware Feedback Control**: Adjusts early exit thresholds based on real-time queue metrics

## Key Features

- **Request Prioritizer**: Sorts requests by SLO deadline urgency
- **Adaptive Batcher**: Dynamically adjusts batch sizes based on predicted early exit confidence
- **Queue State Monitor**: Tracks queue length, wait times, SLO violations
- **Feedback Adjuster**: Dynamically adjusts early exit thresholds based on queue state

## How It Reduces Latency

The workflow control layer reduces latency through three complementary mechanisms:

1. **SLO-Aware Prioritization**: By processing urgent requests (those closer to their deadline) first, the system minimizes SLO violations and ensures time-sensitive requests complete faster. This reduces the average latency for high-priority requests.

2. **Confidence-Aware Adaptive Batching**:

   - For high-confidence requests (likely to exit early), the system uses larger batches, improving throughput without increasing latency
   - For low-confidence requests, smaller batches prevent long waits, reducing tail latency
   - This optimization balances batch efficiency with early exit opportunities

3. **Queue-Aware Feedback Control**:
   - When the queue is congested or SLO violations are high, the system automatically lowers early exit thresholds, allowing more requests to exit early
   - This increases the early exit rate, directly reducing inference latency
   - When the queue is healthy, thresholds can be raised slightly to maintain accuracy

Together, these mechanisms work synergistically: prioritization ensures urgent requests are handled first, adaptive batching optimizes batch processing efficiency, and feedback control dynamically adjusts the early exit behavior to respond to system load.

## Installation

For Mac M1:

```bash
./install_with_old_rust.sh
```

## Usage

Run with workflow control enabled:

```bash
python controller.py --enable_workflow --dataset auburn --arch resnet18 ...
```

## Results

**CV Workload (auburn + resnet18)**:

- Latency improvement: 52.58% → 53.17% (+0.59%)

**NLP Workload (imdb + distilbert-base)**:

- Latency improvement: 9.23% → 16.53% (+7.30%)

## References

### Original Apparate Paper

```
@article{dai2023apparate,
  title={Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving},
  author={Dai, Yinwei and Pan, Rui and Iyer, Anand and Li, Kai and Netravali, Ravi},
  journal={arXiv preprint arXiv:2312.05385},
  year={2023}
}
```

### This Project

This project extends Apparate with a workflow control layer for improved ML serving performance.
