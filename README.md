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

## Installation

For Mac M1:

```bash
./install_with_old_rust.sh
```

For Linux with GPU:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-gpu.txt
```

See [INSTALL.md](INSTALL.md) for detailed instructions.

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
