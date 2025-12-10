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

### Prerequisites

- Python 3.8+
- All Apparate dependencies (see `environment.yml`)

### Setup

For Mac M1:

```bash
./install_with_old_rust.sh
```

For other systems, install dependencies from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate apparate
```

## Running Experiments

### Quick Start: Basic Workflow Experiments

Run baseline and optimized experiments for CV and NLP workloads:

```bash
cd apparate
python run_workflow_experiments.py
```

This script will:

- Run baseline experiments (without workflow control)
- Run optimized experiments (with full workflow control)
- Save results to `workflow_results/`
- Generate a summary in `workflow_results/experiment_summary.json`

### Comprehensive Experiments

For more detailed experiments including QPS sweeps, SLO sweeps, and component analysis:

```bash
# QPS sweep (fixed SLO, varying QPS)
python run_comprehensive_experiments.py \
    --experiment_type qps_sweep \
    --dataset imdb \
    --arch distilbert-base \
    --slo 60 \
    --qps_list 30 90 150

# SLO sweep (fixed QPS, varying SLO)
python run_comprehensive_experiments.py \
    --experiment_type slo_sweep \
    --dataset imdb \
    --arch distilbert-base \
    --qps 90 \
    --slo_list 25 60 120

# Mixed QPS×SLO grid
python run_comprehensive_experiments.py \
    --experiment_type mixed \
    --dataset imdb \
    --arch distilbert-base \
    --qps_list 30 60 90 120 150 \
    --slo_list 25 60 120

# Component analysis (test individual mechanisms)
python run_component_analysis_enhanced.py \
    --dataset imdb \
    --arch distilbert-base \
    --qps 90 \
    --slo_distribution "30:0.3,60:0.4,120:0.3"
```

### Running Individual Experiments

To run a single experiment with workflow control:

```bash
python controller.py \
    --enable_workflow \
    --enable_prioritization \
    --enable_adaptive_batching \
    --enable_feedback \
    --dataset imdb \
    --arch distilbert-base \
    --qps 90 \
    --slo 60 \
    --batch_size 8 \
    --simulation_pickle_path ../simulation_pickles/imdb_distilbert-base.pickle \
    --bootstrap_pickle_path ../bootstrap_pickles/bootstrap_imdb_distilbert-base.pickle \
    --batch_decision_path ../batch_decisions/clockwork_distilbert-base_60_azure_90.pickle
```

### Disabling Specific Features

You can disable individual workflow features:

```bash
python controller.py \
    --enable_workflow \
    --disable_prioritization \      # Disable prioritization only
    --disable_adaptive_batching \    # Disable adaptive batching only
    --disable_feedback              # Disable feedback control only
```

## Generating Results and Figures

### Analyzing Experiment Results

After running experiments, analyze the results:

```bash
python analyze_workflow_results.py
```

This generates:

- Latency CDF plots
- Metric comparison charts
- Analysis report (JSON)

### Generating Paper Figures

#### QPS Sweep Analysis

```bash
# First run QPS sweep experiments
python run_comprehensive_experiments.py \
    --experiment_type qps_sweep \
    --dataset imdb \
    --arch distilbert-base \
    --slo 60 \
    --qps_list 30 90 150

# Then visualize
python visualize_sweep_results.py \
    --results_dir workflow_results/exp-YYYYMMDD_HHMMSS \
    --output_dir figures \
    --plot_type qps_sweep
```

#### SLO Sweep Analysis

```bash
# First run SLO sweep experiments
python run_comprehensive_experiments.py \
    --experiment_type slo_sweep \
    --dataset imdb \
    --arch distilbert-base \
    --qps 90 \
    --slo_list 25 60 120

# Then visualize
python visualize_sweep_results.py \
    --results_dir workflow_results/exp-YYYYMMDD_HHMMSS \
    --output_dir figures \
    --plot_type slo_sweep
```

#### Component-Level Analysis

```bash
# Run component analysis
python run_component_analysis_enhanced.py \
    --dataset imdb \
    --arch distilbert-base \
    --qps 90 \
    --slo_distribution "30:0.3,60:0.4,120:0.3"

# Visualize component contributions
python visualize_component_analysis.py \
    --results_dir workflow_results/component_analysis_enhanced-YYYYMMDD_HHMMSS \
    --output_dir figures
```

#### Mixed QPS×SLO Heatmap

```bash
# Run mixed experiments
python run_comprehensive_experiments.py \
    --experiment_type mixed \
    --dataset imdb \
    --arch distilbert-base \
    --qps_list 30 60 90 120 150 \
    --slo_list 25 60 120

# Generate heatmap
python visualize_sweep_results.py \
    --results_dir workflow_results/exp-YYYYMMDD_HHMMSS \
    --output_dir figures \
    --plot_type heatmap
```

#### Prioritization and Adaptive Batching Analysis

```bash
# Run prioritization and batching tests
python test_prioritization_batching.py

# Visualize results
python visualize_prioritization_batching.py \
    --results_dir workflow_results/prioritization_batching_test_YYYYMMDD_HHMMSS
```

## Reproducing Paper Results

To reproduce the results reported in the paper, follow these steps:

### 1. Prerequisites

Ensure you have:

- All required data files in `../simulation_pickles/`, `../bootstrap_pickles/`, and `../batch_decisions/`
- Python environment with all dependencies installed
- Sufficient disk space for results (experiments generate JSON and log files)

### 2. Main Results (DistilBERT + IMDB)

The paper reports results for DistilBERT-base on IMDB dataset. To reproduce:

```bash
# Component analysis at QPS=90, mixed SLOs (30ms/60ms/120ms)
python run_component_analysis_enhanced.py \
    --dataset imdb \
    --arch distilbert-base \
    --qps 90 \
    --slo_distribution "30:0.3,60:0.4,120:0.3"

# QPS sweep at SLO=60ms
python run_comprehensive_experiments.py \
    --experiment_type qps_sweep \
    --dataset imdb \
    --arch distilbert-base \
    --slo 60 \
    --qps_list 30 90 150

# SLO sweep at QPS=90
python run_comprehensive_experiments.py \
    --experiment_type slo_sweep \
    --dataset imdb \
    --arch distilbert-base \
    --qps 90 \
    --slo_list 25 60 120

# Mixed QPS×SLO grid
python run_comprehensive_experiments.py \
    --experiment_type mixed \
    --dataset imdb \
    --arch distilbert-base \
    --qps_list 30 60 90 120 150 \
    --slo_list 25 60 120
```

### 3. Expected Results

After running the component analysis, you should see:

- **Baseline Apparate**: ~8.74% latency improvement
- **Full workflow control**: ~15.65% latency improvement
- **Feedback control only**: ~15.27% latency improvement
- **Accuracy**: ~98.2-98.7% across all configurations

### 4. Generate All Paper Figures

```bash
# Create figures directory
mkdir -p figures

# Generate all visualizations
python visualize_sweep_results.py \
    --results_dir workflow_results/exp-YYYYMMDD_HHMMSS \
    --output_dir figures

python visualize_component_analysis.py \
    --results_dir workflow_results/component_analysis_enhanced-YYYYMMDD_HHMMSS \
    --output_dir figures

python visualize_prioritization_batching.py \
    --results_dir workflow_results/prioritization_batching_test_YYYYMMDD_HHMMSS \
    --output_dir figures
```

### 5. Verify Results

Check the generated JSON files in `workflow_results/` for:

- `latency_improvement`: Should match paper values
- `overall_accuracy`: Should be ~98.2-98.7%
- `early_exit_rate`: Should increase with workflow control
- `slo_violation_rate`: Should decrease for tight SLOs with prioritization

## Results Summary

**CV Workload (auburn + resnet18)**:

- Latency improvement: 52.58% → 53.17% (+0.59%)

**NLP Workload (imdb + distilbert-base)**:

- Latency improvement: 9.23% → 16.53% (+7.30%)
- Full workflow control: 15.65% (vs 8.74% baseline Apparate)
- 79% relative improvement over baseline

## File Structure

```
apparate/
├── controller.py                    # Main serving controller
├── workflow_controller.py          # Workflow control layer implementation
├── batch_systems.py                # Batch scheduling with prioritization
├── run_workflow_experiments.py     # Basic experiment runner
├── run_comprehensive_experiments.py # Comprehensive experiments (QPS/SLO sweeps)
├── run_component_analysis_enhanced.py # Component-level analysis
├── analyze_workflow_results.py     # Result analysis and metrics
├── visualize_sweep_results.py     # QPS/SLO sweep visualizations
├── visualize_component_analysis.py # Component analysis plots
├── visualize_prioritization_batching.py # Prioritization/batching plots
└── workflow_results/               # Experiment results directory
```

## Troubleshooting

### Missing Data Files

If experiments fail with missing pickle files:

1. Check that `../simulation_pickles/` contains the required dataset files
2. Check that `../bootstrap_pickles/` contains bootstrap statistics
3. Check that `../batch_decisions/` contains batch decision files (or they will be auto-generated)

### Batch Decision Generation

Batch decisions are automatically generated if missing. If generation fails:

- Ensure `batch_systems.py` and `utils.py` are importable
- Check that profile pickles exist in `profile_pickles_bs/`
- Verify the batching scheme matches available profiles

### Simulation Mode

All experiments run in simulation mode (no GPU required). This uses precomputed confidence and latency profiles, making it suitable for Mac M1 and CPU-only environments.

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

This project extends Apparate with a workflow control layer for improved ML serving performance. For more details, see the paper and GitHub repository.
