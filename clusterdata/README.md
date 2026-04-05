# AI Cloud Autoscaling Optimization

## Overview
This project evaluates and compares multiple autoscaling strategies for cloud resource allocation using a trace-driven simulation. The goal is to understand how different policies balance infrastructure cost, SLA violations, and resource utilization under dynamic workloads.

## Objectives
- Compare baseline, predictive, and optimization-based autoscaling policies
- Analyze tradeoffs between cost, reliability, and stability
- Simulate real-world cloud workloads using trace data
- Identify strategies that improve efficiency and reduce SLA violations

## Autoscaling Policies Implemented
- **Static Policy** – Fixed number of instances
- **Rolling Average** – Uses recent demand trends
- **Target Tracking** – Maintains a target utilization level
- **Machine Learning (ML)** – Predicts future demand using regression
- **Hybrid Policy** – Combines ML prediction with reactive scaling
- **OR-Tools Optimization** – Offline optimal solution (upper bound)

## Dataset
The simulation uses workload data derived from the **Alibaba Cluster Trace (2018)**, which captures real-world cloud task arrival patterns.

- Time windows: fixed intervals
- Workload: aggregated task arrivals per interval
- Sampling: consistent subset used across all policies
- Reproducibility: fixed random seed

## Evaluation Metrics
The following metrics are used to evaluate performance:

- **Total Cost** – Infrastructure cost + under-provision penalty
- **SLA Violation Rate (%)** – % of time intervals with unmet demand
- **Under-Provisioning** – Total unmet workload demand
- **Over-Provisioning** – Excess allocated capacity
- **Stability** – Change in resource allocation over time

## How It Works
1. Load workload trace data
2. Simulate demand over time intervals
3. Apply each autoscaling policy
4. Compute performance metrics
5. Compare results across policies

## How to Run

### Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn
- ortools

### Run Simulation
```bash
python run_experiment.py