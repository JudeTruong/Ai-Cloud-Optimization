import os
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import compute_metrics

# Policies
from policies.rolling import run_rolling_policy
from policies.target_tracking import run_target_tracking_policy
from policies.static import run_static_policy
from policies.ortools_policy import run_ortools_policy
from policies.ml_policy import (
    train_ml_model,
    run_ml_policy,
    run_hybrid_policy
)


# ==============================
# LOAD DATA
# ==============================
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "sample_trace.csv")

    df = pd.read_csv(data_path)

    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    if "task_arrivals" not in df.columns:
        raise ValueError("Expected column 'task_arrivals' was not found in sample_trace.csv")

    return df


# ==============================
# MAIN EXPERIMENT
# ==============================
def run_experiment(config):

    # ==============================
    # LOAD DATA
    # ==============================
    df = load_data()

    demand_col = config.get("demand_col", "task_arrivals")

    if demand_col not in df.columns:
        raise ValueError(f"Demand column '{demand_col}' not found in dataframe.")

    # ==============================
    # TRAIN / TEST SPLIT
    # ==============================
    split_idx = int(len(df) * config.get("train_ratio", 0.7))

    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)

    eval_start = config["window"]

    # ==============================
    # STATIC POLICY
    # ==============================
    df_static_eval = df_test.iloc[eval_start:].copy().reset_index(drop=True)
    static_metrics = run_static_policy(df_static_eval, config)

    # ==============================
    # ROLLING POLICY
    # ==============================
    rolling_result = run_rolling_policy(df_test, config)

    df_test["rolling_instances"] = rolling_result["instances"]
    df_test["rolling_capacity"] = (
        df_test["rolling_instances"] * config["capacity_per_instance"]
    )

    df_rolling_eval = df_test.iloc[eval_start:].copy().reset_index(drop=True)

    rolling_metrics = compute_metrics(
        df_rolling_eval,
        "rolling_instances",
        config,
        demand_col=demand_col
    )

    # ==============================
    # TARGET TRACKING POLICY
    # ==============================
    tt_result = run_target_tracking_policy(df_test, config)

    df_test["tt_instances"] = tt_result["instances"]
    df_test["tt_capacity"] = (
        df_test["tt_instances"] * config["capacity_per_instance"]
    )

    df_tt_eval = df_test.iloc[eval_start:].copy().reset_index(drop=True)

    tt_metrics = compute_metrics(
        df_tt_eval,
        "tt_instances",
        config,
        demand_col=demand_col
    )

    # ==============================
    # ML TRAINING
    # ==============================
    model = train_ml_model(df_train, config)

    # ==============================
    # PURE ML POLICY
    # ==============================
    ml_instances = run_ml_policy(
        df_test,
        model,
        config,
        config["window"]
    )

    df_ml = df_test.iloc[eval_start:].copy().reset_index(drop=True)
    df_ml["ml_instances"] = ml_instances

    ml_metrics = compute_metrics(
        df_ml,
        "ml_instances",
        config,
        demand_col=demand_col
    )

    df_test["ml_instances"] = np.nan
    df_test["ml_capacity"] = np.nan

    df_test.loc[eval_start:, "ml_instances"] = ml_instances
    df_test.loc[eval_start:, "ml_capacity"] = (
        ml_instances * config["capacity_per_instance"]
    )

    # ==============================
    # HYBRID POLICY
    # ==============================
    hybrid_instances = run_hybrid_policy(
        df_test,
        model,
        config,
        config["window"]
    )

    df_hybrid = df_test.iloc[eval_start:].copy().reset_index(drop=True)
    df_hybrid["hybrid_instances"] = hybrid_instances

    hybrid_metrics = compute_metrics(
        df_hybrid,
        "hybrid_instances",
        config,
        demand_col=demand_col
    )

    df_test["hybrid_instances"] = np.nan
    df_test["hybrid_capacity"] = np.nan

    df_test.loc[eval_start:, "hybrid_instances"] = hybrid_instances
    df_test.loc[eval_start:, "hybrid_capacity"] = (
        hybrid_instances * config["capacity_per_instance"]
    )

    # ==============================
    # OR-TOOLS POLICY
    # ==============================
    demand = df_test[demand_col].values.astype(int).tolist()

    ort_results = run_ortools_policy(
        demand=demand,
        instance_cost=config["instance_cost"],
        C=config["capacity_per_instance"],
        lambda_penalty=config["under_penalty"],
        x_max=config["max_instances"],
        delta_max=config["max_delta"],
        initial_instances=0
    )

    if ort_results is None:
        raise RuntimeError("OR-Tools failed to find a feasible solution.")

    allocation, total_under, total_instance_cost, sla_rate, total_obj = ort_results

    df_test["ort_instances"] = allocation
    df_test["ort_capacity"] = (
        df_test["ort_instances"] * config["capacity_per_instance"]
    )

    df_ort_eval = df_test.iloc[eval_start:].copy().reset_index(drop=True)

    ort_metrics = compute_metrics(
        df_ort_eval,
        "ort_instances",
        config,
        demand_col=demand_col
    )

    print("OR-Tools full-horizon objective from solver:", total_obj)
    print("OR-Tools evaluation-window cost from shared metrics:", ort_metrics["total_cost"])

    # ==============================
    # RETURN RESULTS
    # ==============================
    return {
        "static": static_metrics,
        "rolling": rolling_metrics,
        "target_tracking": tt_metrics,
        "ml": ml_metrics,
        "hybrid": hybrid_metrics,
        "ortools": ort_metrics,
        "df": df_test,
        "ort_allocation": allocation
    }
# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    CONFIG = {
        "window": 5,
        "train_ratio": 0.7,

        "demand_col": "task_arrivals",

        "capacity_per_instance": 5,
        "max_instances": 2000,
        "max_delta": 25,

        "instance_cost": 0.02,
        "under_penalty": 6,

        "static_instances": 1200,

        "target_util": 0.7,
        "cooldown": 0,
        "delay": 1,

        "ml_safety_buffer": 1.1,
        "ml_smoothing": 0.6,

        "hybrid_threshold": 1.2,
        "hybrid_ml_weight": 0.6,
        "hybrid_reactive_weight": 0.4,
        "hybrid_smoothing": 0.6
    }

    results = run_experiment(CONFIG)

    policies = [
        "static",
        "rolling",
        "target_tracking",
        "ml",
        "hybrid",
        "ortools"
    ]

    # ==============================
    # PRINT RESULTS
    # ==============================
    for name in policies:
        print(f"\n===== {name.upper()} =====")
        for k, v in results[name].items():
            print(f"{k}: {round(v, 4)}")

    print("\n===== End of Simulation =====\n")

    # ==============================
    # SAVE RESULTS
    # ==============================
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parents[2]

    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)

    rows = []

    for name in policies:
        metrics = results[name]

        rows.append({
            "policy": name,
            "total_cost": metrics.get("total_cost"),
            "sla_violation_rate": metrics.get("sla_violation_rate"),
            "total_under": metrics.get("total_under"),
            "total_over": metrics.get("total_over"),
            "stability": metrics.get("stability"),
            "avg_utilization": metrics.get("avg_utilization"),
            "peak_utilization": metrics.get("peak_utilization")
        })

    comparison_df = pd.DataFrame(rows)

    output_path = results_dir / "final_comparison.csv"
    comparison_df.to_csv(output_path, index=False)

    print(f"Saved comparison table to: {output_path}")

    trace_output_path = results_dir / "simulation_trace_output.csv"
    results["df"].to_csv(trace_output_path, index=False)

    print(f"Saved full simulation output to: {trace_output_path}")