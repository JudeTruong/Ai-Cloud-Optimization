import os
import numpy as np
import pandas as pd

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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "data", "sample_trace.csv")

    df = pd.read_csv(data_path)
    df = df.sort_values("time").reset_index(drop=True)

    return df


# ==============================
# ADD CPU DEMAND
# ==============================
def add_cpu_demand(df):
    np.random.seed(42)

    df["cpu_per_task"] = np.random.lognormal(mean=-3, sigma=0.3, size=len(df))
    df["cpu_demand"] = (
        df["task_arrivals"] * df["cpu_per_task"] * 10
    ).astype(int)

    # add controlled spikes
    spike_idx = np.random.choice(len(df), size=10)
    df.loc[spike_idx, "cpu_demand"] *= 2

    return df


# ==============================
# METRICS FUNCTION
# ==============================
def compute_metrics(df, inst_col, config):

    df = df.copy()

    df["capacity"] = df[inst_col] * config["capacity_per_instance"]

    df["under"] = np.maximum(0, df["cpu_demand"] - df["capacity"])
    df["over"] = np.maximum(0, df["capacity"] - df["cpu_demand"])

    df["cost"] = (
        df[inst_col] * config["instance_cost"]
        + df["under"] * config["under_penalty"]
    )

    # utilization
    df["utilization"] = df["cpu_demand"] / (df["capacity"] + 1e-6)

    stability = df[inst_col].diff().abs().sum()

    return {
        "total_cost": df["cost"].sum(),
        "total_under": df["under"].sum(),
        "total_over": df["over"].sum(),
        "sla_violation_rate":
            (df["under"] > 0).sum() / len(df) * 100,
        "stability": stability,
        "avg_utilization": df["utilization"].mean(),
        "peak_utilization": df["utilization"].max(),
    }


# ==============================
# MAIN EXPERIMENT
# ==============================
def run_experiment(config):

    # Load + preprocess
    df = load_data()
    df = add_cpu_demand(df)

    # ==============================
    # TRAIN / TEST SPLIT
    # ==============================
    split_idx = int(len(df) * 0.7)

    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    df_test = df_test.reset_index(drop=True)

    demand = df_test["cpu_demand"].values.tolist()

    # ==============================
    # BASE POLICIES (on TEST)
    # ==============================
    rolling_result = run_rolling_policy(df_test, config)
    tt_result = run_target_tracking_policy(df_test, config)
    static_metrics = run_static_policy(df_test, config)

    df_test["rolling_instances"] = rolling_result["instances"]
    df_test["tt_instances"] = tt_result["instances"]

    df_test["rolling_capacity"] = (
        df_test["rolling_instances"] * config["capacity_per_instance"]
    )
    df_test["tt_capacity"] = (
        df_test["tt_instances"] * config["capacity_per_instance"]
    )

    rolling_metrics = rolling_result["metrics"]
    tt_metrics = tt_result["metrics"]

    # ==============================
    # ML TRAINING (TRAIN ONLY)
    # ==============================
    model = train_ml_model(df_train, config["window"])

    # ==============================
    # ML POLICY (TEST ONLY)
    # ==============================
    ml_instances = run_ml_policy(
        df_test, model, config, config["window"]
    )

    df_ml = df_test.iloc[config["window"]:].copy()
    df_ml["ml_instances"] = ml_instances

    ml_metrics = compute_metrics(df_ml, "ml_instances", config)

    # ==============================
    # HYBRID POLICY
    # ==============================
    hybrid_instances = run_hybrid_policy(
        df_test, model, config, config["window"]
    )

    df_hybrid = df_test.iloc[config["window"]:].copy()
    df_hybrid["hybrid_instances"] = hybrid_instances

    hybrid_metrics = compute_metrics(df_hybrid, "hybrid_instances", config)

    # ==============================
    # ADD ML + HYBRID TO FULL DF
    # ==============================
    df_test["ml_capacity"] = 0
    df_test.loc[config["window"]:, "ml_capacity"] = (
        df_ml["ml_instances"].values * config["capacity_per_instance"]
    )

    df_test["hybrid_capacity"] = 0
    df_test.loc[config["window"]:, "hybrid_capacity"] = (
        df_hybrid["hybrid_instances"].values * config["capacity_per_instance"]
    )

    # ==============================
    # OR-TOOLS
    # ==============================
    ort_results = run_ortools_policy(
        demand=demand,
        instance_cost=config["instance_cost"],
        C=config["capacity_per_instance"],
        lambda_penalty=config["under_penalty"],
        x_max=config["max_instances"],
        delta_max=config["max_delta"]
    )

    allocation, total_under, _, sla_rate, total_obj = ort_results

    df_test["ort_capacity"] = [
        x * config["capacity_per_instance"] for x in allocation
    ]

    ort_stability = sum(
        abs(allocation[i] - allocation[i-1])
        for i in range(1, len(allocation))
    )

    ort_metrics = {
        "total_cost": total_obj,
        "total_under": total_under,
        "total_over": 0,
        "sla_violation_rate": sla_rate * 100,
        "stability": ort_stability
    }

    # ==============================
    # RETURN RESULTS
    # ==============================
    return {
        "rolling": rolling_metrics,
        "static": static_metrics,
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
        "capacity_per_instance": 5,
        "max_instances": 2000,
        "max_delta": 25,
        "instance_cost": 0.02,
        "under_penalty": 6
    }
    results = run_experiment(CONFIG)

    for name in ["rolling", "static", "target_tracking", "ml", "hybrid", "ortools"]:
        print(f"\n===== {name.upper()} =====")
        for k, v in results[name].items():
            print(f"{k}: {round(v, 4)}")

    print("\n===== End of Simulation =====\n")