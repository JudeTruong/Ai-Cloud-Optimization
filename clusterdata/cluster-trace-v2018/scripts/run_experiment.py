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


SYSTEM_CONFIG = {
    "scaling_delay": 2,
    "cooldown": 3
}

# ==============================
# LOAD DATA
# ==============================
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "data", "sample_trace.csv")

    df = pd.read_csv(data_path)
    df = df.sort_values("time").reset_index(drop=True)

    df = df.iloc[:300].copy()

    return df


# ==============================
# ADD CPU DEMAND
# ==============================
def add_cpu_demand(df):
    np.random.seed(42)

    # realistic heavy-tailed distribution
    df["cpu_per_task"] = np.random.lognormal(mean=-3, sigma=0.1, size=len(df))
    df["cpu_per_task"] = np.clip(df["cpu_per_task"], 0, 0.2)

    df["cpu_demand"] = (
        df["task_arrivals"] * df["cpu_per_task"] * 10
    ).astype(int)

    # 🔥 add burst spikes AFTER computing demand
    spike_idx = np.random.choice(len(df), size=10)
    df.loc[spike_idx, "cpu_demand"] *= 1

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

    # ======================
    # NEW REALISTIC METRICS
    # ======================

    # utilization
    df["utilization"] = df["cpu_demand"] / (df["capacity"] + 1e-6)

    avg_util = df["utilization"].mean()
    peak_util = df["utilization"].max()

    # over-provision ratio
    over_ratio = df["over"].sum() / (df["capacity"].sum() + 1e-6)

    # SLA severity (how bad violations are)
    if (df["under"] > 0).any():
        avg_under = df.loc[df["under"] > 0, "under"].mean()
    else:
        avg_under = 0

    stability = df[inst_col].diff().abs().sum()

    return {
        "total_cost": df["cost"].sum(),
        "total_under": df["under"].sum(),
        "total_over": df["over"].sum(),
        "sla_violation_rate":
            (df["under"] > 0).sum() / len(df) * 100,
        "stability": stability,

        # 🔥 new metrics
        "avg_utilization": avg_util,
        "peak_utilization": peak_util,
        "over_provision_ratio": over_ratio,
        "avg_sla_severity": avg_under
    }


# ==============================
# MAIN EXPERIMENT
# ==============================
def run_experiment(config):

    # Load data
    df = load_data()
    df = add_cpu_demand(df)

    demand = df["cpu_demand"].values.tolist()

    # ==============================
    # BASE POLICIES
    # ==============================
    rolling_result = run_rolling_policy(df, config)
    static_metrics = run_static_policy(df, config)
    tt_result = run_target_tracking_policy(df, config)

    # Extract instances
    df["rolling_instances"] = rolling_result["instances"]
    df["tt_instances"] = tt_result["instances"]

    # Build capacity (for animation)
    df["rolling_capacity"] = (
        df["rolling_instances"] * config["capacity_per_instance"]
    )
    df["tt_capacity"] = (
        df["tt_instances"] * config["capacity_per_instance"]
    )

    rolling_metrics = rolling_result["metrics"]
    tt_metrics = tt_result["metrics"]

    # ==============================
    # ML + HYBRID
    # ==============================
    model = train_ml_model(df, config["window"])

    # ML
    ml_instances = run_ml_policy(df, model, config, config["window"])
    df_ml = df.iloc[config["window"]:].copy()
    df_ml["ml_instances"] = ml_instances

    ml_metrics = compute_metrics(df_ml, "ml_instances", config)

    # Hybrid
    hybrid_instances = run_hybrid_policy(df, model, config, config["window"])
    df_hybrid = df.iloc[config["window"]:].copy()
    df_hybrid["hybrid_instances"] = hybrid_instances

    hybrid_metrics = compute_metrics(df_hybrid, "hybrid_instances", config)

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

    df["ort_capacity"] = [
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

        # ML
    df["ml_capacity"] = np.nan
    df.loc[config["window"]:, "ml_capacity"] = (
        df_ml["ml_instances"] * config["capacity_per_instance"]
    )

    # Hybrid
    df["hybrid_capacity"] = np.nan
    df.loc[config["window"]:, "hybrid_capacity"] = (
        df_hybrid["hybrid_instances"] * config["capacity_per_instance"]
    )

    # ==============================
    # RETURN EVERYTHING
    # ==============================
    return {
        "rolling": rolling_metrics,
        "static": static_metrics,
        "target_tracking": tt_metrics,
        "ml": ml_metrics,
        "hybrid": hybrid_metrics,
        "ortools": ort_metrics,
        "df": df,
        "ort_allocation": allocation
    }


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    CONFIG = {
        "window": 5,
        "capacity_per_instance": 4,   # moderate VM size
        "max_instances": 2000,         # realistic cluster limit
        "max_delta": 25,              # moderate scaling speed
        "instance_cost": 0.02,        # normalized cost
        "under_penalty": 6            # SLA penalty
    }

    results = run_experiment(CONFIG)
    for name in ["rolling", "static", "target_tracking", "ml", "hybrid", "ortools"]:
        print(f"\n===== {name.upper()} =====")
        for k, v in results[name].items():
            print(f"{k}: {round(v, 4) if isinstance(v, float) else v}")

    print("\n===== End of Simulation =====\n")