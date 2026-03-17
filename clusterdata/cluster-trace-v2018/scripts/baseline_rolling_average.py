import os
import numpy as np
import pandas as pd

from ortools_policy import run_ortools_policy


def run_experiment(config, static_instances=10):

    # ==============================
    # LOAD DATA (ONCE)
    # ==============================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "data", "sample_trace.csv")

    df = pd.read_csv(data_path)
    df = df.sort_values("time").reset_index(drop=True)

    # Use EXACT same 300 windows as OR-Tools
    df = df.iloc[:300].copy()

    demand = df["task_arrivals"].values.tolist()

    # ==============================
    # ROLLING PREDICTOR
    # ==============================

    df["prediction"] = (
        df["task_arrivals"]
        .rolling(window=config["window"])
        .max()
        .shift(1)
    )

    df["prediction"] = df["prediction"].bfill()

    # ==============================
    # ROLLING POLICY
    # ==============================

    instances = []
    prev_x = 0

    for pred in df["prediction"]:
        desired = int(np.ceil(pred / config["capacity_per_instance"]))

        desired = min(desired, config["max_instances"])

        if desired > prev_x:
            x_t = min(prev_x + config["max_delta"], desired)
        else:
            x_t = max(prev_x - config["max_delta"], desired)

        instances.append(x_t)
        prev_x = x_t

    df["required_instances"] = instances

    df["capacity"] = (
        df["required_instances"] * config["capacity_per_instance"]
    )

    df["under"] = np.maximum(0, df["task_arrivals"] - df["capacity"])
    df["over"] = np.maximum(0, df["capacity"] - df["task_arrivals"])

    df["cost"] = (
        df["required_instances"] * config["instance_cost"]
        + df["under"] * config["under_penalty"]
    )

    rolling_metrics = {
        "total_cost": df["cost"].sum(),
        "total_under": df["under"].sum(),
        "total_over": df["over"].sum(),
        "sla_violation_rate":
            (df["under"] > 0).sum() / len(df) * 100
    }
    # ==============================
    # TARGET TRACKING POLICY
    # ==============================

    TARGET_UTIL = 0.7
    COOLDOWN = 0
    DELAY = 1

    instances = []
    prev_x = 0
    last_scale_time = -COOLDOWN

    pending = []  # (time_to_apply, new_instances)

    current_instances = 0

    for t in range(len(df)):

        demand_t = df.loc[t, "task_arrivals"]

        # --- Apply delayed changes ---
        for change in pending[:]:
            apply_time, new_x = change
            if t >= apply_time:
                current_instances = new_x
                pending.remove(change)

        # --- Compute required instances ---
        desired = int(np.ceil(
            demand_t / (TARGET_UTIL * config["capacity_per_instance"])
        ))

        desired = min(desired, config["max_instances"])

        # --- Cooldown + delta constraint ---
        if t - last_scale_time >= COOLDOWN:

            if desired > current_instances:
                new_x = min(current_instances + config["max_delta"], desired)
            else:
                new_x = max(current_instances - config["max_delta"], desired)

            if new_x != current_instances:
                pending.append((t + DELAY, new_x))
                last_scale_time = t

        instances.append(current_instances)

    df["tt_instances"] = instances

    df["tt_capacity"] = (
        df["tt_instances"] * config["capacity_per_instance"]
    )

    df["tt_under"] = np.maximum(
        0, df["task_arrivals"] - df["tt_capacity"]
    )

    df["tt_over"] = np.maximum(
        0, df["tt_capacity"] - df["task_arrivals"]
    )

    df["tt_cost"] = (
        df["tt_instances"] * config["instance_cost"]
        + df["tt_under"] * config["under_penalty"]
    )

    tt_metrics = {
        "total_cost": df["tt_cost"].sum(),
        "total_under": df["tt_under"].sum(),
        "total_over": df["tt_over"].sum(),
        "sla_violation_rate":
            (df["tt_under"] > 0).sum() / len(df) * 100
    }
    # ==============================
    # STATIC POLICY
    # ==============================

    static_capacity = (
        static_instances * config["capacity_per_instance"]
    )

    static_under = np.maximum(
        0, df["task_arrivals"] - static_capacity
    )

    static_over = np.maximum(
        0, static_capacity - df["task_arrivals"]
    )

    static_cost = (
        static_instances * config["instance_cost"]
        + static_under * config["under_penalty"]
    )

    static_metrics = {
        "total_cost": static_cost.sum(),
        "total_under": static_under.sum(),
        "total_over": static_over.sum(),
        "sla_violation_rate":
            (static_under > 0).sum() / len(df) * 100
    }

    # ==============================
    # OR-TOOLS POLICY
    # ==============================

    ort_results = run_ortools_policy(
        demand=demand,
        instance_cost=config["instance_cost"],
        C=config["capacity_per_instance"],
        lambda_penalty=config["under_penalty"],
        x_max=config["max_instances"],
        delta_max=config["max_delta"]
    )

    allocation, total_under, total_instance_cost, sla_rate, total_obj = ort_results

    ort_metrics = {
        "total_cost": total_obj,
        "total_under": total_under,
        "total_over": 0,
        "sla_violation_rate": sla_rate * 100
    }

    return {
        "rolling": rolling_metrics,
        "static": static_metrics,
        "ortools": ort_metrics,
        "target_tracking": tt_metrics
    }


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    CONFIG = {
        "window": 3,
        "capacity_per_instance": 100,
        "max_instances": 250,
        "max_delta": 10,
        "instance_cost": 10,
        "under_penalty": 50
    }

    results = run_experiment(CONFIG)

    print("\n===== Rolling Policy =====")
    for k, v in results["rolling"].items():
        print(f"{k}: {round(v, 2)}")

    print("\n===== Static Policy =====")
    for k, v in results["static"].items():
        print(f"{k}: {round(v, 2)}")

    print("\n===== OR-Tools Offline Optimal =====")
    for k, v in results["ortools"].items():
        print(f"{k}: {round(v, 2)}")
    print("\n===== COMPARISON  =====")

    rolling_cost = results["rolling"]["total_cost"]
    optimal_cost = results["ortools"]["total_cost"]

    gap = (rolling_cost - optimal_cost) / optimal_cost

    print("Optimality gap:", gap)
    gap_percent = gap * 100
    print("Optimality gap (%):", round(gap_percent, 2))
    print(results["rolling"]["total_under"])
    print(results["rolling"]["sla_violation_rate"])
    print("\n===== Target Tracking Policy =====")
    for k, v in results["target_tracking"].items():
        print(f"{k}: {round(v, 2)}")
    print("\n===== End of Simulation =====\n")
