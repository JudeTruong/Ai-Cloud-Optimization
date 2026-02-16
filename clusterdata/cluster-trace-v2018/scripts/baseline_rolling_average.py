import os
import numpy as np
import pandas as pd


def run_experiment(config, static_instances=10):

    # ==============================
    # LOAD DATA
    # ==============================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "data", "sample_trace.csv")

    df = pd.read_csv(data_path)
    df = df.sort_values("time").reset_index(drop=True)

    # ==============================
    # ROLLING AVERAGE PREDICTOR
    # ==============================

    df["prediction"] = (
        df["task_arrivals"]
        .rolling(window=config["window"])
        .mean()
        .shift(1)
    )

    df = df.dropna().reset_index(drop=True)

    # ==============================
    # ROLLING POLICY WITH SMOOTHNESS
    # ==============================

    instances = []
    prev_x = 0

    for pred in df["prediction"]:
        desired = int(np.ceil(pred / config["capacity_per_instance"]))

        # max bound
        desired = min(desired, config["max_instances"])

        # smoothness constraint
        if desired > prev_x:
            x_t = min(prev_x + config["max_delta"], desired)
        else:
            x_t = max(prev_x - config["max_delta"], desired)

        instances.append(x_t)
        prev_x = x_t

    df["required_instances"] = instances

    # ==============================
    # ROLLING METRICS
    # ==============================

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
    # STATIC BASELINE
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

    return {
        "rolling": rolling_metrics,
        "static": static_metrics
    }


# =========================================
# MAIN EXECUTION
# =========================================

if __name__ == "__main__":

    CONFIG = {
        "window": 3,
        "capacity_per_instance": 100,
        "max_instances": 50,
        "max_delta": 5,
        "instance_cost": 1,
        "under_penalty": 5
    }

    np.random.seed(42)

    results = run_experiment(CONFIG)

    print("\n===== Rolling Policy =====")
    for k, v in results["rolling"].items():
        print(f"{k}: {round(v, 2)}")

    print("\n===== Static Policy =====")
    for k, v in results["static"].items():
        print(f"{k}: {round(v, 2)}")

    print("\n===== End of Simulation =====\n")
