import numpy as np


def compute_metrics(df, inst_col, config, demand_col=None):
    """
    Compute all autoscaling evaluation metrics in one consistent place.

    demand_col is the workload column used for evaluation.
    For the main experiment, we use task_arrivals consistently.
    """

    df = df.copy()

    if demand_col is None:
        demand_col = config.get("demand_col", "task_arrivals")

    if demand_col not in df.columns:
        raise ValueError(f"Demand column '{demand_col}' not found in dataframe.")

    df["capacity"] = df[inst_col] * config["capacity_per_instance"]

    df["under"] = np.maximum(0, df[demand_col] - df["capacity"])
    df["over"] = np.maximum(0, df["capacity"] - df[demand_col])

    df["cost"] = (
        df[inst_col] * config["instance_cost"]
        + df["under"] * config["under_penalty"]
    )

    df["utilization"] = df[demand_col] / (df["capacity"] + 1e-6)

    stability = df[inst_col].diff().abs().sum()

    return {
        "total_cost": df["cost"].sum(),
        "total_under": df["under"].sum(),
        "total_over": df["over"].sum(),
        "sla_violation_rate": (df["under"] > 0).sum() / len(df) * 100,
        "stability": stability,
        "avg_utilization": df["utilization"].mean(),
        "peak_utilization": df["utilization"].max(),
    }