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

    # Utilization is only meaningful when allocated capacity is greater than 0.
    # If capacity is 0 and demand is positive, that interval is already captured
    # by under-provisioning and SLA violation metrics.
    df["utilization"] = np.where(
        df["capacity"] > 0,
        df[demand_col] / df["capacity"],
        np.nan
    )

    stability = df[inst_col].diff().abs().sum()

    return {
        "total_cost": df["cost"].sum(),
        "total_under": df["under"].sum(),
        "total_over": df["over"].sum(),
        "sla_violation_rate": (df["under"] > 0).sum() / len(df) * 100,
        "stability": stability,
        "avg_utilization": np.nanmean(df["utilization"]),
        "peak_utilization": np.nanmax(df["utilization"]),
    }