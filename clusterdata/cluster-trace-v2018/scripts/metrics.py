import numpy as np


def compute_metrics(df, inst_col, config, demand_col="cpu_demand"):
    """
    Compute all autoscaling evaluation metrics in one consistent place.

    demand_col:
        The workload demand column used for evaluation.
        Currently this is cpu_demand, but we will revisit this in the next step.
    """

    df = df.copy()

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