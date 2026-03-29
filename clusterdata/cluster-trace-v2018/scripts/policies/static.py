import numpy as np

def run_static_policy(df, config, static_instances=1200):

    static_capacity = static_instances * config["capacity_per_instance"]

    under = np.maximum(0, df["cpu_demand"] - static_capacity)
    over = np.maximum(0, static_capacity - df["cpu_demand"])

    cost = (
        static_instances * config["instance_cost"]
        + under * config["under_penalty"]
    )

    return {
        "total_cost": cost.sum(),
        "total_under": under.sum(),
        "total_over": over.sum(),
        "sla_violation_rate":
            (under > 0).sum() / len(df) * 100
    }