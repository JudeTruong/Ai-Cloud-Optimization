import numpy as np

def run_rolling_policy(df, config):

    df = df.copy()

    df["prediction"] = (
        df["task_arrivals"]
        .rolling(window=config["window"])
        .max()
        .shift(1)
    )

    df["prediction"] = df["prediction"].bfill()

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

    df["instances"] = instances

    # Metrics
    df["capacity"] = df["instances"] * config["capacity_per_instance"]
    df["under"] = np.maximum(0, df["cpu_demand"] - df["capacity"])
    df["over"] = np.maximum(0, df["capacity"] - df["cpu_demand"])

    df["cost"] = (
        df["instances"] * config["instance_cost"]
        + df["under"] * config["under_penalty"]
    )

    stability = df["instances"].diff().abs().sum()

    return {
    "metrics": {
        "total_cost": df["cost"].sum(),
        "total_under": df["under"].sum(),
        "total_over": df["over"].sum(),
        "sla_violation_rate":
            (df["under"] > 0).sum() / len(df) * 100,
        "stability": stability
    },
    "instances": df["instances"].values
}