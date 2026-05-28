import numpy as np
from metrics import compute_metrics


def run_rolling_policy(df, config):
    df = df.copy()

    demand_col = config.get("demand_col", "task_arrivals")

    df["prediction"] = (
        df[demand_col]
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

    return {
        "metrics": compute_metrics(df, "instances", config),
        "instances": df["instances"].values
    }