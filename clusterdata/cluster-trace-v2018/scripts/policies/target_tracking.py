import numpy as np
from metrics import compute_metrics


def run_target_tracking_policy(df, config):
    df = df.copy()

    demand_col = config.get("demand_col", "task_arrivals")

    target_util = config.get("target_util", 0.7)
    cooldown = config.get("cooldown", 0)
    delay = config.get("delay", 1)

    instances = []
    last_scale_time = -cooldown
    pending = []

    current_instances = 0

    for t in range(len(df)):
        demand_t = df.loc[t, demand_col]

        for change in pending[:]:
            apply_time, new_x = change
            if t >= apply_time:
                current_instances = new_x
                pending.remove(change)

        desired = int(np.ceil(
            demand_t / (target_util * config["capacity_per_instance"])
        ))

        desired = min(desired, config["max_instances"])

        if t - last_scale_time >= cooldown:
            if desired > current_instances:
                new_x = min(current_instances + config["max_delta"], desired)
            else:
                new_x = max(current_instances - config["max_delta"], desired)

            if new_x != current_instances:
                pending.append((t + delay, new_x))
                last_scale_time = t

        instances.append(current_instances)

    df["instances"] = instances

    return {
        "metrics": compute_metrics(df, "instances", config),
        "instances": df["instances"].values
    }